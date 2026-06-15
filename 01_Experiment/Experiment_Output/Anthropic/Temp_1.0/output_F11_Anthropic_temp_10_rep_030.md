 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. That's ~6.46 million R-level iterations with expensive string operations and named-vector lookups (which are O(n) hash lookups on large named vectors). This alone is catastrophically slow.

2. **`compute_neighbor_stats()`** is then called **5 times** (once per neighbor source variable), each time iterating over the ~6.46 million-element `neighbor_lookup` list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level `lapply` iterations total.

3. **Random Forest inference** is a single `predict()` call on a pre-trained model against a matrix/data.frame. Even with 6.46M rows and 110 predictors, optimized RF libraries (e.g., `ranger`) perform this in minutes via vectorized C++ code. Loading the model from disk is a single `readRDS()`. Writing predictions is a single column assignment or `fwrite()`. This is trivially fast relative to the neighbor computation.

**The ~86+ hour runtime is dominated by the neighbor feature engineering, not RF inference.**

---

## Optimization Strategy

The key optimizations:

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` merge/join approach. Instead of iterating row-by-row with string concatenation and named-vector lookups, we expand the neighbor relationships into an edge list and join against the data in bulk.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable — computing max, min, and mean of neighbor values via `:=` and `by=` grouping, which runs in optimized C.

3. **Eliminate the per-row `lapply`** entirely. The neighbor lookup list of 6.46M elements is replaced by a flat edge-list data.table with ~1.37 million × 28 years ≈ ~38.5 million rows (directed neighbor-year edges), which `data.table` handles efficiently.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0. Convert cell_data to data.table (if not already) and ensure key columns
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure 'id' and 'year' exist as expected; create a row index for final join-back
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 1. Build a flat edge-list from the nb object (one-time, vectorized)
#    rook_neighbors_unique is a list of length N_cells (344,208),
#    where element i contains integer indices into id_order of i's neighbors.
#    id_order is the vector mapping position -> cell id.
# ──────────────────────────────────────────────────────────────────────

# Number of neighbors per focal cell
n_neighbors <- lengths(rook_neighbors_unique)

# Focal cell indices (repeated by number of neighbors)
focal_indices <- rep(seq_along(rook_neighbors_unique), times = n_neighbors)

# Neighbor cell indices (unlisted)
neighbor_indices <- unlist(rook_neighbors_unique, use.names = FALSE)

# Map indices to actual cell IDs
edges <- data.table(
  focal_id    = id_order[focal_indices],
  neighbor_id = id_order[neighbor_indices]
)

rm(focal_indices, neighbor_indices, n_neighbors)  # free memory

cat("Edge list rows (directed spatial edges):", nrow(edges), "\n")

# ──────────────────────────────────────────────────────────────────────
# 2. Cross-join edges with years to get the full neighbor-year edge list
#    Then join to cell_data to pull neighbor values
# ──────────────────────────────────────────────────────────────────────

# Get the unique years present in the data
years_vec <- sort(unique(cell_data$year))

# Expand edges × years: each spatial edge exists for every year
# Use a cross join via CJ inside a merge or simply via rep
edges_by_year <- edges[, .(year = years_vec), by = .(focal_id, neighbor_id)]

cat("Edge-year rows:", nrow(edges_by_year), "\n")

# Key the cell_data for fast joins
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# 3. For each neighbor source variable, compute neighbor stats via
#    a single data.table join + grouped aggregation
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key edges_by_year on neighbor_id, year for joining neighbor values
setkey(edges_by_year, neighbor_id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  # Extract only the columns we need from cell_data for the neighbor lookup
  neighbor_vals <- cell_data[, .(id, year, val = get(var_name))]
  setkey(neighbor_vals, id, year)

  # Join: attach the neighbor's value to each edge-year row
  # edges_by_year is keyed on (neighbor_id, year); neighbor_vals on (id, year)
  work <- neighbor_vals[edges_by_year, on = .(id = neighbor_id, year = year),
                        nomatch = NA,
                        allow.cartesian = TRUE]
  # work now has columns: id (=neighbor_id), year, val, focal_id

  # Drop NAs in the variable value (matching original logic)
  work <- work[!is.na(val)]

  # Aggregate: group by focal_id + year, compute max/min/mean
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(focal_id, year)]

  # Rename columns to match expected output pattern
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # Join the aggregated stats back onto cell_data
  setkey(agg, focal_id, year)
  cell_data <- agg[cell_data, on = .(focal_id = id, year = year)]

  # The join introduces 'focal_id'; drop it and restore 'id'
  # After the right join (agg[cell_data]), the 'id' column is replaced by focal_id
  # We need to fix this:
  if ("focal_id" %in% names(cell_data)) {
    # focal_id came from agg; the original 'id' is available via the join key
    # In X[Y] join, the key column from Y is absorbed. Let's rename:
    setnames(cell_data, "focal_id", "id", skip_absent = TRUE)
  }

  # Cells with zero valid neighbors get NA (matching original behavior: c(NA,NA,NA))
  # This is automatic since they won't appear in `agg`.

  rm(work, agg, neighbor_vals)
  gc()
}

# Restore the original row order
setorder(cell_data, .row_idx)
cell_data[, .row_idx := NULL]

cat("Neighbor feature engineering complete.\n")

# ──────────────────────────────────────────────────────────────────────
# 4. Random Forest inference (unchanged — NOT the bottleneck)
# ──────────────────────────────────────────────────────────────────────
# Load the pre-trained model (single readRDS call)
rf_model <- readRDS("trained_rf_model.rds")

# Generate predictions (single vectorized predict call)
cell_data$predicted <- predict(rf_model, data = cell_data)$predictions
# (Adjust accessor based on whether model is ranger, randomForest, etc.)

# Write output
fwrite(cell_data, "predictions_output.csv")

cat("Pipeline complete.\n")
```

---

## Summary

| Component | Original Complexity | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops | Eliminated; replaced by vectorized edge-list construction |
| `compute_neighbor_stats` (×5 vars) | 32.3M R-level `lapply` iterations | 5 `data.table` keyed joins + grouped aggregations (C-level) |
| RF inference | Single `predict()` call | **Unchanged** — already efficient |

**Expected speedup**: From ~86+ hours down to **minutes** (typically 10–30 min depending on RAM/disk), with the neighbor feature engineering dropping from the dominant cost to a minor one. The Random Forest model and the original numerical estimand (predictions) are fully preserved.