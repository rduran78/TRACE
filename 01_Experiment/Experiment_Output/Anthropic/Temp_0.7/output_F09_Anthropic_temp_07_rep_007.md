 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is an O(N) `lapply` over ~6.46 million rows
For every single cell-year row, the function:
- Looks up the cell's spatial reference index.
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes together character keys (`"cellid_year"`) for every neighbor × every year.
- Performs named-vector lookups (which are hash-based but still slow at scale with millions of names).

This means ~6.46M iterations, each doing string concatenation and named-vector indexing. The entire lookup table is **year-redundant**: the neighbor *structure* is purely spatial (rook contiguity doesn't change across years), yet the code rebuilds neighbor-row mappings for every cell-year combination, duplicating the spatial logic 28 times.

### 2. `compute_neighbor_stats` is an O(N) `lapply` over ~6.46 million rows
Each iteration subsets a numeric vector by index, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is notoriously slow in R due to memory allocation overhead.

### Root cause summary
The pipeline treats a **spatial** problem as a **panel** problem. Neighbor relationships are invariant across years, so the correct approach is:

> **Build the adjacency table once (344K cells), then join yearly attributes and compute grouped statistics using vectorized/data.table operations — never loop over 6.46M rows.**

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| **A** | Build a two-column `data.table` of directed edges: `(id, neighbor_id)` — ~1.37M rows. Do this **once**. | Spatial structure is time-invariant. |
| **B** | For each year, join cell attributes onto the edge table by `(neighbor_id, year)`, then compute `max`, `min`, `mean` grouped by `(id, year)` using `data.table`. | Vectorized grouped aggregation; no R-level loops. |
| **C** | Join the resulting neighbor-stats columns back onto the main panel `data.table`. | Preserves the original data structure for the pre-trained Random Forest. |

**Expected speedup:** The 6.46M-row `lapply` is replaced by a `data.table` grouped join-and-aggregate over ~1.37M edges × 28 years ≈ 38.5M edge-year rows, which `data.table` handles in seconds to low minutes. Total runtime should drop from **86+ hours to under 5 minutes** on a standard laptop.

**Numerical equivalence:** `max`, `min`, and `mean` are computed on exactly the same neighbor sets with the same NA-removal logic, so the trained Random Forest receives identical inputs.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP A: Build the spatial edge table ONCE (time-invariant)
# ===========================================================================
# rook_neighbors_unique : spdep nb object (list of integer vectors), length = 344,208
# id_order              : vector of cell IDs aligned with the nb object

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total directed edges
  n_edges <- sum(lengths(neighbors))  # ~1,373,394
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"; skip those
    nb_i <- nb_i[nb_i != 0L]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    idx <- pos:(pos + n_i - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges among %d cells\n",
            nrow(edge_dt), length(id_order)))

# ===========================================================================
# STEP B: Convert the panel to data.table (if not already)
# ===========================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ===========================================================================
# STEP C: Compute neighbor stats for each source variable — vectorized
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will expand the edge table by year, join neighbor attributes, and
# aggregate.  To keep peak RAM manageable (~16 GB laptop), we process
# one variable at a time and can even chunk by year if needed.

# Create a minimal lookup keyed by (id, year) for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  
  cat(sprintf("Computing neighbor stats for: %s ...\n", var_name))
  
  # --- Minimal attribute table: only the column we need ----
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  
  # --- Cross join edges × years, then join neighbor values ----
  # Instead of a full cross join (which would be 38.5M rows at once),
  # we can do this efficiently: for every year, join edges to attributes.
  
  years <- sort(unique(cell_data$year))
  
  stats_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    # Neighbor values for this year
    attr_yr <- attr_dt[year == yr, .(neighbor_id, value)]
    setkey(attr_yr, neighbor_id)
    
    # Join edge table to neighbor values
    merged <- attr_yr[edge_dt, on = .(neighbor_id), nomatch = NA,
                      allow.cartesian = TRUE]
    # merged has columns: neighbor_id, value, id
    # Aggregate by focal cell (id)
    agg <- merged[!is.na(value),
                  .(nb_max  = max(value),
                    nb_min  = min(value),
                    nb_mean = mean(value)),
                  by = .(id)]
    agg[, year := yr]
    stats_list[[j]] <- agg
  }
  
  stats_dt <- rbindlist(stats_list, use.names = TRUE)
  
  # Name columns to match original pipeline conventions
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats_dt, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # --- Remove old columns if they exist (idempotent re-runs) ----
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # --- Join back to the main panel ----
  setkey(stats_dt, id, year)
  cell_data <- stats_dt[cell_data, on = .(id, year)]
  # Cells with no neighbors (or all-NA neighbors) get NA — same as original
  
  setkey(cell_data, id, year)
  
  cat(sprintf("  Done. Columns added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# ===========================================================================
# STEP D: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# The trained model object (e.g., `rf_model`) and its expected feature names
# are unchanged.  cell_data now contains the same neighbor_max_*, neighbor_min_*,
# neighbor_mean_* columns with numerically identical values.

# Example (adjust to your actual model object and prediction call):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete. Ready for Random Forest prediction.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor sets** | `edge_dt` is built from the same `rook_neighbors_unique` nb object using the same `id_order` mapping. Every directed edge is identical. |
| **Same NA handling** | The `!is.na(value)` filter in the aggregation mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors receive `NA` for all three stats (the left join leaves them as `NA`). |
| **Same aggregation functions** | `max`, `min`, `mean` in `data.table` are numerically identical to base R's `max`, `min`, `mean`. |
| **Trained RF untouched** | No retraining occurs; the model object is used as-is for `predict()`. |

## Expected Performance

| Stage | Rows/Operations | Estimated Time |
|-------|----------------|----------------|
| Build edge table | 1.37M edges | < 1 second |
| Per-variable, per-year join + aggregate | ~1.37M edges × 28 years × 5 vars | ~1–3 minutes total |
| Join back to panel | 6.46M rows × 5 vars | ~30 seconds |
| **Total** | | **~2–5 minutes** |

This represents a speedup of roughly **1,000×–2,500×** over the original 86+ hour estimate, well within the 16 GB RAM budget (peak usage ~2–3 GB for the edge-year intermediates).