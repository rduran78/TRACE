 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times × ~4 neighbors each ≈ **25+ million hash lookups on a massive character vector**.

Character pasting and named-vector lookups at this scale are catastrophically slow in R. The resulting `neighbor_lookup` list itself consumes substantial memory (6.46M list elements, each an integer vector).

### 2. `compute_neighbor_stats` is called 5 times over 6.46M list elements
Each call iterates over 6.46M list entries, subsetting a numeric vector and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is slow due to R's per-element overhead and the final row-binding of 6.46M 3-element vectors.

### Core insight: The neighbor topology is **static across years**
The rook-neighbor relationships are purely spatial — cell A's neighbors are the same in 1992 as in 2019. The current code redundantly encodes this 28 times (once per year per cell). The entire lookup should be built **once at the cell level** (344K cells), and yearly attribute joins should be vectorized.

---

## Optimization Strategy

1. **Build a spatial adjacency edge-list once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-independent.

2. **For each variable, join yearly attributes onto the edge-list** using `data.table` keyed joins — this is vectorized C-level code, not R-level `lapply`.

3. **Aggregate neighbor stats with `data.table` grouped operations** — `[, .(max, min, mean), by = .(cell_id, year)]` runs in seconds on 1.37M × 28 ≈ 38M rows.

4. **Join aggregated stats back** to the main dataset.

This replaces ~6.46M R-level iterations with a handful of vectorized `data.table` joins and group-by aggregations. Expected runtime: **minutes, not days**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a static spatial edge-list ONCE
#
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns (id, neighbor_id), ~1.37M rows
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains positional indices of neighbors of id_order[i]
  n <- length(neighbors)
  from_idx <- rep(seq_len(n), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  # Remove zero-length / 0-coded "no neighbor" entries if present (spdep convention)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# ~1,373,394 rows — small and fast

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each neighbor source variable, compute neighbor stats
#         using vectorized data.table joins + grouped aggregation,
#         then attach results back to cell_data.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main data for fast joins
setkey(cell_data, id, year)

# Pre-expand edge list by year (all 28 years) — ~38.5M rows, but only 3 columns
# This is the "reusable neighbor table joined with year" concept.
years <- sort(unique(cell_data$year))
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year_dt[, `:=`(
  id          = edge_dt$id[edge_idx],
  neighbor_id = edge_dt$neighbor_id[edge_idx]
)]
edge_year_dt[, edge_idx := NULL]
setkey(edge_year_dt, neighbor_id, year)

# ~38.5M rows × 3 columns ≈ 0.9 GB — fits comfortably in 16 GB RAM

for (var_name in neighbor_source_vars) {

  message("Computing neighbor stats for: ", var_name)

  # Extract only the columns we need for the join
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join neighbor attribute values onto the edge-year table
  # For each (id, neighbor_id, year) row, get the neighbor's value
  edge_vals <- merge(
    edge_year_dt,
    attr_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  # Aggregate: for each (id, year), compute max/min/mean of neighbor values
  stats <- edge_vals[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Join stats back to cell_data
  setkey(stats, id, year)
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE, sort = FALSE)

  # Clean up per-iteration temporaries

  rm(attr_dt, edge_vals, stats)
}

# Re-sort to original order if needed
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the already-trained Random Forest
#         (model object unchanged — no retraining)
# ──────────────────────────────────────────────────────────────────────
# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original pipeline produced, with identical numerical values.
#
# Example (adjust to your actual model object and predict call):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of neighbor values per (cell, year) as the original code. The edge-list encodes the identical rook-neighbor topology; the `data.table` join retrieves the identical year-matched attribute values. |
| **Trained RF model** | The model object is never touched. Only the input feature columns are recomputed (identically), then passed to `predict()`. |
| **Column names** | Output columns follow the same `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` naming convention. |
| **NA handling** | `!is.na(value)` in the aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with no valid neighbors get `NA` via the `all.x = TRUE` merge, matching the original `return(c(NA, NA, NA))`. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| `build_edge_list` | < 1 second |
| `CJ` expansion (38.5M rows) | ~5–10 seconds |
| Per-variable merge + aggregate (×5) | ~30–90 seconds each |
| Total neighbor feature computation | **~3–8 minutes** |

This is a **~600–1700× speedup** over the original 86+ hour estimate, achieved entirely through vectorization and eliminating redundant per-row R-level iteration.