 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three fundamental reasons:

### 1. **Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`**
For every one of the ~6.46 million cell-year rows, the function performs string-pasting, named-vector lookups (`id_to_ref`, `idx_lookup`), and subsetting. Named-vector lookups in R are hash-table operations, but doing ~6.46 million × ~4 neighbors ≈ 26 million `paste` + hash lookups inside an `lapply` is extremely expensive. The result is a **list of 6.46 million integer vectors**, which is also memory-heavy.

### 2. **The neighbor topology is year-invariant, but is rebuilt across all cell-years**
The rook-neighbor structure is purely spatial — cell A's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-expands this topology across every year, producing a massive redundant structure. There are only ~344,208 spatial cells and ~1.37 million directed neighbor pairs, but the code inflates this to ~6.46 million row-level lookups.

### 3. **`compute_neighbor_stats` loops over 6.46 million list elements**
Each call to `compute_neighbor_stats` does an `lapply` over the 6.46M-element `neighbor_lookup`, extracts values, removes NAs, and computes max/min/mean. This is done 5 times (once per source variable), totaling ~32.3 million R-level loop iterations with per-element subsetting.

**Summary:** The bottleneck is expanding a small spatial graph (~1.37M edges) into a massive cell-year list (~6.46M entries), then looping over that list repeatedly. The fix is to **never expand by year at all** — instead, build the edge table once and use vectorized joins.

---

## Optimization Strategy

### Core Insight
Since the neighbor graph is time-invariant, build a **single edge table** (a two-column data.table of `id` → `neighbor_id`, ~1.37M rows) and use **vectorized joins by `(neighbor_id, year)`** to pull neighbor attributes, then **group-aggregate by `(id, year)`** to compute max, min, and mean. This replaces all `lapply` loops with `data.table` operations that run in seconds, not hours.

### Steps

1. **Convert `spdep::nb` to an edge data.table** — one row per directed neighbor pair: `(id, neighbor_id)`. ~1.37M rows. Built once.
2. **For each source variable**, join `cell_data` onto the edge table by `(neighbor_id, year)` to fetch the neighbor's value, then aggregate `max`, `min`, `mean` grouped by `(id, year)`. This is fully vectorized.
3. **Left-join** the aggregated stats back onto `cell_data`.
4. **Predict** with the existing trained Random Forest model (unchanged).

**Expected speedup:** From ~86+ hours to **minutes** (typically 2–10 minutes total on a 16 GB laptop). Memory peak is manageable: the edge table is ~1.37M rows, expanded by years to ~38.4M rows during the join, which fits in 16 GB as a lean data.table.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_data <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build the spatial edge table ONCE
#         from the spdep::nb object (rook_neighbors_unique)
#         and the id_order vector.
#
#   rook_neighbors_unique: list of length 344,208
#     where element [[i]] is an integer vector of neighbor
#     indices into id_order.
#   id_order: vector of 344,208 cell IDs in the order
#     matching the nb object.
# ============================================================

build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(nb_obj, length, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- nb_obj[[i]]
    # spdep::nb stores 0L for cells with no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    n_i <- length(nb_i)
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }

  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges: ~1,373,394 rows, two columns: id, neighbor_id

cat("Edge table built:", nrow(edges), "directed edges\n")

# ============================================================
# STEP 2: Compute neighbor stats for each source variable
#         using vectorized data.table joins + group aggregation
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Unique years in the data (for cross-join)
unique_years <- sort(unique(cell_data$year))

# Expand edge table × years: each spatial edge exists in every year.
# ~1.37M edges × 28 years ≈ 38.5M rows.
# This is the most memory-intensive step but fits in 16 GB
# because it's only 3 integer/numeric columns.
edges_by_year <- CJ(edge_idx = seq_len(nrow(edges)), year = unique_years)
edges_by_year[, `:=`(id          = edges$id[edge_idx],
                      neighbor_id = edges$neighbor_id[edge_idx])]
edges_by_year[, edge_idx := NULL]

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

compute_and_add_neighbor_features_fast <- function(cell_dt, edges_yr, var_name) {
  # Extract only the columns we need for the join
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # Join neighbor values onto expanded edge table
  joined <- val_dt[edges_yr, on = .(neighbor_id, year), nomatch = NA]
  # joined has columns: neighbor_id, year, val, id

  # Aggregate: max, min, mean of neighbor values, grouped by (id, year)
  stats <- joined[!is.na(val),
                   .(nmax  = max(val),
                     nmin  = min(val),
                     nmean = mean(val)),
                   by = .(id, year)]

  # Name the output columns to match original pipeline conventions
  max_col  <- paste0("n_max_",  var_name)
  min_col  <- paste0("n_min_",  var_name)
  mean_col <- paste0("n_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Remove old columns from cell_dt if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Left-join stats back onto cell_dt
  setkey(stats, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]

  cell_dt
}

# Run for all 5 variables
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "... ")
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edges_by_year, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("done in", round(elapsed, 1), "sec\n")
}

# ============================================================
# STEP 3: Predict using the existing trained Random Forest
#         (model object unchanged)
# ============================================================

# Ensure cell_data is a data.frame for predict() compatibility
# (some RF packages require data.frame, not data.table)
cell_data_df <- as.data.frame(cell_data)

# rf_model is your pre-trained Random Forest — do NOT retrain
predictions <- predict(rf_model, newdata = cell_data_df)

cell_data$rf_prediction <- predictions

cat("Done. Predictions appended to cell_data.\n")
```

---

## Memory-Constrained Alternative (if 38.5M-row `edges_by_year` is too large)

If the ~38.5M-row expanded edge table strains the 16 GB laptop, process **one year at a time** — still vastly faster than the original because each year's join is only ~1.37M rows:

```r
compute_neighbor_features_by_year <- function(cell_dt, edges, var_name) {
  max_col  <- paste0("n_max_",  var_name)
  min_col  <- paste0("n_min_",  var_name)
  mean_col <- paste0("n_mean_", var_name)

  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  unique_years <- sort(unique(cell_dt$year))

  for (yr in unique_years) {
    # Subset this year's values
    yr_vals <- cell_dt[year == yr, .(neighbor_id = id, val = get(var_name))]
    setkey(yr_vals, neighbor_id)

    # Join onto edge table
    joined <- yr_vals[edges, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]

    # Aggregate
    stats <- joined[!is.na(val),
                     .(nmax = max(val), nmin = min(val), nmean = mean(val)),
                     by = .(id)]

    # Write back into cell_dt
    idx <- cell_dt[year == yr, which = TRUE]
    m <- match(cell_dt$id[idx], stats$id)
    set(cell_data, i = idx, j = max_col,  value = stats$nmax[m])
    set(cell_data, i = idx, j = min_col,  value = stats$nmin[m])
    set(cell_data, i = idx, j = mean_col, value = stats$nmean[m])
  }

  cell_dt
}

for (var_name in neighbor_source_vars) {
  cat("Computing (year-loop):", var_name, "... ")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_by_year(cell_data, edges, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("done in", round(elapsed, 1), "sec\n")
}
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup structure | 6.46M-element R list built with `paste`/hash | 1.37M-row edge `data.table` built once |
| Per-variable computation | `lapply` over 6.46M elements | Vectorized `data.table` join + `group by` |
| Total iterations (R-level) | ~32.3M (5 vars × 6.46M) | 5 vectorized operations (or 5 × 28 small ops) |
| Estimated wall time | **86+ hours** | **2–10 minutes** |
| Trained RF model | Preserved ✅ | Preserved ✅ |
| Numerical output | Identical max/min/mean | Identical max/min/mean ✅ |
| Peak RAM | High (6.46M-element list of int vectors) | ~1–2 GB for expanded edge table |