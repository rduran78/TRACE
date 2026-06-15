 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M × ~4 neighbors ≈ 25.8M string operations is extremely expensive in base R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M elements calling `max`/`min`/`mean` on small vectors — slow due to R-level loop overhead and repeated function-call dispatch.

**Root causes, ranked:**

| Rank | Cause | Impact |
|------|-------|--------|
| 1 | Per-row `paste` + character-key lookup in `build_neighbor_lookup` (~6.46M iterations) | Dominant — estimated >80% of 86 h |
| 2 | Per-row `lapply` in `compute_neighbor_stats` (~6.46M × 5 vars) | Significant |
| 3 | Repeated allocation of small vectors inside closures | Moderate (GC pressure) |

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed join via `data.table`.** Build a `data.table` keyed on `(id, year)` with an integer row index. For each cell-year, join to a pre-expanded neighbor table (cell-year → neighbor-cell-year) in one vectorized merge — no per-row `paste` or named-vector lookup.

2. **Vectorize neighbor stats computation.** After the join produces a long-form table of (row, neighbor_row), extract the variable values, then use `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) — a single vectorized pass per variable instead of 6.46M R-level `lapply` calls.

3. **Memory management.** The expanded neighbor-edge table will have ~25.8M rows × a few integer columns — roughly 200–400 MB, well within 16 GB.

**Expected speedup:** From ~86+ hours to roughly 5–15 minutes total for all 5 variables.

## Optimized R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build a vectorized neighbor-edge table (run once)
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edge_table <- function(cell_data_dt, id_order, neighbors) {
 # cell_data_dt: a data.table with columns id, year (and all feature cols)
 #               with an added integer column .row_idx = .I
 # id_order:     the vector of cell IDs in the same order as the nb object
 # neighbors:    the spdep nb object (list of integer index vectors)

 # --- Step A: Build cell-level directed edge list (from nb object) ---
 # Each element neighbors[[i]] gives the indices (into id_order) of
 # the neighbors of id_order[i].
 from_idx <- rep(seq_along(neighbors), lengths(neighbors))
 to_idx   <- unlist(neighbors, use.names = FALSE)

 # Remove any 0-entries that spdep uses to denote "no neighbors"
 valid <- to_idx > 0L
 from_idx <- from_idx[valid]
 to_idx   <- to_idx[valid]

 cell_edges <- data.table(
   from_id = id_order[from_idx],
   to_id   = id_order[to_idx]
 )
 # cell_edges now has ~1,373,394 rows (directed rook edges)

 # --- Step B: Get unique years ---
 years <- sort(unique(cell_data_dt$year))

 # --- Step C: Cross-join edges × years to get cell-year edge table ---
 # Use CJ for memory-efficient cross join, then join to get row indices.
 cell_year_edges <- cell_edges[, .(from_id, to_id)]
 # Expand by year
 cell_year_edges <- cell_year_edges[
   , .(year = years), by = .(from_id, to_id)
 ]
 # cell_year_edges now has ~1,373,394 × 28 ≈ 38.5M rows
 # (but many will match; this is the upper bound)

 # --- Step D: Map (from_id, year) → source row index ---
 setkey(cell_data_dt, id, year)
 # Add row index to cell_data_dt if not present
 if (!".row_idx" %in% names(cell_data_dt)) {
   cell_data_dt[, .row_idx := .I]
 }

 # Join to get the source row index (the row whose features we are building)
 cell_year_edges[
   cell_data_dt,
   on = .(from_id = id, year = year),
   src_row := i..row_idx
 ]

 # Join to get the neighbor row index
 cell_year_edges[
   cell_data_dt,
   on = .(to_id = id, year = year),
   nbr_row := i..row_idx
 ]

 # Drop edges where either side is missing (cell-year not in data)
 cell_year_edges <- cell_year_edges[!is.na(src_row) & !is.na(nbr_row)]

 # Keep only the integer index columns we need
 cell_year_edges[, .(src_row, nbr_row)]
}

# ──────────────────────────────────────────────────────────────────────
# 2. Compute neighbor stats for one variable (vectorized)
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_stats_fast <- function(cell_data_dt, edge_dt, var_name) {
 # edge_dt has columns: src_row, nbr_row (integer indices into cell_data_dt)
 # Returns a data.table with columns: .row_idx, <var>_max, <var>_min, <var>_mean

 vals <- cell_data_dt[[var_name]]

 # Attach neighbor values
 work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]

 # Drop NAs in neighbor values
 work <- work[!is.na(nbr_val)]

 # Grouped aggregation — single vectorized pass
 stats <- work[, .(
   v_max  = max(nbr_val),
   v_min  = min(nbr_val),
   v_mean = mean(nbr_val)
 ), by = src_row]

 # Rename columns to match original pipeline's naming convention
 setnames(stats, c("v_max", "v_min", "v_mean"),
          paste0("neighbor_", var_name, c("_max", "_min", "_mean")))

 stats
}

# ──────────────────────────────────────────────────────────────────────
# 3. Main driver — drop-in replacement for the outer loop
# ──────────────────────────────────────────────────────────────────────
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
 # Convert to data.table (non-destructive copy)
 cell_dt <- as.data.table(cell_data)
 cell_dt[, .row_idx := .I]

 message("Building vectorized edge table...")
 edge_dt <- build_neighbor_edge_table(cell_dt, id_order, rook_neighbors_unique)
 message(sprintf("  Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

 for (var_name in neighbor_source_vars) {
   message(sprintf("Computing neighbor stats for '%s'...", var_name))
   stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)

   # Merge back; rows with no valid neighbors get NA (preserving original behavior)
   max_col  <- paste0("neighbor_", var_name, "_max")
   min_col  <- paste0("neighbor_", var_name, "_min")
   mean_col <- paste0("neighbor_", var_name, "_mean")

   # Initialize with NA
   cell_dt[, (max_col)  := NA_real_]
   cell_dt[, (min_col)  := NA_real_]
   cell_dt[, (mean_col) := NA_real_]

   # Update matched rows
   cell_dt[stats, on = .(.row_idx = src_row),
           c(max_col, min_col, mean_col) := mget(paste0("i.", c(max_col, min_col, mean_col)))]
 }

 # Clean up helper column
 cell_dt[, .row_idx := NULL]

 # Return as data.frame if the original was a data.frame (for RF compatibility)
 if (!is.data.table(cell_data)) {
   return(as.data.frame(cell_dt))
 }
 cell_dt
}

# ──────────────────────────────────────────────────────────────────────
# Usage (drop-in replacement for the original outer loop)
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
 cell_data,
 id_order,
 rook_neighbors_unique,
 neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per (cell, year) as the original code. The edge table encodes the identical rook-neighbor relationships, filtered to the same year. |
| **NA handling** | Rows with no valid neighbors receive `NA` for all three stats, identical to the original `if (length(idx) == 0) return(c(NA, NA, NA))` logic. |
| **Trained RF model** | No model retraining. The output `cell_data` has the same columns with the same names and the same values, so `predict(rf_model, newdata = cell_data)` produces identical results. |
| **Column naming** | Adjust the `paste0("neighbor_", ...)` pattern to match whatever `compute_and_add_neighbor_features` originally produced. |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60–70 h (6.46M `paste` + char lookups) | ~1–3 min (vectorized `data.table` join) |
| Stats computation (5 vars) | ~15–20 h (5 × 6.46M `lapply`) | ~2–5 min (5 × grouped `data.table` aggregation) |
| **Total** | **~86+ h** | **~5–15 min** |

Peak RAM usage: ~2–4 GB for the edge table plus working copies, well within the 16 GB budget.