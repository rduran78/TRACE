 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries per-call overhead. Doing this 6.46 million times with string allocation and matching is extremely slow. The result is a list of ~6.46M integer vectors, which also consumes substantial memory.

### 2. `compute_neighbor_stats` — repeated row-level `lapply` over the same 6.46M-element list
For each of the 5 source variables, another full pass over the 6.46M-element neighbor lookup list is made, each time subsetting a numeric vector and computing `max`, `min`, `mean`. That is ~32.3 million R-level function calls across all variables.

### Memory
The neighbor lookup list alone (6.46M elements, each a small integer vector) can consume several GB due to R's per-object overhead (~128 bytes minimum per SEXP). Combined with the 6.46M × 110 data frame, 16 GB RAM is tight.

### Summary of root causes
| Issue | Cause |
|---|---|
| Slow lookup build | Per-row string operations and named-vector hash lookups in pure R |
| Slow stats | Per-row `lapply` with R-level function calls × 5 variables |
| High memory | 6.46M-element R list of integer vectors; redundant copies |

---

## Optimization Strategy

**Core idea: Replace the row-level list with a flat sparse-matrix (CSR) representation and vectorize all neighbor statistics using sparse matrix–vector multiplication and grouped operations via `data.table`.**

### Step-by-step

1. **Flatten the neighbor lookup into a sparse adjacency matrix (cell-year × cell-year).** Instead of building a list of 6.46M elements, build a single `dgCMatrix` (compressed sparse column) or use two integer vectors (`i`, `j`) representing directed edges between row indices. This eliminates millions of small R objects.

2. **Build the adjacency only at the cell level (344K × 344K), then expand to cell-year via a join.** Since neighbors are time-invariant, the neighbor structure is identical across all 28 years. We build cell-level edges once, then join on year to get cell-year edges. This avoids the expensive per-row `paste`/lookup.

3. **Compute neighbor stats vectorially.** Using the flat edge list and `data.table` grouped aggregation (`max`, `min`, `mean` by target row), we process each variable in one vectorized pass — no `lapply` over 6.46M rows.

4. **Process variables sequentially** to limit peak memory.

**Expected improvement:**
- Build time: minutes instead of tens of hours.
- Stats computation per variable: seconds to low minutes.
- Memory: the edge list is ~2 × 1.37M × 28 ≈ 77M integers ≈ ~600 MB, manageable on 16 GB.
- The trained Random Forest model is untouched; the numerical outputs (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert cell_data to data.table (if not already) and create a
#     compact integer row-index.  Preserve original column order.
# ──────────────────────────────────────────────────────────────────────
orig_cols <- copy(names(cell_data))
setDT(cell_data)

# Absolute row position — we will write results back by this index.
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a CELL-level directed edge list from the nb object.
#     rook_neighbors_unique is a list of length 344,208; element k
#     contains the integer indices of the neighbors of cell k
#     (referring to positions in id_order).
#     id_order is the vector that maps position -> cell id.
# ──────────────────────────────────────────────────────────────────────
build_cell_edge_list <- function(id_order, neighbors) {
  # neighbors[[k]] gives neighbor positions for cell at position k.
  from_pos <- rep(seq_along(neighbors),
                  lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)

  # Map positions to actual cell ids.
  data.table(
    from_id = id_order[from_pos],
    to_id   = id_order[to_pos]
  )
}

cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
# cell_edges has ~1,373,394 rows (directed pairs).

# ──────────────────────────────────────────────────────────────────────
# 2.  Expand to cell-year edges by joining on every year present in
#     the data.  Because neighbors are time-invariant, each directed
#     cell pair appears once per year.
# ──────────────────────────────────────────────────────────────────────
# Thin index table: row_idx, id, year — needed for the join.
idx_table <- cell_data[, .(row_idx = .row_idx, id, year)]
setkey(idx_table, id, year)

# Attach the "from" row index (the focal / target row).
cell_edges_yr <- cell_edges[
  , .(from_id, to_id, year = rep(list(sort(unique(cell_data$year))),
                                  .N))
][, .(from_id, to_id, year = unlist(year)), by = .I][, I := NULL]

# --- more memory-friendly alternative (cross-join years once) --------
years_dt <- data.table(year = sort(unique(cell_data$year)))

# CJ-like expansion without materialising a huge intermediate:
cell_edges_yr <- cell_edges[, .(from_id, to_id)]
cell_edges_yr <- cell_edges_yr[
  rep(seq_len(.N), each = nrow(years_dt))
]
cell_edges_yr[, year := rep(years_dt$year, times = nrow(cell_edges))]

# Join to get the ROW INDEX of the "from" (target) row:
setkey(cell_edges_yr, from_id, year)
setkey(idx_table, id, year)
cell_edges_yr[idx_table, target_row := i.row_idx,
              on = .(from_id = id, year)]

# Join to get the ROW INDEX of the "to" (neighbor) row:
cell_edges_yr[idx_table, neighbor_row := i.row_idx,
              on = .(to_id = id, year)]

# Drop edges where either side is missing (cell-year not in panel).
cell_edges_yr <- cell_edges_yr[!is.na(target_row) & !is.na(neighbor_row)]

# Keep only what we need — free memory.
edge_dt <- cell_edges_yr[, .(target_row, neighbor_row)]
rm(cell_edges_yr, cell_edges, idx_table, years_dt)
gc()

# ──────────────────────────────────────────────────────────────────────
# 3.  Vectorised neighbor statistics for each source variable.
#     For every target row, compute max / min / mean of the variable
#     across its neighbors (excluding NAs), exactly replicating the
#     original compute_neighbor_stats logic.
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_data,
                                                   var_name,
                                                   edge_dt) {
  vals <- cell_data[[var_name]]

  # Attach the neighbor's value to every edge.
  work <- copy(edge_dt)
  work[, nval := vals[neighbor_row]]

  # Remove edges where the neighbor value is NA (matches original logic).
  work <- work[!is.na(nval)]

  # Grouped aggregation — one pass.
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = target_row]

  # Initialise result columns to NA (matches original: rows with no

  # valid neighbors get NA).
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Write aggregated values into the correct rows.
  set(cell_data, i = agg$target_row, j = max_col,  value = agg$nb_max)
  set(cell_data, i = agg$target_row, j = min_col,  value = agg$nb_min)
  set(cell_data, i = agg$target_row, j = mean_col, value = agg$nb_mean)

  invisible(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Outer loop — identical variable list, same column names produced.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  gc()                       
  cat("Done:", var_name, "\n")
}

# Clean up helper column.
cell_data[, .row_idx := NULL]
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per target row. The grouping key (`target_row`) is a 1:1 map to the original row. No approximation is introduced. |
| **Trained RF model** | No model code is touched. Only the feature-engineering step is optimized. The resulting columns have the same names and identical values, so `predict(rf_model, cell_data)` produces the same predictions. |
| **NA handling** | Rows with zero valid neighbors receive `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M `lapply` + string ops) | ~1–3 min (vectorized integer expansion + `data.table` keyed join) |
| Stats per variable | ~10+ hours (6.46M `lapply`) | ~30–90 sec (`data.table` grouped agg on ~38M edge rows) |
| **Total (5 vars)** | **86+ hours** | **~10–15 minutes** |
| Peak RAM | >16 GB (list overhead) | ~6–10 GB (flat integer edge table + one working copy) |

The optimization replaces millions of per-row R function calls and small-object allocations with bulk vectorized operations in `data.table`, which delegates to highly optimized C routines internally. This brings the workload comfortably within the capacity of a 16 GB laptop.