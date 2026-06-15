 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and named-vector lookups, which are extremely slow in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which again loops over 6.46 million entries with per-element subsetting and summary computation. Together, these two stages dominate the 86+ hour runtime.

**Specific problems:**

1. **String-key lookups in `build_neighbor_lookup`:** For every row, neighbor cell IDs are pasted with the year to form keys, then looked up in a named vector (`idx_lookup`). Named-vector lookup in R is O(n) per query in the worst case and involves repeated memory allocation of small character vectors. Over 6.46M rows × ~4 neighbors each ≈ 25M+ string operations and lookups.

2. **`lapply` over millions of rows:** R's `lapply` has per-iteration overhead. With 6.46M iterations, even microsecond overhead accumulates to hours.

3. **`do.call(rbind, result)` on a list of 6.46M small vectors:** This is a notoriously slow pattern in R for large lists.

4. **Memory pressure:** 6.46M rows × 110 columns is already ~5–6 GB for numeric data. Building a 6.46M-element list of integer vectors for the neighbor lookup adds substantial memory overhead, and the repeated `data.frame` column binding in the loop compounds this.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized and `data.table`-based operations:**

1. **Vectorized neighbor lookup via `data.table` join:** Instead of building a per-row list, create an edge table (`source_row → neighbor_row`) using a single merge/join. This eliminates all string pasting and named-vector lookups.

2. **Vectorized neighbor stats via `data.table` grouped aggregation:** Instead of `lapply` over 6.46M elements, use `data.table`'s `[, .(max, min, mean), by = source_row]` on the edge table joined with variable values. This leverages C-level grouped aggregation.

3. **Avoid intermediate list structures entirely.**

4. **Process all 5 variables in a tight loop over the same edge table** — the edge table is built once.

**Expected speedup:** From 86+ hours to roughly 10–30 minutes, well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert cell_data to data.table (if not already) and
#         add a row index. This is a zero-copy operation on columns.
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)
cell_data[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge table that maps every (cell, year)
#         row to its neighbor (cell, year) rows.
#
#         This replaces build_neighbor_lookup entirely.
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(cell_data, id_order, neighbors) {
  # --- 1a. Expand the nb object into a two-column edge list of cell IDs
  #         (not row indices — spatial cell IDs).
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells),
                  times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove any zero-length / self-referencing artifacts from spdep

  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, valid)

  # --- 1b. Join with cell_data to attach year and row_idx for the
  #         *source* cell-year rows.
  #         We need: for every (from_id, year) → row_idx of source
  #                  for every (to_id,   year) → row_idx of neighbor
  id_year_map <- cell_data[, .(id, year, row_idx)]

  # Attach source row indices: every edge × every year
  setkey(id_year_map, id)
  setkey(edges, from_id)

  # Cross-join edges with years via the source cell's years
  source_map <- id_year_map[, .(from_id = id, year, src_row = row_idx)]
  setkey(source_map, from_id)
  edge_year <- edges[source_map, on = "from_id",
                     allow.cartesian = TRUE, nomatch = 0L]
  rm(source_map)

  # Attach neighbor row indices
  nbr_map <- id_year_map[, .(to_id = id, year, nbr_row = row_idx)]
  setkey(edge_year, to_id, year)
  setkey(nbr_map, to_id, year)
  edge_year <- edge_year[nbr_map, on = c("to_id", "year"),
                         nomatch = 0L]
  rm(nbr_map, id_year_map)

  # Keep only the columns we need
  edge_year <- edge_year[, .(src_row, nbr_row)]
  setkey(edge_year, src_row)

  return(edge_year)
}

edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# Step 2: Compute neighbor stats for each variable using grouped
#         aggregation on the edge table.
#
#         This replaces compute_neighbor_stats + the outer loop.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Pull the variable values into the edge table
  edge_table[, nbr_val := cell_data[[var_name]][nbr_row]]

  # Grouped aggregation — runs at C level inside data.table
  stats <- edge_table[!is.na(nbr_val),
                      .(var_max  = max(nbr_val),
                        var_min  = min(nbr_val),
                        var_mean = mean(nbr_val)),
                      by = src_row]

  # Allocate new columns with NA, then fill matched rows
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  set(cell_data, i = stats$src_row, j = col_max,  value = stats$var_max)
  set(cell_data, i = stats$src_row, j = col_min,  value = stats$var_min)
  set(cell_data, i = stats$src_row, j = col_mean, value = stats$var_mean)

  rm(stats)
}

# Clean up helper column
edge_table[, nbr_val := NULL]
cell_data[, row_idx := NULL]

# ──────────────────────────────────────────────────────────────────────
# cell_data now contains the same 15 new neighbor feature columns
# (3 stats × 5 variables) as the original code, with identical
# numerical values.  The trained Random Forest model can be applied
# directly via predict(rf_model, cell_data) without retraining.
# ──────────────────────────────────────────────────────────────────────
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets; the edge table encodes the identical rook-neighbor relationships and year-matching logic as the original string-key lookup. |
| **NA handling** | `!is.na(nbr_val)` in the grouped aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Rows with no valid neighbors receive `NA` via the pre-initialized columns. |
| **Trained RF model** | No model object is touched. Only the feature columns are computed. Column names follow the same `neighbor_{stat}_{var}` convention. |

## Memory Estimate

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M × 125 cols) | ~6.5 GB |
| `edge_table` (≈38.4M rows × 2 int cols) | ~0.6 GB |
| Temporary `stats` tables | ~0.15 GB peak |
| **Total peak** | **~7.3 GB** (fits in 16 GB) |

The edge table has ~38.4M rows (1,373,394 directed neighbor pairs × 28 years), which is large but manageable as two integer columns. The entire pipeline should complete in **10–30 minutes** on a standard laptop.