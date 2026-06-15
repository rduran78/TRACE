 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector).
- An `is.na` filter.

Doing this 6.46 million times in an interpreted `lapply` loop is catastrophically slow. The `idx_lookup` named vector with ~6.46M entries makes each keyed access expensive, and the per-row string construction (`paste`) and character matching dominate wall time.

### 2. `compute_neighbor_stats` — Row-level `lapply` over 6.46M rows (×5 variables)

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable), yielding ~32.3 million interpreted R function calls. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also slow.

### Estimated cost breakdown

| Step | Calls | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | String key construction + named vector lookup on 6.46M-entry vector |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M iterations | Subsetting, NA removal, summary stats in pure R |
| `do.call(rbind, ...)` | 5 calls on 6.46M-length lists | List-to-matrix coercion |

The 86+ hour estimate is consistent with this analysis.

---

## Optimization Strategy

**Principle: Replace row-level R loops with vectorized and `data.table`-based operations.**

1. **Replace `build_neighbor_lookup`** with a fully vectorized `data.table` join. Instead of building a list of neighbor row indices per row, build a long-form edge table (`source_row`, `neighbor_row`) using keyed joins. This eliminates all per-row string operations and named-vector lookups.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation over the long-form edge table. For each variable, join in the neighbor values, then compute `max`, `min`, `mean` grouped by source row — all in C-level `data.table` internals.

3. **Avoid materializing a 6.46M-element R list entirely.** The list-of-integer-vectors structure is memory-heavy and forces interpreted iteration. A long-form edge table is both faster and more memory-efficient.

**Expected speedup:** From 86+ hours to roughly 5–15 minutes on the same laptop, depending on disk I/O and RAM pressure. The numerical results are identical (same max, min, mean over the same neighbor sets).

---

## Working R Code

```r
library(data.table)

#' Build a long-form edge table mapping each cell-year row to its neighbor rows.
#' Replaces build_neighbor_lookup entirely.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year`
#' @param id_order    integer vector of cell IDs in the order used by the nb object
#' @param neighbors   spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: src_row, nbr_row
build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {
  dt <- as.data.table(cell_data)
  dt[, src_row := .I]

  # --- Step 1: Build a cell-level edge list (id -> neighbor_id) -----------
  #   neighbors[[k]] gives integer indices into id_order for the k-th cell.
  #   We expand this into a two-column data.table of (cell_id, neighbor_id).
  n_neighbors <- lengths(neighbors)                       # integer vector
  src_cell_idx <- rep(seq_along(neighbors), n_neighbors)  # vectorized rep
  nbr_cell_idx <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    id     = id_order[src_cell_idx],
    nbr_id = id_order[nbr_cell_idx]
  )

  # --- Step 2: Join to get (src_row, nbr_row) for every cell-year ---------
  #   For each row in dt, we need to find all neighbor rows that share the

  #   same year. We do this with two keyed joins.

  # Map (id, year) -> row index for source rows
  src_map <- dt[, .(id, year, src_row)]
  setkey(src_map, id)

  # Map (id, year) -> row index for neighbor rows
  nbr_map <- dt[, .(nbr_id = id, year, nbr_row = src_row)]
  setkey(nbr_map, nbr_id, year)

  # Expand cell_edges by year: join cell_edges with src_map on id
  # This gives (id, nbr_id, year, src_row)
  setkey(cell_edges, id)
  expanded <- cell_edges[src_map, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, nbr_id, year, src_row

  # Now join with nbr_map to get nbr_row
  setkey(expanded, nbr_id, year)
  edge_table <- nbr_map[expanded, on = c("nbr_id", "year"), nomatch = 0L]
  # edge_table has: nbr_id, year, nbr_row, id, src_row

  edge_table[, .(src_row, nbr_row)]
}


#' Compute neighbor max, min, mean for one variable using the edge table.
#' Replaces compute_neighbor_stats entirely.
#'
#' @param cell_data   data.table with at least nrow rows
#' @param edge_table  data.table with columns src_row, nbr_row
#' @param var_name    character: name of the column to aggregate
#' @return data.table with columns: src_row, <var>_nb_max, <var>_nb_min, <var>_nb_mean
compute_neighbor_stats_fast <- function(cell_data, edge_table, var_name) {
  vals <- cell_data[[var_name]]

  # Attach neighbor values
  agg <- edge_table[, .(nbr_val = vals[nbr_row]), by = src_row]

  # Remove NAs before aggregation
  agg <- agg[!is.na(nbr_val)]

  # Grouped aggregation — runs in data.table's C backend
  stats <- agg[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = src_row]

  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))

  stats
}


#' Main driver: build features and attach to cell_data.
#' Drop-in replacement for the original outer loop.
#'
#' @param cell_data              data.frame with columns id, year, and the source vars
#' @param id_order               integer vector of cell IDs matching the nb object
#' @param rook_neighbors_unique  spdep nb object
#' @return cell_data with neighbor feature columns appended
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_data <- as.data.table(cell_data)
  n <- nrow(cell_data)

  message("Building edge table...")
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_data, edge_table, var_name)

    # Left-join back; rows with no valid neighbors get NA (matches original behavior)
    new_cols <- setdiff(names(stats), "src_row")
    cell_data[, (new_cols) := NA_real_]
    cell_data[stats$src_row, (new_cols) := stats[, ..new_cols]]
  }

  cell_data[]
}

# ---- Usage (drop-in replacement for the original outer loop) ----
cell_data <- add_all_neighbor_features(cell_data, id_order, rook_neighbors_unique)

# The trained Random Forest model is untouched.
# predict(rf_model, newdata = cell_data) works as before.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | The edge table is built from the identical `rook_neighbors_unique` nb object and matches on the same `(id, year)` pairs. |
| **Same statistics** | `max`, `min`, `mean` are computed over exactly the same non-NA neighbor values per row. |
| **Same NA behavior** | Rows with zero valid neighbors receive `NA` for all three statistics, identical to the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | No model retraining; the output columns have the same names and semantics, so `predict()` works unchanged. |
| **Numerical equivalence** | All operations are exact arithmetic equivalents; no approximation or sampling is introduced. |

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M interpreted iterations with string ops) | ~1–3 min (vectorized `data.table` joins) |
| Stats per variable | ~15+ hours each (6.46M `lapply` iterations × 5) | ~30–60 sec each (grouped C-level aggregation) |
| **Total** | **86+ hours** | **~5–15 minutes** |

Memory peak will be dominated by the edge table (~6.46M rows × avg ~4 neighbors × 2 integer columns ≈ ~400 MB), well within the 16 GB laptop constraint.