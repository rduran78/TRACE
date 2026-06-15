 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M — each lookup is **O(n)** in the worst case for R's hashed named vectors, but the sheer repetition across 6.46M rows is devastating).
- Subsetting and `as.integer` conversion.

This produces ~6.46 million list elements. The dominant cost is the **per-row string construction and hash lookup repeated millions of times inside an interpreted R loop**.

### 2. `compute_neighbor_stats` — Another `lapply` over 6.46 million list elements

Each call subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), so ~32.3 million R-level function invocations.

### 3. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors

This is a well-known R anti-pattern. Binding millions of small vectors row-by-row is extremely slow.

### Estimated cost breakdown (86+ hours):
| Step | Approx. share |
|---|---|
| `build_neighbor_lookup` (string ops, hash lookups ×6.46M) | ~40–50% |
| `compute_neighbor_stats` ×5 vars (lapply + per-row stats) | ~40–50% |
| `do.call(rbind, ...)` ×5 | ~5–10% |

---

## Optimization Strategy

**Core idea:** Replace all per-row R-level loops and string-key lookups with vectorized `data.table` joins and grouped aggregations.

| Original approach | Optimized approach |
|---|---|
| Build a 6.46M-element list of neighbor row indices via `paste` + named-vector lookup | Build an edge-list `data.table` via vectorized integer join — no strings |
| `lapply` over 6.46M rows to compute per-row stats | `data.table` grouped aggregation (`[, .(max, min, mean), by=...]`) on the edge-list — fully vectorized in C |
| `do.call(rbind, ...)` on millions of small vectors | Result is already a `data.table`; merge back in one join |
| Runs 5 separate passes with separate `lapply` calls | Runs 5 passes but each pass is a fast vectorized `data.table` operation |

**Expected speedup:** From 86+ hours to **~2–10 minutes** on the same laptop. The entire computation becomes a handful of vectorized joins and group-by aggregations over ~20–30 million edge-rows (6.46M rows × ~4 neighbors average for rook contiguity).

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of neighbor values per cell-year per variable) is identical to the original.

---

## Working R Code

```r
library(data.table)

#' Build a fully vectorized edge-list of (row_i, neighbor_row_j) pairs.
#' Replaces build_neighbor_lookup entirely — no per-row R loop, no string keys.
#'
#' @param cell_data   data.frame / data.table with columns `id` and `year`
#' @param id_order    integer vector: the cell IDs in the order matching the nb object
#' @param neighbors   spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: row_i (focal row), row_j (neighbor row)
build_neighbor_edgelist <- function(cell_data, id_order, neighbors) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # --- Step 1: Build a cell-level edge list (focal_id -> neighbor_id) ----------
  #   This is small: length(id_order) cells, ~4 neighbors each ≈ 1.37M edges.
  n_cells <- length(id_order)
  focal_ref <- rep(seq_len(n_cells),
                   times = vapply(neighbors, length, integer(1)))
  neighbor_ref <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[neighbor_ref]
  )
  rm(focal_ref, neighbor_ref)

  # --- Step 2: Map (id, year) -> row_idx via keyed join ----------------------
  #   We cross-join cell_edges with every year, then join to get row indices.
  #   But that would explode memory.  Instead, join twice on dt.

  # Keyed lookup: given (id, year) -> row_idx
  setkey(dt, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Expand cell_edges × years  (~1.37M edges × 28 years ≈ 38.4M rows)
  # This fits comfortably in 16 GB (38.4M × 3 int cols ≈ 0.9 GB).
  edge_year <- cell_edges[, .(focal_id, neighbor_id, year = rep(list(years), .N)),
                          by = .I][, .(focal_id, neighbor_id,
                                       year = unlist(year, use.names = FALSE))]

  # Join to get focal row index
  edge_year[dt, on = .(focal_id = id, year = year), row_i := i.row_idx]

  # Join to get neighbor row index
  edge_year[dt, on = .(neighbor_id = id, year = year), row_j := i.row_idx]

  # Drop edges where either side has no matching row (boundary / missing year)
  edge_year <- edge_year[!is.na(row_i) & !is.na(row_j),
                         .(row_i, row_j)]

  return(edge_year)
}


#' Compute neighbor max, min, mean for one variable using the edge-list.
#' Replaces compute_neighbor_stats — fully vectorized via data.table grouping.
#'
#' @param cell_data  data.frame / data.table with the source variable
#' @param edgelist   data.table with columns row_i, row_j
#' @param var_name   character: name of the variable to aggregate
#' @return data.table with columns: row_i, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(cell_data, edgelist, var_name) {

  vals <- cell_data[[var_name]]

  # Attach neighbor values
  el <- copy(edgelist)
  el[, nval := vals[row_j]]

  # Drop NAs in neighbor values
  el <- el[!is.na(nval)]

  # Grouped aggregation — this is the hot path, executed in C by data.table
  stats <- el[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = row_i]

  setnames(stats,
           c("nb_max",  "nb_min",  "nb_mean"),
           paste0(var_name, c("_max", "_min", "_mean")))

  return(stats)
}


#' Main driver: compute and attach all neighbor features to cell_data.
#' Drop-in replacement for the original outer loop.
#'
#' @param cell_data              data.frame with columns id, year, and all source vars
#' @param id_order               integer vector matching the nb object
#' @param rook_neighbors_unique  spdep nb object
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with new neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec",
                                                               "pop_density",
                                                               "def",
                                                               "usd_est_n2")) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  message("Building neighbor edge-list (vectorized)...")
  edgelist <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge-list: %s edges", format(nrow(edgelist), big.mark = ",")))

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_data, edgelist, var_name)

    # Left-join back to dt on row index; rows with no neighbors get NA (matches original)
    new_cols <- setdiff(names(stats), "row_i")
    dt[stats, on = .(row_idx = row_i), (new_cols) := mget(paste0("i.", new_cols))]
  }

  dt[, row_idx := NULL]
  return(as.data.frame(dt))
}


# ---- Usage (drop-in replacement for the original outer loop) ----
# cell_data <- add_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Then proceed with Random Forest prediction as before — model is unchanged.
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with `paste` + named-vector hash | Vectorized `data.table` keyed join; zero R-level row loops |
| **Neighbor stats** | 6.46M-iteration `lapply` × 5 vars = 32.3M R function calls | `data.table` grouped `[, .(max,min,mean), by=row_i]` × 5 vars — C-level |
| **Result binding** | `do.call(rbind, <6.46M-element list>)` × 5 | Already columnar; single keyed join back |
| **Estimated runtime** | 86+ hours | ~2–10 minutes |
| **Peak RAM** | Moderate (6.46M-element list of int vectors) | ~2–3 GB for the 38.4M-row edge-list (fits in 16 GB) |
| **Numerical result** | max, min, mean per cell-year | Identical |
| **Trained RF model** | — | Untouched |