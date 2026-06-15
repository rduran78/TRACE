 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) — for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** — each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of R's interpreted loop is enormous at this scale.
3. **Memory**: The resulting `neighbor_lookup` list of 6.46M integer vectors, plus the intermediate character vectors, can easily consume several GB.

Then `compute_neighbor_stats` does a second `lapply` over 6.46M elements — less expensive per iteration, but still slow in interpreted R.

**Root cause summary:**
- ~6.46M iterations of character-key construction and named-vector lookup → O(n²)-like behavior.
- Everything is done in interpreted R with no vectorization or hashing.
- The 86+ hour estimate is credible given the scale.

---

## Optimization Strategy

### 1. Replace character-key lookup with integer-key lookup via `data.table`

Use `data.table` to create a fast integer join between `(id, year)` and row indices. This replaces all `paste`/`setNames`/named-vector operations with O(1) hash-based lookups.

### 2. Vectorize the neighbor expansion

Instead of looping row-by-row, **expand all neighbor relationships into a single edge table** (a two-column data.frame of `(source_row, neighbor_row)`), then use grouped vectorized operations (`data.table` aggregation) to compute max, min, and mean in one pass per variable.

This turns the entire pipeline into:
- One `data.table` merge to map `(neighbor_cell_id, year)` → row index.
- One grouped aggregation per variable.

### 3. Memory estimate

The directed rook-neighbor edge list has ~1.37M spatial edges × 28 years ≈ **~38.5M rows** (two integer columns ≈ 308 MB). This fits comfortably in 16 GB.

### 4. Preserve the trained RF model and numerical estimand

The code only changes **how** neighbor features are computed, not **what** is computed. The max/min/mean values are identical, so the RF model's input features are unchanged. No retraining is needed.

---

## Working R Code

```r
library(data.table)

#' Build a vectorized edge table mapping each (cell, year) row to its
#' neighbor (cell, year) rows.  Replaces build_neighbor_lookup entirely.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year`
#'                    (and all predictor columns).
#' @param id_order    integer vector: the cell IDs in the order used by
#'                    the spdep::nb object (i.e., id_order[k] is the
#'                    cell-ID of the k-th element of rook_neighbors_unique).
#' @param neighbors   spdep::nb list (rook_neighbors_unique).  neighbors[[k]]
#'                    is an integer vector of indices into id_order.
#' @return A data.table with columns  src_row  and  nbr_row  (integer row
#'         indices into cell_data).

build_edge_table <- function(cell_data, id_order, neighbors) {

  ## ---- 1.  Build spatial edge list (cell-ID level) ----------------------
  n_cells <- length(id_order)
  # Pre-allocate: count total directed edges
  n_edges_spatial <- sum(lengths(neighbors))

  src_id <- integer(n_edges_spatial)
  nbr_id <- integer(n_edges_spatial)
  pos <- 1L
  for (k in seq_len(n_cells)) {
    nb <- neighbors[[k]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    len <- length(nb)
    idx <- pos:(pos + len - 1L)
    src_id[idx] <- id_order[k]
    nbr_id[idx] <- id_order[nb]
    pos <- pos + len
  }
  # Trim if any nb objects had 0-neighbor entries
  if (pos - 1L < n_edges_spatial) {
    src_id <- src_id[seq_len(pos - 1L)]
    nbr_id <- nbr_id[seq_len(pos - 1L)]
  }
  spatial_edges <- data.table(src_id = src_id, nbr_id = nbr_id)

  ## ---- 2.  Map (id, year) → row index -----------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # We only need id, year, row_idx for the join
  key_dt <- dt[, .(id, year, row_idx)]

  ## ---- 3.  Cross-join spatial edges with years ---------------------------
  years <- sort(unique(dt$year))
  edge_year <- spatial_edges[, CJ(year = years), by = .(src_id, nbr_id)]
  #   columns: src_id, nbr_id, year

  ## ---- 4.  Join to get row indices for source and neighbor ---------------
  setnames(key_dt, c("id", "year", "row_idx"),
                    c("src_id", "year", "src_row"))
  edge_year <- key_dt[edge_year, on = .(src_id, year), nomatch = 0L]

  setnames(key_dt, c("src_id", "year", "src_row"),
                    c("nbr_id", "year", "nbr_row"))
  edge_year <- key_dt[edge_year, on = .(nbr_id, year), nomatch = 0L]

  # Clean up: keep only the mapping columns

  edge_year <- edge_year[, .(src_row, nbr_row)]
  setkey(edge_year, src_row)

  return(edge_year)
}


#' Compute neighbor max, min, mean for one variable using the edge table,
#' and add the three new columns to cell_data (in place if data.table).
#'
#' @param cell_dt     data.table version of cell_data (with row order preserved).
#' @param edge_table  data.table from build_edge_table (src_row, nbr_row).
#' @param var_name    character: name of the source variable.
#' @return  cell_dt, invisibly (modified in place).

compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_table) {

  # Pull neighbor values via the edge table
  vals <- cell_dt[[var_name]]
  et   <- copy(edge_table)
  et[, nbr_val := vals[nbr_row]]

  # Drop NAs in neighbor values before aggregation

  et <- et[!is.na(nbr_val)]

  # Grouped aggregation — one pass
  agg <- et[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), keyby = src_row]

  # Initialize new columns with NA
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = col_max,  value = NA_real_)
  set(cell_dt, j = col_min,  value = NA_real_)
  set(cell_dt, j = col_mean, value = NA_real_)

  # Fill in computed values at the correct rows
  set(cell_dt, i = agg$src_row, j = col_max,  value = agg$nb_max)
  set(cell_dt, i = agg$src_row, j = col_min,  value = agg$nb_min)
  set(cell_dt, i = agg$src_row, j = col_mean, value = agg$nb_mean)

  invisible(cell_dt)
}


## =========================================================================
##  MAIN PIPELINE  (drop-in replacement for the original outer loop)
## =========================================================================

# Convert once to data.table (preserves row order)
cell_dt <- as.data.table(cell_data)

# Build the edge table — runs once (~30 s – 2 min on a laptop)
message("Building edge table …")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("  neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_table)
}

# Convert back to data.frame if downstream code expects one
cell_data <- as.data.frame(cell_dt)

message("Done — neighbor features added.
")
```

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (6.46M list elements via character keys) | ~60–80 hours | **Replaced** by `build_edge_table`: ~1–2 min |
| `compute_neighbor_stats` (6.46M `lapply` iterations × 5 vars) | ~6–10 hours | **Replaced** by `data.table` grouped aggregation: ~1–3 min per variable |
| **Total** | **86+ hours** | **~10–20 minutes** |

## Why the Estimand Is Preserved

The three statistics computed per `(cell, year)` row are:

```
max(neighbor_vals[!is.na(neighbor_vals)])
min(neighbor_vals[!is.na(neighbor_vals)])
mean(neighbor_vals[!is.na(neighbor_vals)])
```

The optimized code computes **exactly the same quantities** — the `data.table` aggregation after `et <- et[!is.na(nbr_val)]` is algebraically identical. Rows with zero non-NA neighbors receive `NA` in all three columns, matching the original `c(NA, NA, NA)` return. The trained Random Forest model receives identical input features and requires no retraining.