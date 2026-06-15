 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector — **O(1) amortized but with constant overhead from `as.character()` and named-vector hashing repeated 6.46M times**.
2. Constructs `paste(..., sep="_")` key strings for every neighbor of every row — **millions of string allocations**.
3. Indexes into `idx_lookup` (a named vector) with those string keys — **hash lookups with string keys, repeated for every neighbor of every row**.

With ~6.46M rows and an average of ~4 rook neighbors each, that's ~25.8M string constructions and hash lookups inside a serial `lapply`. The `compute_neighbor_stats` function is comparatively cheap (just numeric indexing), but `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also unnecessarily slow.

**Root causes ranked by impact:**
1. **Row-level `lapply` with per-element string operations** in `build_neighbor_lookup` (~6.46M iterations, each doing string paste + named-vector lookup).
2. **Rebuilding the lookup is unnecessary if only the variable changes** — the neighbor topology is time-invariant, so the lookup should be built once and reused (it is reused, which is good).
3. **`do.call(rbind, ...)` on millions of small vectors** in `compute_neighbor_stats` is slow; a preallocated matrix is better.
4. **`compute_neighbor_stats` uses an R-level loop** over 6.46M entries — could be vectorized via sparse matrix multiplication.

## Optimization Strategy

**Replace the string-key lookup approach with integer-indexed sparse-matrix arithmetic.** A sparse adjacency matrix `W` (6.46M × 6.46M) where entry `W[i,j] = 1` if row `j` is a spatial-temporal neighbor of row `i` allows computing neighbor means, maxes, and mins via vectorized operations. For the **mean**, it's a single sparse matrix-vector multiply. For **max** and **min**, we use grouped operations via `data.table`.

Key ideas:
- Build a mapping from `(id, year)` → row index using `data.table` (vectorized, no per-row string ops).
- Expand the neighbor list into an edge list `(from_row, to_row)` vectorially, grouped by year.
- Store as a sparse matrix for mean computation; use `data.table` grouped operations for max/min.
- This reduces runtime from ~86+ hours to minutes.

## Optimized R Code

```r
library(data.table)
library(Matrix)

#' Build a sparse neighbor adjacency matrix at the cell-year row level.
#' Each entry W[i,j] = 1 means row j is a rook neighbor of row i in the same year.
build_neighbor_sparse <- function(data, id_order, neighbors) {
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Integer mapping: cell id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # For each unique cell id, get its neighbor cell ids (topology is time-invariant)
  unique_ids <- unique(dt$id)
  edge_list_cell <- rbindlist(lapply(unique_ids, function(cid) {
    ref <- id_to_ref[as.character(cid)]
    nb_ids <- id_order[neighbors[[ref]]]
    nb_ids <- nb_ids[!is.na(nb_ids)]
    if (length(nb_ids) == 0L) return(NULL)
    data.table(from_id = cid, to_id = nb_ids)
  }))

  if (nrow(edge_list_cell) == 0L) {
    stop("No neighbor edges found.")
  }

  # Now expand to row-level edges by joining on year
  # For every (from_id, to_id) pair and every year, find (from_row, to_row)
  setkey(dt, id, year)

  # Join from_id -> row_idx for each year
  from_dt <- dt[, .(from_id = id, year, from_row = row_idx)]
  setkey(from_dt, from_id, year)

  to_dt <- dt[, .(to_id = id, year, to_row = row_idx)]
  setkey(to_dt, to_id, year)

  # Cross with edge_list_cell: for each edge (from_id, to_id), for each year,
  # look up from_row and to_row
  edges <- edge_list_cell
  setkey(edges, from_id)

  # Merge from side
  edges_yr <- merge(edges, from_dt, by = "from_id", allow.cartesian = TRUE)
  # edges_yr now has: from_id, to_id, year, from_row

  # Merge to side
  setkey(edges_yr, to_id, year)
  edges_yr <- merge(edges_yr, to_dt, by = c("to_id", "year"), nomatch = 0L)
  # edges_yr now has: to_id, from_id, year, from_row, to_row

  n <- nrow(dt)

  # Build sparse matrix (rows = from_row, cols = to_row)
  W <- sparseMatrix(
    i = edges_yr$from_row,
    j = edges_yr$to_row,
    x = 1,
    dims = c(n, n)
  )

  # Also return the edge table for max/min computation
  list(W = W, edges = edges_yr[, .(from_row, to_row)], n = n)
}

#' Compute neighbor mean, max, min for a variable using sparse matrix + data.table
compute_neighbor_stats_fast <- function(vals, nb_info) {
  W <- nb_info$W
  edges <- nb_info$edges
  n <- nb_info$n

  # --- MEAN via sparse matrix-vector multiply ---
  # Handle NAs: replace with 0 for sum, track counts of non-NA
  not_na <- as.numeric(!is.na(vals))
  vals_clean <- ifelse(is.na(vals), 0, vals)

  neighbor_sum   <- as.numeric(W %*% vals_clean)
  neighbor_count <- as.numeric(W %*% not_na)

  nb_mean <- ifelse(neighbor_count == 0, NA_real_, neighbor_sum / neighbor_count)

  # --- MAX and MIN via data.table grouped operations ---
  edge_dt <- copy(edges)
  edge_dt[, val := vals[to_row]]
  edge_dt <- edge_dt[!is.na(val)]

  max_dt <- edge_dt[, .(nb_max = max(val)), by = from_row]
  min_dt <- edge_dt[, .(nb_min = min(val)), by = from_row]

  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  nb_max[max_dt$from_row] <- max_dt$nb_max
  nb_min[min_dt$from_row] <- min_dt$nb_min

  data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

#' Main optimized pipeline: build features and append to cell_data
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
  message("Building sparse neighbor structure...")
  t0 <- proc.time()
  nb_info <- build_neighbor_sparse(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Done in %.1f seconds.", (proc.time() - t0)[3]))

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()

    stats <- compute_neighbor_stats_fast(cell_data[[var_name]], nb_info)

    cell_data[[paste0(var_name, "_nb_max")]]  <- stats$nb_max
    cell_data[[paste0(var_name, "_nb_min")]]  <- stats$nb_min
    cell_data[[paste0(var_name, "_nb_mean")]] <- stats$nb_mean

    message(sprintf("  Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  cell_data
}

# ---- USAGE (drop-in replacement for original outer loop) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# The trained Random Forest model is unchanged — just call predict() as before.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `mean`, `max`, `min` are computed over exactly the same neighbor sets (same rook topology, same year matching, same NA handling). The sparse matrix encodes the identical adjacency. |
| **Trained RF model** | No retraining. The code only reconstructs the same predictor columns with the same names and values. `predict(rf_model, ...)` works identically. |
| **Column naming** | Uses the same `_nb_max`, `_nb_min`, `_nb_mean` suffixes — adjust if your `compute_and_add_neighbor_features` uses different names. |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (6.46M string ops) | ~hours | Replaced by vectorized `data.table` merge: **~2–5 min** |
| `compute_neighbor_stats` per variable | ~hours (R-level loop) | Sparse mat-vec for mean (~seconds), `data.table` grouped max/min (~1–2 min each): **~3–5 min/var** |
| **Total for 5 variables** | **86+ hours** | **~20–35 minutes** |
| Peak RAM | Moderate | ~3–6 GB for sparse matrix + edge table (fits in 16 GB) |

The ~200–300× speedup comes from eliminating per-row string construction/hashing and replacing R-level loops with vectorized sparse linear algebra and `data.table` grouped aggregation.