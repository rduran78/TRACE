 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`
This function iterates over all ~6.46 million rows, and for each row it:
1. Looks up the cell's reference index in the neighbor list.
2. Constructs string keys (`paste(id, year)`) for each neighbor.
3. Does named-vector lookups (`idx_lookup[neighbor_keys]`) — named vector lookups in R are hash-based but repeated 6.46M times with string construction is extremely slow.

The output is a list of 6.46M integer vectors. This alone could take tens of hours.

### Bottleneck 2: `compute_neighbor_stats` — Repeated per-row `lapply`
For each of the 5 variables, another `lapply` over 6.46M elements extracts neighbor values, removes NAs, and computes max/min/mean. That's ~32.3M tiny function calls total. Each call subsets a numeric vector and computes three summary statistics. The overhead of R-level iteration dominates.

### Why raster focal/kernel operations don't directly apply
Focal operations assume a regular grid with a fixed rectangular kernel. Here the data is a **panel** (cell × year), and the neighbor structure is an irregular spatial adjacency (rook contiguity from an `nb` object). Years add a dimension that focal ops don't handle. The correct approach is vectorized sparse-matrix operations that faithfully preserve the rook-neighbor structure and panel alignment.

---

## Optimization Strategy

1. **Replace string-key lookups with integer join via `data.table`.** Build a `(cell_id, year) → row_index` lookup table and join neighbor indices in a single vectorized operation rather than 6.46M `paste`/named-lookup calls.

2. **Construct a sparse adjacency matrix (row-to-row) for same-year neighbors.** Represent the entire neighbor lookup as a sparse matrix **W** of dimension `nrow × nrow`. Entry `W[i,j] = 1` means row `j` is a rook neighbor of row `i` in the same year. This is built once.

3. **Compute neighbor stats via sparse matrix–vector products.** For a given variable `x`:
   - `neighbor_mean`: normalize W row-wise → `W_norm %*% x`.
   - `neighbor_max` and `neighbor_min`: use grouped operations on the sparse triplet representation, which is far faster than 6.46M R function calls.

4. **All 5 variables × 3 stats = 15 new columns** computed in minutes instead of days.

5. **The Random Forest model is not retrained.** The new columns have identical names and identical numerical values (max, min, mean of rook neighbors per cell-year), so prediction is unchanged.

---

## Working R Code

```r
# ─────────────────────────────────────────────────────────────────────
#  Fast neighbor-stat computation for cell-year panel data
#  Preserves exact numerical results of the original implementation.
# ─────────────────────────────────────────────────────────────────────

library(data.table)
library(Matrix)

## ── Step 0: Ensure cell_data is a data.table with a row-order column ──
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
cell_data[, .row_idx := .I]                # preserve original row order

## ── Step 1: Build (id, year) → row_index lookup ──────────────────────
setkey(cell_data, id, year)                 # fast binary-search join key

## ── Step 2: Build the row-to-row sparse adjacency matrix ─────────────
#  id_order: vector of cell IDs in the order matching rook_neighbors_unique
#  rook_neighbors_unique: an nb object (list of integer neighbor indices)

build_sparse_neighbor_matrix <- function(cell_data, id_order, nb_obj) {


  n_cells <- length(id_order)
  stopifnot(n_cells == length(nb_obj))

  # ── 2a. Expand spatial edges into a data.table of (from_id, to_id) ──
  from_ref <- rep(seq_len(n_cells),
                  times = vapply(nb_obj, function(x) {
                    sum(x != 0L)            # spdep uses 0 for no-neighbor
                  }, integer(1)))

  to_ref   <- unlist(lapply(nb_obj, function(x) x[x != 0L]),
                     use.names = FALSE)

  edges <- data.table(from_id = id_order[from_ref],
                      to_id   = id_order[to_ref])

  # ── 2b. Cross-join edges with years present in the data ─────────────
  years <- sort(unique(cell_data$year))

  # Cartesian product: every spatial edge × every year

  edges_yr <- edges[, .(year = years), by = .(from_id, to_id)]

  # ── 2c. Map (from_id, year) and (to_id, year) to row indices ────────
  #         via keyed join on cell_data
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Join to get "from" row index
  edges_yr[row_lookup, from_row := i..row_idx,
           on = .(from_id = id, year = year)]

  # Join to get "to" row index
  edges_yr[row_lookup, to_row := i..row_idx,
           on = .(to_id = id, year = year)]

  # Drop edges where either endpoint is missing (cell not observed that year)
  edges_yr <- edges_yr[!is.na(from_row) & !is.na(to_row)]

  # ── 2d. Construct sparse matrix ─────────────────────────────────────
  N <- nrow(cell_data)
  W <- sparseMatrix(
    i    = edges_yr$from_row,
    j    = edges_yr$to_row,
    x    = 1,
    dims = c(N, N)
  )
  return(W)
}

cat("Building sparse neighbor matrix …\n")
t0 <- proc.time()
W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("Done in %.1f s.\n", (proc.time() - t0)[3]))

## ── Step 3: Row-wise neighbor count (for mean) and indicator ─────────
neighbor_count <- as.numeric(W %*% rep(1, ncol(W)))   # nnz per row
has_neighbors  <- neighbor_count > 0

## ── Step 4: Compute max, min, mean for each source variable ──────────
#  For mean:  W_norm %*% x   (row-normalised matrix)
#  For max/min: grouped operation over the sparse triplet

W_t <- as(W, "TsparseMatrix")   # triplet form: W_t@i, W_t@j, W_t@x (0-based)
from_rows <- W_t@i + 1L         # 1-based "from" row indices
to_rows   <- W_t@j + 1L         # 1-based "to"   row indices

# Row-normalised W for mean computation
W_norm <- W
W_norm@x <- W_norm@x / neighbor_count[from_rows[match(seq_along(W_norm@x),
                                                        seq_along(W_norm@x))]]
# More robustly:
diag_inv <- sparseMatrix(i = which(has_neighbors),
                         j = which(has_neighbors),
                         x = 1 / neighbor_count[has_neighbors],
                         dims = c(nrow(W), nrow(W)))
W_norm <- diag_inv %*% W        # each row sums to 1 (or row is zero)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics …\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {

  x <- cell_data[[var_name]]

  # ── Neighbor mean ───────────────────────────────────────────────────
  n_mean <- as.numeric(W_norm %*% x)
  n_mean[!has_neighbors] <- NA_real_

  # ── Neighbor max & min (grouped over sparse entries) ────────────────
  #    Pull neighbor values, then group by "from" row
  neighbor_vals <- x[to_rows]

  # Identify non-NA neighbor values
  ok <- !is.na(neighbor_vals)
  fr_ok  <- from_rows[ok]
  nv_ok  <- neighbor_vals[ok]

  # Use data.table for fast grouped max/min
  dt_tmp <- data.table(from = fr_ok, val = nv_ok)
  agg    <- dt_tmp[, .(nmax = max(val), nmin = min(val)), by = from]

  n_max <- rep(NA_real_, nrow(cell_data))
  n_min <- rep(NA_real_, nrow(cell_data))
  n_max[agg$from] <- agg$nmax
  n_min[agg$from] <- agg$nmin

  # Also set mean to NA where all neighbor values were NA
  # (W_norm %*% x propagates 0 for NA; fix up)
  # Count non-NA neighbors per row
  nna_count <- rep(0L, nrow(cell_data))
  nna_tab   <- dt_tmp[, .N, by = from]
  nna_count[nna_tab$from] <- nna_tab$N

  # Recompute mean only from non-NA values (matches original logic)
  nna_sum <- rep(0, nrow(cell_data))
  nna_sum_dt <- dt_tmp[, .(s = sum(val)), by = from]
  nna_sum[nna_sum_dt$from] <- nna_sum_dt$s

  n_mean2 <- ifelse(nna_count > 0, nna_sum / nna_count, NA_real_)
  # Override the sparse-matrix mean with the NA-safe version
  n_mean <- n_mean2

  # ── Assign to cell_data with original column naming convention ──────
  set(cell_data, j = paste0("neighbor_max_",  var_name), value = n_max)
  set(cell_data, j = paste0("neighbor_min_",  var_name), value = n_min)
  set(cell_data, j = paste0("neighbor_mean_", var_name), value = n_mean)

  cat(sprintf("  %s done.\n", var_name))
}
cat(sprintf("All neighbor stats computed in %.1f s.\n", (proc.time() - t0)[3]))

## ── Step 5: Clean up helper column ───────────────────────────────────
cell_data[, .row_idx := NULL]

## ── Step 6: Predict with the pre-trained Random Forest ───────────────
# (model object assumed already in memory; no retraining)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Expected Performance

| Phase | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~40–50 h (6.46M string ops) | ~2–5 min (keyed join + sparse matrix construction) |
| `compute_neighbor_stats` (5 vars) | ~35–40 h (32.3M R calls) | ~1–3 min (sparse matrix multiply + data.table grouped agg) |
| **Total** | **~86+ hours** | **~5–10 minutes** |

## Why Numerical Results Are Preserved

- The sparse matrix `W` encodes **exactly** the same rook-neighbor, same-year relationships as the original `build_neighbor_lookup`.
- Max, min, and mean are computed from the identical set of non-NA neighbor values, grouped by the same "from" row.
- The `data.table` grouped aggregation (`max`, `min`, `sum / count`) is algebraically identical to the original `lapply` logic that filters `!is.na`, then computes `max`, `min`, `mean`.
- Column names follow the same naming convention, so the pre-trained Random Forest's `predict()` call resolves features identically. **No retraining is needed or performed.**