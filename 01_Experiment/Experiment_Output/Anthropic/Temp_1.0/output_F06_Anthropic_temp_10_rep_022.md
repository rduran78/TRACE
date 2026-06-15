 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row neighbor indices

`build_neighbor_lookup` iterates over **~6.46 million rows**, performing character key lookups for each row's rook neighbors. The `paste` + named-vector lookup pattern is extremely slow at this scale. The output is a list of 6.46M integer vectors — itself a large, fragile object.

### Bottleneck B: `compute_neighbor_stats` — Row-wise `lapply` over 6.46M rows, repeated 5 times

`compute_neighbor_stats` loops over every row, subsets a numeric vector by neighbor indices, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per source variable), producing **~32.3 million** individual R function calls (`lapply` body executions), each with subsetting, NA removal, and three summary computations.

### Why raster focal/kernel operations are *not* a direct replacement

Focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel window. Here the data is a **panel** (cell × year), neighbor structures come from a precomputed `spdep::nb` object (which may reflect irregular boundaries, coastal cells, etc.), and the computation is "within the same year." A focal approach would require reshaping to a 3D raster stack, applying focal per layer, then reshaping back — and would silently alter results at boundary cells where the nb structure differs from a regular 3×3 rook kernel. **The row-index approach is correct in principle; it just needs vectorized implementation.**

### Root cause summary

| Component | Calls | Cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + named-vector lookups | ~30–40 min |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M R-level iterations | ~80+ hours |

---

## 2. Optimization Strategy

### Strategy: Vectorized sparse-matrix multiplication and grouped min/max

Replace both functions with **a single sparse matrix** (cell-year row → neighbor cell-year rows) built once, then use:

- **Sparse matrix–vector multiply** for `mean` (sum of neighbor values / number of neighbors).
- **Grouped vectorized operations** via `data.table` for `max` and `min`.

This eliminates all R-level row iteration.

**Key design decisions:**

1. Build an integer join using `data.table` instead of character key pasting — O(n log n) merge vs O(n) named-vector lookup with huge constant factor.
2. Construct a sparse adjacency matrix `W` (CSC format, `Matrix` package) of dimension `nrow × nrow` where entry (i, j) = 1 if row j is a rook neighbor of row i in the same year.
3. For **mean**: `W %*% x / rowSums(W)` — a single sparse matrix-vector multiply (~seconds for 6.46M rows with ~1.37M nonzeros per year × 28 years ≈ ~20M nonzeros total).
4. For **max** and **min**: Expand the sparse matrix to an edge list, join values, and compute grouped extremes with `data.table`.

**Expected runtime: ~2–5 minutes** (down from 86+ hours).

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# ──────────────────────────────────────────────────────────────
# STEP 1: Build the sparse neighbor matrix (row-to-row, same year)
# ──────────────────────────────────────────────────────────────

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year (and be ordered consistently)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Map cell id -> position in id_order (i.e., index into the nb list)
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Expand the nb object into an edge list of (cell_ref, neighbor_cell_ref)
  # rook_neighbors is an nb object: a list where element i is an integer

  # vector of neighbor indices (positions in id_order), with 0L meaning none.
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb <- rook_neighbors[[i]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(from_ref = i, to_ref = nb)
  }))

  # Translate ref indices back to cell ids
  edge_list[, from_id := id_order[from_ref]]
  edge_list[, to_id   := id_order[to_ref]]
  edge_list[, c("from_ref", "to_ref") := NULL]

  # Create a lookup from (id, year) -> row_idx
  setkey(dt, id, year)
  lookup <- dt[, .(id, year, row_idx)]

  # Get unique years
  years <- unique(dt$year)

  # For each year, join edges to get (from_row_idx, to_row_idx)
  sparse_edges <- rbindlist(lapply(years, function(yr) {
    lu_yr <- lookup[year == yr]
    setkey(lu_yr, id)

    # Join from side
    edges_yr <- merge(edge_list, lu_yr[, .(id, row_idx)],
                      by.x = "from_id", by.y = "id", all.x = FALSE)
    setnames(edges_yr, "row_idx", "from_row")

    # Join to side
    edges_yr <- merge(edges_yr, lu_yr[, .(id, row_idx)],
                      by.x = "to_id", by.y = "id", all.x = FALSE)
    setnames(edges_yr, "row_idx", "to_row")

    edges_yr[, .(from_row, to_row)]
  }))

  n <- nrow(dt)

  # Build sparse matrix: W[i, j] = 1 means row j is a neighbor of row i

  W <- sparseMatrix(
    i    = sparse_edges$from_row,
    j    = sparse_edges$to_row,
    x    = 1,
    dims = c(n, n)
  )

  # Also return the edge list for min/max computation
  list(W = W, edges = sparse_edges, dt = dt)
}

# ──────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats (max, min, mean) for one variable
# ──────────────────────────────────────────────────────────────

compute_neighbor_stats_fast <- function(dt, W, edges, var_name) {
  x <- dt[[var_name]]

  # ---- MEAN ----
  # Replace NA with 0 for the multiply, but track non-NA counts
  x_nona <- ifelse(is.na(x), 0, x)
  not_na <- as.numeric(!is.na(x))

  neighbor_sum   <- as.numeric(W %*% x_nona)
  neighbor_count <- as.numeric(W %*% not_na)

  # Where all neighbors are NA or no neighbors exist, result is NA
  n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

  # ---- MAX and MIN ----
  # Use the edge list: attach the neighbor's value, then group by from_row
  edge_dt <- copy(edges)
  edge_dt[, val := x[to_row]]
  # Remove edges where neighbor value is NA
  edge_dt <- edge_dt[!is.na(val)]

  # Grouped max and min
  agg <- edge_dt[, .(n_max = max(val), n_min = min(val)), by = from_row]

  # Initialize result vectors with NA
  n <- length(x)
  n_max <- rep(NA_real_, n)
  n_min <- rep(NA_real_, n)

  n_max[agg$from_row] <- agg$n_max
  n_min[agg$from_row] <- agg$n_min

  data.table(
    n_max  = n_max,
    n_min  = n_min,
    n_mean = n_mean
  )
}

# ──────────────────────────────────────────────────────────────
# STEP 3: Main pipeline
# ──────────────────────────────────────────────────────────────

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  message("Building sparse neighbor matrix...")
  t0 <- proc.time()
  res <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
  W     <- res$W
  edges <- res$edges
  dt    <- res$dt
  message(sprintf("  Done in %.1f seconds.", (proc.time() - t0)[3]))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()

    stats <- compute_neighbor_stats_fast(dt, W, edges, var_name)

    # Add columns with the same naming convention as the original code
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")

    dt[, (col_max)  := stats$n_max]
    dt[, (col_min)  := stats$n_min]
    dt[, (col_mean) := stats$n_mean]

    message(sprintf("  Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  # Return as data.frame to match downstream expectations
  as.data.frame(dt[, row_idx := NULL])
}

# ──────────────────────────────────────────────────────────────
# USAGE (drop-in replacement)
# ──────────────────────────────────────────────────────────────

cell_data <- run_neighbor_feature_pipeline(
  cell_data, id_order, rook_neighbors_unique
)

# Then proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Correctness

| Original computation | Replacement | Equivalence |
|---|---|---|
| `max(neighbor_vals)` after NA removal | `data.table` grouped `max` after NA removal on edge list | Identical — same set of non-NA neighbor values |
| `min(neighbor_vals)` after NA removal | `data.table` grouped `min` after NA removal on edge list | Identical |
| `mean(neighbor_vals)` after NA removal | `(W %*% x_nona) / (W %*% not_na)` | Identical — sum of non-NA values divided by count of non-NA values |
| Rows with no neighbors or all-NA neighbors → `NA` | `neighbor_count == 0 → NA`; absent from `agg` → `NA` | Identical |
| Trained RF model | Untouched — features are numerically identical | Preserved |

## 5. Expected Performance

| Stage | Estimated time |
|---|---|
| Build sparse matrix (~20M nonzero entries) | ~30–60 seconds |
| `W %*% x` for mean (per variable) | ~1–2 seconds |
| Edge-list grouped max/min (per variable) | ~5–10 seconds |
| **Total for 5 variables** | **~2–5 minutes** |

This is a **~1,000×–2,500× speedup** over the original 86+ hour estimate, running comfortably within 16 GB RAM (the sparse matrix with ~20M entries occupies ~200–300 MB; the edge `data.table` ~150 MB).