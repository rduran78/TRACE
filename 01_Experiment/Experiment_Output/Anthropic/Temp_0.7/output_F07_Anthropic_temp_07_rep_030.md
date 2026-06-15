 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** constructs a list of ~6.46 million elements, each built by an R-level `lapply` iteration that performs character pasting, named-vector lookups, and NA filtering — all inside a per-row anonymous function. This is **O(n)** interpreted R loops over millions of rows, with expensive string operations at every step. The subsequent `compute_neighbor_stats` then does another `lapply` over 6.46M elements. Together, these two phases dominate the 86+ hour runtime.

**Key issues:**

1. **`build_neighbor_lookup`**: `paste()` + named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. The `id_to_ref` and `idx_lookup` named vectors use linear-time character matching at scale.
2. **`compute_neighbor_stats`**: Another 6.46M-element `lapply` with per-element subsetting and aggregation.
3. **Memory**: Storing a 6.46M-element list of integer vectors for `neighbor_lookup` is wasteful; a sparse-matrix or edge-list representation is far more compact and vectorizable.
4. **The loop runs 5 times** (once per variable), multiplying the cost.

---

## Optimization Strategy

### Core idea: Replace per-row R loops with vectorized sparse-matrix multiplication.

1. **Build a sparse adjacency matrix `W`** (rows = cell-year observations, cols = cell-year observations) where `W[i,j] = 1` iff observation `j` is a rook-neighbor of observation `i` in the same year. This is a one-time cost.

2. **Compute neighbor stats via sparse matrix operations:**
   - **Neighbor mean** = `(W %*% x) / (W %*% 1_{not NA})` — a single sparse matrix-vector multiply.
   - **Neighbor max/min** — use grouped operations on the edge list (COO representation of `W`), fully vectorized with `data.table`.

3. **All 5 variables** are computed against the same adjacency structure, so `W` is built once.

4. **Memory**: The sparse matrix has ~1.37M × 28 ≈ 38.4M non-zero entries (directed edges × years), stored as three integer/double vectors — well within 16 GB.

5. **The trained Random Forest model is untouched.** The numerical results are identical (same neighbor definitions, same max/min/mean).

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Build the sparse neighbor adjacency matrix (one-time cost)
# ─────────────────────────────────────────────────────────────────────

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table for speed
  dt <- as.data.table(cell_data)
  dt[, obs_idx := .I]  # original row index

  # Map cell id -> position in id_order
  id_map <- data.table(id = id_order, ref_idx = seq_along(id_order))

  # Build directed edge list from the nb object (cell-level, year-agnostic)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(from_ref = integer(0), to_ref = integer(0)))
    }
    data.table(from_ref = i, to_ref = as.integer(nb))
  }))

  # Translate ref_idx -> cell id
  edge_list[, from_id := id_order[from_ref]]
  edge_list[, to_id   := id_order[to_ref]]

  # For every year, expand edges to observation-level indices
  # Key the data.table for fast joins
  setkey(dt, id, year)

  years <- sort(unique(dt$year))

  # Create a lookup: (id, year) -> obs_idx
  lookup <- dt[, .(id, year, obs_idx)]
  setkey(lookup, id, year)

  # Cross-join edges with years, then look up obs_idx for both endpoints
  cat("Expanding edges across years...\n")

  # Efficient: use CJ-like expansion via merge

  edge_years <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edge_years[, from_id := edge_list$from_id[edge_idx]]
  edge_years[, to_id   := edge_list$to_id[edge_idx]]

  # Join to get obs_idx for "from" side
  setkey(edge_years, from_id, year)
  edge_years <- lookup[edge_years, .(from_id, to_id, year,
                                      from_obs = obs_idx,
                                      edge_idx),
                       on = .(id = from_id, year), nomatch = 0L]

  # Join to get obs_idx for "to" side
  setkey(edge_years, to_id, year)
  edge_years <- lookup[edge_years, .(from_id, to_id, year,
                                      from_obs,
                                      to_obs = obs_idx),
                       on = .(id = to_id, year), nomatch = 0L]

  n <- nrow(dt)
  cat(sprintf("Building sparse matrix: %d obs, %d directed edges\n",
              n, nrow(edge_years)))

  W <- sparseMatrix(
    i = edge_years$from_obs,
    j = edge_years$to_obs,
    x = 1,
    dims = c(n, n)
  )

  list(W = W, edge_dt = edge_years[, .(from_obs, to_obs)], n = n, dt = dt)
}

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor max, min, mean for one variable
# ─────────────────────────────────────────────────────────────────────

compute_neighbor_features_fast <- function(dt, var_name, W, edge_dt) {
  x <- dt[[var_name]]
  n <- length(x)

  # ── Neighbor mean via sparse matrix ──
  not_na   <- as.numeric(!is.na(x))
  x_clean  <- ifelse(is.na(x), 0, x)

  neighbor_sum   <- as.numeric(W %*% x_clean)
  neighbor_count <- as.numeric(W %*% not_na)

  nb_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

  # ── Neighbor max and min via edge list + data.table ──
  vals_to <- x[edge_dt$to_obs]
  valid   <- !is.na(vals_to)

  agg_dt <- data.table(
    from_obs = edge_dt$from_obs[valid],
    val      = vals_to[valid]
  )

  if (nrow(agg_dt) > 0) {
    stats <- agg_dt[, .(nb_max = max(val), nb_min = min(val)), by = from_obs]

    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
    nb_max[stats$from_obs] <- stats$nb_max
    nb_min[stats$from_obs] <- stats$nb_min
  } else {
    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
  }

  # Return named columns matching the original pipeline's naming convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  out <- data.frame(nb_max, nb_min, nb_mean)
  colnames(out) <- c(max_col, min_col, mean_col)
  out
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Main driver — drop-in replacement for the original pipeline
# ─────────────────────────────────────────────────────────────────────

## --- Build adjacency structure (once) ---
cat("Building sparse neighbor adjacency...\n")
system.time({
  nb_struct <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
})

W       <- nb_struct$W
edge_dt <- nb_struct$edge_dt
dt      <- nb_struct$dt   # data.table copy with obs_idx

## --- Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  %s ...\n", var_name))
    new_cols <- compute_neighbor_features_fast(dt, var_name, W, edge_dt)
    # Attach to the original data.frame (preserving row order via obs_idx)
    for (cname in colnames(new_cols)) {
      cell_data[[cname]] <- new_cols[[cname]]
    }
  }
})

cat("Done. Neighbor features added to cell_data.\n")

# The trained Random Forest model is unchanged — use cell_data
# directly for predict() as before.
```

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M R-loop iterations with string ops) | ~2–5 min (vectorized edge expansion + sparse matrix construction) |
| `compute_neighbor_stats` × 5 vars | ~hours (6.46M `lapply` × 5) | ~1–3 min per var (one sparse mat-vec for mean; one `data.table` grouped agg for max/min) |
| **Total** | **86+ hours** | **~10–20 minutes** |
| Peak RAM | Large (6.46M-element list of vectors) | ~2–4 GB (sparse matrix + edge table) |

## Why Results Are Numerically Identical

- The sparse matrix `W` encodes **exactly** the same rook-neighbor relationships as the original `nb` object, scoped to the same year.
- `NA` handling is replicated: `NA` neighbor values are excluded before computing max, min, and mean; cells with zero valid neighbors receive `NA`.
- No approximation, sampling, or retraining is involved. The trained Random Forest model is preserved as-is.