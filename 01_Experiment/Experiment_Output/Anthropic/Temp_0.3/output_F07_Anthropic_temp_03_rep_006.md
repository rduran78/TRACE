 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by character-key lookups inside an `lapply` over every row. Specifically:

1. **`paste` + named-vector lookup per row**: For each of the 6.46M rows, the code converts neighbor cell IDs to character keys (`paste(id, year, "_")`), then does named-vector indexing into `idx_lookup` (a named vector of length 6.46M). Named-vector lookup in R is **O(n)** per query in the worst case and involves repeated hashing. Doing this 6.46M times is catastrophic.

2. **The resulting `neighbor_lookup` list is enormous**: ~6.46M list elements, each containing a variable-length integer vector. This alone consumes several GB of RAM and takes hours to build.

3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the 6.46M-element list. This is slow but secondary to problem #1.

4. **The entire approach is row-centric and scalar**: It never exploits vectorized or columnar operations.

**Estimated complexity**: `build_neighbor_lookup` is roughly O(N × k) with large constant factors from character operations, where N ≈ 6.46M and k ≈ average neighbors (~4). The 86+ hour estimate is credible.

---

## Optimization Strategy

**Key insight**: The neighbor structure is *time-invariant*. Cell `i`'s neighbors are the same in every year. Therefore, we should:

1. **Work at the cell level, not the cell-year level.** Build a sparse adjacency matrix once for the 344,208 cells.
2. **Use sparse matrix–dense matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean from that.
3. **For max and min**: Use a grouped operation via `data.table` with an edge list, which is vectorized and fast.

**Concrete plan**:

- Convert the `nb` object to a sparse adjacency matrix `W` (344,208 × 344,208) using `spdep::nb2listw` → `spdep::listw2mat` or directly via `Matrix::sparseMatrix`.
- Reshape the data so that for each year, the variable values are aligned to the cell ordering.
- **Neighbor mean**: For each year, multiply `W %*% x` and divide by the row-sum of `W` (number of neighbors). This is a single sparse matrix–vector multiply per year per variable — extremely fast.
- **Neighbor max/min**: Build an edge-list `data.table` (from, to) and join values, then aggregate with `max`/`min` grouped by `from`.

This reduces the runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ── 1. Build sparse adjacency matrix from the nb object ──────────────────────
# id_order: character or integer vector of cell IDs in the order matching
#           rook_neighbors_unique (the nb object).
# rook_neighbors_unique: an nb object (list of integer index vectors).

build_sparse_adj <- function(id_order, nb_obj) {
  n <- length(id_order)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor entries (nb encodes no-neighbor as 0L)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  list(W = W, from = from, to = to)
}

adj <- build_sparse_adj(id_order, rook_neighbors_unique)
W      <- adj$W
edge_from <- adj$from
edge_to   <- adj$to

# Number of neighbors per cell (time-invariant)
n_neighbors <- as.integer(rowSums(W))   # length = 344,208

# ── 2. Convert cell_data to data.table and create cell index ─────────────────
cell_dt <- as.data.table(cell_data)

# Map each cell ID to its positional index in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_pos := id_to_pos[as.character(id)]]

# Sort for efficient by-year operations
setkey(cell_dt, year, cell_pos)

# ── 3. Edge list as data.table (reused for every variable) ───────────────────
# This represents directed edges: for cell `from`, cell `to` is a neighbor.
edge_dt <- data.table(from = edge_from, to = edge_to)

# ── 4. Compute neighbor stats for each variable ─────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))

for (var_name in neighbor_source_vars) {

  cat("Processing:", var_name, "\n")

  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]

  for (yr in years) {

    # Extract the rows for this year, aligned to cell_pos
    yr_idx <- cell_dt[.(yr), which = TRUE]  # row indices in cell_dt for this year
    yr_sub <- cell_dt[yr_idx, .(cell_pos, val = get(var_name))]

    # Build a full-length vector aligned to cell positions (NA for missing cells)
    vals_vec <- rep(NA_real_, n_cells)
    vals_vec[yr_sub$cell_pos] <- yr_sub$val

    # ── Neighbor MEAN via sparse matrix multiply ──
    # Replace NA with 0 for multiplication, track non-NA counts
    vals_nona <- vals_vec
    vals_nona[is.na(vals_nona)] <- 0
    not_na <- as.numeric(!is.na(vals_vec))

    neighbor_sum   <- as.numeric(W %*% vals_nona)
    neighbor_count <- as.numeric(W %*% not_na)

    neighbor_mean_vec <- ifelse(neighbor_count > 0,
                                neighbor_sum / neighbor_count,
                                NA_real_)

    # ── Neighbor MAX and MIN via edge list ──
    # Get neighbor values for every edge
    edge_vals <- data.table(
      from = edge_from,
      val  = vals_vec[edge_to]
    )
    # Drop edges where neighbor value is NA
    edge_vals <- edge_vals[!is.na(val)]

    if (nrow(edge_vals) > 0) {
      agg <- edge_vals[, .(nmax = max(val), nmin = min(val)), by = from]

      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
      neighbor_max_vec[agg$from] <- agg$nmax
      neighbor_min_vec[agg$from] <- agg$nmin
    } else {
      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
    }

    # ── Write results back, aligned by cell_pos ──
    yr_positions <- cell_dt[yr_idx, cell_pos]
    set(cell_dt, i = yr_idx, j = col_max,  value = neighbor_max_vec[yr_positions])
    set(cell_dt, i = yr_idx, j = col_min,  value = neighbor_min_vec[yr_positions])
    set(cell_dt, i = yr_idx, j = col_mean, value = neighbor_mean_vec[yr_positions])
  }
}

# ── 5. Clean up helper column and convert back if needed ─────────────────────
cell_dt[, cell_pos := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | Mean = sum(neighbor values) / count(non-NA neighbors), identical to the original. Max and min are computed over the same non-NA neighbor sets. |
| **Irregular topology** | The sparse matrix `W` and edge list are built directly from the `nb` object — no rectangular-grid assumption. Missing cells (not in a given year) produce `NA` and are excluded from aggregation, exactly as the original code does. |
| **Trained RF model** | No model code is touched. Only the feature-engineering pipeline is optimized. Column names are identical (`neighbor_max_ntl`, etc.). |
| **RAM** | The sparse matrix is ~1.4M non-zeros (~22 MB). The edge `data.table` is ~1.4M rows (~11 MB). Per-year vectors are 344K doubles (~2.6 MB). Peak usage is well under 16 GB. |

## Expected Runtime

- **28 years × 5 variables = 140 iterations.**
- Each iteration: one sparse mat-vec (~0.01s), one `data.table` grouped aggregation over ~1.4M edges (~0.05s).
- **Total: ~2–5 minutes** (down from 86+ hours), a speedup of roughly **1,000×–2,500×**.