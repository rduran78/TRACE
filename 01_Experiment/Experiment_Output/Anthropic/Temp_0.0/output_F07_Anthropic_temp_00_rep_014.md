 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by character-key lookups inside an `lapply` over every row. Specifically:

1. **`paste` + named-vector lookup per row**: For each of the 6.46M rows, the code converts neighbor cell IDs to character keys (`paste(id, year, sep="_")`), then does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) per query in the worst case and involves repeated string hashing. Over 6.46M rows × ~4 neighbors each ≈ 26M string operations.

2. **`lapply` over 6.46M rows for `compute_neighbor_stats`**: Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. The per-element overhead of R-level `lapply` (closure call, allocation of 3-element vectors) across 6.46M iterations is enormous.

3. **Memory**: Building a 6.46M-element list of integer vectors, plus the `do.call(rbind, ...)` on 6.46M 3-element vectors, creates massive intermediate allocations.

**Estimated cost**: The 86+ hour runtime is almost entirely attributable to the row-level R loop in `build_neighbor_lookup` and the repeated `lapply` in `compute_neighbor_stats`.

---

## Optimization Strategy

### Key Insight: Vectorize via Sparse Matrix Multiplication

The neighbor relationships are a **fixed spatial graph** (344,208 cells, ~1.37M directed edges). The panel has 28 years. For each year, the neighbor-max, neighbor-min, and neighbor-mean of a variable can be computed by operating on the **sparse adjacency matrix** directly — no per-row R loop needed.

**Plan:**

1. **Build a sparse binary adjacency matrix `W`** (344,208 × 344,208) from `rook_neighbors_unique` once. This is tiny (~1.37M non-zero entries).

2. **For each variable and each year**, extract the value vector `v` (length 344,208), then:
   - **Neighbor mean** = `(W %*% v) / (W %*% ones)` — sparse matrix-vector multiply, microseconds.
   - **Neighbor max / min** — use a grouped operation on the sparse matrix's structure (CSC column indices), or use `data.table` grouped operations on an edge list.

3. **Join results back** to the panel `data.table` by `(id, year)`.

This replaces 6.46M R-level iterations with ~28 sparse matrix-vector multiplies per variable (one per year), each taking milliseconds. Total runtime drops from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ── 0. Convert to data.table if not already ──────────────────────────────────
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# ── 1. Build sparse adjacency matrix from spdep nb object (once) ─────────────
build_sparse_adjacency <- function(nb_obj, id_order) {

  # nb_obj: list of integer vectors (neighbor indices into id_order)
  # Returns: sparse binary matrix W (n x n), where W[i,j]=1 means j is

  #          a rook neighbor of i.
  n <- length(nb_obj)
  stopifnot(n == length(id_order))

  # Build COO triplets
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)

  # Remove 0-neighbor entries (spdep uses integer(0) or 0L for no neighbors)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

W <- build_sparse_adjacency(rook_neighbors_unique, id_order)

# Precompute the number of neighbors per cell (constant across years)
ones_vec    <- rep(1, length(id_order))
n_neighbors <- as.numeric(W %*% ones_vec)  # length = n_cells

# ── 2. Build edge list for max/min (grouped ops) ────────────────────────────
# Extract COO from W
W_coo <- summary(W)  # data.frame with columns i, j, x
edge_dt <- data.table(focal = W_coo$i, neighbor = W_coo$j)
# focal's neighbor is 'neighbor', so for focal cell i we want values at j.

# Map from cell index (1..344208) to id_order value
idx_to_id <- data.table(cell_idx = seq_along(id_order), id = id_order)

# ── 3. Compute neighbor stats for all variables ─────────────────────────────

# Ensure id_order mapping in cell_data
# Create a cell_idx column: position of each cell's id in id_order
id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
cell_data <- merge(cell_data, id_map, by = "id", all.x = TRUE, sort = FALSE)

# Key for fast subsetting
setkey(cell_data, year, cell_idx)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

for (var_name in neighbor_source_vars) {

  cat("Processing variable:", var_name, "\n")

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  for (yr in years) {

    # Extract the value vector for this year, ordered by cell_idx
    yr_data <- cell_data[.(yr)]  # keyed on year first
    # Ensure we have a full vector aligned to cell_idx 1..n_cells
    v <- rep(NA_real_, n_cells)
    v[yr_data$cell_idx] <- yr_data[[var_name]]

    # ── Neighbor mean via sparse mat-vec ──
    Wv <- as.numeric(W %*% v)
    # Handle NA propagation: count only non-NA neighbors
    not_na   <- as.numeric(!is.na(v))
    Wv_notna <- as.numeric(W %*% not_na)          # count of non-NA neighbors
    # Sum only non-NA values: replace NA with 0 for the multiply
    v_zero        <- v
    v_zero[is.na(v_zero)] <- 0
    Wv_sum        <- as.numeric(W %*% v_zero)
    neighbor_mean <- ifelse(Wv_notna > 0, Wv_sum / Wv_notna, NA_real_)

    # ── Neighbor max and min via edge-list grouped ops ──
    # Look up neighbor values
    edge_vals <- data.table(
      focal = edge_dt$focal,
      val   = v[edge_dt$neighbor]
    )
    # Remove edges where neighbor value is NA
    edge_vals <- edge_vals[!is.na(val)]

    agg <- edge_vals[, .(nmax = max(val), nmin = min(val)), by = focal]

    # Build full-length vectors
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[agg$focal] <- agg$nmax
    neighbor_min[agg$focal] <- agg$nmin

    # ── Write back into cell_data ──
    # yr_data$cell_idx gives the cell indices present this year
    cidx <- yr_data$cell_idx
    # Use data.table set() for speed — need the actual row numbers in cell_data
    row_nums <- which(cell_data$year == yr)
    # But these are already aligned because we subset by key.
    # Safer: match back by cell_idx
    set(cell_data, i = row_nums, j = max_col,  value = neighbor_max[cidx])
    set(cell_data, i = row_nums, j = min_col,  value = neighbor_min[cidx])
    set(cell_data, i = row_nums, j = mean_col, value = neighbor_mean[cidx])
  }

  cat("  Done:", var_name, "\n")
}

# ── 4. Clean up helper column ────────────────────────────────────────────────
# Remove cell_idx if you don't want it in the final data
# cell_data[, cell_idx := NULL]

cat("All neighbor features computed.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | For each cell-year, we compute max, min, and mean over exactly the same set of rook-neighbor values as the original code. The sparse matrix `W` encodes the identical adjacency from `rook_neighbors_unique`. |
| **NA handling** | Matches original: NAs are excluded before computing stats; if all neighbors are NA (or a cell has no neighbors), the result is `NA`. |
| **Irregular topology / masked cells** | The sparse matrix naturally handles cells with 0–4 neighbors, missing cells, and boundary irregularities — no rectangular-raster assumption. |
| **Trained Random Forest** | No model retraining. We only recompute the input features identically, then feed them to the existing model. |

## Expected Performance

| Step | Cost |
|---|---|
| Build sparse matrix | ~1 second (1.37M entries) |
| Per variable, per year: 2 sparse mat-vec multiplies | ~5 ms each × 28 years × 5 vars = ~1.4 seconds total |
| Per variable, per year: edge-list grouped max/min | ~0.1 s each × 28 × 5 = ~14 seconds total |
| Write-back via `set()` | Negligible |
| **Total** | **< 1 minute** (down from 86+ hours) |

Memory peak stays well under 4 GB (sparse matrix ~22 MB, edge list ~22 MB, value vectors ~2.6 MB each, `cell_data` itself is the dominant object).