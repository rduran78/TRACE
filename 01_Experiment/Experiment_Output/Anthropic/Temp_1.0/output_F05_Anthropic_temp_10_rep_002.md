 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The repeated string-key work is **not merely a local inefficiency — it is a symptom of a larger, fundamentally misaligned algorithmic pattern**. Here is the full chain of waste:

### The Core Problem: Row-Level Iteration Over 6.46M Rows in Pure R

`build_neighbor_lookup` calls `lapply` over every one of the ~6.46 million rows. Inside each iteration it:

1. **Constructs paste keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — string allocation and concatenation for every neighbor of every row.
2. **Looks up those keys** in a named character vector (`idx_lookup[neighbor_keys]`) — this is an O(1)-amortized hash lookup *per key*, but the constant factor of R's named-vector hashing on character strings is large, and it is invoked **~6.46M × avg_neighbors ≈ 25+ billion character operations**.

The `idx_lookup` named vector itself (`paste(data$id, data$year, sep = "_")` with 6.46M entries) is built once — that part is fine. But the *per-row consumption* of it is the bottleneck.

### Why It's a Broader Pattern

The neighbor lookup produces a **list of integer row-index vectors** — one per row. This is then consumed by `compute_neighbor_stats`, which itself iterates over all 6.46M rows again, subsetting a numeric vector and computing `max/min/mean`. This is repeated for **each of the 5 variables**, meaning 5 × 6.46M additional R-level iterations.

So the full cost is:

| Phase | Iterations | Cost Driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M | String paste + hash lookup per neighbor |
| `compute_neighbor_stats` | 5 × 6.46M | R-level `lapply`, vector subset, `max/min/mean` |
| **Total R-level loop iterations** | **~38.8M** | Plus billions of string ops in phase 1 |

At ~86+ hours, this is dominated by the string construction in phase 1 and the interpretive overhead of pure-R loops in both phases.

### Key Insight for Reformulation

The string-keyed lookup is solving a simple structural problem: **"given a cell ID and a year, find the row index."** Because the panel is balanced (344,208 cells × 28 years = 9,637,824 potential slots, with 6.46M populated), and years are a small integer domain, this can be solved with **integer arithmetic instead of string hashing**. Moreover, the neighbor structure is **invariant across years** — the same cell has the same neighbors in every year. This means:

> **We don't need to build a per-row neighbor list at all. We need a per-cell neighbor list (344K entries), and then we can join across all years simultaneously using vectorized operations.**

---

## 2. Optimization Strategy

### Algorithmic Reformulation

**Replace the row-level string-key approach with a vectorized sparse-matrix multiplication.**

The neighbor relationships form a **sparse adjacency matrix** `W` of dimension 344,208 × 344,208. Computing the mean of a variable across neighbors is simply:

```
neighbor_mean = (W %*% x) / (W %*% 1)  # where 1 is a vector of non-NA indicators
```

And max/min can be computed via a sparse row-sweep using the adjacency structure.

For a balanced panel, we compute these statistics **per year** (each year has the same cell set and the same adjacency), which reduces to 28 sparse matrix–vector multiplications per variable — not 6.46M R-level iterations.

### Concrete Steps

1. **Build the sparse adjacency matrix once** from `rook_neighbors_unique` (a `nb` object → `nb2listw` → sparse matrix, or directly).
2. **For each year and each variable**, extract the 344,208-length vector, apply sparse operations to get `neighbor_max`, `neighbor_min`, `neighbor_mean`.
3. **Write results back** into the data frame.
4. **No string keys. No per-row `lapply`. No `paste`.**

### Complexity Comparison

| | Original | Reformulated |
|---|---|---|
| Neighbor lookup | 6.46M string-hash lookups | 1 sparse matrix (built once) |
| Stats computation | 5 vars × 6.46M R iterations | 5 vars × 28 years × 3 sparse ops |
| Dominant cost | ~25B character ops | ~5 × 28 × 2 sparse mat-vec products (~1.37M nonzeros each) |
| Expected time | 86+ hours | **Minutes** |

### RAM Check

- Sparse matrix: 1,373,394 nonzeros × 12 bytes ≈ 16 MB.
- Data frame: 6.46M rows × 110 cols × 8 bytes ≈ 5.7 GB. Already in memory.
- Year-sliced vectors: 344K × 8 bytes ≈ 2.7 MB each. Negligible.
- Well within 16 GB.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites:
#   cell_data           — data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   rook_neighbors_unique — nb object (from spdep) with 344,208 entries
#   id_order            — integer/numeric vector of cell IDs in the order matching rook_neighbors_unique
#
# This code preserves the exact same numerical output as the original:
#   For each row, neighbor_max_<var>, neighbor_min_<var>, neighbor_mean_<var>
# =============================================================================

library(Matrix)  # for sparse matrices
library(data.table)  # for fast grouped operations

# ---- Step 1: Build sparse adjacency matrix (once) --------------------------

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer neighbor index vectors)
  # n: number of spatial units
  # Returns: sparse logical/numeric adjacency matrix (n x n), row i has 1s at neighbor columns
  
  # Enumerate all (i, j) pairs
  i_idx <- rep(seq_len(n), times = lengths(nb_obj))
  j_idx <- unlist(nb_obj)
  
  # Remove zero-length / zero entries (spdep uses 0L for no-neighbor indicator in some versions)
  valid <- j_idx > 0L
  i_idx <- i_idx[valid]
  j_idx <- j_idx[valid]
  
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)  # 344,208
W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# ---- Step 2: Map cell IDs to matrix row/col indices ------------------------

# id_order[k] is the cell ID for the k-th row/col of W
# We need a lookup: cell_id -> matrix index
cell_id_to_mat_idx <- setNames(seq_along(id_order), as.character(id_order))

# ---- Step 3: Convert to data.table for fast year-grouped operations ---------

dt <- as.data.table(cell_data)

# Add matrix index column (maps each row's cell ID to the row/col index in W)
dt[, mat_idx := cell_id_to_mat_idx[as.character(id)]]

# Sort by year and mat_idx to enable fast vectorized access
setkey(dt, year, mat_idx)

# ---- Step 4: Compute neighbor stats per variable ----------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns
for (var_name in neighbor_source_vars) {
  dt[, paste0("neighbor_max_", var_name) := NA_real_]
  dt[, paste0("neighbor_min_", var_name) := NA_real_]
  dt[, paste0("neighbor_mean_", var_name) := NA_real_]
}

# Get sorted unique years
years <- sort(unique(dt$year))

# For max and min via sparse matrix, we need a direct approach since
# sparse mat-vec only gives sums. We iterate over years (only 28)
# and use the sparse structure directly.

# Extract the adjacency list from the sparse matrix for row-wise max/min
# This is done once and is fast.
W_dgC <- as(W, "dgCMatrix")  # ensure CSC format for column slicing
W_dgR <- as(W, "dgRMatrix")  # CSR format for fast row access

# Actually, for max/min we need to iterate over rows of W.
# With 344K cells and avg ~4 neighbors each (rook), this is very fast in vectorized R.
# We'll extract the neighbor list from the sparse matrix once.

# Extract neighbor indices from sparse matrix (CSR format)
# W_dgR@j is 0-based column indices, W_dgR@p is row pointer
get_neighbor_list_from_sparse <- function(W_csr) {
  n <- nrow(W_csr)
  p <- W_csr@p
  j <- W_csr@j + 1L  # convert to 1-based
  lapply(seq_len(n), function(i) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end >= start) j[start:end] else integer(0)
  })
}

# This gives us neighbor mat_idx for each cell (same as rook_neighbors_unique but guaranteed consistent)
nb_list <- get_neighbor_list_from_sparse(W_dgR)

# ---- Step 5: Year-by-year, variable-by-variable computation ----------------

# Strategy:
#   For each year, the subset of dt with that year has cells ordered by mat_idx (due to setkey).
#   We build a full-length vector (length n_cells) with the variable values placed at their mat_idx.
#   Then we compute neighbor stats using the nb_list.
#
#   For mean: use sparse matrix-vector product for sum and count, then divide.
#   For max/min: use vectorized C-level operations via vapply on nb_list.
#
#   Since nb_list has only 344K entries with ~4 neighbors each, vapply over it is ~344K iterations
#   with trivial work — takes < 1 second per variable per year.

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  # Get the rows for this year
  yr_rows <- dt[.(yr)]  # keyed access: all rows with this year
  
  # Map mat_idx to positions within this year-slice
  # yr_rows is sorted by mat_idx due to setkey
  yr_mat_indices <- yr_rows$mat_idx
  
  # For each variable, build a full vector of length n_cells (NA for missing cells)
  for (var_name in neighbor_source_vars) {
    
    # Full vector (NA-initialized)
    full_vec <- rep(NA_real_, n_cells)
    full_vec[yr_mat_indices] <- yr_rows[[var_name]]
    
    # -- Neighbor mean via sparse matrix-vector multiply --
    # Replace NA with 0 for sum, track non-NA for count
    non_na <- !is.na(full_vec)
    vec_zero <- full_vec
    vec_zero[!non_na] <- 0
    
    neighbor_sum   <- as.numeric(W %*% vec_zero)       # sum of neighbor values
    neighbor_count <- as.numeric(W %*% as.numeric(non_na))  # count of non-NA neighbors
    
    neighbor_mean_full <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # -- Neighbor max and min via nb_list --
    # vapply over 344K cells — fast because each has ~4 neighbors
    neighbor_max_full <- vapply(nb_list, function(idx) {
      if (length(idx) == 0L) return(NA_real_)
      v <- full_vec[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0L) return(NA_real_)
      max(v)
    }, numeric(1))
    
    neighbor_min_full <- vapply(nb_list, function(idx) {
      if (length(idx) == 0L) return(NA_real_)
      v <- full_vec[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0L) return(NA_real_)
      min(v)
    }, numeric(1))
    
    # -- Write results back to the year-slice rows --
    # yr_mat_indices tells us which mat_idx each row corresponds to
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Use set() for in-place modification (no copy)
    row_indices <- dt[year == yr, which = TRUE]
    set(dt, i = row_indices, j = max_col,  value = neighbor_max_full[yr_mat_indices])
    set(dt, i = row_indices, j = min_col,  value = neighbor_min_full[yr_mat_indices])
    set(dt, i = row_indices, j = mean_col, value = neighbor_mean_full[yr_mat_indices])
  }
}

# ---- Step 6: Convert back to data.frame if needed --------------------------
cell_data <- as.data.frame(dt)

# ---- Done. The trained Random Forest model is untouched. --------------------
cat("Neighbor feature construction complete.\n")
```

---

## 4. Complexity and Runtime Estimate

| Operation | Count | Unit Cost | Total |
|---|---|---|---|
| Build sparse matrix | 1 | O(1.37M) | < 1 sec |
| Sparse mat-vec (`W %*% x`) | 5 vars × 28 yrs × 2 = 280 | O(1.37M) | ~30 sec |
| `vapply` for max/min | 5 vars × 28 yrs × 2 × 344K cells | O(4) per cell | ~5–10 min |
| `data.table` keyed access/set | 28 years × 5 vars | O(230K) per slice | ~1 min |
| **Total** | | | **~10–15 minutes** |

This is a **~350×–500× speedup** over the original 86+ hours.

---

## 5. Why This Preserves the Numerical Estimand

- The sparse matrix `W` encodes **exactly the same neighbor relationships** as `rook_neighbors_unique`.
- For each cell-year, the set of neighbor values considered is identical.
- `max`, `min`, and `mean` are computed on the same non-NA neighbor values.
- The mean computation `(W %*% x) / (W %*% 1_{non-NA})` is algebraically equivalent to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- The resulting columns are identically named and placed into `cell_data`.
- **The trained Random Forest model is never touched** — only the feature-construction step is replaced.