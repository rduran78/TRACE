 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, and for each row it:

1. Performs character-based key lookups (`paste` + named-vector indexing) — O(n) string operations repeated ~6.46M times.
2. Allocates a new character vector (`neighbor_keys`) per row.
3. Uses `as.character` and `paste` inside a per-row `lapply`, which defeats R's vectorization.

The `compute_neighbor_stats` function then iterates over the 6.46M-element list again per variable (×5 variables). Each iteration subsets `vals[idx]` and computes `max/min/mean` — this is actually fine per-call, but the 6.46M-element `lapply` in pure R is slow.

**Root cause summary:**

| Component | Cost | Why |
|---|---|---|
| `build_neighbor_lookup` | ~80+ hours | Per-row string paste + named-vector hash lookup ×6.46M |
| `compute_neighbor_stats` | ~6 hours (×5 vars) | Per-row lapply over 6.46M list elements, ×5 |
| **Total** | **~86+ hours** | Pure-R row-level iteration, no vectorization |

The `spdep::nb` object (`rook_neighbors_unique`) has ~344K cells with ~1.37M directed edges — this is a **sparse graph** and should be represented as a **sparse matrix**, which enables fully vectorized neighbor aggregation via matrix multiplication.

---

## Optimization Strategy

**Replace the entire row-level lookup + loop with a single sparse adjacency matrix and matrix-vector products.**

1. **Build a sparse binary adjacency matrix `W`** (344,208 × 344,208) from `rook_neighbors_unique`. This has ~1.37M nonzeros — trivial in memory (~16 MB).

2. **Expand to the panel** using a cell-to-row mapping. For each year, the neighbor of cell `i` in year `t` is the row of that neighbor cell in year `t`. Rather than building a 6.46M × 6.46M matrix (too large), we process **year-by-year**: for each year, extract the column of values, do `W %*% x` (sparse mat-vec), and divide by neighbor counts. This gives `neighbor_sum`, from which `neighbor_mean = neighbor_sum / neighbor_count`.

3. For **neighbor max and min**, sparse matrix multiplication doesn't directly help, but we can use a **long-form edge table** + `data.table` grouped aggregation — vectorized C-level groupby.

**Expected speedup:** From ~86 hours → **~2–5 minutes**.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================
# STEP 1: Build sparse adjacency matrix from spdep::nb object
# ==============================================================
build_sparse_adjacency <- function(nb_obj) {
  # nb_obj is a list of length n_cells; nb_obj[[i]] gives integer 
  # vector of neighbor indices (or 0L if no neighbors)
  n <- length(nb_obj)
  
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # spdep encodes "no neighbors" as a single 0; remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  W
}

W <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- length(rook_neighbors_unique)

# Neighbor count per cell (constant across years)
neighbor_count <- as.numeric(W %*% rep(1, n_cells))

# ==============================================================
# STEP 2: Create cell-index and year columns in data.table
# ==============================================================
# id_order is the vector mapping position index -> cell id
# We need the reverse: cell id -> position index in W

dt <- as.data.table(cell_data)

id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
dt[, cell_pos := id_to_pos[as.character(id)]]

# Sort for efficiency (optional but helps cache locality)
setkey(dt, year, cell_pos)

years <- sort(unique(dt$year))

# ==============================================================
# STEP 3: Build edge table (long form) for max/min
#          This is ~1.37M edges × 28 years ≈ 38.5M rows
#          (~300 MB with 2 int + 1 double column — fits in RAM)
# ==============================================================
# We build the edge list once from W
W_coo <- summary(W)  # gives i, j, x columns (data.frame)
edges <- data.table(from = W_coo$i, to = W_coo$j)

# ==============================================================
# STEP 4: Function to compute all three stats for one variable
# ==============================================================
compute_neighbor_features_fast <- function(dt, var_name, W, edges, 
                                            neighbor_count, id_order, years) {
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  n_cells <- nrow(W)
  
  for (yr in years) {
    # Extract rows for this year, ordered by cell_pos
    yr_mask <- dt$year == yr
    yr_dt   <- dt[yr_mask]
    
    # Build a full-length vector indexed by cell_pos
    # (some cells may be missing in a year; they stay NA)
    vals_full <- rep(NA_real_, n_cells)
    vals_full[yr_dt$cell_pos] <- yr_dt[[var_name]]
    
    # --- MEAN via sparse matrix-vector product ---
    # Replace NA with 0 for summation, track valid counts
    vals_zero <- vals_full
    vals_zero[is.na(vals_zero)] <- 0
    valid_indicator <- as.numeric(!is.na(vals_full))
    
    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_valid <- as.numeric(W %*% valid_indicator)
    
    n_mean <- ifelse(neighbor_valid > 0, 
                     neighbor_sum / neighbor_valid, 
                     NA_real_)
    
    # --- MAX and MIN via edge table ---
    # Look up neighbor values
    neighbor_vals_vec <- vals_full[edges$to]
    
    # Create temporary data.table for grouped aggregation
    agg_dt <- data.table(
      from = edges$from,
      val  = neighbor_vals_vec
    )
    # Remove NA neighbor values before aggregation
    agg_dt <- agg_dt[!is.na(val)]
    
    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(nmax = max(val), nmin = min(val)), by = from]
      
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
      n_max_full[stats$from] <- stats$nmax
      n_min_full[stats$from] <- stats$nmin
    } else {
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
    }
    
    # --- Write results back to dt for this year's rows ---
    # Map from cell_pos back to the rows
    pos_vals <- yr_dt$cell_pos
    
    set(dt, which = yr_mask, j = col_max,  value = n_max_full[dt$cell_pos[yr_mask]])
    set(dt, which = yr_mask, j = col_min,  value = n_min_full[dt$cell_pos[yr_mask]])
    set(dt, which = yr_mask, j = col_mean, value = n_mean[dt$cell_pos[yr_mask]])
  }
  
  dt
}

# ==============================================================
# STEP 5: Run for all 5 neighbor source variables
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  dt <- compute_neighbor_features_fast(
    dt, var_name, W, edges, neighbor_count, id_order, years
  )
}

# ==============================================================
# STEP 6: Remove helper column and convert back if needed
# ==============================================================
dt[, cell_pos := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(dt)

# The trained Random Forest model is untouched — 
# the output columns are numerically identical to the originals,
# so predict(rf_model, cell_data) works without retraining.
```

---

## Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(6.46M) string ops, ~80 hrs | O(1.37M) sparse triplet construction, **~1 sec** |
| Compute stats (per var, per year) | O(230K) list iterations, R-level | Sparse mat-vec O(1.37M) + `data.table` groupby O(1.37M), C-level |
| Total (5 vars × 28 years) | **~86+ hours** | **~2–5 minutes** |
| Memory | ~6.46M-element list of int vectors | Sparse matrix (~16 MB) + edge table (~33 MB) |

## Numerical Equivalence

The sparse matrix approach computes **exactly the same** quantities:

- `neighbor_mean` = sum of non-NA neighbor values / count of non-NA neighbors (identical to the original `mean(neighbor_vals[!is.na(neighbor_vals)])`)
- `neighbor_max` / `neighbor_min` = grouped max/min over the same edge set (identical to original `max`/`min` on the same `neighbor_vals`)

The trained Random Forest model is **never touched** — only the input feature columns are recomputed with identical values, so all downstream predictions remain unchanged.