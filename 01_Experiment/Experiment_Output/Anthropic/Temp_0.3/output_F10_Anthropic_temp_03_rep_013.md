 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query on large vectors, and `paste()`-based key construction for 6.46M rows × ~4 neighbors each is extremely expensive. This single step likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting subsets of a numeric vector. While each individual operation is small, the R-level loop overhead across 6.46M iterations, repeated for 5 variables (×3 stats = 15 new columns), compounds severely.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it were year-dependent.** Every cell has the same rook neighbors every year. The code re-resolves neighbor row indices per cell-year, even though the spatial adjacency graph is static. This means the topology work is repeated 28× implicitly.

**Root cause summary:** The design conflates the spatial graph (344K nodes, ~1.37M edges) with the panel structure (28 years). The lookup should be built once on the 344K-cell spatial graph and then projected across years via vectorized integer arithmetic, not string matching.

## Optimization Strategy

1. **Build a sparse adjacency structure once** over the 344K cells using a two-column integer edge list (from, to). This is O(E) where E ≈ 1.37M.

2. **Exploit the panel's regular structure.** If data is sorted by `(id, year)` — or `(year, id)` — then the row index of any `(cell_i, year_t)` can be computed by arithmetic: `offset[cell_i] + (year_t - min_year)`. No string keys needed.

3. **Vectorize the aggregation using sparse matrix multiplication.** Construct a sparse `N×N` adjacency matrix `A` (where N = 344,208). For each year, extract the variable column as a vector over cells, then use `A` to compute neighbor sums and neighbor counts in one matrix-vector multiply. Max and min require a grouped operation, but can be done efficiently with `data.table` or a compiled C++ snippet via `Rcpp`.

4. **For max/min:** Use `data.table` joins on the integer edge list — expand edges, join variable values, and group-aggregate. This is highly optimized internally in `data.table` (radix-based, in-place).

5. **Process all 28 years in a vectorized batch** per variable, or loop over 28 years (not 6.46M rows).

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes 5 variables × 28 years × ~1.37M edge expansions with `data.table` group-by, which is trivial).

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 0: Ensure cell_data is a data.table sorted by (id, year) ---------
cell_dt <- as.data.table(cell_data)
setkeyv(cell_dt, c("id", "year"))

# Unique cell IDs in sorted order and unique years
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_cells      <- length(unique_ids)
n_years      <- length(unique_years)

stopifnot(nrow(cell_dt) == n_cells * n_years)  # balanced panel check

# Map cell id -> integer index 1..n_cells
id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Map year -> integer index 1..n_years
year_to_idx <- setNames(seq_along(unique_years), as.character(unique_years))

# Assign a sequential row locator: for sorted (id, year), row of (cell i, year t)
# = (i-1)*n_years + t   where i = id_to_idx[id], t = year_to_idx[year]
# Verify this matches the actual row order:
cell_dt[, row_check := (id_to_idx[as.character(id)] - 1L) * n_years +
                         year_to_idx[as.character(year)]]
stopifnot(all(cell_dt$row_check == seq_len(nrow(cell_dt))))
cell_dt[, row_check := NULL]

# ---- Step 1: Build edge list from rook_neighbors_unique (spdep nb object) --
# rook_neighbors_unique is a list of length n_cells; element [[i]] contains
# integer indices (into id_order) of neighbors of cell i.
# id_order is the vector of cell IDs in the order matching the nb object.

# Map id_order positions to our sorted unique_ids positions
id_order_to_sorted <- id_to_idx[as.character(id_order)]

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(data.table(from = integer(0), to = integer(0)))
  }
  data.table(
    from = id_order_to_sorted[i],
    to   = id_order_to_sorted[nb]
  )
}))

# Remove any NA edges (boundary cells whose neighbors don't exist)
edges <- edges[!is.na(from) & !is.na(to)]
setkey(edges, from)

cat(sprintf("Edge list: %d directed edges over %d cells\n", nrow(edges), n_cells))

# ---- Step 2: Function to compute neighbor max, min, mean for one variable ---
#
# For each cell i and year t, we need max/min/mean of variable values at
# neighbors of i in year t.
#
# Strategy: loop over 28 years (not 6.46M rows). For each year, extract the
# variable vector (length n_cells), join onto edge list, and group-aggregate.

add_neighbor_features <- function(dt, var_name, edges, n_cells, n_years,
                                  unique_years, year_to_idx) {
  # Pre-extract the full variable column as a matrix: n_cells x n_years
  # Row i, col t = value for cell i in year t
  # Because dt is keyed by (id, year), values are laid out as:

  #   cell1-year1, cell1-year2, ..., cell1-yearT, cell2-year1, ...
  vals_vec <- dt[[var_name]]
  # Reshape to matrix: rows=cells, cols=years (byrow=FALSE reads column-major,
  # which matches our layout since consecutive rows = consecutive years for same cell)
  vals_mat <- matrix(vals_vec, nrow = n_years, ncol = n_cells)
  # vals_mat[t, i] = value for cell i in year t
  # (R fills matrices column-major: first n_years entries -> column 1 = cell 1)

  # Prepare output matrices
  max_mat  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
  min_mat  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
  mean_mat <- matrix(NA_real_, nrow = n_years, ncol = n_cells)

  # edges$from, edges$to are integer cell indices
  e_from <- edges$from
  e_to   <- edges$to

  for (t in seq_len(n_years)) {
    # Neighbor values: for each edge (from -> to), get value at 'to' in year t
    nb_vals <- vals_mat[t, e_to]

    # Build a data.table for fast grouped aggregation
    agg_dt <- data.table(from = e_from, val = nb_vals)

    # Remove NAs before aggregation (matches original: neighbor_vals[!is.na()])
    agg_dt <- agg_dt[!is.na(val)]

    if (nrow(agg_dt) > 0L) {
      stats <- agg_dt[, .(nb_max  = max(val),
                           nb_min  = min(val),
                           nb_mean = mean(val)),
                       by = from]

      max_mat[t,  stats$from] <- stats$nb_max
      min_mat[t,  stats$from] <- stats$nb_min
      mean_mat[t, stats$from] <- stats$nb_mean
    }
  }

  # Flatten back to vector (column-major matches our row layout)
  max_col  <- paste0("max_",  var_name)
  min_col  <- paste0("min_",  var_name)
  mean_col <- paste0("mean_", var_name)

  dt[, (max_col)  := as.vector(max_mat)]
  dt[, (min_col)  := as.vector(min_mat)]
  dt[, (mean_col) := as.vector(mean_mat)]

  invisible(dt)
}

# ---- Step 3: Run for all 5 neighbor source variables ------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  t0 <- proc.time()
  add_neighbor_features(cell_dt, var_name, edges, n_cells, n_years,
                        unique_years, year_to_idx)
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Done in %.1f seconds\n", elapsed))
}

# ---- Step 4: Convert back to data.frame if the RF model expects one ---------
cell_data <- as.data.frame(cell_dt)

# ---- Step 5: Apply the pre-trained Random Forest (unchanged) ----------------
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` extracts neighbor values for cell `i` in year `t` | `vals_mat[t, e_to]` extracts the same values via the same edge pairs | Same values selected |
| `neighbor_vals[!is.na(neighbor_vals)]` | `agg_dt[!is.na(val)]` | Same NA removal |
| `max()`, `min()`, `mean()` on filtered vector | `max(val)`, `min(val)`, `mean(val)` in `data.table` group-by | Identical R functions, identical floating-point results |
| Cells with no valid neighbors → `c(NA, NA, NA)` | Matrix initialized to `NA_real_`; only cells appearing in `stats$from` are overwritten | Same NA pattern |

## Performance Analysis

| Component | Original | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string-paste + named-vector lookups → **hours** | 1.37M-row edge list built once → **seconds** |
| Per-variable aggregation | 6.46M R-level `lapply` iterations × 5 vars → **hours** | 28 `data.table` group-by operations on 1.37M rows × 5 vars = 140 calls → **minutes** |
| Memory | 6.46M-element list of integer vectors (~large) | One 1.37M × 2 integer edge table + temporary matrices (~modest) |
| **Total estimated time** | **86+ hours** | **5–15 minutes on 16 GB laptop** |

The key insight: the spatial graph has **344K nodes and 1.37M edges** — it's the panel expansion to 6.46M row-level list entries with string-key lookups that created the bottleneck. By separating the static topology (edge list) from the temporal dimension (28-iteration loop), we reduce the problem by a factor of ~4,700× in loop iterations and eliminate all string operations entirely.