 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It constructs a lookup of **row indices** for every cell-year combination — all ~6.46 million rows — even though the **neighbor topology is invariant across years**. The function pastes IDs and years into keys, hashes them, and then does ~6.46 million lookups into a named character vector of length ~6.46 million. This produces a list of ~6.46 million elements, each containing row indices into the full panel. The `compute_neighbor_stats` function then loops over that same list of ~6.46 million elements per variable.

**Root cause:** The code conflates the **static spatial graph** (which cell is neighbor to which cell — 344,208 cells, ~1.37M edges, year-invariant) with the **dynamic panel data** (variable values that change by year). By rebuilding this mapping per cell-year row instead of per cell, the cost is multiplied by 28× and all string operations (paste, hash lookup) are applied to millions of rows unnecessarily.

**Specific costs:**
1. `build_neighbor_lookup`: Creates ~6.46M string keys, does ~6.46M hash lookups → very slow.
2. `compute_neighbor_stats`: Iterates an R-level `lapply` over ~6.46M elements per variable → slow.
3. Memory: The `neighbor_lookup` list holds ~6.46M integer vectors → large.

---

## Optimization Strategy

**Separate static topology from dynamic values:**

1. **Build the neighbor graph once at the cell level (344K cells, not 6.46M cell-years).** Create a mapping from each cell's position in `id_order` to its neighbors' positions in `id_order`. This is a simple re-index of `rook_neighbors_unique` — essentially free.

2. **Organize data so that values for each year can be extracted as a matrix.** Sort the data by `(id, year)` or `(year, id)` and reshape the variable columns into matrices of dimension `[n_cells × n_years]`. Then neighbor stats become matrix operations on indexed rows.

3. **Vectorize neighbor stat computation.** For each variable, build a sparse matrix or use fast row-indexed operations: for each cell, gather neighbor rows from the matrix, compute max/min/mean across neighbors for all 28 years simultaneously.

4. **Use `data.table` for speed** in reshaping and joining results back.

**Complexity reduction:**
- Current: O(n_cells × n_years) string hashing and list construction → ~6.46M operations in slow R.
- Proposed: O(n_cells) integer list construction + O(n_cells × n_years) vectorized numeric operations → ~344K list elements, vectorized column operations.

**Expected speedup:** From 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure data.table format and sort consistently
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# Ensure id_order is the canonical ordering (positions 1..N match rook_neighbors_unique)
# id_order: integer/character vector of cell IDs in the order matching rook_neighbors_unique
n_cells <- length(id_order)

# Create a map: cell_id -> position in id_order (1-based index into rook_neighbors_unique)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# ==============================================================================
# STEP 1: Build STATIC neighbor index at the cell level (not cell-year level)
#
# cell_neighbor_pos[[i]] = integer vector of positions (in id_order) that are
#                          neighbors of the cell at position i.
# This is just rook_neighbors_unique itself (an nb object is already a list of
# integer position vectors), but we ensure it's clean.
# ==============================================================================
cell_neighbor_pos <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]
})
# cell_neighbor_pos[[i]] gives neighbor positions for cell at position i in id_order


# ==============================================================================
# STEP 2: Reshape each variable into a matrix: n_cells rows × n_years columns
#
# Row i corresponds to id_order[i].
# Column j corresponds to the j-th year in sorted order.
# ==============================================================================
years_sorted <- sort(unique(cell_dt$year))
n_years <- length(years_sorted)

# Add position column to data.table
cell_dt[, cell_pos := id_to_pos[as.character(id)]]

# Sort by cell_pos and year for consistent matrix filling
setkey(cell_dt, cell_pos, year)

# Verify we have complete panel (each cell appears in each year)
# If not complete, the matrix approach still works but needs NA-filling
stopifnot(nrow(cell_dt) == n_cells * n_years)

# Build year-to-column-index map
year_to_col <- setNames(seq_along(years_sorted), as.character(years_sorted))
cell_dt[, year_col := year_to_col[as.character(year)]]

# Function to extract a variable as an [n_cells x n_years] matrix
var_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]
  mat
}

# ==============================================================================
# STEP 3: Compute neighbor stats per variable using the static graph + matrices
#
# For each cell i (row i in the matrix), gather the rows of its neighbors,
# then compute columnwise (i.e., per-year) max, min, mean.
#
# To avoid an R-level loop over 344K cells being too slow, we use a sparse
# approach: build an edge list and use data.table grouping.
# ==============================================================================

# Build edge list: (focal_pos, neighbor_pos) — one row per directed edge
# This is static and reused for every variable.
edge_focal <- rep(seq_len(n_cells), times = lengths(cell_neighbor_pos))
edge_neighbor <- unlist(cell_neighbor_pos, use.names = FALSE)
n_edges <- length(edge_focal)

cat(sprintf("Edge list built: %d directed edges\n", n_edges))

# For each variable, we need neighbor max/min/mean per cell per year.
# Strategy: index into the matrix using the edge list, then group by focal cell.
#
# edge_values[e, y] = var_matrix[edge_neighbor[e], y]
# Then group by edge_focal[e] and compute max/min/mean per column (year).
#
# With ~1.37M edges × 28 years this is ~38.4M values — fits comfortably in RAM.

compute_neighbor_stats_fast <- function(var_matrix, edge_focal, edge_neighbor,
                                        n_cells, n_years) {
  # Extract neighbor values for all edges: matrix [n_edges x n_years]
  neighbor_vals <- var_matrix[edge_neighbor, , drop = FALSE]  # n_edges x n_years

  # We need to group rows of neighbor_vals by edge_focal and compute
  # max, min, mean per year-column.
  #
  # Use a C-level split via data.table or manual approach.
  # Since n_edges ~ 1.37M and n_years = 28, we can use matrix splitting.

  # Pre-allocate result matrices
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Use data.table for fast grouped operations
  # Melt neighbor_vals into long form: (edge_id, year_col, value)
  # Then group by (focal, year_col)
  #
  # But 1.37M × 28 = 38.4M rows is manageable.

  # Alternative: loop over years (only 28 iterations — very fast)
  for (y in seq_len(n_years)) {
    col_vals <- neighbor_vals[, y]  # length n_edges

    # Use data.table for fast grouped max/min/mean
    dt_tmp <- data.table(focal = edge_focal, val = col_vals)
    # Remove NAs before aggregation
    dt_tmp <- dt_tmp[!is.na(val)]

    if (nrow(dt_tmp) > 0) {
      agg <- dt_tmp[, .(
        vmax  = max(val),
        vmin  = min(val),
        vmean = mean(val)
      ), by = focal]

      max_mat[agg$focal, y]  <- agg$vmax
      min_mat[agg$focal, y]  <- agg$vmin
      mean_mat[agg$focal, y] <- agg$vmean
    }
  }

  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# ==============================================================================
# STEP 4: Run for all neighbor source variables and attach results to cell_dt
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Build the [n_cells x n_years] matrix for this variable
  var_matrix <- var_to_matrix(cell_dt, var_name, n_cells, n_years)

  # Compute neighbor stats
  stats <- compute_neighbor_stats_fast(var_matrix, edge_focal, edge_neighbor,
                                       n_cells, n_years)

  # Map results back to cell_dt rows using (cell_pos, year_col)
  idx <- cbind(cell_dt$cell_pos, cell_dt$year_col)

  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (max_col)  := stats$max[idx]]
  cell_dt[, (min_col)  := stats$min[idx]]
  cell_dt[, (mean_col) := stats$mean[idx]]

  cat(sprintf("  Done: %s\n", var_name))
}

# ==============================================================================
# STEP 5: Clean up helper columns and convert back if needed
# ==============================================================================
cell_dt[, c("cell_pos", "year_col") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object and prediction code remain exactly as before.
# e.g.:
# predictions <- predict(rf_model, newdata = cell_data)
#
# The column names (var_neighbor_max, var_neighbor_min, var_neighbor_mean)
# and their numerical values are identical to the original implementation,
# preserving the original numerical estimand.
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup level** | Cell-year (6.46M entries) | Cell (344K entries, static) |
| **String hashing** | ~6.46M paste + hash ops | Zero |
| **Lookup list size** | ~6.46M elements | Edge list: ~1.37M pairs (reused) |
| **Stat computation loop** | R `lapply` over 6.46M per variable | 28 `data.table` grouped aggregations per variable |
| **Total inner iterations** | ~32.3M (5 vars × 6.46M) | 140 (5 vars × 28 years), each vectorized over ~1.37M edges |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** |
| **Peak RAM** | Large (6.46M-element list + overhead) | ~1–2 GB (matrices + edge list) |
| **Numerical output** | Baseline | **Identical** — same max/min/mean over same neighbor sets |
| **RF model** | Pre-trained | **Unchanged, not retrained** |

The key insight: **factoring the invariant spatial topology out of the per-year computation** eliminates millions of redundant string operations, reduces the data structure from 6.46M entries to 344K + a reusable edge list, and enables vectorized year-by-year aggregation via `data.table`.