 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine).
2. Constructs character key strings via `paste()` for every neighbor × that row's year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching on a vector of length ~6.46M.

The result is a **list of 6.46 million integer vectors**. The `paste()`-based key construction and named-vector lookup are extremely slow at this scale. Named vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of character hashing over millions of entries is enormous). Doing this ~6.46M × ~4 neighbors ≈ 25+ million times is the primary time sink.

### Bottleneck B: `compute_neighbor_stats` — repeated `lapply` over 6.46M elements

For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the index vectors from the lookup. This is done 5 times. The per-element overhead of `lapply` with anonymous functions over millions of elements is substantial, though secondary to Bottleneck A.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a **complete regular grid with uniform time dimension**. Here the panel is cell × year, stored long, with an irregular neighbor structure (`spdep::nb`). Focal operations would require reshaping into a 3D raster stack per variable per year, running focal, then reshaping back — and would not naturally handle missing cells or the panel structure. The analogy is useful conceptually (we want a "moving window" summary), but the implementation should use **sparse matrix multiplication and vectorized operations** on the long-format panel, which preserves the exact numerical results.

---

## 2. Optimization Strategy

### Step 1: Replace the character-key lookup with integer arithmetic

Instead of `paste(id, year)` keys, exploit the panel structure. If we map each `(cell_index, year_index)` pair to a row number via a **dense integer matrix or direct arithmetic**, lookup becomes O(1) with no string operations.

Specifically, if we sort the data by `(id, year)` and the panel is balanced (344,208 cells × 28 years = 9,637,824 potential rows; actual = 6.46M so it's unbalanced), we use an **integer matrix** `row_matrix[cell_index, year_index]` that maps to the row number in `cell_data`. This matrix has 344,208 × 28 ≈ 9.6M entries (just ~38 MB as integers) — trivially fits in RAM.

### Step 2: Build the neighbor lookup via vectorized matrix indexing

Instead of looping over 6.46M rows, we:
1. Build `row_matrix` (cell × year → row number, NA if missing).
2. Build a sparse neighbor edge list from the `nb` object (just ~1.37M directed edges).
3. For each row in `cell_data`, its neighbors are determined by its cell's spatial neighbors (from the edge list) at the same year. We can compute all ~25M neighbor-row indices in one vectorized operation using the edge list and `row_matrix`.

### Step 3: Replace `lapply`-based stats with sparse matrix multiplication

For `max`, `min`, and `mean`:
- Build a **sparse adjacency matrix** W of dimension (n_rows × n_rows) where `W[i, j] = 1` if row j is a rook neighbor of row i (same year).
- **Mean**: `W_rowstandardized %*% x` gives neighbor means in one matrix-vector multiply.
- **Max/Min**: Use grouped operations via `data.table` on the edge list, which is fully vectorized.

This replaces 5 × 6.46M `lapply` iterations with a handful of vectorized operations.

### Expected speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | ~40-60 hours | ~10-30 seconds |
| Stats computation (5 vars × 3 stats) | ~20-30 hours | ~30-60 seconds |
| **Total** | **~86+ hours** | **~1-3 minutes** |

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)
library(spdep)    # for nb object structure
library(ranger)   # or randomForest — for prediction only

# ============================================================
# STEP 0: Ensure cell_data is a data.table with id and year
# ============================================================
cell_data <- as.data.table(cell_data)

# id_order: the vector of unique cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: the spdep nb object (list of integer index vectors)

# ============================================================
# STEP 1: Build integer mappings
# ============================================================

# Map cell IDs to integer indices (matching id_order)
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Map years to integer indices
all_years  <- sort(unique(cell_data$year))
year_to_idx <- setNames(seq_along(all_years), as.character(all_years))

n_cells <- length(id_order)
n_years <- length(all_years)

# Add integer indices to cell_data
cell_data[, cell_idx := id_to_idx[as.character(id)]]
cell_data[, year_idx := year_to_idx[as.character(year)]]

# Assign a row identifier (preserve original order for final output)
cell_data[, row_id := .I]

# ============================================================
# STEP 2: Build row_matrix — maps (cell_idx, year_idx) -> row_id
#   This is a dense integer matrix; NA means that cell-year is absent.
# ============================================================

row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_matrix[cbind(cell_data$cell_idx, cell_data$year_idx)] <- cell_data$row_id

cat("Row matrix built:", n_cells, "x", n_years, "\n")

# ============================================================
# STEP 3: Build directed edge list from nb object
#   Each entry: (from_cell_idx, to_cell_idx)
# ============================================================

edge_from <- rep(seq_along(rook_neighbors_unique),
                 lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique)

# Remove any 0-neighbor entries (spdep uses integer(0) for islands)
valid <- edge_to > 0L
edge_from <- edge_from[valid]
edge_to   <- edge_to[valid]

n_edges <- length(edge_from)
cat("Spatial directed edges:", n_edges, "\n")

# ============================================================
# STEP 4: Expand edge list across all years — vectorized
#   For each spatial edge (i -> j) and each year t,
#   if both row_matrix[i, t] and row_matrix[j, t] exist,
#   then row row_matrix[i, t] has neighbor row row_matrix[j, t].
# ============================================================

# We'll iterate over years (only 28 iterations — trivial)
edge_list_parts <- vector("list", n_years)

for (t in seq_len(n_years)) {
  from_rows <- row_matrix[edge_from, t]  # vectorized column extraction
  to_rows   <- row_matrix[edge_to,   t]
  
  both_exist <- !is.na(from_rows) & !is.na(to_rows)
  
  edge_list_parts[[t]] <- data.table(
    from_row = from_rows[both_exist],
    to_row   = to_rows[both_exist]
  )
}

edges_dt <- rbindlist(edge_list_parts)
rm(edge_list_parts)

cat("Total cell-year directed neighbor edges:", nrow(edges_dt), "\n")

# ============================================================
# STEP 5: Compute neighbor stats for each variable — vectorized
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
n_rows <- nrow(cell_data)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Extract the neighbor values for every edge
  vals <- cell_data[[var_name]]
  edges_dt[, val := vals[to_row]]
  
  # Remove edges where the neighbor value is NA
  valid_edges <- edges_dt[!is.na(val)]
  
  # Compute grouped stats: max, min, mean by from_row
  stats <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row]
  
  # Initialize result columns with NA
  max_col  <- rep(NA_real_, n_rows)
  min_col  <- rep(NA_real_, n_rows)
  mean_col <- rep(NA_real_, n_rows)
  
  # Fill in computed values
  max_col[stats$from_row]  <- stats$nb_max
  min_col[stats$from_row]  <- stats$nb_min
  mean_col[stats$from_row] <- stats$nb_mean
  
  # Add to cell_data with the expected column names
  # (Adjust naming convention to match what the trained RF model expects)
  set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = max_col)
  set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = min_col)
  set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  
  cat("  Done:", var_name, "\n")
}

# Clean up temporary columns
edges_dt[, val := NULL]
cell_data[, c("cell_idx", "year_idx", "row_id") := NULL]

# ============================================================
# STEP 6: Predict with the pre-trained Random Forest
#   (Model is NOT retrained — only used for prediction)
# ============================================================

# Example (adjust to your actual model object and column names):
# predictions <- predict(trained_rf_model, data = cell_data)
# cell_data[, prediction := predictions$predictions]

cat("Neighbor feature engineering complete.\n")
cat("Rows:", nrow(cell_data), " Columns:", ncol(cell_data), "\n")
```

---

## Summary of Key Design Decisions

| Decision | Rationale |
|---|---|
| **Integer matrix `row_matrix`** instead of character key lookup | Eliminates ~25M `paste()` calls and named-vector lookups. Matrix indexing is O(1). ~38 MB fits easily in 16 GB RAM. |
| **Loop over 28 years** (not 6.46M rows) to expand edges | Each iteration is fully vectorized column extraction from the matrix. 28 iterations is negligible. |
| **`data.table` grouped aggregation** for max/min/mean | `data.table`'s radix-sort-based grouping on integer keys is orders of magnitude faster than 6.46M `lapply` calls. Single pass per variable. |
| **No raster focal approach** | The panel is unbalanced, stored long-format, with an irregular `nb` structure. Reshaping to raster stacks per year would add complexity and potential for floating-point discrepancies. The `data.table` approach computes the **identical** max, min, and mean values. |
| **Pre-trained RF model untouched** | Only the feature engineering pipeline is optimized. The model object, its hyperparameters, and the prediction call remain unchanged. |
| **Numerical equivalence preserved** | `max()`, `min()`, and `mean()` are applied to the exact same sets of non-NA neighbor values as in the original code, producing identical results. |