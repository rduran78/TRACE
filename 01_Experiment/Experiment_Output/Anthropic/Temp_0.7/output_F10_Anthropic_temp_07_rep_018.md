 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(n²) neighbor lookup construction**: `build_neighbor_lookup` creates a lookup for every row (~6.46M rows) by string-pasting keys and matching them. The `paste()`/`setNames()`/indexing pattern over 6.46M rows with list-of-character-vector lookups is extremely slow.

2. **Row-wise `lapply` over 6.46M rows for each variable**: `compute_neighbor_stats` iterates row-by-row in pure R, computing `max`, `min`, `mean` for each node's neighbors. With 5 variables, that's ~32.3M R-level loop iterations, each involving subsetting, NA removal, and three summary functions.

3. **Redundant topology per row**: The neighbor graph topology is **year-invariant** (rook adjacency depends only on spatial position), yet the lookup is built at the cell-year level, duplicating the same spatial adjacency structure 28 times.

**Key insight**: The adjacency graph has only 344,208 nodes and ~1.37M directed edges. This is a **sparse graph**. The neighbor aggregation (max, min, mean of neighbor attributes) is a **sparse matrix–vector operation** that can be vectorized using sparse matrix algebra and `data.table` grouped operations—eliminating all R-level row loops.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** (344,208 × 344,208 with ~1.37M nonzeros). This encodes the rook neighbor topology at the cell level, not the cell-year level.

2. **For each variable and each year**, extract the attribute vector (length 344,208), then use:
   - Sparse matrix–vector multiply for **mean** (with a degree-normalized adjacency matrix) or sum (then divide by degree).
   - For **max** and **min**: use `data.table` with an edge list and grouped aggregation, which is vectorized C-level code.

3. **Join results back** to the panel `data.table` by `(id, year)`.

This reduces ~86 hours to minutes by:
- Replacing 6.46M × 5 R-level `lapply` calls with vectorized sparse operations.
- Building topology once at the cell level (344K nodes), not the cell-year level (6.46M rows).
- Leveraging `data.table` and `Matrix` package internals (C/Fortran).

---

## Optimized R Code

```r
# ==============================================================================
# Optimized Sparse-Graph Neighbor Aggregation Pipeline
# Numerically equivalent to the original; preserves trained RF model.
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and all
#         predictor variables. id_order is the vector of unique cell IDs
#         (same order as rook_neighbors_unique). rook_neighbors_unique is an
#         spdep::nb object (list of integer index vectors).
# --------------------------------------------------------------------------

setDT(cell_data)

# Unique cell IDs in canonical order (matching nb object indexing)
# id_order: length 344,208
n_cells <- length(id_order)

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency structure ONCE at the cell level
# --------------------------------------------------------------------------

cat("Building sparse adjacency structure...\n")

# Create edge list from the nb object
# Each entry rook_neighbors_unique[[i]] gives the indices (into id_order)
#   of the neighbors of cell id_order[i].
edge_from <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique)

# Map cell index -> cell id
# id_from and id_to are the actual cell IDs on each edge
id_from <- id_order[edge_from]
id_to   <- id_order[edge_to]

# Edge table at the cell level (no year dimension yet)
# "from" node wants to aggregate attributes of "to" nodes
edges_dt <- data.table(from_id = id_from, to_id = id_to)

# Degree of each node (number of neighbors), for computing mean
degree_dt <- edges_dt[, .(degree = .N), by = from_id]

# Create a mapping from cell id -> integer index (1..n_cells)
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# Sparse adjacency matrix (for potential sum-based mean computation)
# A[i,j] = 1 means j is a neighbor of i (i aggregates from j)
adj_sparse <- sparseMatrix(
  i = id_to_idx[as.character(edges_dt$from_id)],
  j = id_to_idx[as.character(edges_dt$to_id)],
  x = 1,
  dims = c(n_cells, n_cells)
)

cat("Adjacency structure built:", length(edge_from), "directed edges.\n")

# --------------------------------------------------------------------------
# STEP 2: Prepare the panel data for fast lookups
# --------------------------------------------------------------------------

# Ensure sorted for fast keyed joins
setkey(cell_data, id, year)

# Get sorted unique years
years <- sort(unique(cell_data$year))

# Create integer cell index column for sparse matrix operations
cell_data[, cell_idx := id_to_idx[as.character(id)]]

# Pre-allocate output columns
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("nb_max_", var_name)
  col_min  <- paste0("nb_min_", var_name)
  col_mean <- paste0("nb_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor statistics per variable per year
#          Using vectorized data.table grouped operations on the edge list.
# --------------------------------------------------------------------------

cat("Computing neighbor statistics...\n")

# We iterate over years (28) × variables (5) = 140 iterations.
# Each iteration is fully vectorized over ~1.37M edges.

for (yr in years) {
  
  # Extract the slice for this year
  # Key lookup is fast
  yr_data <- cell_data[year == yr, ]
  
  # Build a lookup from cell id -> row index in cell_data for this year
  # (so we can write results back)
  yr_row_indices <- which(cell_data$year == yr)
  yr_id_to_row   <- setNames(yr_row_indices, as.character(yr_data$id))
  
  # For each neighbor source variable
  for (var_name in neighbor_source_vars) {
    
    col_max  <- paste0("nb_max_", var_name)
    col_min  <- paste0("nb_min_", var_name)
    col_mean <- paste0("nb_mean_", var_name)
    
    # Build a lookup: cell_id -> variable value for this year
    # Use a named numeric vector for fast lookup
    val_vec <- setNames(yr_data[[var_name]], as.character(yr_data$id))
    
    # Look up the "to" (neighbor) node's value for each edge
    neighbor_vals <- val_vec[as.character(edges_dt$to_id)]
    
    # Build a temporary edge table with values
    # Only keep edges where the "from" node exists in this year's data
    # and the neighbor value is not NA
    work_dt <- data.table(
      from_id = edges_dt$from_id,
      val     = as.numeric(neighbor_vals)
    )
    
    # Remove edges where neighbor value is NA
    work_dt <- work_dt[!is.na(val)]
    
    # Also only keep edges whose from_id exists in this year
    # (should be all of them for a balanced panel, but be safe)
    work_dt <- work_dt[as.character(from_id) %chin% as.character(yr_data$id)]
    
    # Grouped aggregation: max, min, mean per from_id
    # This is a single vectorized pass in data.table (C level)
    agg <- work_dt[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from_id]
    
    # Write results back into cell_data at the correct rows
    match_rows <- yr_id_to_row[as.character(agg$from_id)]
    
    set(cell_data, i = match_rows, j = col_max,  value = agg$nb_max)
    set(cell_data, i = match_rows, j = col_min,  value = agg$nb_min)
    set(cell_data, i = match_rows, j = col_mean, value = agg$nb_mean)
  }
  
  cat("  Year", yr, "done.\n")
}

cat("Neighbor feature computation complete.\n")

# --------------------------------------------------------------------------
# STEP 4: Apply the pre-trained Random Forest model (unchanged)
# --------------------------------------------------------------------------

# The RF model object (e.g., `rf_model`) is already in memory.
# Predict using the augmented cell_data which now has all ~110 predictors
# including the 15 new neighbor-aggregated features.

# Ensure column order/names match what the RF expects:
# predictor_names <- rf_model$forest$independent.variable.names  # if ranger
# or similar accessor for randomForest

cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# Adjust the predict() call to match your RF package (ranger, randomForest, etc.)
```

---

## Further Optimization: Sparse Matrix Approach for Mean (Optional)

If memory allows, the **mean** can be computed even faster via sparse matrix–vector multiplication, avoiding the `data.table` groupby for that statistic:

```r
# --------------------------------------------------------------------------
# ALTERNATIVE: Sparse matrix-vector multiply for neighbor mean
# --------------------------------------------------------------------------

# Build degree vector aligned to cell index
degree_vec <- rep(0, n_cells)
deg_tab <- table(edge_from)
degree_vec[as.integer(names(deg_tab))] <- as.integer(deg_tab)

# For a given year and variable:
#   1. Construct value vector v of length n_cells (NA -> 0, track missingness)
#   2. neighbor_sum = adj_sparse %*% v
#   3. neighbor_count = adj_sparse %*% (!is.na(original_v))
#   4. neighbor_mean = neighbor_sum / neighbor_count

compute_neighbor_mean_sparse <- function(cell_data_yr, var_name,
                                          adj_sparse, id_to_idx, n_cells) {
  # Value vector aligned to cell index
  v <- rep(NA_real_, n_cells)
  idx <- id_to_idx[as.character(cell_data_yr$id)]
  v[idx] <- cell_data_yr[[var_name]]
  
  # Replace NA with 0 for multiplication; track non-NA
  v_clean <- ifelse(is.na(v), 0, v)
  v_valid <- as.numeric(!is.na(v))
  
  nb_sum   <- as.numeric(adj_sparse %*% v_clean)
  nb_count <- as.numeric(adj_sparse %*% v_valid)
  
  nb_mean <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
  
  # Return aligned to the cells present in cell_data_yr
  nb_mean[idx]
}
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Topology construction | 6.46M string-paste keys | 1.37M integer edge list, built once |
| Per-variable aggregation | 6.46M R-level `lapply` calls | 140 vectorized `data.table` group-bys on ~1.37M edges |
| Memory pattern | 6.46M-element list of integer vectors | Single `data.table` + sparse matrix (~50 MB) |
| Estimated runtime | 86+ hours | **~5–15 minutes** on 16 GB laptop |
| Numerical equivalence | Baseline | Identical `max`, `min`, `mean` over non-NA neighbor values |
| RF model | Unchanged | Unchanged—predict only, no retraining |

The key transformation: **lift the topology from the cell-year level (6.46M) to the cell level (344K), cross it with per-year attribute vectors using vectorized operations, and eliminate all R-level row loops.**