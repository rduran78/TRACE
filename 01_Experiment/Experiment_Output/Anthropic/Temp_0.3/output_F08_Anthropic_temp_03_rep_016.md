 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor graph is static.** Cell *i*'s neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination (~6.46M entries), each time doing string-pasting and hash lookups.

2. **`compute_neighbor_stats` iterates over 6.46M list entries** with an `lapply` + `rbind` pattern, which is slow in R.

3. **The string-key lookup (`paste` + named vector indexing)** is O(n) in construction and O(1) amortized per lookup, but the sheer volume (6.46M × ~4 neighbors each ≈ 26M string operations) makes it a bottleneck.

**The key insight:** Since the neighbor topology is year-invariant, we should separate the problem into:

- **A static neighbor index** over the 344,208 cells (built once), and
- **Year-varying variable matrices** where we can vectorize the neighbor aggregation per year using matrix indexing.

This reduces the problem from 6.46M list iterations to 28 year-iterations over a 344K-cell vector, using fast vectorized matrix operations.

## Optimization Strategy

1. **Build a static cell-level neighbor structure once** — a simple mapping from cell position (1…344,208) to neighbor positions. This is essentially what `rook_neighbors_unique` (the `nb` object) already is.

2. **Convert the `nb` list to a sparse "edge list"** of (cell_index, neighbor_index) pairs (~1.37M rows). This allows fully vectorized aggregation.

3. **For each year and each variable**, extract the variable vector for that year, then use the edge list to gather all neighbor values, and compute grouped max/min/mean via `data.table` or base `tapply`-style vectorized operations.

4. **Join results back** into the main data.

This eliminates all per-row `lapply`, all string-key construction, and leverages vectorized C-level operations. Expected speedup: from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Ensure cell_data is a data.table with correct order
# ============================================================
setDT(cell_data)

# id_order is the vector of 344,208 cell IDs in the order matching
# rook_neighbors_unique (the nb object). This is already available.

# ============================================================
# STEP 1: Build a STATIC edge list from the nb object (once)
# ============================================================
# rook_neighbors_unique is a list of length 344,208.
# rook_neighbors_unique[[i]] gives integer indices (into id_order)
#   of the neighbors of cell id_order[i].

build_static_edge_list <- function(nb_obj) {
  # nb_obj: list of integer vectors (neighbor indices), length = n_cells
  n_neighbors <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    ni <- nb_obj[[i]]
    len <- length(ni)
    if (len > 0L) {
      from_idx[pos:(pos + len - 1L)] <- i
      to_idx[pos:(pos + len - 1L)]   <- ni
      pos <- pos + len
    }
  }
  
  data.table(from_cell_pos = from_idx, to_cell_pos = to_idx)
}

cat("Building static edge list from nb object...\n")
edge_list <- build_static_edge_list(rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

# ============================================================
# STEP 2: Build a cell-position lookup for cell_data
# ============================================================
# Map each cell ID to its position in id_order (1..344208)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add cell_pos to cell_data (static per cell, same across years)
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Ensure data is keyed for fast subsetting by year
setkey(cell_data, year, cell_pos)

# ============================================================
# STEP 3: Vectorized neighbor stat computation
# ============================================================
# For each year, for each variable:
#   - Extract the value vector indexed by cell_pos
#   - Use edge_list to gather neighbor values
#   - Compute grouped max, min, mean by from_cell_pos
#   - Write results back into cell_data

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-create output columns (initialized to NA)
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  if (!col_max  %in% names(cell_data)) cell_data[, (col_max)  := NA_real_]
  if (!col_min  %in% names(cell_data)) cell_data[, (col_min)  := NA_real_]
  if (!col_mean %in% names(cell_data)) cell_data[, (col_mean) := NA_real_]
}

years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

cat(sprintf("Computing neighbor stats: %d years x %d variables...\n",
            length(years), length(neighbor_source_vars)))

for (yr in years) {
  # Extract the subset for this year, ordered by cell_pos
  # Because we keyed on (year, cell_pos), this is fast
  yr_rows <- cell_data[.(yr)]  # keyed lookup
  
  # Build a full-length value vector indexed by cell_pos
  # (some cell_pos values might be missing if data is incomplete)
  # yr_rows$cell_pos gives positions; we need a vector of length n_cells
  
  for (var_name in neighbor_source_vars) {
    # Create a lookup vector: position -> value
    val_vec <- rep(NA_real_, n_cells)
    val_vec[yr_rows$cell_pos] <- yr_rows[[var_name]]
    
    # Gather neighbor values using the static edge list
    neighbor_vals <- val_vec[edge_list$to_cell_pos]
    
    # Compute grouped stats using data.table
    # edge_list$from_cell_pos identifies which cell each edge belongs to
    stats_dt <- data.table(
      from_cell_pos = edge_list$from_cell_pos,
      nval = neighbor_vals
    )
    
    # Remove NA neighbor values before aggregation
    stats_dt <- stats_dt[!is.na(nval)]
    
    # Aggregate
    agg <- stats_dt[, .(
      n_max  = max(nval),
      n_min  = min(nval),
      n_mean = mean(nval)
    ), by = from_cell_pos]
    
    # Build full-length result vectors (NA for cells with no valid neighbors)
    res_max  <- rep(NA_real_, n_cells)
    res_min  <- rep(NA_real_, n_cells)
    res_mean <- rep(NA_real_, n_cells)
    
    res_max[agg$from_cell_pos]  <- agg$n_max
    res_min[agg$from_cell_pos]  <- agg$n_min
    res_mean[agg$from_cell_pos] <- agg$n_mean
    
    # Write back into cell_data for this year's rows
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Use the row indices in the original cell_data
    row_indices <- cell_data[.(yr), which = TRUE]
    cell_pos_for_rows <- cell_data$cell_pos[row_indices]
    
    set(cell_data, i = row_indices, j = col_max,  value = res_max[cell_pos_for_rows])
    set(cell_data, i = row_indices, j = col_min,  value = res_min[cell_pos_for_rows])
    set(cell_data, i = row_indices, j = col_mean, value = res_mean[cell_pos_for_rows])
  }
  
  cat(sprintf("  Year %d done.\n", yr))
}

# Clean up helper column if not needed downstream
# cell_data[, cell_pos := NULL]

cat("Neighbor feature computation complete.\n")

# ============================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model is used as-is. No retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup** | 6.46M-entry list with string keys | 1.37M-row static edge list (built once) |
| **Per-variable computation** | `lapply` over 6.46M rows | Vectorized `data.table` grouped aggregation over 28 year-slices |
| **String operations** | ~26M `paste()` calls | Zero |
| **Time complexity** | O(rows × avg_neighbors) with R-loop overhead | O(years × edges) with C-level vectorized ops |
| **Expected runtime** | 86+ hours | ~5–15 minutes |
| **Numerical results** | max, min, mean of non-NA neighbor values | Identical — same aggregation functions, same NA handling |
| **RF model** | Pre-trained, not retrained | **Preserved exactly** — only the feature-engineering step is optimized |

The redesign exploits the **static-vs-changing distinction**: the neighbor graph (which cells are adjacent) is built once as a lightweight edge list, and only the variable values (which change by year) are gathered and aggregated per year using fast vectorized operations.