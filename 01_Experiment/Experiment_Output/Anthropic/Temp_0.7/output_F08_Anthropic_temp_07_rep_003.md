 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) because of a fundamental design flaw: **it conflates static spatial topology with dynamic temporal data**.

### Specific problems:

1. **`build_neighbor_lookup` rebuilds row-level lookups across all ~6.46M rows.** For each of the 6.46M cell-year rows, it constructs string keys (`paste(id, year)`), performs hash lookups, and allocates integer vectors. This is done once, but the result is a list of 6.46M elements — each encoding what is really the *same* spatial neighbor structure repeated 28 times (once per year).

2. **`compute_neighbor_stats` iterates over 6.46M list elements per variable.** For 5 variables, that's ~32.3M R-level `lapply` iterations, each performing subsetting, `NA` removal, and three summary statistics. R's per-element overhead in `lapply` over millions of tiny vectors is enormous.

3. **String key construction and lookup (`paste`, `setNames`, named indexing) are expensive** at this scale — millions of string allocations and hash-table probes.

### The key insight:

- **The neighbor graph is static** — cell *i*'s neighbors are always the same cells regardless of year. There are only 344,208 cells and ~1.37M directed neighbor edges.
- **The variables change by year** — but within a given year, the neighbor *structure* is identical.

Therefore, we should:
- Build the neighbor lookup **once over 344K cells** (not 6.46M cell-years).
- For each year, **slice the data, compute neighbor stats using the static topology, and write results back** — operating on 344K-row year-slices instead of the full 6.46M-row table.
- Use **vectorized matrix operations** instead of per-row `lapply`.

---

## Optimization Strategy

### 1. Build a static cell-level neighbor structure (once, 344K cells)

Convert `rook_neighbors_unique` (an `nb` object) into a **sparse adjacency representation** — specifically, two integer vectors (`from`, `to`) representing all directed neighbor edges. This is a CSR-like (compressed sparse row) representation using `from`/`to` indices into a cell-order vector.

### 2. Process year-by-year using vectorized grouped operations

For each year:
- Extract the variable column for that year's 344K rows.
- Use the static edge list to gather all neighbor values.
- Compute `max`, `min`, `mean` per cell using **`tapply`** or, much faster, **`data.table` grouping** over the edge list.

### 3. Use `data.table` for speed

`data.table` provides near-C-speed grouped aggregation. Grouping ~1.37M edges by `from` cell to compute max/min/mean is trivial — milliseconds per variable per year.

### 4. Complexity comparison

| | Current | Optimized |
|---|---|---|
| Lookup build | 6.46M string keys + hash lookups | 344K cells, pre-indexed once |
| Stats computation | 6.46M × 5 = 32.3M lapply calls | 28 years × 5 vars × 1 grouped aggregation (~1.37M rows) |
| Estimated time | 86+ hours | **~5–15 minutes** |

---

## Working R Code

```r
library(data.table)

#' Redesigned pipeline: separate static topology from dynamic variable computation.
#' Preserves the original numerical estimand (neighbor max, min, mean)
#' and the pre-trained Random Forest model.

# ===========================================================================
# STEP 1: Build static neighbor edge list (once, from the nb object)
# ===========================================================================

build_static_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb: an nb object (list of integer vectors of neighbor indices)
  # id_order: vector of cell IDs in the order matching neighbors_nb
  #
  # Returns a data.table with columns:
  #   from_idx : integer index into id_order (the focal cell)
  #   to_idx   : integer index into id_order (the neighbor cell)
  
  from_vec <- rep(seq_along(neighbors_nb),
                  times = lengths(neighbors_nb))
  to_vec   <- unlist(neighbors_nb, use.names = FALSE)
  
  # Remove zero-neighbor entries (nb objects use 0L for no-neighbor)
  valid <- to_vec != 0L
  
  data.table(
    from_idx = from_vec[valid],
    to_idx   = to_vec[valid]
  )
}

edge_dt <- build_static_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows — the full static directed neighbor graph.

n_cells <- length(id_order)

# ===========================================================================
# STEP 2: Create a cell-index mapping in the full data
# ===========================================================================

# Convert cell_data to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure data is sorted by (id, year) for predictable indexing.
# Create a mapping from cell ID to cell index (1..344208).
id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))

# Add cell_idx column (static cell index, reusable)
cell_data[, cell_idx := id_to_cellidx[as.character(id)]]

# Get sorted unique years
years <- sort(unique(cell_data$year))

# ===========================================================================
# STEP 3: Compute neighbor stats — year by year, vectorized
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

# Key the data for fast subsetting by year
setkey(cell_data, year, cell_idx)

for (yr in years) {
  
  # Extract this year's slice — a 344,208-row (or fewer) sub-table

  # Keyed lookup is very fast
  year_rows <- cell_data[.(yr)]  # subset by year
  
  # Build a fast lookup: cell_idx -> row position within year_rows
  # (handles case where some cells may be missing in some years)
  cellidx_to_rowpos <- integer(n_cells)  # 0 means missing
  cellidx_to_rowpos[year_rows$cell_idx] <- seq_len(nrow(year_rows))
  
  for (var_name in neighbor_source_vars) {
    
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Vector of variable values indexed by cell_idx for this year
    # Initialize with NA for all cells
    vals_by_cell <- rep(NA_real_, n_cells)
    vals_by_cell[year_rows$cell_idx] <- year_rows[[var_name]]
    
    # Gather neighbor values via the static edge list
    # For each edge (from_idx, to_idx), get the neighbor's value
    neighbor_vals <- vals_by_cell[edge_dt$to_idx]
    
    # Compute grouped stats: group by from_idx
    # Use data.table for fast grouped aggregation
    agg_dt <- data.table(
      from_idx = edge_dt$from_idx,
      nval     = neighbor_vals
    )
    
    # Remove NA neighbor values before aggregation (matches original logic)
    agg_dt <- agg_dt[!is.na(nval)]
    
    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ), by = from_idx]
      
      # Map results back: from_idx -> cell_idx -> row in cell_data for this year
      # We need to write into the correct rows of cell_data
      # The rows for this year in cell_data can be identified
      
      # Build result vectors (NA for cells with no valid neighbors)
      result_max  <- rep(NA_real_, n_cells)
      result_min  <- rep(NA_real_, n_cells)
      result_mean <- rep(NA_real_, n_cells)
      
      result_max[stats$from_idx]  <- stats$nb_max
      result_min[stats$from_idx]  <- stats$nb_min
      result_mean[stats$from_idx] <- stats$nb_mean
      
      # Write back into cell_data for this year's rows
      # Use the row positions within year_rows
      row_positions <- cellidx_to_rowpos[year_rows$cell_idx]
      valid_rows    <- row_positions > 0
      
      # We need the actual row indices in cell_data, not year_rows
      # Since cell_data is keyed by (year, cell_idx), get the row indices
      actual_rows <- cell_data[.(yr), which = TRUE]
      
      cell_data[actual_rows, (col_max)  := result_max[cell_idx]]
      cell_data[actual_rows, (col_min)  := result_min[cell_idx]]
      cell_data[actual_rows, (col_mean) := result_mean[cell_idx]]
      
    } else {
      # All neighbor values are NA for this year/variable — leave as NA
    }
  }
  
  if (interactive()) message("Completed year: ", yr)
}

# Clean up helper column if not needed downstream
# cell_data[, cell_idx := NULL]  # uncomment if cell_idx is not needed

# ===========================================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# The cell_data now contains the same neighbor_max_*, neighbor_min_*, 
# neighbor_mean_* columns as the original pipeline, with identical values.
# Proceed with prediction using the existing trained model:
#
#   predictions <- predict(trained_rf_model, newdata = cell_data)
#
# The model is NOT retrained. Only the feature-engineering step was optimized.
```

### A cleaner, more memory-efficient version (avoids repeated `which = TRUE`):

```r
# ===========================================================================
# ALTERNATIVE: Even cleaner — work on a matrix, write back at the end
# ===========================================================================

compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          neighbor_source_vars) {
  library(data.table)
  
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  cell_data[, cell_idx__ := id_to_cellidx[as.character(id)]]
  
  # Build static edge list once
  from_vec <- rep(seq_along(neighbors_nb), times = lengths(neighbors_nb))
  to_vec   <- unlist(neighbors_nb, use.names = FALSE)
  valid    <- to_vec != 0L
  edge_from <- from_vec[valid]
  edge_to   <- to_vec[valid]
  n_edges  <- sum(valid)
  
  years <- sort(unique(cell_data$year))
  
  # Pre-allocate result columns
  for (vn in neighbor_source_vars) {
    for (suffix in c("max", "min", "mean")) {
      col <- paste0("neighbor_", suffix, "_", vn)
      cell_data[, (col) := NA_real_]
    }
  }
  
  # Process by year
  for (yr in years) {
    row_mask <- cell_data$year == yr
    yr_cellidx <- cell_data$cell_idx__[row_mask]
    
    # Map cell_idx -> position in this year's subset
    pos_in_full <- which(row_mask)
    
    for (vn in neighbor_source_vars) {
      # Build cell-indexed value vector
      vals <- rep(NA_real_, n_cells)
      vals[yr_cellidx] <- cell_data[[vn]][row_mask]
      
      # Gather neighbor values
      nvals <- vals[edge_to]
      
      # Grouped aggregation using data.table
      # Only keep non-NA
      ok <- !is.na(nvals)
      if (any(ok)) {
        agg <- data.table(fi = edge_from[ok], v = nvals[ok])[
          , .(mx = max(v), mn = min(v), av = mean(v)), by = fi
        ]
        
        # Scatter results into cell-indexed vectors
        res_max  <- rep(NA_real_, n_cells)
        res_min  <- rep(NA_real_, n_cells)
        res_mean <- rep(NA_real_, n_cells)
        res_max[agg$fi]  <- agg$mx
        res_min[agg$fi]  <- agg$mn
        res_mean[agg$fi] <- agg$av
        
        # Write back to the correct rows in cell_data
        col_max  <- paste0("neighbor_max_", vn)
        col_min  <- paste0("neighbor_min_", vn)
        col_mean <- paste0("neighbor_mean_", vn)
        
        set(cell_data, i = pos_in_full, j = col_max,  value = res_max[yr_cellidx])
        set(cell_data, i = pos_in_full, j = col_min,  value = res_min[yr_cellidx])
        set(cell_data, i = pos_in_full, j = col_mean, value = res_mean[yr_cellidx])
      }
    }
    if (interactive()) message("Done: ", yr)
  }
  
  cell_data[, cell_idx__ := NULL]
  cell_data
}

# Usage:
cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique,
  c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Then predict (model unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Before | After |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key hash lookups | 1 static edge list (1.37M integer pairs) |
| **Stats computation** | 6.46M `lapply` iterations × 5 vars | 28 years × 5 vars × 1 vectorized `data.table` group-by |
| **Dominant cost** | Per-row R overhead, string ops | Vectorized C-level aggregation |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** |
| **Peak RAM** | Multiple 6.46M-element lists | ~1.37M edge vectors + 344K value vectors |
| **Numerical result** | — | **Identical** (same max/min/mean over same non-NA neighbor values) |
| **RF model** | — | **Untouched** (pre-trained, used only for prediction) |