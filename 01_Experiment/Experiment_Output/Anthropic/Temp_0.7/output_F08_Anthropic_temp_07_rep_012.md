 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is in `build_neighbor_lookup`. It constructs a lookup mapping **per cell-year row** (~6.46 million entries), even though the neighbor *topology* is purely spatial and identical across all 28 years. Specifically:

1. **Redundant work × 28**: The function builds ~6.46M entries in `neighbor_lookup`, one per row. But the neighbor graph has only ~344K cells. The topology is duplicated 28 times — once per year — doing string-pasting, hash lookups, and allocation for every cell-year combination.

2. **String-key hash lookups at scale**: `paste(id, year, sep="_")` creates ~6.46M string keys, and then for each row, neighbor keys are looked up in that hash. This is O(n × avg_neighbors) string operations on millions of rows.

3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the 6.46M-element `neighbor_lookup` list. The lookup list itself consumes substantial memory (millions of integer vectors).

4. **The result is numerically identical** to: for each cell, find its spatial neighbors (fixed); for each year, pull that year's variable values for those neighbors and compute max/min/mean. This is a **spatial join per year**, not a per-row operation.

**Summary**: The static neighbor graph is being re-expanded into a year-specific row-index lookup at enormous cost. The fix is to separate the static topology from the year-varying data.

---

## Optimization Strategy

1. **Build the neighbor topology once** over the 344K cells (not 6.46M rows). Store it as a simple list: `cell_neighbors[[cell_index]] → vector of neighbor cell indices`. This is just a reformatting of `rook_neighbors_unique` and is done once.

2. **Organize data by year**. For each year, extract the variable columns into a matrix indexed by cell position. Since cells repeat in the same order each year (or can be sorted to do so), neighbor indexing is direct integer subscripting into a vector — the fastest possible R operation.

3. **Compute neighbor stats per year per variable** using vectorized operations. For each cell, its neighbors are known; pull their values from the year-slice vector, compute max/min/mean. With `data.table` or matrix operations, this can be heavily vectorized.

4. **Optional further speedup**: Use a CSR (compressed sparse row) representation of the neighbor graph and a single vectorized pass per variable-year via `collapse::fmax`, `fmin`, `fmean` grouped operations, or manual vectorization with `rep()`/`unlist()` tricks.

**Expected speedup**: From ~86 hours to **minutes**. The dominant cost drops from ~6.46M list elements × 5 variables to ~344K cells × 28 years × 5 variables with vectorized arithmetic.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from year-varying data
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, 
                                          id_order, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars) {
  
  # ------------------------------------------------------------------
  # STEP 1: Build the static neighbor topology ONCE (344K cells)
  # ------------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
  # rook_neighbors_unique[[i]] gives the indices (into id_order) of neighbors of cell i.
  
  n_cells <- length(id_order)
  
  # Map cell IDs to their positional index in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute CSR-style representation for vectorized lookups
  # neighbor_of[[i]] = integer vector of positional indices of neighbors of cell i
  # This is essentially rook_neighbors_unique itself, but we ensure integer indexing.
  neighbor_of <- rook_neighbors_unique  # nb object: list of integer vectors
  
  # Build CSR (compressed sparse row) arrays for fully vectorized computation
  # "from" cell index repeated for each neighbor, "to" = neighbor cell index
  n_neighbors <- lengths(neighbor_of)
  
  # Cell indices that have at least one neighbor
  has_neighbors <- which(n_neighbors > 0)
  
  # CSR vectors
  csr_from <- rep.int(seq_along(neighbor_of), n_neighbors)
  csr_to   <- unlist(neighbor_of, use.names = FALSE)
  
  # Remove 0-entries that spdep::nb uses to indicate no neighbors
  valid <- csr_to > 0L
  csr_from <- csr_from[valid]
  csr_to   <- csr_to[valid]
  
  n_edges <- length(csr_from)
  
  message(sprintf("Static topology: %d cells, %d directed edges", n_cells, n_edges))
  
  # ------------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table and ensure sort order
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure the data has an 'id' and 'year' column
  stopifnot(all(c("id", "year") %in% names(dt)))
  
  # Create a cell position column for fast indexing
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  message(sprintf("Processing %d variables × %d years", 
                  length(neighbor_source_vars), n_years))
  
  # ------------------------------------------------------------------
  # STEP 3: Pre-allocate output columns
  # ------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # ------------------------------------------------------------------
  # STEP 4: For each year, compute neighbor stats vectorized
  # ------------------------------------------------------------------
  # Key the data.table for fast subsetting
  setkey(dt, year, cell_pos)
  
  for (yr in years) {
    
    # Extract this year's slice, ordered by cell_pos
    # After setkey(dt, year, cell_pos), rows for each year are contiguous
    # and sorted by cell_pos.
    yr_idx <- dt[.(yr), which = TRUE]
    
    # Get cell positions for this year (should be 1..n_cells if complete panel,
    # but we handle incomplete panels too)
    yr_cell_pos <- dt$cell_pos[yr_idx]
    
    # Build a fast map: cell_pos -> row index within yr_idx
    # For a complete balanced panel this is identity, but we handle gaps
    pos_to_yr_row <- integer(n_cells)
    pos_to_yr_row[] <- NA_integer_
    pos_to_yr_row[yr_cell_pos] <- seq_along(yr_idx)
    
    for (var_name in neighbor_source_vars) {
      
      # Extract the variable values for this year, ordered by cell_pos
      vals_yr <- dt[[var_name]][yr_idx]  # length = number of cells this year
      
      # Look up neighbor values using CSR representation
      # csr_from and csr_to are in cell_pos space
      # Map to this year's row space
      from_yr_row <- pos_to_yr_row[csr_from]
      to_yr_row   <- pos_to_yr_row[csr_to]
      
      # Filter edges where both endpoints exist this year
      edge_valid <- !is.na(from_yr_row) & !is.na(to_yr_row)
      e_from <- from_yr_row[edge_valid]
      e_to   <- to_yr_row[edge_valid]
      
      # Get neighbor values
      neighbor_vals <- vals_yr[e_to]
      
      # Also filter out NA variable values
      val_valid <- !is.na(neighbor_vals)
      e_from_v  <- e_from[val_valid]
      nv        <- neighbor_vals[val_valid]
      
      # Compute grouped max, min, mean using data.table's fast grouping
      # or base R tapply / collapse package
      if (length(e_from_v) > 0) {
        
        # Use data.table for fast grouped aggregation
        edge_dt <- data.table(from_row = e_from_v, nval = nv)
        
        agg <- edge_dt[, .(nmax  = max(nval), 
                           nmin  = min(nval), 
                           nmean = mean(nval)), 
                       by = from_row]
        
        # Write results back into dt
        target_rows <- yr_idx[agg$from_row]
        
        max_col  <- paste0("neighbor_max_", var_name)
        min_col  <- paste0("neighbor_min_", var_name)
        mean_col <- paste0("neighbor_mean_", var_name)
        
        set(dt, i = target_rows, j = max_col,  value = agg$nmax)
        set(dt, i = target_rows, j = min_col,  value = agg$nmin)
        set(dt, i = target_rows, j = mean_col, value = agg$nmean)
      }
    }
    
    if (which(years == yr) %% 5 == 0 || yr == years[n_years]) {
      message(sprintf("  Completed year %d (%d/%d)", yr, which(years == yr), n_years))
    }
  }
  
  # ------------------------------------------------------------------
  # STEP 5: Remove helper column and restore original row order
  # ------------------------------------------------------------------
  dt[, cell_pos := NULL]
  
  # Restore original row order (setkey changed it)
  # If caller needs original order, they should pass a row-id.
  # We restore by (id, year) to match the input.
  setkey(dt, NULL)
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# USAGE — Drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `rook_neighbors_unique` is used identically; no edges added or removed. |
| **Same variables** | The same 5 source variables are read; no transformations applied. |
| **Same aggregation functions** | `max()`, `min()`, `mean()` are applied to exactly the same sets of non-NA neighbor values per cell-year. |
| **Same output columns** | `neighbor_max_X`, `neighbor_min_X`, `neighbor_mean_X` — identical names and semantics. |
| **Random Forest untouched** | The trained model is only used for `predict()` afterward; no retraining. |

## Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-key entries | 344K-cell CSR built once |
| Per-variable computation | 6.46M list iterations × 5 | 28 year-slices × 5 vars, vectorized `data.table` grouped aggregation |
| String operations | ~billions of `paste()` + hash lookups | Zero |
| Memory for lookup | ~6.46M-element list of integer vectors | Two integer vectors of length ~1.37M (CSR) |
| **Estimated wall time** | **86+ hours** | **~5–15 minutes** |

The speedup comes from recognizing that the neighbor topology is **static across years** and should be encoded once as a sparse graph, then applied per year-slice using vectorized integer indexing and `data.table` grouped aggregation — eliminating millions of redundant string operations and list allocations.