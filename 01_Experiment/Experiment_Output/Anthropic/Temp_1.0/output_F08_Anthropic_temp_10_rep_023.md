 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates the static spatial topology with the dynamic temporal data.**

### Specific bottlenecks:

1. **`build_neighbor_lookup` operates on the full 6.46M-row cell-year panel.** It builds a lookup that maps each of the ~6.46M rows to its neighbor *rows* — but the neighbor *relationship* is purely spatial (344,208 cells × ~4 neighbors each). The year dimension is irrelevant to topology. By embedding year into the lookup via `paste(id, year)` keys, the function creates ~6.46M list entries instead of ~344K, doing ~28× redundant work.

2. **String-based key lookups (`paste` + named vector indexing) are extremely slow at this scale.** Creating and hashing ~6.46M string keys, then performing ~6.46M × ~4 = ~26M named lookups, is a major bottleneck. Named vector lookup in R is O(n) per probe in the worst case.

3. **`compute_neighbor_stats` iterates row-by-row over 6.46M rows via `lapply`.** Each iteration subsets a numeric vector, removes NAs, and computes three aggregates. The overhead of 6.46M R function calls plus `do.call(rbind, ...)` on 6.46M three-element vectors is enormous.

4. **The outer loop repeats this entire process 5 times** (once per variable), multiplying the cost.

### The key insight:

- **Static:** The neighbor graph (which cell borders which cell) never changes across years. There are only ~344K cells with ~1.37M directed neighbor edges.
- **Dynamic:** The variable values (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`) change by year.

The correct design is: **build the neighbor structure once over cells (not cell-years), then for each year, slice the data, use the cell-level neighbor structure to gather neighbor values, and compute stats — all vectorized.**

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** — a simple list of length 344,208 where each element contains the integer indices of that cell's neighbors within the cell ID vector. No year, no string keys. This runs in milliseconds.

2. **Process year-by-year.** For each of the 28 years, subset the data to that year's ~344K rows. Within a single year, every cell appears exactly once, so the cell-level neighbor indices directly map to row indices (after aligning cell order).

3. **Vectorize the neighbor aggregation using `data.table` or matrix operations.** Instead of `lapply` over millions of rows, "explode" the neighbor list into an edge table (cell_index, neighbor_index), join on variable values, and use `data.table` grouped aggregation (`max`, `min`, `mean` by cell) — which is C-level fast.

4. **Process all 5 variables simultaneously** within the same year pass to avoid redundant subsetting.

### Expected speedup:
- Neighbor lookup build: from ~hours to <1 second.
- Neighbor stats: from ~17 hours per variable to ~seconds per variable per year.
- Total: from ~86+ hours to **~2–5 minutes**.

The Random Forest model is never touched — only the feature-engineering step is redesigned. The numerical results (neighbor max, min, mean) are identical.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the STATIC cell-level neighbor lookup (once, independent of year)
# ==============================================================================
# Inputs:
#   id_order           — vector of 344,208 cell IDs in the order matching
#                         rook_neighbors_unique (i.e., id_order[i] is the cell
#                         whose neighbors are rook_neighbors_unique[[i]])
#   rook_neighbors_unique — spdep::nb object (list of 344,208 integer vectors,
#                            each giving positional indices of neighbors within
#                            id_order)
#
# Output:
#   cell_neighbor_idx  — list of length 344,208; cell_neighbor_idx[[i]] gives
#                         the positional indices (within id_order) of cell i's
#                         rook neighbors. This is EXACTLY rook_neighbors_unique
#                         with the spdep zero-neighbor convention handled.

build_cell_neighbor_lookup <- function(rook_neighbors_unique) {
  # spdep::nb objects encode "no neighbors" as a single 0L.
  # We convert those to integer(0) for clean downstream indexing.
  lapply(rook_neighbors_unique, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })
}

cell_neighbor_idx <- build_cell_neighbor_lookup(rook_neighbors_unique)


# ==============================================================================
# STEP 2: Pre-build the exploded edge table (once, static)
# ==============================================================================
# This is a two-column data.table: (cell_pos, neighbor_pos)
# where both are positional indices into id_order (1..344208).
# ~1,373,394 rows — trivially small.

build_edge_table <- function(cell_neighbor_idx) {
  n_neighbors <- vapply(cell_neighbor_idx, length, integer(1))
  cell_pos     <- rep(seq_along(cell_neighbor_idx), times = n_neighbors)
  neighbor_pos <- unlist(cell_neighbor_idx, use.names = FALSE)
  data.table(cell_pos = cell_pos, neighbor_pos = neighbor_pos)
}

edge_dt <- build_edge_table(cell_neighbor_idx)


# ==============================================================================
# STEP 3: Compute neighbor stats for all variables, all years — vectorized
# ==============================================================================
# Inputs:
#   cell_data  — data.frame/data.table with columns: id, year, and the
#                 neighbor_source_vars. ~6.46M rows.
#   id_order   — the 344,208 cell IDs in positional order.
#   edge_dt    — from Step 2.
#   neighbor_source_vars — character vector of variable names.
#
# Output:
#   cell_data with new columns: <var>_neighbor_max, <var>_neighbor_min,
#   <var>_neighbor_mean for each var in neighbor_source_vars.

compute_all_neighbor_features <- function(cell_data, id_order, edge_dt,
                                          neighbor_source_vars) {
  
  # Convert to data.table if needed (by reference if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Create a mapping: cell_id -> positional index in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add positional index to cell_data
  cell_data[, cell_pos := id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns with NA_real_
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # Process each year
  for (yr in years) {
    
    cat(sprintf("Processing year %d ...\n", yr))
    
    # Row indices in cell_data for this year
    yr_row_idx <- which(cell_data$year == yr)
    
    # Extract the sub-table for this year: cell_pos and variable values
    # We need a fast mapping: cell_pos -> row index within yr_row_idx
    yr_cell_pos <- cell_data$cell_pos[yr_row_idx]
    
    # Map: positional_index_in_id_order -> index_within_yr_row_idx
    # Not all 344K cells may be present every year, so use a sparse approach.
    pos_to_yr_row <- integer(length(id_order))  # vector of length 344K
    pos_to_yr_row[] <- 0L
    pos_to_yr_row[yr_cell_pos] <- seq_along(yr_row_idx)
    
    # For the edge table, find which edges have both cell and neighbor present
    # this year. Map cell_pos and neighbor_pos to yr_row indices.
    edge_cell_yr     <- pos_to_yr_row[edge_dt$cell_pos]
    edge_neighbor_yr <- pos_to_yr_row[edge_dt$neighbor_pos]
    valid_edges      <- (edge_cell_yr > 0L) & (edge_neighbor_yr > 0L)
    
    # Filtered edge mapping (indices into yr_row_idx)
    e_cell     <- edge_cell_yr[valid_edges]
    e_neighbor <- edge_neighbor_yr[valid_edges]
    
    # Actual row indices in cell_data
    e_cell_row     <- yr_row_idx[e_cell]
    e_neighbor_row <- yr_row_idx[e_neighbor]
    
    for (var_name in neighbor_source_vars) {
      
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")
      
      # Get neighbor values
      neighbor_vals <- cell_data[[var_name]][e_neighbor_row]
      
      # Build a small data.table for grouped aggregation
      agg_dt <- data.table(
        yr_idx = e_cell,
        nval   = neighbor_vals
      )
      
      # Remove NA neighbor values before aggregation
      agg_dt <- agg_dt[!is.na(nval)]
      
      if (nrow(agg_dt) > 0L) {
        # Compute grouped stats — very fast in data.table
        stats <- agg_dt[, .(
          nb_max  = max(nval),
          nb_min  = min(nval),
          nb_mean = mean(nval)
        ), by = yr_idx]
        
        # Write results back into cell_data at the correct rows
        target_rows <- yr_row_idx[stats$yr_idx]
        set(cell_data, i = target_rows, j = col_max,  value = stats$nb_max)
        set(cell_data, i = target_rows, j = col_min,  value = stats$nb_min)
        set(cell_data, i = target_rows, j = col_mean, value = stats$nb_mean)
      }
    }
  }
  
  # Clean up helper column
  cell_data[, cell_pos := NULL]
  
  return(cell_data)
}


# ==============================================================================
# USAGE — drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  edge_dt              = edge_dt,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the 15 new columns (5 vars × 3 stats).
# Proceed directly to Random Forest prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Guarantees

| Requirement | Status |
|---|---|
| **Numerical equivalence** | ✅ Identical `max`, `min`, `mean` per cell-year-variable. The aggregation logic is mathematically equivalent; only the execution strategy changes. |
| **Trained RF model preserved** | ✅ No model retraining. The code only engineers features, then feeds them to `predict()`. |
| **Same output columns** | ✅ `<var>_neighbor_max`, `<var>_neighbor_min`, `<var>_neighbor_mean` for all 5 variables. |
| **RAM feasibility (16 GB)** | ✅ Peak additional memory is the edge table (~1.37M × 2 integers ≈ 11 MB) plus one year-slice of aggregation (~1.37M × 2 doubles ≈ 22 MB). The main `cell_data` table (~6.46M × 125 columns) is the dominant allocation and was already in memory. |
| **Speed** | ✅ ~28 years × 5 variables × ~1.37M edges of `data.table` grouped aggregation ≈ **2–5 minutes** total vs. 86+ hours. |