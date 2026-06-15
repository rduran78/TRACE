 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) relationships.** It creates a list of 6.46 million entries, one per cell-year row. But rook-neighbor relationships are purely spatial — they don't change across years. There are only 344,208 unique cells. The function is doing 28× redundant work by re-resolving the same neighbor topology for every year.

2. **String-key hashing is expensive.** The function constructs `paste(id, year)` keys for all 6.46M rows, builds a named lookup vector, and then for each row, constructs neighbor keys by pasting neighbor IDs with the current year. This involves ~6.46M string concatenations for the index, plus ~6.46M × avg_neighbors more concatenations during lookup — hundreds of millions of string operations.

3. **`compute_neighbor_stats` iterates row-by-row over 6.46M rows** using `lapply`, calling `max`, `min`, `mean` individually per row. This is done 5 times (once per variable), totaling ~32.3 million R-level function calls with per-element subsetting.

4. **The neighbor lookup list itself consumes massive memory** — 6.46M list entries of integer vectors, with overhead per element.

### Key Insight

The neighbor graph is **static** (cell-to-cell, year-invariant). The variable values are **dynamic** (change by year). The correct design is:

- Build the neighbor lookup **once** over 344,208 cells (not 6.46M cell-years).
- For each variable, extract values **per year**, apply the cell-level neighbor lookup to compute stats, then write results back.

This reduces the lookup construction from O(6.46M) to O(344K) and makes the stats computation naturally vectorizable.

---

## Optimization Strategy

### 1. Separate Static Topology from Dynamic Data

Build a **cell-level** neighbor index once (344K entries instead of 6.46M). This is just a direct reformatting of `rook_neighbors_unique` — it's already an `nb` object indexed by cell position.

### 2. Vectorized Year-Sliced Computation

For each year:
- Extract the column vector of values for that variable (344K values, one per cell, in cell-order).
- Use the cell-level neighbor index to compute max/min/mean via vectorized operations.

### 3. Use Matrix Operations or data.table for Speed

Instead of `lapply` over 344K cells per year, use a **sparse adjacency approach** or a pre-flattened index with `vapply`/C-level grouping. A sparse matrix multiply can compute neighbor means in one shot; max and min require a grouped approach but can be done efficiently with pre-built flat indices.

### 4. Estimated Speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Lookup construction | 6.46M string-key entries | 344K integer entries (reuse `nb` directly) | ~19× |
| Stats computation | 6.46M × 5 vars = 32.3M R calls | 344K × 28 years × 5 vars, vectorized | ~50-200× |
| Memory | ~6.46M list elements + string keys | ~344K list + dense year-vectors | ~10× less |

**Expected total runtime: 1–5 minutes** instead of 86+ hours.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE: Separate static topology from dynamic (yearly) variables
# ==============================================================================
#
# Assumptions carried forward:
#   - cell_data is a data.frame/data.table with columns: id, year, ntl, ec,
#     pop_density, def, usd_est_n2, and ~110 predictor columns.
#   - id_order is a vector of cell IDs in the order matching rook_neighbors_unique.
#   - rook_neighbors_unique is an spdep::nb object (list of length 344,208),
#     where each element is an integer vector of neighbor *positions* into id_order.
#   - cell_data is sorted by (id, year) or at minimum has consistent ordering.
#   - The pre-trained Random Forest model (rf_model) is loaded and untouched.
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table if not already (for performance)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --------------------------------------------------------------------------
# STEP 1: Build the STATIC cell-level neighbor index (done ONCE)
#
# rook_neighbors_unique is already an nb object indexed by position in id_order.
# Each element rook_neighbors_unique[[i]] gives the positional indices of
# neighbors of id_order[i]. We just need to map cell IDs to positions.
#
# We also pre-build "flat" index vectors for fast grouped operations.
# --------------------------------------------------------------------------

build_cell_neighbor_flat_index <- function(nb_obj) {
  # nb_obj: list of length N_cells, each element is integer vector of neighbor

  # positions (0 means no neighbors in spdep convention; we handle that).
  
  n_cells <- length(nb_obj)
  
  # Count neighbors per cell
  n_neighbors <- vapply(nb_obj, function(x) {
    x <- x[x > 0L]  # spdep uses 0 for "no neighbors"
    length(x)
  }, integer(1))
  
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate flat vectors
  # cell_idx: which cell "owns" this neighbor entry (repeated for each neighbor)
  # neighbor_idx: the positional index of the neighbor cell
  cell_idx     <- integer(total_edges)
  neighbor_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    k <- length(nbrs)
    if (k > 0L) {
      cell_idx[pos:(pos + k - 1L)]     <- i
      neighbor_idx[pos:(pos + k - 1L)] <- nbrs
      pos <- pos + k
    }
  }
  
  list(
    cell_idx     = cell_idx,
    neighbor_idx = neighbor_idx,
    n_neighbors  = n_neighbors,
    n_cells      = n_cells
  )
}

cat("Building static cell-level neighbor flat index...\n")
flat_nb <- build_cell_neighbor_flat_index(rook_neighbors_unique)
cat(sprintf("  %d cells, %d directed neighbor edges\n",
            flat_nb$n_cells, length(flat_nb$cell_idx)))

# --------------------------------------------------------------------------
# STEP 2: Ensure cell_data has a cell-position column for fast indexing
# --------------------------------------------------------------------------

# Map each cell ID to its position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify all cells are mapped
stopifnot(!anyNA(cell_data$cell_pos))

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_cells <- flat_nb$n_cells

cat(sprintf("Processing %d years × %d cells = %d cell-years\n",
            length(years), n_cells, nrow(cell_data)))

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor stats efficiently (year-by-year, vectorized)
#
# For each variable and each year:
#   1. Extract a dense vector of values indexed by cell_pos (length = n_cells).
#   2. Look up neighbor values using the flat index.
#   3. Compute grouped max, min, mean using data.table's fast grouping.
#   4. Write results back into cell_data.
# --------------------------------------------------------------------------

compute_neighbor_stats_optimized <- function(cell_data, flat_nb, var_name, years, n_cells) {
  
  cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns with NA
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
  
  # Key cell_data by year and cell_pos for fast subsetting
  # We'll iterate by year
  
  cell_idx     <- flat_nb$cell_idx
  neighbor_idx <- flat_nb$neighbor_idx
  n_neighbors  <- flat_nb$n_neighbors
  
  for (yr in years) {
    # Get row indices in cell_data for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Build a dense vector: values_by_pos[cell_pos] = value
    # This assumes each cell appears exactly once per year
    yr_subset <- cell_data[yr_rows, .(cell_pos, val = get(var_name))]
    
    values_by_pos <- rep(NA_real_, n_cells)
    values_by_pos[yr_subset$cell_pos] <- yr_subset$val
    
    # Look up neighbor values using flat index
    neighbor_vals <- values_by_pos[neighbor_idx]
    
    # Compute grouped stats using data.table (very fast C-level grouping)
    # cell_idx tells us which cell each neighbor_val belongs to
    edge_dt <- data.table(
      cell = cell_idx,
      val  = neighbor_vals
    )
    
    # Remove NA values before aggregation
    edge_dt <- edge_dt[!is.na(val)]
    
    # Compute max, min, mean per cell in one pass
    stats <- edge_dt[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = cell]
    
    # Map results back: stats$cell is cell_pos
    # We need to find the row in cell_data for this year and this cell_pos
    # Build a mapping from cell_pos to yr_rows index
    yr_cell_pos <- cell_data$cell_pos[yr_rows]
    
    # Create a pos-to-row-index lookup (dense, since cell_pos ∈ 1:n_cells)
    pos_to_yr_row <- rep(NA_integer_, n_cells)
    pos_to_yr_row[yr_cell_pos] <- yr_rows
    
    # Write results
    matched_rows <- pos_to_yr_row[stats$cell]
    valid <- !is.na(matched_rows)
    
    if (any(valid)) {
      set(cell_data, i = matched_rows[valid], j = col_max,  value = stats$nb_max[valid])
      set(cell_data, i = matched_rows[valid], j = col_min,  value = stats$nb_min[valid])
      set(cell_data, i = matched_rows[valid], j = col_mean, value = stats$nb_mean[valid])
    }
  }
  
  return(cell_data)
}

# --------------------------------------------------------------------------
# STEP 4: Run the optimized pipeline
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (optimized)...\n")
t_start <- Sys.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_optimized(
    cell_data, flat_nb, var_name, years, n_cells
  )
}

t_end <- Sys.time()
cat(sprintf("Neighbor feature computation completed in %.1f minutes\n",
            as.numeric(difftime(t_end, t_start, units = "mins"))))

# Clean up helper column
cell_data[, cell_pos := NULL]

# --------------------------------------------------------------------------
# STEP 5: Predict using the pre-trained Random Forest (UNCHANGED)
#
# The RF model is already trained. We only feed it the updated cell_data
# with the same feature columns it expects. The neighbor_max_*, neighbor_min_*,
# neighbor_mean_* columns are numerically identical to the original pipeline's
# output, preserving the original numerical estimand.
# --------------------------------------------------------------------------

cat("Running pre-trained Random Forest predictions...\n")

# Example (adjust to your actual model object and prediction call):
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data[, prediction := predictions]

cat("Pipeline complete.\n")
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup scope** | 6.46M cell-year rows | 344K cells (static, built once) |
| **Lookup data structure** | Named character vector + paste keys | Integer flat index (cell_idx, neighbor_idx) |
| **Stats computation** | `lapply` over 6.46M rows per variable | `data.table` grouped aggregation over ~1.37M edges per year |
| **Total iterations** | ~32.3M R-level function calls | 28 years × 5 vars = 140 vectorized passes |
| **Estimated runtime** | 86+ hours | **1–5 minutes** |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** — same max, min, mean of same neighbor sets |
| **RF model** | Pre-trained, not retrained | **Preserved exactly** — only `predict()` is called |
| **Memory** | ~6.46M list entries + string keys | ~1.37M × 2 integer vectors + 344K dense vector per year-pass |

The optimized code produces **exactly the same `neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*` columns** as the original, ensuring the pre-trained Random Forest receives identical input features and the original numerical estimand is preserved.