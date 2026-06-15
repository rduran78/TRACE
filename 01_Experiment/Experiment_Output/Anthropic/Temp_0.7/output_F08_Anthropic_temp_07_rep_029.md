 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static spatial topology with dynamic temporal data.**

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows)** rather than per cell (~344K cells). Since rook-neighbor adjacency is purely spatial and never changes across years, this is doing **28× redundant work** on the topology side.

2. **String-key hashing is expensive.** The function creates `paste(id, year)` keys for all 6.46M rows, builds a named lookup vector, and then for each of the 6.46M rows, constructs neighbor keys via `paste()` again and does named-vector lookups. This involves massive string allocation and hash-table probing.

3. **`lapply` over 6.46M rows** in `compute_neighbor_stats` is inherently slow in R — each iteration has overhead from function dispatch, subsetting, and `is.na` checks.

4. **The neighbor lookup list itself is ~6.46M entries**, each containing integer vectors. This consumes substantial memory and creates GC pressure.

### What Is Static vs. What Changes

| Aspect | Static (invariant across years) | Dynamic (changes by year) |
|---|---|---|
| Cell IDs | ✓ | |
| Neighbor adjacency (rook) | ✓ | |
| Variable values (ntl, ec, …) | | ✓ |
| Neighbor stats (max, min, mean) | | ✓ |

**Key insight:** We only need **one** neighbor lookup of ~344K cells (not 6.46M cell-years). Then for each year, we index into that year's data slice using the static topology.

---

## Optimization Strategy

### 1. Build the neighbor lookup once, at the cell level only (~344K entries)

Create a mapping from each cell's positional index (1…344,208) to its neighbors' positional indices. This is done **once** and is year-independent.

### 2. Organize data so that each year's values can be accessed by cell index

Sort/ensure `cell_data` is ordered by `(id, year)` or `(year, id)`. With a consistent cell ordering, for any given year we can extract a contiguous block of rows and index into it by cell position.

### 3. Vectorize neighbor stat computation using sparse matrix multiplication

Represent the neighbor adjacency as a **sparse matrix** `W` of dimension 344,208 × 344,208. Then for each year and each variable:

- `neighbor_mean = (W %*% x) / (W %*% ones)` (where non-neighbor entries are 0)
- `neighbor_max` and `neighbor_min` via grouped operations on the sparse structure

This replaces 6.46M R-level `lapply` iterations with ~28 matrix operations per variable.

### 4. Alternative: use `data.table` grouped operations with the static adjacency edge list

Convert the `nb` object to an edge list (from_cell, to_cell). Join against each year's variable values. Compute grouped max/min/mean via `data.table` — extremely fast.

### Expected Speedup

| Step | Old | New |
|---|---|---|
| Build lookup | ~6.46M string ops | ~344K integer ops (once) |
| Compute stats | ~6.46M × 5 lapply calls | 28 × 5 vectorized group-bys |
| Estimated time | 86+ hours | **~2–10 minutes** |

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Convert the static nb object to a static edge list (ONCE)
# ==============================================================================
# rook_neighbors_unique: an nb object (list of length 344,208)
# id_order: vector of cell IDs in the order matching the nb object

build_static_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (positional indices)
  # id_order[i] is the cell ID for position i
  
  from_pos <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  to_pos   <- unlist(neighbors_nb)
  
  # Return as data.table with positional indices and cell IDs
  data.table(
    from_pos = from_pos,
    to_pos   = to_pos,
    from_id  = id_order[from_pos],
    to_id    = id_order[to_pos]
  )
}

edge_dt <- build_static_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows and is year-independent

cat(sprintf("Static edge list: %d directed neighbor pairs\n", nrow(edge_dt)))

# ==============================================================================
# STEP 2: Ensure cell_data is a data.table with consistent ordering
# ==============================================================================
cell_data <- as.data.table(cell_data)

# Create a positional index for each cell ID (matching the nb object order)
id_pos_map <- data.table(id = id_order, cell_pos = seq_along(id_order))
cell_data  <- merge(cell_data, id_pos_map, by = "id", all.x = TRUE, sort = FALSE)

# ==============================================================================
# STEP 3: Compute neighbor stats efficiently — static topology, dynamic values
# ==============================================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # cell_dt must have columns: id, year, cell_pos, and all source_vars
  # edge_dt must have columns: from_pos, to_pos (static, year-independent)
  
  years <- sort(unique(cell_dt$year))
  
  # Pre-allocate result columns in cell_dt
  for (var in source_vars) {
    cell_dt[, paste0("neighbor_max_",  var) := NA_real_]
    cell_dt[, paste0("neighbor_min_",  var) := NA_real_]
    cell_dt[, paste0("neighbor_mean_", var) := NA_real_]
  }
  
  # Key cell_dt for fast lookups by (year, cell_pos)
  # We'll process year by year to keep memory bounded
  
  for (yr in years) {
    cat(sprintf("  Processing year %d ...\n", yr))
    
    # Extract this year's data: a vector of values indexed by cell_pos
    # Get row indices in cell_dt for this year
    yr_rows <- cell_dt[year == yr]
    
    # Build a lookup: cell_pos -> row index in cell_dt
    # (we need to write results back)
    yr_row_indices <- which(cell_dt$year == yr)
    
    # Build a value lookup by cell_pos for this year
    # Create a vector of length max(cell_pos), indexed by cell_pos
    n_cells <- length(unique(cell_dt$cell_pos))
    max_pos <- max(cell_dt$cell_pos, na.rm = TRUE)
    
    # Map cell_pos -> row index within yr_rows
    pos_to_yr_row <- integer(max_pos)
    pos_to_yr_row[yr_rows$cell_pos] <- seq_len(nrow(yr_rows))
    
    for (var in source_vars) {
      # Build value vector indexed by cell_pos
      val_vec <- rep(NA_real_, max_pos)
      val_vec[yr_rows$cell_pos] <- yr_rows[[var]]
      
      # Look up neighbor values using the STATIC edge list
      # For each edge (from_pos, to_pos), get the value at to_pos
      neighbor_vals_dt <- data.table(
        from_pos = edge_dt$from_pos,
        val      = val_vec[edge_dt$to_pos]
      )
      
      # Remove NA neighbor values before aggregation
      neighbor_vals_dt <- neighbor_vals_dt[!is.na(val)]
      
      # Compute grouped stats: max, min, mean per from_pos
      if (nrow(neighbor_vals_dt) > 0) {
        stats <- neighbor_vals_dt[, .(
          nmax  = max(val),
          nmin  = min(val),
          nmean = mean(val)
        ), by = from_pos]
        
        # Map from_pos back to yr_row_indices in cell_dt
        # Find which yr_rows correspond to these from_pos values
        matched_yr_local <- pos_to_yr_row[stats$from_pos]
        matched_global   <- yr_row_indices[matched_yr_local]
        
        # Write results directly into cell_dt
        set(cell_dt, i = matched_global,
            j = paste0("neighbor_max_", var),  value = stats$nmax)
        set(cell_dt, i = matched_global,
            j = paste0("neighbor_min_", var),  value = stats$nmin)
        set(cell_dt, i = matched_global,
            j = paste0("neighbor_mean_", var), value = stats$nmean)
      }
    }
  }
  
  cell_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (static topology, dynamic values)...\n")
t0 <- proc.time()

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ==============================================================================
# STEP 4: Clean up helper column, proceed to prediction
# ==============================================================================
cell_data[, cell_pos := NULL]

# The trained Random Forest model is unchanged — use it directly:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | How it is met |
|---|---|
| **Same numerical estimand** | For each cell-year row, the neighbor max/min/mean is computed over the same set of rook-adjacent cells' values for that same year — identical to the original logic. The `data.table` grouped aggregation (`max`, `min`, `mean`) produces the same IEEE 754 results. |
| **Trained RF model unchanged** | No model retraining occurs. The output columns have the same names and values, so `predict()` on the pre-trained model works identically. |
| **NA handling preserved** | `neighbor_vals_dt[!is.na(val)]` mirrors the original `neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` (the pre-allocated default). |

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string-key entries | 1 sparse edge list (~1.37M rows, built once) |
| Stat computation iterations | 6.46M × 5 = 32.3M `lapply` calls | 28 years × 5 vars = 140 vectorized `data.table` group-bys |
| Memory for lookup | ~6.46M-element list of int vectors | ~1.37M-row `data.table` (reused) |
| Estimated wall time | **86+ hours** | **~2–10 minutes** |