 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves the neighbor cell IDs from the `nb` object.
3. Pastes neighbor IDs with the current year to form keys.
4. Looks up those keys in `idx_lookup` (a named vector).

The `paste()`-based key construction and named-vector lookup for every single row is extremely slow at this scale. The output is a list of ~6.46 million integer vectors — itself a large, fragmented memory structure.

### Bottleneck B: `compute_neighbor_stats` — per-row `lapply` over 6.46M elements, repeated 5 times

For each of the 5 source variables, this function iterates over every row, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. That's **~32.3 million** R-level function calls with repeated subsetting.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume data lives on a regular grid with a single time slice. Here the data is a **panel** (grid × year), and the neighbor structure is defined by an irregular `spdep::nb` object (not necessarily a regular 4-connected raster grid). However, the *concept* is the same: we can vectorize the neighbor aggregation using sparse matrix multiplication, which is the algebraic equivalent of a focal operation and works on arbitrary neighbor graphs.

### Root cause summary

| Component | Calls | Cost per call | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | `paste` + named lookup | ~hours |
| `compute_neighbor_stats` | 6.46M × 5 vars | subset + max/min/mean | ~hours |

---

## 2. Optimization Strategy

### Key insight: Replace per-row R loops with sparse-matrix operations

A rook-neighbor aggregation (mean, max, min over neighbors) can be expressed as operations on a **sparse adjacency matrix W** of dimension `N_rows × N_rows` (6.46M × 6.46M), where entry `W[i,j] = 1` if row `j` is a rook neighbor of row `i` **in the same year**.

- **Mean**: `W_norm %*% x` where `W_norm` is row-normalized (each row sums to 1). This is a single sparse matrix-vector multiply — seconds, not hours.
- **Max and Min**: Cannot be done with matrix multiply directly, but can be done efficiently with `data.table` group-by operations on an edge list.

### Plan

1. **Build an edge list** (from-row → to-row) once, using vectorized `data.table` joins instead of per-row `paste`/lookup. This replaces `build_neighbor_lookup`.
2. **Compute neighbor stats** using `data.table` grouped aggregation on the edge list. For each variable, join the edge list with the variable values, then group by `from_row` and compute `max`, `min`, `mean`. This replaces `compute_neighbor_stats`.
3. The result is numerically identical (same max, min, mean over the same neighbor sets).
4. The trained Random Forest model is untouched — we only change how predictor columns are computed, not their values.

**Expected speedup**: From ~86 hours to **~2–10 minutes** on the same laptop.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# ─────────────────────────────────────────────────────────────
# STEP 1: Build a vectorized edge list (replaces build_neighbor_lookup)
# ─────────────────────────────────────────────────────────────

build_edge_list <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
  
  # 1a. Build cell-level edge list from nb object
  #     For cell i (in id_order), neighbors are id_order[rook_neighbors_unique[[i]]]
  n_cells <- length(id_order)
  from_cell <- rep(id_order, times = lengths(rook_neighbors_unique))
  to_cell   <- id_order[unlist(rook_neighbors_unique)]
  
  cell_edges <- data.table(from_id = from_cell, to_id = to_cell)
  
  # 1b. Create row index for cell_data_dt
  cell_data_dt[, row_idx := .I]
  
  # 1c. Cross-join cell edges with years via merge
  #     For each (from_id, to_id) pair, we need all years where BOTH exist.
  #     Since this is a balanced panel (all cells × all years), we can do:
  
  # Get unique years
  years <- sort(unique(cell_data_dt$year))
  
  # Expand cell edges × years
  # To avoid a massive cross-join in memory, we join twice:
  #   edge (from_id, to_id) → join cell_data on (from_id, year) → join on (to_id, year)
  
  # Create lookup: (id, year) → row_idx
  id_year_lookup <- cell_data_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)
  
  # Replicate edges for each year
  edge_years <- CJ(edge_idx = seq_len(nrow(cell_edges)), year = years)
  edge_years[, from_id := cell_edges$from_id[edge_idx]]
  edge_years[, to_id   := cell_edges$to_id[edge_idx]]
  
  # Join to get from_row and to_row
  setkey(edge_years, from_id, year)
  edge_years <- id_year_lookup[edge_years, 
                                .(from_row = row_idx, 
                                  to_id = i.to_id, 
                                  year = i.year), 
                                nomatch = NA]
  
  # Drop edges where from_row is NA (cell not present in that year)
  edge_years <- edge_years[!is.na(from_row)]
  
  setkey(edge_years, to_id, year)
  edge_years <- id_year_lookup[edge_years,
                                .(from_row = i.from_row,
                                  to_row = row_idx),
                                nomatch = NA]
  
  edge_years <- edge_years[!is.na(to_row)]
  
  return(edge_years)  # data.table with columns: from_row, to_row
}

# ─────────────────────────────────────────────────────────────
# STEP 1 (alternative, more memory-friendly for balanced panels)
# ─────────────────────────────────────────────────────────────

build_edge_list_balanced <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # Optimized version assuming a balanced panel (every cell appears in every year)
  
  cell_data_dt[, row_idx := .I]
  
  # Build lookup: id → list of (year, row_idx), sorted by year
  setkey(cell_data_dt, id, year)
  
  # For a balanced panel, each cell has the same set of years.
  years <- sort(unique(cell_data_dt$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  
  # Map id → position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # For balanced panel: row indices for cell id_order[k] are a contiguous block
  # if data is sorted by (id, year). Let's ensure that and compute offsets.
  cell_data_dt <- cell_data_dt[order(id, year)]
  cell_data_dt[, row_idx := .I]  # re-index after sort
  
  # Offset for cell k: rows (k-1)*n_years + 1 through k*n_years
  # Verify balance
  rows_per_cell <- cell_data_dt[, .N, by = id]
  if (!all(rows_per_cell$N == n_years)) {
    message("Panel is not perfectly balanced; falling back to general method.")
    return(build_edge_list(cell_data_dt, id_order, rook_neighbors_unique))
  }
  
  # Reorder so that cell_data_dt is in id_order order within each year
  cell_data_dt[, id_pos := id_to_pos[as.character(id)]]
  setkey(cell_data_dt, id_pos, year)
  cell_data_dt[, row_idx := .I]
  
  # Now row for cell k, year t is: (k-1)*n_years + t_index
  # Build cell-level edges
  from_pos <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_pos   <- unlist(rook_neighbors_unique)
  
  # Expand over years: for year index t (1..n_years),
  #   from_row = (from_pos - 1) * n_years + t
  #   to_row   = (to_pos - 1)   * n_years + t
  
  n_cell_edges <- length(from_pos)
  
  from_row <- rep((from_pos - 1L) * n_years, each = n_years) +
              rep(seq_len(n_years), times = n_cell_edges)
  to_row   <- rep((to_pos - 1L)   * n_years, each = n_years) +
              rep(seq_len(n_years), times = n_cell_edges)
  
  edge_list <- data.table(from_row = from_row, to_row = to_row)
  
  # Return both the edge list and the re-sorted data
  return(list(edges = edge_list, data = cell_data_dt))
}

# ─────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats via data.table grouped aggregation
# ─────────────────────────────────────────────────────────────

compute_neighbor_stats_fast <- function(cell_data_dt, edge_list, var_name) {
  # edge_list: data.table with from_row, to_row
  # Returns a data.table with columns: from_row, max_var, min_var, mean_var
  
  vals <- cell_data_dt[[var_name]]
  
  # Attach neighbor values
  edge_list[, neighbor_val := vals[to_row]]
  
  # Remove NA neighbor values
  valid <- edge_list[!is.na(neighbor_val)]
  
  # Aggregate
  stats <- valid[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]
  
  # Initialize result columns with NA
  n <- nrow(cell_data_dt)
  result_max  <- rep(NA_real_, n)
  result_min  <- rep(NA_real_, n)
  result_mean <- rep(NA_real_, n)
  
  result_max[stats$from_row]  <- stats$nb_max
  result_min[stats$from_row]  <- stats$nb_min
  result_mean[stats$from_row] <- stats$nb_mean
  
  # Clean up temporary column
  edge_list[, neighbor_val := NULL]
  
  return(data.table(
    nb_max  = result_max,
    nb_min  = result_min,
    nb_mean = result_mean
  ))
}

# ─────────────────────────────────────────────────────────────
# STEP 3: Full pipeline (replaces the outer loop)
# ─────────────────────────────────────────────────────────────

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cell_data_dt <- as.data.table(cell_data)
  
  cat("Building edge list...\n")
  t0 <- proc.time()
  
  result <- build_edge_list_balanced(cell_data_dt, id_order, rook_neighbors_unique)
  
  if (is.list(result) && "edges" %in% names(result)) {
    edge_list   <- result$edges
    cell_data_dt <- result$data
  } else {
    edge_list <- result
  }
  
  cat("Edge list built in", (proc.time() - t0)[3], "seconds.\n")
  cat("Edge list has", nrow(edge_list), "rows.\n")
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    t1 <- proc.time()
    
    stats <- compute_neighbor_stats_fast(cell_data_dt, edge_list, var_name)
    
    # Add columns with standard naming convention
    # (adjust naming to match whatever compute_and_add_neighbor_features used)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    cell_data_dt[, (max_col)  := stats$nb_max]
    cell_data_dt[, (min_col)  := stats$nb_min]
    cell_data_dt[, (mean_col) := stats$nb_mean]
    
    cat("  Done in", (proc.time() - t1)[3], "seconds.\n")
  }
  
  # Remove helper columns
  cell_data_dt[, c("row_idx", "id_pos") := NULL]
  
  return(cell_data_dt)
}

# ─────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then predict with the already-trained Random Forest (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | Rook neighbors from `spdep::nb` object, same year | Identical — same `nb` object, same year matching |
| `max` computation | `max(neighbor_vals[!is.na(...)])` | `max(neighbor_val)` after filtering `!is.na` — identical |
| `min` computation | `min(neighbor_vals[!is.na(...)])` | `min(neighbor_val)` after filtering `!is.na` — identical |
| `mean` computation | `mean(neighbor_vals[!is.na(...)])` | `mean(neighbor_val)` after filtering `!is.na` — identical |
| No-neighbor case | Returns `c(NA, NA, NA)` | Row absent from `stats` → `NA` assigned — identical |
| RF model | Not retrained | Not retrained |

The optimized code computes **exactly the same values** for every cell-year row. The only change is the computational mechanism: vectorized `data.table` grouped aggregation over an edge list instead of 6.46 million individual R function calls.

---

## 5. Memory Considerations (16 GB laptop)

| Object | Approximate size |
|---|---|
| `cell_data_dt` (6.46M × 110 cols) | ~5.7 GB |
| Edge list (1.37M cell-edges × 28 years = ~38.5M rows × 2 int cols) | ~0.6 GB |
| Temporary `neighbor_val` column on edge list | ~0.3 GB |
| Working headroom | ~9.4 GB |

This fits comfortably in 16 GB. If memory is tighter (e.g., unbalanced panel inflating the edge list), process years in chunks:

```r
# Optional: chunk by year to reduce peak memory
for (yr in unique(cell_data_dt$year)) {
  year_edges <- edge_list[cell_data_dt$year[from_row] == yr]
  # ... compute stats for this year only, assign back
}
```

---

## Summary

| Aspect | Before | After |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `paste` + named vector lookups | Vectorized `data.table` join or arithmetic indexing |
| `compute_neighbor_stats` | ~32.3M R-level `lapply` iterations | 5 × one `data.table` grouped aggregation |
| Estimated runtime | **86+ hours** | **2–10 minutes** |
| Numerical results | Baseline | **Identical** |
| RF model | Preserved | **Preserved** |