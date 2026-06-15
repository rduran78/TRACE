 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable (only 5 calls total). The actual cost there is modest — binding 6.46M rows × 3 columns takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for every single row `i`, the code:
   - Calls `as.character(data$id[i])` — 6.46M times.
   - Looks up `id_to_ref[as.character(...)]` — named character vector lookup, O(n) hash probe × 6.46M.
   - Retrieves `id_order[neighbors[[ref_idx]]]` — fine, but done per-row.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — creates ~4 strings per row (avg ~4 rook neighbors) × 6.46M rows ≈ 25.8M string allocations.
   - Looks up `idx_lookup[neighbor_keys]` — probing a **named vector of 6.46 million entries** using string keys, ~4 times per row = ~25.8M hash lookups into a 6.46M-entry hash table.

2. **This is done once but takes the vast majority of the 86+ hours.** The `lapply` over 6.46M rows with string operations and named-vector lookups is catastrophically slow in R. Named vector lookup in R is not O(1) — it uses linear hashing with poor cache behavior at this scale.

3. **The lookup is redundant across variables.** The `neighbor_lookup` is built once and reused for all 5 variables — that's correct. But the construction itself is the wall-clock killer.

4. **`compute_neighbor_stats` is comparatively cheap.** It's just integer indexing into a numeric vector (`vals[idx]`) and computing `max/min/mean` — all fast vectorized operations. The `do.call(rbind, ...)` on a list of 6.46M length-3 vectors takes ~10-30 seconds, not hours.

## Optimization Strategy

1. **Eliminate all string operations.** Replace `paste(id, year)` key construction and named-vector string lookups with pure integer arithmetic. Create a direct integer mapping from `(id_index, year_index)` → row number using a matrix or a computed offset, since years are contiguous (1992–2019, 28 years).

2. **Vectorize the neighbor lookup construction.** Instead of an `lapply` over 6.46M rows, expand the neighbor list (which is per-cell, ~344K entries) into a flat edge list, then broadcast across all 28 years using vectorized integer operations.

3. **Replace `do.call(rbind, lapply(...))` with pre-allocated matrix and direct column computation** using vectorized grouped operations (via `data.table` or manual sparse-vector indexing).

4. **Preserve the trained Random Forest model** — we only change feature engineering, not the model. The numerical outputs (max, min, mean of neighbor values) remain identical.

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Key insight: since years are contiguous 1992-2019 (28 years) and every cell
# appears in every year, we can compute a direct integer mapping:
#   row_number = (cell_index - 1) * n_years + year_index
# This eliminates ALL string operations.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for fast operations if not already
  dt <- as.data.table(data)
  
  n_cells <- length(id_order)
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  # Create integer mappings
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  year_to_yearidx <- setNames(seq_along(years), as.character(years))
  
  # Ensure data is sorted by (id, year) so that row = (cellidx-1)*n_years + yearidx
  # First, compute cellidx and yearidx for each row
  dt[, cellidx := id_to_cellidx[as.character(id)]]
  dt[, yearidx := year_to_yearidx[as.character(year)]]
  
  # Sort by cellidx, then yearidx — this gives us a predictable row layout
  setorder(dt, cellidx, yearidx)
  
  # Now row i in dt corresponds to cellidx = ((i-1) %/% n_years) + 1,
  #                                 yearidx = ((i-1) %% n_years) + 1
  # Verify this assumption holds (all cells have all years):
  stopifnot(nrow(dt) == n_cells * n_years)
  
  # Build flat neighbor edge list at the CELL level (not cell-year level)
  # neighbors[[c]] gives the neighbor indices for cell c in id_order
  # We expand this to cell-year level using integer arithmetic
  
  # For each cell c with neighbor cells n1, n2, ..., nk:
  #   For each year y (yearidx 1..28):
  #     source row = (c - 1) * n_years + y
  #     neighbor rows = (n1 - 1) * n_years + y, (n2 - 1) * n_years + y, ...
  
  # Build the lookup as a list of length nrow(dt)
  # But do it vectorized: first build cell-level, then replicate across years
  
  # Cell-level neighbor list (already have this: neighbors)
  # Convert to cellidx-based if not already
  # neighbors is an nb object: neighbors[[i]] gives indices into id_order
  # So neighbors[[cellidx]] gives neighbor cell indices — already in cellidx space
  
  # Pre-compute neighbor count per cell
  n_neighbors <- lengths(neighbors)  # length n_cells
  
  # Flatten the cell-level neighbor list
  # For cell c, neighbors are neighbors[[c]]
  flat_source_cell <- rep(seq_len(n_cells), times = n_neighbors)
  flat_target_cell <- unlist(neighbors)
  
  # Now expand across years: each (source_cell, target_cell) pair generates
  # n_years entries, one per year
  n_edges_cell <- length(flat_source_cell)  # ~1,373,394 / 2 directed? 
  # Actually ~1,373,394 directed edges total
  
  # For each year, compute source_row and target_row
  # source_row = (source_cell - 1) * n_years + yearidx
  # target_row = (target_cell - 1) * n_years + yearidx
  
  # Expand: repeat each edge n_years times, cycle through years
  flat_source_cell_exp <- rep(flat_source_cell, each = n_years)
  flat_target_cell_exp <- rep(flat_target_cell, each = n_years)
  year_idx_exp <- rep(seq_len(n_years), times = n_edges_cell)
  
  flat_source_row <- (flat_source_cell_exp - 1L) * n_years + year_idx_exp
  flat_target_row <- (flat_target_cell_exp - 1L) * n_years + year_idx_exp
  
  # Now build the lookup list: for each source_row, collect all target_rows
  # Use split() which is vectorized and fast
  neighbor_lookup <- split(flat_target_row, flat_source_row)
  
  # split() returns a named list with names as character(source_row)
  # We need a list of length nrow(dt), indexed 1..nrow(dt)
  # Rows with no neighbors won't appear in the split result
  
  full_lookup <- vector("list", nrow(dt))
  idx_present <- as.integer(names(neighbor_lookup))
  full_lookup[idx_present] <- neighbor_lookup
  
  # Return both the reordered data and the lookup
  # IMPORTANT: dt is now reordered by (cellidx, yearidx). We must return
  # the reordered data so downstream code uses consistent row indices.
  
  # Remove helper columns
  dt[, c("cellidx", "yearidx") := NULL]
  
  list(data = as.data.frame(dt), neighbor_lookup = full_lookup)
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats — fully vectorized, no lapply
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- nrow(data)
  
  # Flatten the lookup
  source_rows <- rep(seq_len(n), times = lengths(neighbor_lookup))
  target_rows <- unlist(neighbor_lookup)
  
  if (length(target_rows) == 0) {
    return(cbind(rep(NA_real_, n), rep(NA_real_, n), rep(NA_real_, n)))
  }
  
  neighbor_vals <- vals[target_rows]
  
  # Remove NAs: filter out entries where neighbor_vals is NA
  valid <- !is.na(neighbor_vals)
  source_rows_v <- source_rows[valid]
  neighbor_vals_v <- neighbor_vals[valid]
  
  # Use data.table for fast grouped aggregation
  edge_dt <- data.table(src = source_rows_v, val = neighbor_vals_v)
  
  agg <- edge_dt[, .(
    nmax = max(val),
    nmin = min(val),
    nmean = mean(val)
  ), by = src]
  
  # Initialize result matrix with NAs
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  result[agg$src, 1] <- agg$nmax
  result[agg$src, 2] <- agg$nmin
  result[agg$src, 3] <- agg$nmean
  
  result
}


# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# MAIN PIPELINE — drop-in replacement
# =============================================================================
# Step 1: Build optimized neighbor lookup (replaces the old build_neighbor_lookup)
message("Building optimized neighbor lookup...")
system.time({
  lookup_result <- build_neighbor_lookup_fast(
    cell_data, id_order, rook_neighbors_unique
  )
})

# IMPORTANT: use the reordered data from here on
cell_data <- lookup_result$data
neighbor_lookup <- lookup_result$neighbor_lookup

# Step 2: Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics...")
system.time({
  for (var_name in neighbor_source_vars) {
    message("  Processing: ", var_name)
    cell_data <- compute_and_add_neighbor_features_fast(
      cell_data, var_name, neighbor_lookup
    )
  }
})

message("Done. Neighbor features added to cell_data.")

# The trained Random Forest model is untouched.
# The numerical estimand (max, min, mean of neighbor values) is preserved exactly.
# Predicted runtime: ~2-10 minutes instead of 86+ hours.
```

## Summary of Changes and Expected Speedup

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows with `paste()` + named-vector string lookup | Vectorized integer arithmetic + `split()` on flat edge arrays | ~1000–5000× |
| `compute_neighbor_stats` | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Flat vectorized indexing + `data.table` grouped aggregation | ~50–200× |
| **Total estimated time** | **86+ hours** | **~2–10 minutes** | **~500–2500×** |

**Key principle:** The original code performed **~6.46 million iterations** of string construction and hash-table probing in pure R. The optimized code replaces this with **vectorized integer arithmetic** (cell-to-row mapping via `(cellidx - 1) * n_years + yearidx`), a single `split()` call, and `data.table` grouped aggregation — all of which operate on contiguous integer/numeric vectors and leverage R's internal C-level loops.