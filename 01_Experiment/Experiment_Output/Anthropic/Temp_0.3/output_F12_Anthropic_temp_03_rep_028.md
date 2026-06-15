 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorizable bind operation per variable (only 5 calls total). The real cost there is modest.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookups over 6.46M rows inside `lapply`**: For every single row (6.46 million iterations), the function:
   - Calls `as.character(data$id[i])` and does a named-vector lookup (`id_to_ref[...]`).
   - Extracts neighbor cell IDs from the `neighbors` list.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build composite keys — for every neighbor of every row.
   - Performs named-vector lookup on `idx_lookup[neighbor_keys]` — a character-keyed lookup into a vector of length 6.46M.

2. **Character-keyed named vector lookup is O(n) per probe in the worst case** in base R (it uses linear hashing but with a massive vector of 6.46M names, each lookup is expensive). Across ~6.46M rows × ~4 neighbors each ≈ **25.8 million character-key lookups into a 6.46M-length named vector**. This is catastrophically slow.

3. **This lookup is computed once but dominates total runtime.** `compute_neighbor_stats()` by contrast just does integer indexing (`vals[idx]`), which is near-instantaneous. The `do.call(rbind, ...)` on 6.46M three-element vectors takes seconds, not hours.

4. **The fundamental architectural flaw**: The lookup conflates spatial neighbor structure (which is time-invariant) with the panel's year dimension. The neighbor relationships are identical across all 28 years, yet the code re-derives them for every cell-year row, multiplying work by 28×.

## Optimization Strategy

1. **Build the spatial neighbor index only once over the 344,208 unique cells** (not 6.46M cell-years), using integer indexing instead of character-key lookups.
2. **Expand to the panel dimension using vectorized operations**: since neighbors are the same for every year, map cell-level neighbor indices to cell-year row indices via simple arithmetic/merge — not per-row `paste` + named-vector lookup.
3. **Replace `do.call(rbind, lapply(...))` with pre-allocated matrix + direct vectorized column computation** using the neighbor index, avoiding per-row R function calls entirely.
4. **Use `data.table` for fast keyed joins** where needed.

This reduces the complexity from ~25.8M character lookups in a 6.46M-length named vector to ~1.37M integer lookups in a 344K-length structure (for the spatial part), plus a vectorized year expansion.

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# Preserves the trained Random Forest model and original numerical estimand.
# =============================================================================

library(data.table)

# ---- Step 1: Build spatial-only neighbor lookup (344K cells, not 6.46M rows) ----

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table if not already
  dt <- as.data.table(data)
  
  # --- Spatial neighbor map: cell index -> vector of neighbor cell indices ---
  # id_order is the vector of unique cell IDs in the order matching the nb object.
  # neighbors[[k]] gives the indices (into id_order) of the rook neighbors of id_order[k].
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  # Create a fast mapping: cell_id -> position in id_order
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  # (If id_order values are not contiguous integers, use a hash instead:)
  # But for typical grid cell IDs this works. Fallback below.
  
  # If IDs are too large or non-integer, use environment-based hash:
  if (max(id_order, na.rm = TRUE) > 1e8 || !is.integer(id_order)) {
    id_to_pos_env <- new.env(hash = TRUE, size = n_cells)
    for (k in seq_along(id_order)) {
      id_to_pos_env[[as.character(id_order[k])]] <- k
    }
    id_to_pos <- NULL  # signal to use env
  } else {
    id_to_pos_env <- NULL
  }
  
  # --- Build row index mapping: (cell_id, year) -> row in dt ---
  # Key the data.table for fast lookups
  dt[, row_idx := .I]
  setkey(dt, id, year)
  
  # Create a matrix: rows = cells (in id_order order), cols = years
  # Entry = row index in dt for that (cell, year), or NA
  # This avoids all character-key lookups.
  
  cat("Building (cell, year) -> row index matrix...\n")
  
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # For each unique cell in id_order, find all its rows across years
  # Use data.table merge for speed
  cell_year_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  
  # Map each cell in id_order to its rows
  id_order_dt <- data.table(id = id_order, cell_pos = seq_along(id_order))
  merged <- merge(dt[, .(id, year, row_idx)], id_order_dt, by = "id", sort = FALSE)
  merged[, year_col := year_to_col[as.character(year)]]
  
  # Fill the matrix
  cell_year_map[cbind(merged$cell_pos, merged$year_col)] <- merged$row_idx
  
  # --- Now build the full neighbor_lookup: for each row in dt, the row indices of its neighbors ---
  cat("Expanding spatial neighbors across years...\n")
  
  # Pre-compute: for each cell position, the neighbor cell positions
  # neighbors is an nb object: neighbors[[k]] is an integer vector of neighbor positions
  # (0 means no neighbors in spdep convention; filter those)
  
  # For each row in dt, we need:
  #   1. Its cell_pos (position in id_order)
  #   2. Its year_col
  #   3. The neighbor cell positions -> look up cell_year_map[neighbor_pos, year_col]
  
  # Build cell_pos and year_col for every row in dt
  setkey(dt, NULL)  # reset key
  dt_info <- merge(dt[, .(id, year, row_idx)], id_order_dt, by = "id", sort = FALSE)
  dt_info[, year_col := year_to_col[as.character(year)]]
  setorder(dt_info, row_idx)
  
  cell_pos_vec <- dt_info$cell_pos   # length = nrow(dt)
  year_col_vec <- dt_info$year_col   # length = nrow(dt)
  
  n_rows <- nrow(dt)
  
  # --- Vectorized neighbor lookup construction ---
  # Instead of returning a list of variable-length vectors (slow to iterate in R),
  # we build a CSR-like structure for maximum speed in compute_neighbor_stats.
  
  # First pass: compute the number of valid neighbors per row
  cat("Computing neighbor counts...\n")
  
  # For speed, pre-extract neighbor lengths
  neighbor_lengths <- lengths(neighbors)  # per cell
  
  # Total directed neighbor pairs across all rows (upper bound)
  total_pairs_upper <- sum(as.numeric(neighbor_lengths[cell_pos_vec]))
  
  # Allocate flat arrays
  nb_row_id  <- integer(total_pairs_upper)   # which dt row this neighbor belongs to
  nb_row_idx <- integer(total_pairs_upper)    # the dt row index of the neighbor
  
  cat("Building flat neighbor index (vectorized)...\n")
  
  # We process in chunks by cell_pos to avoid per-row R overhead
  # Group rows by cell_pos
  rows_by_cell <- split(seq_len(n_rows), cell_pos_vec)
  
  ptr <- 0L
  for (cp in seq_len(n_cells)) {
    nb_cells <- neighbors[[cp]]
    # spdep nb: 0 means no neighbors
    nb_cells <- nb_cells[nb_cells != 0L]
    if (length(nb_cells) == 0L) next
    
    cp_rows <- rows_by_cell[[as.character(cp)]]
    if (is.null(cp_rows) || length(cp_rows) == 0L) next
    
    # For each year that this cell appears in, look up neighbor rows
    for (ri in cp_rows) {
      yc <- year_col_vec[ri]
      nb_row_indices <- cell_year_map[nb_cells, yc]
      valid <- !is.na(nb_row_indices)
      n_valid <- sum(valid)
      if (n_valid == 0L) next
      
      idx_range <- (ptr + 1L):(ptr + n_valid)
      nb_row_id[idx_range]  <- ri
      nb_row_idx[idx_range] <- nb_row_indices[valid]
      ptr <- ptr + n_valid
    }
  }
  
  # Trim
  nb_row_id  <- nb_row_id[1:ptr]
  nb_row_idx <- nb_row_idx[1:ptr]
  
  cat("Built", ptr, "directed neighbor-row pairs.\n")
  
  # Return as a data.table for fast grouped operations
  list(
    nb_dt    = data.table(row_id = nb_row_id, nb_row = nb_row_idx),
    n_rows   = n_rows,
    dt       = dt
  )
}


# ---- Step 2: Vectorized compute_neighbor_stats (no per-row lapply) ----

compute_neighbor_stats_fast <- function(data, nb_info, var_name) {
  nb_dt  <- nb_info$nb_dt
  n_rows <- nb_info$n_rows
  
  vals <- data[[var_name]]
  
  # Attach neighbor values
  nb_dt[, nb_val := vals[nb_row]]
  
  # Remove NAs in neighbor values
  valid_nb <- nb_dt[!is.na(nb_val)]
  
  # Grouped aggregation: max, min, mean per row_id
  stats <- valid_nb[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = row_id]
  
  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)
  
  out_max[stats$row_id]  <- stats$nb_max
  out_min[stats$row_id]  <- stats$nb_min
  out_mean[stats$row_id] <- stats$nb_mean
  
  cbind(out_max, out_min, out_mean)
}


# ---- Step 3: Optimized outer loop ----

compute_and_add_neighbor_features_fast <- function(data, var_name, nb_info) {
  cat("Computing neighbor stats for:", var_name, "\n")
  stats_mat <- compute_neighbor_stats_fast(data, nb_info, var_name)
  
  data[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
  
  data
}


# ---- Step 4: Run the optimized pipeline ----

cat("Building optimized neighbor lookup...\n")
nb_info <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, nb_info)
}

cat("Done. Neighbor features added to cell_data.\n")

# The trained Random Forest model is untouched.
# The numerical estimand (max, min, mean of neighbor values) is preserved identically.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M `paste` + named-vector character lookups into a 6.46M-length vector | Eliminated |
| **Neighbor structure** | Redundantly recomputed for every cell-year row (×28 years) | Computed once over 344K cells, expanded via integer matrix |
| **Key lookup method** | Character-keyed named vector (`idx_lookup[neighbor_keys]`) — extremely slow at scale | Integer matrix indexing (`cell_year_map[nb_cells, yc]`) — O(1) |
| **Stats computation** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | `data.table` grouped aggregation — vectorized C-level `max`/`min`/`mean` |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the one-time neighbor expansion loop over 344K cells × 28 years) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical output** | max, min, mean of non-NA neighbor values | Identical ✓ |