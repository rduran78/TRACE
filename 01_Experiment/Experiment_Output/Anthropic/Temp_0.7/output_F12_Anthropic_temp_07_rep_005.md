 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step and runs in seconds to low minutes at most. The `lapply` inside `compute_neighbor_stats()` is also relatively lean — it's just indexing into a pre-extracted numeric vector and computing three summary statistics.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive per-row string operations**: For each of the ~6.46 million rows, it calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to construct string keys, then does **named-vector lookup** (`idx_lookup[neighbor_keys]`) against a character vector of 6.46 million names. Named vector lookup by character key in R is O(n) per probe in the worst case (it uses a linear hash with potential collisions), and this is done for every neighbor of every row. With ~1.37 million directed neighbor relationships spread across 28 years, the total number of key lookups is on the order of **tens of millions**, each probing a 6.46-million-entry named character vector.

2. **Redundant work across years**: The neighbor *structure* is identical across all 28 years — the same grid cell has the same rook neighbors every year. Yet `build_neighbor_lookup()` recomputes the string-key-based lookup independently for every cell-year row, doing 28× the work necessary.

3. **`as.character()` and `paste()` allocations**: Each of the 6.46 million iterations allocates new character vectors for `as.character(data$id[i])`, the `paste(...)` call, and the subsetting result. This creates enormous GC (garbage collection) pressure.

In contrast, `compute_neighbor_stats()` merely indexes a numeric vector and computes `max/min/mean` — these are trivially fast operations. And `do.call(rbind, result)` on a list of 6.46M three-element vectors is equivalent to `matrix(unlist(result), ncol=3, byrow=TRUE)`, which takes seconds.

**Conclusion**: The bottleneck is `build_neighbor_lookup()` — specifically its per-row string construction and named-character-vector lookup over a 6.46M-entry table, repeated redundantly for all 28 years. This is where the 86+ hours are being spent.

---

## Optimization Strategy

1. **Eliminate string keys entirely.** Replace the character-based `idx_lookup` with integer arithmetic. Since the data is a panel (cell × year), we can map `(id, year)` → row index using an integer hash (e.g., via `data.table` or a direct integer-keyed environment/match).

2. **Exploit year-invariant neighbor structure.** Build the neighbor mapping once at the cell level (344,208 cells), then expand to cell-year rows via vectorized integer indexing — no per-row `lapply` needed.

3. **Vectorize `compute_neighbor_stats()`.** Replace the per-row `lapply` with a single grouped operation using `data.table` or vectorized indexing with `rowMeans`-style operations.

4. **Replace `do.call(rbind, ...)` with `matrix(unlist(...), ...)`** as a minor secondary improvement.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Key insight: neighbor structure is IDENTICAL across all 28 years.
# We build a cell-level neighbor list once, then expand to row-level
# using pure integer arithmetic — no string keys, no paste, no named vectors.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  dt <- as.data.table(data)
  
  # Step 1: Create integer mappings
  # Map each unique id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map each unique year to a sequential integer
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_idx <- setNames(seq_along(years_sorted), as.character(years_sorted))
  
  # Step 2: Build a fast (id, year) -> row_index lookup using data.table
  # Add row indices to dt
  dt[, row_idx := .I]
  setkey(dt, id, year)
  
  # Step 3: Build cell-level neighbor ID list (done once for 344K cells, not 6.46M rows)
  # For each cell in id_order, get the IDs of its rook neighbors
  n_cells <- length(id_order)
  
  # Precompute: for each cell index in id_order, what are the neighbor cell IDs?
  cell_neighbor_ids <- lapply(seq_len(n_cells), function(ref) {
    nb_indices <- neighbors[[ref]]
    # Remove 0s (spdep uses 0 for no-neighbor sentinel)
    nb_indices <- nb_indices[nb_indices > 0L]
    if (length(nb_indices) == 0L) return(integer(0))
    id_order[nb_indices]
  })
  names(cell_neighbor_ids) <- as.character(id_order)
  
  # Step 4: For each row in data, look up neighbor rows using data.table join
  # Instead of lapply over 6.46M rows, we do this vectorized:
  
  # Build an edge list: (focal_id, focal_year, neighbor_id) 
  # Then join to get neighbor row indices
  
  # First, create a compact representation: for each cell, its neighbors
  # Expand to an edge data.table
  from_ids <- rep(id_order, times = vapply(cell_neighbor_ids, length, integer(1)))
  to_ids   <- unlist(cell_neighbor_ids, use.names = FALSE)
  
  edges <- data.table(focal_id = from_ids, neighbor_id = to_ids)
  
  # Cross with years to get (focal_id, year, neighbor_id)
  # But this could be huge. Instead, we work row-by-row more cleverly:
  # For each row i with (id_i, year_i), neighbors are rows with (neighbor_id, year_i)
  
  # Build the lookup: for each (id, year), what is the row index?
  row_lookup <- dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)
  
  # For each row, get its cell's neighbor IDs, then find rows with those IDs and same year
  # We do this via a merge rather than per-row lapply
  
  # Create focal edge table: each row's id and year, crossed with its neighbors
  # focal_row_idx, focal_id, focal_year -> expand by neighbor_id
  
  focal_info <- dt[, .(focal_row = row_idx, focal_id = id, focal_year = year)]
  
  # Map focal_id to its ref index in id_order
  focal_info[, ref_idx := id_to_ref[as.character(focal_id)]]
  
  # Now we need to expand: for each focal row, one record per neighbor
  # This is the key vectorized step
  
  # Get number of neighbors per cell
  n_neighbors_per_cell <- vapply(cell_neighbor_ids, length, integer(1))
  
  # Map each focal row to its cell's neighbor count
  focal_info[, n_nb := n_neighbors_per_cell[ref_idx]]
  
  # Expand focal_info: repeat each row n_nb times
  expanded <- focal_info[rep(seq_len(.N), n_nb)]
  
  # Add the neighbor_id column
  # For each focal row, the neighbor IDs come from cell_neighbor_ids[[ref_idx]]
  # We need to generate the neighbor_id vector in the same order as the expansion
  neighbor_id_vec <- unlist(cell_neighbor_ids[focal_info$ref_idx], use.names = FALSE)
  expanded[, neighbor_id := neighbor_id_vec]
  
  # Now join to find the row index of (neighbor_id, focal_year)
  setnames(expanded, "focal_year", "year")
  expanded[, id := neighbor_id]
  
  # Keyed join
  expanded[row_lookup, neighbor_row := i.row_idx, on = .(id, year)]
  
  # Remove NAs (neighbors that don't exist in data for that year)
  expanded <- expanded[!is.na(neighbor_row)]
  
  # Now split by focal_row to get the neighbor_lookup list
  # Sort by focal_row for efficient splitting
  setkey(expanded, focal_row)
  
  # Pre-allocate result list
  n_rows <- nrow(dt)
  neighbor_lookup <- vector("list", n_rows)
  
  # Split neighbor_row by focal_row
  split_result <- split(expanded$neighbor_row, expanded$focal_row)
  
  # Fill in the lookup (rows with no neighbors remain NULL -> handle in stats)
  for (nm in names(split_result)) {
    neighbor_lookup[[as.integer(nm)]] <- as.integer(split_result[[nm]])
  }
  
  return(neighbor_lookup)
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================================
# Replace lapply + do.call(rbind, ...) with fully vectorized data.table operations

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Build a flat edge table: (focal_row, neighbor_value)
  focal_rows <- rep(seq_len(n), times = vapply(neighbor_lookup, function(x) {
    if (is.null(x)) 0L else length(x)
  }, integer(1)))
  
  neighbor_rows <- unlist(neighbor_lookup, use.names = FALSE)
  
  if (length(neighbor_rows) == 0) {
    # Edge case: no neighbors at all
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
    return(out)
  }
  
  neighbor_vals <- vals[neighbor_rows]
  
  # Use data.table for grouped aggregation
  edge_dt <- data.table(
    focal = focal_rows,
    nval  = neighbor_vals
  )
  
  # Remove NA neighbor values
  edge_dt <- edge_dt[!is.na(nval)]
  
  # Compute grouped stats
  stats_dt <- edge_dt[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal]
  
  # Map back to full row set
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[stats_dt$focal, 1] <- stats_dt$nb_max
  out[stats_dt$focal, 2] <- stats_dt$nb_min
  out[stats_dt$focal, 3] <- stats_dt$nb_mean
  
  colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
  return(out)
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  for (j in seq_len(ncol(stats_mat))) {
    data[[colnames(stats_mat)[j]]] <- stats_mat[, j]
  }
  return(data)
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================
# Build the lookup ONCE (this is the former bottleneck — now fast)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched — use it for prediction as before.
# predict(rf_model, cell_data)  # unchanged
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, result)` and list binding in `compute_neighbor_stats()` | **Wrong.** This step is a minor cost. |
| **True bottleneck** | — | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + named-character-vector lookup against a 6.46M-entry index, redundantly recomputing year-invariant structure 28×. |
| **Root cause** | — | O(tens of millions) string allocations and hash lookups in a single-threaded `lapply`. |
| **Fix** | — | Replace string keys with integer-keyed `data.table` joins; exploit year-invariant neighbor structure via vectorized edge-list expansion and merge. |
| **Expected speedup** | — | From ~86+ hours to **minutes** (vectorized joins on ~38M edges). |
| **RF model** | — | Preserved unchanged; only the feature-engineering pipeline is optimized. |
| **Numerical equivalence** | — | Identical `max`, `min`, `mean` computed on the same neighbor sets → same estimand. |