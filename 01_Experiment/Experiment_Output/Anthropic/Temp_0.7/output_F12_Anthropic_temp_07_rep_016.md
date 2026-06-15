 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors isn't free, it's a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M three-element vectors into a matrix — that's a well-known pattern and completes in seconds.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookup over 6.46M rows:** `idx_lookup` is a named integer vector of length ~6.46M. Named vector lookup in R via `idx_lookup[neighbor_keys]` is **O(n)** per lookup due to linear name matching (R uses hashing for named vectors, but the construction and repeated character key generation is extremely expensive at this scale).

2. **`lapply` over 6.46M rows with per-row character operations:** For each of the 6.46 million rows, the function:
   - Calls `as.character(data$id[i])` — character conversion per row.
   - Looks up `id_to_ref[...]` — named vector lookup.
   - Retrieves `id_order[neighbors[[ref_idx]]]` — subsetting.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — creates character keys for every neighbor of every row.
   - Looks up `idx_lookup[neighbor_keys]` — named vector hash lookup with character keys.

   With ~1.37M directed neighbor relationships spread across 344K cells and 28 years, the average cell has ~4 neighbors. That means ~6.46M × 4 = ~25.8 million `paste` + hash-lookup operations, all inside a sequential `lapply`. The character allocation and hashing alone dominate runtime.

3. **This lookup is called once but produces a list of 6.46M elements**, each constructed via expensive string operations. This single call dwarfs the cost of `compute_neighbor_stats()`, which simply does numeric indexing (`vals[idx]`) — an O(1) vector subset per neighbor.

**In summary:** `build_neighbor_lookup()` is the dominant bottleneck due to per-row character key construction and named-vector string hashing over 6.46M rows. `compute_neighbor_stats()` is comparatively cheap, and `do.call(rbind, ...)` is a minor cost.

---

## Optimization Strategy

1. **Replace all character/paste key lookups with pure integer arithmetic.** Instead of `paste(id, year, sep="_")` → named vector lookup, compute a direct integer index: create an integer hash map from `(id, year)` to row number using `match()` on a precomputed integer key (e.g., `id * 100 + (year - 1991)`), or better, use a two-level integer lookup table: `row_index_matrix[id_ref, year_ref]` where dimensions are `(n_cells, n_years)`.

2. **Vectorize the neighbor lookup construction** by expanding the neighbor list once (via `data.table` or vectorized operations) instead of using `lapply` over 6.46M rows.

3. **Vectorize `compute_neighbor_stats()`** using the same group-based approach — avoid per-element `lapply` over 6.46M entries.

4. **Preserve the trained Random Forest model** — we only change feature engineering / data prep, not the model.

5. **Preserve the original numerical estimand** — the same max, min, mean neighbor statistics are computed identically, just faster.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup using integer-indexed matrix (no strings)
# ==============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of unique cell IDs in the order matching neighbors (nb object)
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)
  
  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  # Step 1: Build a matrix that maps (cell_ref_index, year_index) -> row in data
  # cell_ref_index: position in id_order (1..n_cells)
  # year_index: position in years vector (1..n_years)
  
  id_to_ref  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_idx <- setNames(seq_along(years), as.character(years))
  
  # Map each row of data to (cell_ref, year_ref)
  cell_refs <- id_to_ref[as.character(data$id)]
  year_refs <- year_to_idx[as.character(data$year)]
  
  # Build the lookup matrix: row_matrix[cell_ref, year_ref] = row_number_in_data
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(cell_refs, year_refs)] <- seq_len(nrow(data))
  
  # Step 2: Expand neighbor relationships into a flat edge table
  # For each row i in data, we need: neighbors of cell_refs[i], at year_refs[i]
  # 
  # Instead of lapply over 6.46M rows, we vectorize:
  # - For each cell_ref c, get its neighbor cell_refs from neighbors[[c]]
  # - For each year y, the neighbor rows are row_matrix[neighbors[[c]], y]
  # - Row i with (cell_ref=c, year_ref=y) maps to these neighbor rows
  
  # Precompute: for each cell_ref, the neighbor cell_refs (as integer vectors)
  # neighbors is already in this form (nb object: list of integer vectors)
  
  # Build edge list: (from_row_in_data, to_row_in_data)
  # We iterate over cells (344K) not rows (6.46M)
  
  # For each cell, get all years it appears in, and for each year, 
  # map neighbor cells to their rows at that year.
  
  # Efficient approach: iterate over cells (344K), not rows (6.46M)
  
  cat("Building neighbor lookup (vectorized)...\n")
  
  # Preallocate: estimate total edges

  # Average ~4 neighbors per cell, 28 years, 344K cells ≈ 38.5M edges
  # But some neighbors may be NA (missing year), so we'll collect and filter
  
  # Use data.table for speed
  dt <- data.table(
    row_id   = seq_len(nrow(data)),
    cell_ref = cell_refs,
    year_ref = year_refs
  )
  
  # For each cell_ref, get neighbor cell_refs
  # Build a data.table of (cell_ref, neighbor_cell_ref)
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(c) {
    nb <- neighbors[[c]]
    if (length(nb) == 0 || (length(nb) == 1 && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(cell_ref = c, nb_cell_ref = nb)
  }))
  
  # Now cross-join with years: for each (cell_ref, year_ref), 
  # the neighbor rows are row_matrix[nb_cell_ref, year_ref]
  
  # Merge edge_list with dt to get (row_id, year_ref, nb_cell_ref)
  setkey(dt, cell_ref)
  setkey(edge_list, cell_ref)
  
  expanded <- edge_list[dt, on = "cell_ref", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: cell_ref, nb_cell_ref, row_id, year_ref
  
  # Look up the neighbor's row in data
  expanded[, nb_row := row_matrix[cbind(nb_cell_ref, year_ref)]]
  
  # Remove edges where neighbor row doesn't exist
  expanded <- expanded[!is.na(nb_row)]
  
  # Now build the lookup list: for each row_id, collect all nb_row values
  setkey(expanded, row_id)
  
  # Convert to list indexed by row_id
  n_rows <- nrow(data)
  lookup <- vector("list", n_rows)
  
  # Split nb_row by row_id
  split_result <- split(expanded$nb_row, expanded$row_id)
  
  # Assign to lookup (some row_ids may have no neighbors)
  idx <- as.integer(names(split_result))
  lookup[idx] <- split_result
  
  # Fill remaining with integer(0)
  empty_idx <- setdiff(seq_len(n_rows), idx)
  lookup[empty_idx] <- list(integer(0))
  
  cat("Neighbor lookup complete.\n")
  return(lookup)
}


# ==============================================================================
# OPTIMIZED compute_neighbor_stats: vectorized via data.table grouping
# ==============================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  n <- nrow(data)
  vals <- data[[var_name]]
  
  # Build edge table: (row_id, neighbor_row_id)
  lens <- lengths(neighbor_lookup)
  
  if (sum(lens) == 0) {
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- paste0(var_name, c("_max", "_min", "_mean"))
    return(out)
  }
  
  from_row <- rep.int(seq_len(n), lens)
  to_row   <- unlist(neighbor_lookup, use.names = FALSE)
  
  # Get neighbor values
  nb_vals <- vals[to_row]
  
  # Use data.table for grouped aggregation
  dt_edges <- data.table(from_row = from_row, nb_val = nb_vals)
  
  # Remove NA neighbor values
  dt_edges <- dt_edges[!is.na(nb_val)]
  
  # Compute stats per from_row
  stats <- dt_edges[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = from_row]
  
  # Initialize output matrix with NAs
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[stats$from_row, 1] <- stats$nb_max
  out[stats$from_row, 2] <- stats$nb_min
  out[stats$from_row, 3] <- stats$nb_mean
  
  colnames(out) <- paste0(var_name, c("_max", "_min", "_mean"))
  return(out)
}


# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  
  col_max  <- paste0("neighbor_", var_name, "_max")
  col_min  <- paste0("neighbor_", var_name, "_min")
  col_mean <- paste0("neighbor_", var_name, "_mean")
  
  data[[col_max]]  <- stats_mat[, 1]
  data[[col_min]]  <- stats_mat[, 2]
  data[[col_mean]] <- stats_mat[, 3]
  
  return(data)
}


# ==============================================================================
# OPTIMIZED OUTER LOOP
# ==============================================================================

# Build the lookup ONCE (this is the former bottleneck, now fast)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is preserved — no retraining needed.
# The numerical estimand is preserved — identical max/min/mean computations.
```

---

## Summary of Changes and Expected Speedup

| Component | Original | Optimized | Why |
|---|---|---|---|
| **`build_neighbor_lookup`** | `lapply` over 6.46M rows, `paste()` + named-vector string hashing per row | Integer matrix `row_matrix[cell_ref, year_ref]`, vectorized expansion via `data.table` with iteration over 344K cells (not 6.46M rows) | Eliminates ~25.8M `paste()` calls and string hash lookups. Pure integer indexing. |
| **`compute_neighbor_stats`** | `lapply` over 6.46M elements, `do.call(rbind, ...)` | Flat edge table + `data.table` grouped aggregation (`[, .(...), by=from_row]`) | Vectorized C-level grouping replaces 6.46M R-level function calls. |
| **String operations** | `paste(id, year, sep="_")` for every row and every neighbor | `cbind(integer, integer)` matrix indexing | O(1) integer matrix access vs. O(k) string hashing. |

**Expected runtime:** From 86+ hours down to approximately **10–30 minutes** on the same 16 GB laptop, depending on I/O. The Random Forest model and all numerical outputs (neighbor max, min, mean) are preserved exactly.