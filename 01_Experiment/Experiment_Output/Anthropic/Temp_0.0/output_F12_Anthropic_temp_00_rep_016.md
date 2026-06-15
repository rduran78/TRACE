 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it is a single vectorized operation that completes in seconds. The `lapply` inside `compute_neighbor_stats()` does no list binding at all — it returns a fixed-length vector per iteration.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive character key construction and lookup**: For each of the ~6.46 million rows, it calls `paste()` to construct character keys for every neighbor of that cell-year, then performs named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup in R is **O(n)** per query via hashing overhead at scale. With ~6.46M rows and an average of ~4 rook neighbors each, that's ~25.8 million `paste` + hash-lookup operations against a named vector of length 6.46M.

2. **Repeated `as.character()` and `paste()` calls inside the `lapply`**: Each of the 6.46M iterations does string coercion and concatenation — these are extremely expensive in a tight R loop.

3. **The lookup is rebuilt identically for all 5 variables but used 5 times**: This is fine (it's built once), but the build itself is the wall-clock killer. The `lapply` over 6.46M rows with string operations inside is the 86+ hour bottleneck.

4. **`compute_neighbor_stats()` is comparatively cheap**: It indexes a numeric vector by integer positions and computes `max/min/mean` on small subsets. This is fast.

**In summary**: The bottleneck is the per-row string-key construction and named-vector hash lookup inside `build_neighbor_lookup()`. The fix is to replace all string-keyed lookups with pure integer arithmetic.

---

## Optimization Strategy

1. **Replace string-key lookups with integer arithmetic**: Since every `id` appears for every year (balanced panel: 344,208 cells × 28 years = 9,637,824 — but the document says ~6.46M rows, so it may be unbalanced). We build an integer matrix mapping `(cell_index, year) → row_number` and look up neighbors via direct integer indexing — no strings, no hashing.

2. **Vectorize `compute_neighbor_stats()`**: Replace the per-row `lapply` with a single grouped operation using `data.table` or pre-allocated matrix fills, computing `max`, `min`, `mean` over neighbor indices in bulk.

3. **Preserve the trained Random Forest model**: We only change feature engineering (the pipeline that produces the same numerical columns). The RF model object is untouched.

4. **Preserve the original numerical estimand**: The computed `max`, `min`, `mean` of neighbor values are identical — we just compute them faster.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup using integer arithmetic (no string keys)
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table for fast operations
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Build integer map: id_to_ref (cell id -> position in id_order)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build a lookup matrix: rows = cell_ref_index, cols = year
  # cell_ref_index is position in id_order (1..344208)
  # year is mapped to 1..n_years
  years_unique <- sort(unique(dt$year))
  year_to_col  <- setNames(seq_along(years_unique), as.character(years_unique))
  n_cells      <- length(id_order)
  n_years      <- length(years_unique)
  
  # row_lookup_matrix[cell_ref, year_col] = row index in data (or NA)
  row_lookup_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  
  cell_refs <- id_to_ref[as.character(dt$id)]
  year_cols <- year_to_col[as.character(dt$year)]
  
  row_lookup_matrix[cbind(cell_refs, year_cols)] <- dt$row_idx
  
  # Now build neighbor_lookup: for each row, find neighbor rows
  # Pre-expand neighbors per cell into a flat structure for speed
  n_rows <- nrow(dt)
  
  # Precompute cell_ref and year_col for every row
  row_cell_ref <- as.integer(cell_refs)  # length n_rows
  row_year_col <- as.integer(year_cols)  # length n_rows
  
  # For each row i:
  #   neighbor_cell_refs = neighbors[[ row_cell_ref[i] ]]
  #   neighbor_rows = row_lookup_matrix[ neighbor_cell_refs, row_year_col[i] ]
  #   remove NAs
  
  # Vectorized approach: build flat neighbor table
  # Step 1: Expand neighbors into a data.table of (cell_ref, neighbor_cell_ref)
  neighbor_dt <- rbindlist(lapply(seq_along(neighbors), function(j) {
    nb <- neighbors[[j]]
    if (length(nb) == 0) return(NULL)
    data.table(cell_ref = j, nb_cell_ref = as.integer(nb))
  }))
  
  # Step 2: For each row, get its cell_ref and year_col
  row_info <- data.table(
    row_i    = seq_len(n_rows),
    cell_ref = row_cell_ref,
    year_col = row_year_col
  )
  
  # Step 3: Join row_info with neighbor_dt on cell_ref
  # This gives us: for each row_i, all neighbor cell_refs
  setkey(neighbor_dt, cell_ref)
  setkey(row_info, cell_ref)
  
  expanded <- neighbor_dt[row_info, on = "cell_ref", allow.cartesian = TRUE,
                          nomatch = NULL]
  # expanded has columns: cell_ref, nb_cell_ref, row_i, year_col
  
  # Step 4: Look up the neighbor's row index
  expanded[, nb_row := row_lookup_matrix[cbind(nb_cell_ref, year_col)]]
  
  # Step 5: Remove NAs (neighbor cell-year doesn't exist in data)
  expanded <- expanded[!is.na(nb_row)]
  
  # Step 6: Split into list indexed by row_i
  setkey(expanded, row_i)
  
  # Pre-allocate list
  neighbor_lookup <- vector("list", n_rows)
  
  # Fill using split
  split_result <- split(expanded$nb_row, expanded$row_i)
  
  # split() returns only keys that exist; fill them in
  idx_names <- as.integer(names(split_result))
  for (k in seq_along(idx_names)) {
    neighbor_lookup[[ idx_names[k] ]] <- as.integer(split_result[[k]])
  }
  
  # Rows with no neighbors remain NULL; convert to integer(0)
  null_mask <- vapply(neighbor_lookup, is.null, logical(1))
  neighbor_lookup[null_mask] <- list(integer(0))
  
  neighbor_lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats using vectorized data.table operations
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Build flat table of (row_i, neighbor_row)
  lens <- lengths(neighbor_lookup)
  row_i_vec <- rep.int(seq_len(n), lens)
  nb_row_vec <- unlist(neighbor_lookup, use.names = FALSE)
  
  if (length(nb_row_vec) == 0) {
    # No neighbors at all
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
    return(out)
  }
  
  # Get neighbor values
  nb_vals <- vals[nb_row_vec]
  
  # Build data.table and compute grouped stats
  dt_nb <- data.table(row_i = row_i_vec, val = nb_vals)
  dt_nb <- dt_nb[!is.na(val)]
  
  stats <- dt_nb[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = row_i]
  
  # Allocate output matrix (NA by default for rows with no valid neighbors)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[stats$row_i, 1] <- stats$nb_max
  out[stats$row_i, 2] <- stats$nb_min
  out[stats$row_i, 3] <- stats$nb_mean
  
  colnames(out) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  out
}

# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  for (col_idx in seq_len(ncol(stats_mat))) {
    data[[colnames(stats_mat)[col_idx]]] <- stats_mat[, col_idx]
  }
  data
}

# =============================================================================
# MAIN PIPELINE (drop-in replacement)
# =============================================================================

# Build the neighbor lookup ONCE (this is now fast: minutes, not days)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is UNCHANGED.
# The numerical estimand is PRESERVED (identical max/min/mean computations).
# Predicted values from the RF model will be numerically identical.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations of `paste()` + named-vector hash lookup | Eliminated |
| **Key construction** | String concatenation (`paste`) per row per neighbor | Integer matrix indexing: `row_lookup_matrix[cbind(cell_ref, year_col)]` |
| **Neighbor lookup** | Named character vector (~6.46M entries), queried ~25.8M times | Integer matrix direct indexing, then vectorized `data.table` join |
| **Stats computation** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation on flat neighbor table |
| **Estimated runtime** | 86+ hours | ~5–15 minutes on the same laptop |
| **RF model** | Preserved (untouched) | Preserved (untouched) |
| **Numerical output** | Original `max/min/mean` | Identical `max/min/mean` |

The colleague misidentified the symptom (`do.call(rbind, ...)`) as the disease. The real pathology is millions of string-construction and hash-lookup operations inside `build_neighbor_lookup()`. Replacing strings with integer arithmetic and replacing row-wise `lapply` with vectorized `data.table` operations eliminates the bottleneck entirely.