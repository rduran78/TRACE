 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable — only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning pre-allocated vectors. That step is O(n) and relatively fast.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Character key construction and named-vector lookup over 6.46 million rows.** `paste(data$id, data$year, sep = "_")` creates 6.46M character keys. Then, for *each* of the 6.46M rows, it does:
   - `as.character(data$id[i])` — character conversion per row.
   - `id_to_ref[as.character(...)]` — named vector lookup (hash-based, but called 6.46M times individually inside `lapply`).
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` — creates ~4 character keys per row (rook neighbors), totaling ~26M `paste` calls.
   - `idx_lookup[neighbor_keys]` — named vector lookup against a 6.46M-element named vector, called 6.46M times with ~4 keys each = ~26M hash lookups.

2. **Total cost:** ~6.46 million R-level function calls via `lapply`, each doing multiple character allocations, paste operations, and hash lookups. This is the dominant O(n × k) bottleneck with enormous constant factors from R's character handling overhead. On a laptop, this alone accounts for the vast majority of the 86+ hour runtime.

3. By contrast, `compute_neighbor_stats()` simply indexes into a numeric vector with pre-computed integer indices — extremely fast. And `do.call(rbind, result)` on 6.46M length-3 vectors takes seconds, not hours.

## Optimization Strategy

1. **Replace character-key hashing with integer arithmetic.** Map `(id, year)` pairs to integer indices using a direct integer lookup table instead of character paste + named vector lookup. Since years span 1992–2019 (28 years), we can encode `(id, year)` as a single integer and use `match()` or a direct-index table.

2. **Vectorize `build_neighbor_lookup` entirely** — eliminate the per-row `lapply` by expanding all neighbor relationships at once using vectorized operations, then splitting by row.

3. **Replace `do.call(rbind, ...)` with direct matrix pre-allocation** in `compute_neighbor_stats` (a minor but easy win).

4. **Preserve the trained Random Forest model** — we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
# =============================================================================
# OPTIMIZED build_neighbor_lookup — fully vectorized, no character keys
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n <- nrow(data)
  num_ids <- length(id_order)
  
  # Step 1: Create integer mapping from id -> reference index (position in id_order)
  # Use match() once, vectorized over all rows
  id_ref <- match(data$id, id_order)  # length n, integer vector
  
  # Step 2: Create a fast (id, year) -> row mapping using an integer-keyed approach.
  # Map years to 1..28
  years <- sort(unique(data$year))
  year_offset <- match(data$year, years)  # length n, integer vector
  
  num_years <- length(years)
  
  # Encode (id_position_in_id_order, year_offset) as a single integer key
  # key = (id_index_in_id_order - 1) * num_years + year_offset
  # This gives a unique integer in 1..(num_ids * num_years) for each (id, year)
  
  row_keys <- (match(data$id, id_order) - 1L) * num_years + year_offset
  
  # Build reverse lookup: key -> row index
  # Pre-allocate a vector of size num_ids * num_years (344208 * 28 ≈ 9.6M — fits easily)
  max_key <- num_ids * num_years
  key_to_row <- integer(max_key)
  key_to_row[row_keys] <- seq_len(n)
  
  # Step 3: Expand all neighbor relationships vectorized
  # For each row i, we need neighbors[[id_ref[i]]] mapped to the same year.
  # 
  # Strategy: build an edge list (row_index, neighbor_id_ref) for all rows,
  # then compute neighbor keys and look up row indices.
  
  # Precompute neighbor lengths
  nb_lengths <- lengths(neighbors)  # length = num_ids
  row_nb_lengths <- nb_lengths[id_ref]  # length = n, neighbors per row
  
  # Total edges ≈ sum of row_nb_lengths (≈ 6.46M * ~4 = ~26M)
  total_edges <- sum(row_nb_lengths)
  
  # Row indices repeated by their neighbor count
  row_rep <- rep.int(seq_len(n), row_nb_lengths)
  
  # Year offsets repeated
  year_rep <- year_offset[row_rep]
  
  # Neighbor id references: unlist neighbors in id_ref order, then index
  # neighbors is an nb object indexed by position in id_order.
  # For row i, the neighbor refs are neighbors[[id_ref[i]]].
  # We need to unlist neighbors[id_ref] in order.
  
  nb_expanded <- neighbors[id_ref]  # list of length n, reordered
  neighbor_refs <- unlist(nb_expanded, use.names = FALSE)  # integer vector, length = total_edges
  
  # Compute keys for all neighbor (id_ref, year) pairs
  neighbor_keys <- (neighbor_refs - 1L) * num_years + year_rep
  
  # Look up row indices
  neighbor_rows <- key_to_row[neighbor_keys]
  
  # Remove invalid (0 means no matching row in data)
  valid <- neighbor_rows > 0L
  row_rep_valid <- row_rep[valid]
  neighbor_rows_valid <- neighbor_rows[valid]
  
  # Step 4: Split neighbor row indices by source row
  # Use split with factor to preserve all levels (rows with no valid neighbors get integer(0))
  lookup <- split(neighbor_rows_valid, factor(row_rep_valid, levels = seq_len(n)))
  
  # Ensure each element is integer
  lookup <- lapply(lookup, as.integer)
  
  lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats — pre-allocated matrix, no do.call(rbind)
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) next
    result_mat[i, 1L] <- max(neighbor_vals)
    result_mat[i, 2L] <- min(neighbor_vals)
    result_mat[i, 3L] <- mean(neighbor_vals)
  }
  
  result_mat
}

# =============================================================================
# ALTERNATIVE: Fully vectorized compute_neighbor_stats using group operations
# (avoids the R-level for loop entirely for even more speed)
# =============================================================================
compute_neighbor_stats_vectorized <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Build flat vectors: source row and neighbor value
  lens <- lengths(neighbor_lookup)
  src <- rep.int(seq_len(n), lens)
  nb_idx <- unlist(neighbor_lookup, use.names = FALSE)
  nb_vals <- vals[nb_idx]
  
  # Remove NAs
  valid <- !is.na(nb_vals)
  src <- src[valid]
  nb_vals <- nb_vals[valid]
  
  # Use data.table for fast grouped aggregation
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::data.table(src = src, val = nb_vals)
    agg <- dt[, .(mx = max(val), mn = min(val), mn2 = mean(val)), by = src]
    
    result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
    result_mat[agg$src, 1L] <- agg$mx
    result_mat[agg$src, 2L] <- agg$mn
    result_mat[agg$src, 3L] <- agg$mn2
  } else {
    # Fallback: tapply
    f <- factor(src, levels = seq_len(n))
    result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
    result_mat[, 1L] <- as.numeric(tapply(nb_vals, f, max))
    result_mat[, 2L] <- as.numeric(tapply(nb_vals, f, min))
    result_mat[, 3L] <- as.numeric(tapply(nb_vals, f, mean))
  }
  
  result_mat
}

# =============================================================================
# Wrapper: compute_and_add_neighbor_features (preserves original column names)
# =============================================================================
compute_and_add_neighbor_features <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_vectorized(data, neighbor_lookup, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]
  
  data
}

# =============================================================================
# MAIN PIPELINE — drop-in replacement
# =============================================================================

# Build the optimized lookup (runs once; minutes instead of tens of hours)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched — use it directly for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Complexity Improvements

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | O(n) R-level calls, each doing `paste` + hash lookup against 6.46M-element named vector. **~26M character hash lookups.** | **Fully vectorized** integer arithmetic + single `split()`. Zero per-row R function calls. |
| Key encoding | Character paste `"id_year"` → named vector | Integer key `(id_idx-1)*28 + year_idx` → direct integer indexing into pre-allocated vector |
| `compute_neighbor_stats` | `lapply` + `do.call(rbind, ...)` | Vectorized `unlist` → `data.table` grouped aggregation, or pre-allocated matrix loop |
| **Estimated runtime** | **86+ hours** | **~5–15 minutes** on the same laptop |

**Key insight:** The bottleneck was never the `rbind` or the stats computation — it was the 6.46 million individual R-level iterations in `build_neighbor_lookup`, each performing expensive character string construction and hash-table lookups against a massive named vector. Replacing character keys with integer arithmetic and vectorizing the entire expansion eliminates this bottleneck by orders of magnitude.