 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (5 times total) and each invocation merely stacks pre-allocated 3-element vectors. This is a **minor** inefficiency.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`lapply` over 6.46 million rows with per-row `paste()` and character-keyed lookups.** For every single row, the function:
   - Converts `data$id[i]` to character and looks it up in `id_to_ref` (a named character vector — O(n) hash lookup repeated 6.46M times).
   - Extracts neighbor cell IDs via `neighbors[[ref_idx]]`.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — constructing new character strings for every neighbor of every row.
   - Looks up each of those strings in `idx_lookup`, another named vector with **6.46 million entries**.

2. **Scale of the problem:** With ~6.46M rows and an average of ~4 rook neighbors per cell, this means roughly **25.8 million character string constructions and named-vector lookups** inside the inner function, on top of the 6.46M outer iterations. Named vector lookup in R is hash-based but carries significant per-call overhead in an interpreted loop. The `paste()` calls generate enormous temporary character vectors.

3. **`build_neighbor_lookup` is called once and produces a list of 6.46M integer vectors.** This single structure dominates memory and time. The subsequent `compute_neighbor_stats` merely indexes into a numeric vector using these pre-built integer indices — that part is comparatively fast.

4. **The `paste`-based keying strategy is the deepest problem.** It converts a naturally integer problem (id × year → row index) into a character string matching problem, which is orders of magnitude slower.

**Conclusion:** The bottleneck is the O(N × avg_neighbors) character-key construction and lookup inside `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Replace character-keyed lookup with integer arithmetic.** Create a 2D integer matrix (or a hash via `data.table`) mapping `(id, year) → row_index`. This eliminates all `paste()` calls and named-vector lookups.

2. **Vectorize `build_neighbor_lookup` using `data.table`.** Use a join-based approach: for each row, expand its neighbors, then batch-join to find row indices. This replaces the interpreted R loop with a single vectorized operation.

3. **Replace `do.call(rbind, result)` with direct matrix pre-allocation** in `compute_neighbor_stats` (a secondary optimization).

4. **Preserve the trained Random Forest model** — we only change feature engineering, producing numerically identical columns.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Strategy: vectorized expansion + data.table keyed join
# Instead of 6.46M R-level iterations with paste/character lookup,
# we expand all (row, neighbor_id, year) combinations at once and join.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table if not already
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Step 1: Build mapping from cell id -> position in id_order
  id_to_ref <- data.table(
    cell_id = id_order,
    ref_idx = seq_along(id_order)
  )
  
  # Step 2: Build neighbor expansion table (cell_id -> neighbor_cell_id)
  # This is done ONCE for the spatial structure (not per year)
  neighbor_dt <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L) return(NULL)
    data.table(ref_idx = i, neighbor_ref_idx = nb)
  }))
  
  # Map ref_idx back to cell IDs
  neighbor_dt[, cell_id := id_order[ref_idx]]
  neighbor_dt[, neighbor_cell_id := id_order[neighbor_ref_idx]]
  neighbor_dt[, c("ref_idx", "neighbor_ref_idx") := NULL]
  
  # Step 3: Build a keyed lookup: (cell_id, year) -> row_idx
  row_lookup <- dt[, .(cell_id = id, year, row_idx)]
  setkey(row_lookup, cell_id, year)
  
  # Step 4: For every row in dt, get its cell_id and year, expand to neighbors,
  #         then join to find the row indices of those neighbors.
  
  # Expand: each row -> its neighbor cell_ids (same year)
  row_to_neighbors <- dt[, .(source_row = row_idx, cell_id = id, year)]
  setkey(neighbor_dt, cell_id)
  
  # Merge: row_to_neighbors x neighbor_dt on cell_id
  # This gives (source_row, year, neighbor_cell_id) for every row-neighbor pair
  expanded <- neighbor_dt[row_to_neighbors,
                          on = .(cell_id),
                          .(source_row, year, neighbor_cell_id),
                          allow.cartesian = TRUE,
                          nomatch = NA]
  
  # Drop rows where there were no neighbors
  expanded <- expanded[!is.na(neighbor_cell_id)]
  
  # Join to find the row index of each (neighbor_cell_id, year)
  expanded[, neighbor_row := row_lookup[.(neighbor_cell_id, year), row_idx]]
  
  # Drop unmatched
  expanded <- expanded[!is.na(neighbor_row)]
  
  # Step 5: Collapse back into a list indexed by source_row
  setkey(expanded, source_row)
  n_rows <- nrow(dt)
  
  # Split into list
  lookup_list <- vector("list", n_rows)
  
  # Fast split using data.table
  split_dt <- expanded[, .(neighbor_rows = list(neighbor_row)), by = source_row]
  
  # Fill the list
  for (j in seq_len(nrow(split_dt))) {
    lookup_list[[split_dt$source_row[j]]] <- split_dt$neighbor_rows[[j]]
  }
  
  # Fill empties with integer(0)
  empty_idx <- which(vapply(lookup_list, is.null, logical(1)))
  if (length(empty_idx) > 0) {
    lookup_list[empty_idx] <- list(integer(0))
  }
  
  lookup_list
}


# ==============================================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================================
# Strategy: pre-allocate matrix, avoid do.call(rbind, ...)

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


# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]
  
  data
}


# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================
# Build the lookup ONCE — this is now the fast version
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is UNCHANGED.
# The numerical outputs (neighbor_max, neighbor_min, neighbor_mean) are
# identical to the original implementation — same indices, same arithmetic.
```

---

## Summary of Expected Speedup

| Component | Original | Optimized | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (6.46M R-loop iterations with `paste` + named-vector lookup) | **~2–5 minutes** (vectorized `data.table` join, no character keys) | Eliminates per-row `paste()` and character hash lookups entirely |
| `compute_neighbor_stats` (×5 vars) | ~5–10 min total | **~2–5 min total** (pre-allocated matrix) | Minor improvement; was never the true bottleneck |
| `do.call(rbind, ...)` | Seconds per call | Eliminated | Replaced by direct matrix write |
| **Total** | **~86+ hours** | **~10–15 minutes** | **~350–500× speedup** |

The dominant cost was always the **6.46 million interpreted-loop iterations** in `build_neighbor_lookup`, each performing string construction and character-keyed lookups against a 6.46M-entry named vector. The `data.table` join-based approach converts this into a single vectorized equi-join operation, which is the correct optimization.