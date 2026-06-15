 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The string-keyed lookup `idx_lookup[neighbor_keys]` is nominally O(1) amortized per key (R uses hashing for named vectors), but the constant factor is enormous: each call hashes freshly-allocated strings against a 6.46M-name hash table. The total work is roughly **25.8 million string allocations + hash lookups**, all inside an R-level `lapply` with no vectorization.

### The Deeper Structural Insight

The entire string-key scheme is **unnecessary**. The lookup answers one simple question:

> *"Given that row `i` belongs to cell `id` in year `y`, which rows belong to cell `id`'s rook neighbors in the same year `y`?"*

Since the panel is balanced (344,208 cells × 28 years = 9,637,824 potential rows, ~6.46M present), the mapping from `(cell, year)` → row index can be done with **integer arithmetic** on a pre-built integer matrix, completely eliminating string operations.

Furthermore, `compute_neighbor_stats` is called **5 separate times**, each time iterating over the same 6.46M-element `neighbor_lookup`. The neighbor topology doesn't change across variables — only the values do. This means the neighbor-gather pattern should be **vectorized across all variables simultaneously** using matrix indexing.

---

## Optimization Strategy

| Layer | Current | Proposed |
|-------|---------|----------|
| **Cell→index mapping** | String paste + named vector | Integer lookup table `cell_row_matrix[cell_index, year_index]` |
| **Neighbor expansion** | Row-by-row `lapply` with string keys | Vectorized construction of a sparse neighbor-row edge list using `data.table` joins |
| **Stat computation** | `lapply` over 6.46M lists, per variable | Single grouped aggregation via `data.table` over the edge list, all variables at once |
| **Overall complexity** | ~6.46M × (string alloc + hash) × 5 vars | One join to build edge list + one grouped aggregation |

**Estimated speedup**: from 86+ hours to **minutes** (dominated by the `data.table` grouped aggregation over ~25M edges × 5 variables).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves the exact numerical estimand (max, min, mean of non-NA neighbor vals)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (by reference if already one, copy otherwise)
  dt <- as.data.table(cell_data)
  dt[, row_idx__ := .I]
  
  # ------------------------------------------------------------------
  # Step 1: Build an integer mapping from cell id -> position in id_order
  #         This replaces id_to_ref.
  # ------------------------------------------------------------------
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Assign each row its cell-position index (integer, no strings)
  dt[, cell_pos__ := id_to_pos[as.character(id)]]
  
  # ------------------------------------------------------------------
  # Step 2: Build a directed edge list of (cell_pos, neighbor_cell_pos)
  #         from the nb object. This is done once, independent of year.
  # ------------------------------------------------------------------
  # rook_neighbors_unique is a list of length = length(id_order),
  # where element [[j]] gives integer indices into id_order of j's neighbors.
  
  from_pos <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_pos <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  edges <- data.table(cell_pos_from = from_pos, cell_pos_to = to_pos)
  
  cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edges)))
  
  # ------------------------------------------------------------------
  # Step 3: For each row in dt, find its neighbor rows in the same year.
  #
  #   Logic: row i has cell_pos__ = p, year = y.
  #          Its neighbor rows are all rows with cell_pos__ in neighbors(p)

  #          AND year == y.
  #
  #   We achieve this with a join:
  #     dt[, .(row_idx__, cell_pos__, year)]
  #       JOIN edges ON cell_pos__ == cell_pos_from
  #     -> gives (row_idx__=i, year=y, cell_pos_to=q)
  #     Then join back to dt on (cell_pos__=q, year=y) to get neighbor row.
  # ------------------------------------------------------------------
  
  # Slim table for the "from" side: each row's identity
  dt_from <- dt[, .(row_idx_from = row_idx__, cell_pos__, year)]
  setkey(dt_from, cell_pos__)
  
  # Slim table for the "to" side: lookup by (cell_pos, year) -> row index
  dt_to <- dt[, .(row_idx_to = row_idx__, cell_pos__, year)]
  setkey(dt_to, cell_pos__, year)
  
  # Join from-rows to edges: for each row, get its neighbor cell positions
  # This produces ~6.46M * avg_neighbors rows ≈ 25-26M rows
  setkey(edges, cell_pos_from)
  
  cat("Joining rows to neighbor edges...\n")
  expanded <- dt_from[edges, on = .(cell_pos__ = cell_pos_from),
                      .(row_idx_from, year, cell_pos_to = cell_pos_to),
                      nomatch = NULL,
                      allow.cartesian = TRUE]
  
  cat(sprintf("Expanded edge-row table: %d rows\n", nrow(expanded)))
  
  # Now join to find the actual neighbor row index in the same year
  cat("Resolving neighbor rows by (cell_pos, year)...\n")
  expanded[dt_to,
           row_idx_to := i.row_idx_to,
           on = .(cell_pos_to = cell_pos__, year = year)]
  
  # Drop edges where the neighbor cell-year doesn't exist in the data
  expanded <- expanded[!is.na(row_idx_to)]
  
  cat(sprintf("Resolved neighbor pairs: %d\n", nrow(expanded)))
  
  # ------------------------------------------------------------------
  # Step 4: Gather neighbor values for all source variables at once,
  #         then compute grouped stats (max, min, mean of non-NA).
  # ------------------------------------------------------------------
  
  # Extract the variable columns for neighbor rows
  # We do this by direct column indexing into the original dt
  cat("Gathering neighbor variable values...\n")
  
  for (vn in neighbor_source_vars) {
    col_vals <- dt[[vn]]  # full column vector
    expanded[, (vn) := col_vals[row_idx_to]]
  }
  
  # Compute stats grouped by row_idx_from (the focal row)
  cat("Computing neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(vn) {
    list(
      bquote(max(.(as.name(vn)), na.rm = TRUE)),
      bquote(min(.(as.name(vn)), na.rm = TRUE)),
      bquote(mean(.(as.name(vn)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(vn) {
    paste0("neighbor_", c("max_", "min_", "mean_"), vn)
  }))
  
  names(agg_exprs) <- agg_names
  
  # For rows where ALL neighbor values of a variable are NA,
  # max/min with na.rm=TRUE produce Inf/-Inf and mean produces NaN.
  # We'll fix those after aggregation.
  
  stats <- expanded[,
    lapply(agg_exprs, eval, envir = .SD),
    by = row_idx_from
  ]
  
  # Replace Inf/-Inf/NaN with NA to match original behavior
  for (col in agg_names) {
    vals <- stats[[col]]
    vals[is.infinite(vals) | is.nan(vals)] <- NA_real_
    set(stats, j = col, value = vals)
  }
  
  # ------------------------------------------------------------------
  # Step 5: Merge stats back to the main data, preserving row order.
  # ------------------------------------------------------------------
  cat("Merging results back to main data...\n")
  
  # Ensure all rows are represented (rows with no valid neighbors get NA)
  setkey(stats, row_idx_from)
  
  for (col in agg_names) {
    # Initialize with NA
    dt[, (col) := NA_real_]
    # Fill in computed values
    dt[stats, (col) := get(paste0("i.", col)), on = .(row_idx__ = row_idx_from)]
  }
  
  # Clean up helper columns
  dt[, c("row_idx__", "cell_pos__") := NULL]
  
  cat("Done.\n")
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# Usage — drop-in replacement for the original outer loop
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The resulting cell_data now contains columns:
# #   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
# #   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
# #   ... etc.
# #
# # These are numerically identical to the original implementation.
# # The trained Random Forest model can be used directly for prediction
# # as long as the column names match (adjust naming if needed).
```

---

## Handling the Column-Name Convention

If your existing trained Random Forest expects specific column names (e.g., `ntl_neighbor_max` instead of `neighbor_max_ntl`), add a renaming step:

```r
# Adjust column names to match the trained model's expectations
# Example: if the original code produced "ntl_neighbor_max", "ntl_neighbor_min", etc.
rename_map <- unlist(lapply(neighbor_source_vars, function(vn) {
  old <- paste0("neighbor_", c("max_", "min_", "mean_"), vn)
  new <- paste0(vn, "_neighbor_", c("max", "min", "mean"))
  setNames(old, new)
}))

setnames(cell_data, rename_map, names(rename_map))
```

---

## Why This Is Correct (Numerical Equivalence)

| Original | Optimized | Equivalence |
|----------|-----------|-------------|
| For row `i`, find neighbor cell IDs via `id_order[neighbors[[ref_idx]]]` | Edge list built from same `rook_neighbors_unique` and `id_order` | Same neighbor cell set |
| Filter to same year via string key `paste(id, year)` | Equi-join on `(cell_pos, year)` | Same year filter |
| Drop `NA` results (cell-year not in data) | `nomatch = NULL` + `!is.na(row_idx_to)` | Same missing-data handling |
| `max`, `min`, `mean` of non-NA neighbor values | Same functions with `na.rm = TRUE`, then Inf/NaN → NA | Identical output |

---

## Performance Estimate

| Step | Estimated Time | Memory |
|------|---------------|--------|
| Edge list construction | < 1 sec | ~20 MB |
| First join (rows × edges) | ~30 sec | ~1.5 GB (25M rows × 3 int cols) |
| Second join (resolve neighbor rows) | ~30 sec | in-place |
| Gather 5 variable columns | ~10 sec | +1 GB |
| Grouped aggregation (15 stats) | ~2-3 min | ~1 GB |
| Merge back | ~10 sec | in-place |
| **Total** | **~4-5 minutes** | **~4-5 GB peak** |

This fits comfortably within 16 GB RAM and replaces the estimated 86+ hour runtime. The trained Random Forest model is untouched — only the feature-construction pipeline is reformulated.