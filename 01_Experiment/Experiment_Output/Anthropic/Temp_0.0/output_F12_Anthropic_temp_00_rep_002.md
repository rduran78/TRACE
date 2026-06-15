 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on a list of 6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. There is no "repeated list binding" inside `compute_neighbor_stats()` — it uses `lapply` to build a list in one pass, then binds once. This is standard and efficient.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row `paste()` and character lookup in a named vector — 6.46 million times.** For each of the ~6.46M rows, the inner `lapply` function calls `as.character()`, indexes into `id_to_ref`, calls `paste()` to construct `neighbor_keys` (one per neighbor), and then performs named-vector lookup into `idx_lookup` (which itself is a named character vector of length ~6.46M). Named vector lookup in R is O(n) hash-table probing per call, and doing this millions of times with multi-element key vectors is catastrophically slow.

2. **The `idx_lookup` named vector has ~6.46M entries.** Each lookup into it requires hashing a character string and probing a large table. With an average of ~4 rook neighbors per cell, that's ~25.8 million individual character-key lookups into a 6.46M-entry named vector, nested inside a 6.46M-iteration `lapply`. The total character operations (paste + hash + match) dominate runtime.

3. **This function is called once, but it takes the vast majority of the 86+ hours.** `compute_neighbor_stats` is called only 5 times and each call is a simple numeric indexing operation (`vals[idx]`) which is O(1) per element — fast.

**In summary:** The bottleneck is the O(n × k) character-key construction and named-vector lookup inside `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all character key construction and named-vector lookup.** Replace `paste()`-based keys and named-vector indexing with pure integer arithmetic. Since years are contiguous (1992–2019), we can map each `(id, year)` pair to a row index using a precomputed integer matrix or a direct offset formula.

2. **Vectorize the neighbor lookup construction** using `data.table` for fast group-indexed operations, or — even better — use a direct integer-offset scheme: if data is sorted by `(id, year)`, then for a given cell `id` at row position `base_row`, all its year-rows are at known offsets, and neighbor cells' rows can be computed by integer arithmetic alone.

3. **Vectorize `compute_neighbor_stats()`** by unrolling the neighbor list into a long vector, using `grouping` to compute `max/min/mean` in one vectorized pass via `data.table` or `collapse`.

4. **Preserve the trained Random Forest model** — we only change feature-engineering code, not the model or the numerical results.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Strategy: avoid all paste() and named-vector character lookups.
# Use integer arithmetic with a pre-built (id, year) -> row_index hash via
# data.table, then expand neighbor pairs with integer indexing only.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of unique cell IDs in the order matching neighbors (nb object)
  # neighbors: list of integer neighbor indices (spdep nb object)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build a keyed lookup: (id, year) -> row index
  # Using integer-keyed data.table join (very fast)
  setkey(dt, id, year)

  # Map each unique cell id to its position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Get unique years (sorted)
  years <- sort(unique(dt$year))
  n_years <- length(years)

  # --- Build the full neighbor edge list at the cell level ---
  # For each cell position p in id_order, neighbors[[p]] gives neighbor positions
  # Expand to (focal_id, neighbor_id) pairs
  n_cells <- length(id_order)

  focal_pos <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_pos <- unlist(neighbors, use.names = FALSE)

  # Convert positions back to actual IDs
  focal_ids <- id_order[focal_pos]
  neighbor_ids <- id_order[neighbor_pos]

  # --- Cross with years to get (focal_id, year, neighbor_id) triples ---
  # Instead of crossing everything (expensive in memory), we build the lookup
  # row-by-row using a merge approach.

  # Create edge data.table at cell level
  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)

  # For the row-index lookup, we need (id, year) -> row_idx
  # Build this as a keyed data.table for fast joins
  row_map <- dt[, .(id, year, row_idx)]
  setkey(row_map, id, year)

  # Now, for each row in dt, we need:
  #   1. Find which cell position this row's id maps to
  #   2. Get that cell's neighbor IDs
  #   3. Look up (neighbor_id, same year) in row_map
  #
  # We can do this as a large join:

  # Step 1: For each row in dt, attach its neighbor IDs
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Build focal_row -> neighbor_ids mapping
  # edges is at cell level; we need to expand by year
  # Approach: join dt with edges on focal_id, then join with row_map on (neighbor_id, year)

  cat("Building neighbor edge list with year expansion...\n")

  # Add year info to edges by joining with dt (focal side)
  # dt has (id, year, row_idx); edges has (focal_id, neighbor_id)
  setkey(edges, focal_id)

  # For each (focal_id, year) combination, get all neighbor_ids
  # Then look up (neighbor_id, year) -> neighbor_row_idx
  focal_rows <- dt[, .(focal_row = row_idx, focal_id = id, year)]

  # Join focal_rows with edges to get (focal_row, year, neighbor_id)
  setkey(focal_rows, focal_id)
  setkey(edges, focal_id)
  expanded <- edges[focal_rows, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded has: focal_id, neighbor_id, focal_row, year

  # Now join with row_map to get neighbor_row_idx
  setnames(row_map, c("id", "year", "row_idx"), c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(row_map, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  expanded <- row_map[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched rows (neighbor exists in that year)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  cat("Assembling lookup list...\n")

  # Build the lookup list: for each focal_row, a vector of neighbor_row_idx
  setkey(expanded, focal_row)
  n_rows <- nrow(dt)

  # Split neighbor_row_idx by focal_row
  lookup_dt <- expanded[, .(neighbors = list(neighbor_row_idx)), by = focal_row]
  setkey(lookup_dt, focal_row)

  # Initialize full lookup (some rows may have no neighbors)
  neighbor_lookup <- vector("list", n_rows)
  neighbor_lookup[lookup_dt$focal_row] <- lookup_dt$neighbors

  # Fill empties with integer(0)
  empty <- which(vapply(neighbor_lookup, is.null, logical(1)))
  if (length(empty) > 0) {
    neighbor_lookup[empty] <- list(integer(0))
  }

  return(neighbor_lookup)
}


# ==============================================================================
# OPTIMIZED compute_neighbor_stats (vectorized, no per-row lapply)
# ==============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Unroll the neighbor lookup into a long (focal_row, neighbor_row) table
  lens <- lengths(neighbor_lookup)
  focal_rows <- rep(seq_len(n), lens)
  neighbor_rows <- unlist(neighbor_lookup, use.names = FALSE)

  # Get neighbor values
  neighbor_vals <- vals[neighbor_rows]

  # Remove NAs
  valid <- !is.na(neighbor_vals)
  focal_rows <- focal_rows[valid]
  neighbor_vals <- neighbor_vals[valid]

  # Compute grouped stats using data.table
  stats_dt <- data.table(focal = focal_rows, val = neighbor_vals)
  agg <- stats_dt[, .(
    max_val  = max(val),
    min_val  = min(val),
    mean_val = mean(val)
  ), by = focal]

  # Map back to full result matrix
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  result[agg$focal, 1] <- agg$max_val
  result[agg$focal, 2] <- agg$min_val
  result[agg$focal, 3] <- agg$mean_val

  return(result)
}


# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]

  return(data)
}


# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================
# cell_data, id_order, rook_neighbors_unique are assumed to exist in the environment.
# The trained Random Forest model is NOT touched.

cat("Building optimized neighbor lookup...\n")
system.time({
  neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  system.time({
    cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
  })
}

cat("Done. Neighbor features added. RF model unchanged.\n")
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, result)` and list binding in `compute_neighbor_stats()` | **Wrong.** This is a single bind of a pre-built list — seconds at most. |
| **True bottleneck** | — | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + named-vector character lookup into a 6.46M-entry table (~25.8M hash probes). This is where 86+ hours are spent. |
| **Optimization** | — | Replace all character-key operations with integer-indexed `data.table` equi-joins. Vectorize `compute_neighbor_stats()` by unrolling to a long table and using grouped aggregation. |
| **Expected speedup** | — | From 86+ hours to **minutes** (the join-based lookup and vectorized stats should complete in ~2–10 minutes on 16 GB RAM). |
| **RF model** | — | Untouched. Same numerical output (max, min, mean of neighbor values), same column names. |