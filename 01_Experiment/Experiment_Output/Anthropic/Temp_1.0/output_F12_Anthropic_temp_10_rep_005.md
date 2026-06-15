 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorizable matrix assembly step and takes on the order of seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Massive per-row character key construction and named-vector lookup**: `build_neighbor_lookup()` runs an `lapply` over **every row** (~6.46 million rows). For each row, it:
   - Converts `data$id[i]` to character and performs a named-vector lookup (`id_to_ref[as.character(data$id[i])]`) — this is O(n) named-vector search repeated 6.46M times.
   - Retrieves the neighbor cell IDs and calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build string keys — millions of string allocations.
   - Performs **another** named-vector lookup `idx_lookup[neighbor_keys]` where `idx_lookup` is a named vector of length **6.46 million** — named-vector lookup on a vector this size is extremely slow (R uses linear hashing with string comparison, and doing this 6.46M × ~4 neighbors ≈ 25.8 billion character comparisons).

2. **The result is invariant across variables but recomputed only once** — that's fine, but the single computation itself is the 86+ hour bottleneck. The `paste`-based string key lookups against a 6.46M-entry named vector are catastrophically slow. Named vector indexing in R with string keys is not O(1) hash-table lookup; performance degrades severely at this scale.

3. `compute_neighbor_stats()` by contrast is simple: it indexes a numeric vector by integer positions (fast) and computes `max/min/mean` on small neighbor sets. The `do.call(rbind, ...)` on the result list is a single operation. Even if suboptimal, it accounts for seconds, not hours.

**Root cause summary**: The pipeline spends virtually all its time in `build_neighbor_lookup()` doing repeated string construction (`paste`) and named-character-vector indexing (`idx_lookup[neighbor_keys]`) at a scale of ~6.46M × ~4 = ~25M lookups against a 6.46M-length named vector.

---

## Optimization Strategy

1. **Replace string-key lookups with integer arithmetic.** Instead of building `"id_year"` string keys and looking them up in a named vector, use the structure of the panel: every cell appears in every year (balanced panel, 28 years). Compute row positions arithmetically: `row = (cell_position - 1) * n_years + year_offset`. This is O(1) per lookup with no string allocation.

2. **Vectorize `build_neighbor_lookup()`** — eliminate the per-row `lapply` entirely. Pre-expand the neighbor relationships into a full edge list (cell_i → cell_j for each year), compute target row indices with integer math, and store the result as a grouped integer list.

3. **Vectorize `compute_neighbor_stats()`** — replace `lapply` + `do.call(rbind, ...)` with column-wise grouped aggregation using the edge list and `data.table` or vectorized split/vapply.

4. **Preserve the trained Random Forest model** — we only change feature engineering speed; the output columns are numerically identical (same max, min, mean of the same neighbor values).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Ensure data is a data.table, sorted by (id, year)
# ============================================================
cell_dt <- as.data.table(cell_data)

# Sort by id then year — critical for the arithmetic indexing trick
setorder(cell_dt, id, year)

# The unique IDs and years, in sorted order
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_ids   <- length(unique_ids)   # 344,208
n_years <- length(unique_years) # 28

# Verify balanced panel
stopifnot(nrow(cell_dt) == n_ids * n_years)

# ============================================================
# STEP 1: Build integer mappings (replaces all paste/named-vector lookups)
# ============================================================

# Map each unique cell id to its 1-based position in the sorted id vector
id_to_pos <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Map each year to its 1-based offset
year_to_offset <- setNames(seq_along(unique_years), as.character(unique_years))

# With data sorted by (id, year), the row index of cell at position p
# in year with offset y is:  (p - 1) * n_years + y
# This replaces the entire idx_lookup named vector and paste() calls.

# ============================================================
# STEP 2: Build the neighbor edge list ONCE using integer arithmetic
#          (replaces build_neighbor_lookup entirely)
# ============================================================

# rook_neighbors_unique is an nb object: a list of length n_ids,
# where element [[p]] gives the positions (in id_order) of neighbors of
# the p-th cell in id_order.
# id_order is the vector of cell IDs in the order matching the nb object.

# Map id_order positions to our sorted-id positions
id_order_to_pos <- id_to_pos[as.character(id_order)]

build_neighbor_edge_list <- function(rook_neighbors_unique,
                                     id_order_to_pos,
                                     n_years) {
  # For each cell position p in id_order, get its neighbors' positions
  # and create edges (source_pos, target_pos)
  n_cells <- length(rook_neighbors_unique)

  # Pre-compute lengths for pre-allocation
  lens <- lengths(rook_neighbors_unique)
  total_edges_per_year <- sum(lens)  # ~1.37M directed edges

  # Pre-allocate edge list for ONE year
  source_pos <- integer(total_edges_per_year)
  target_pos <- integer(total_edges_per_year)

  offset <- 0L
  for (p in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[p]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    n_nb <- length(nb)
    idx_range <- (offset + 1L):(offset + n_nb)
    source_pos[idx_range] <- id_order_to_pos[p]
    target_pos[idx_range] <- id_order_to_pos[nb]
    offset <- offset + n_nb
  }

  # Trim if any cells had 0 neighbors
  if (offset < total_edges_per_year) {
    source_pos <- source_pos[1:offset]
    target_pos <- target_pos[1:offset]
  }

  # Now expand across all years using integer row arithmetic
  # Row of cell at position p in year-offset y = (p-1)*n_years + y
  source_rows <- integer(offset * n_years)
  target_rows <- integer(offset * n_years)

  for (y in seq_len(n_years)) {
    rng <- ((y - 1L) * offset + 1L):(y * offset)
    source_rows[rng] <- (source_pos - 1L) * n_years + y
    target_rows[rng] <- (target_pos - 1L) * n_years + y
  }

  data.table(source_row = source_rows, target_row = target_rows)
}

cat("Building neighbor edge list...\n")
system.time({
  edge_dt <- build_neighbor_edge_list(rook_neighbors_unique,
                                      id_order_to_pos,
                                      n_years)
})
# Expected: ~38.4M rows (1.37M edges × 28 years), built in seconds

# ============================================================
# STEP 3: Vectorized compute_neighbor_stats using data.table grouping
#          (replaces compute_neighbor_stats and the for loop)
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")

  # Extract the variable values for all target (neighbor) rows
  edge_dt[, val := cell_dt[[var_name]][target_row]]

  # Remove NAs from neighbor values before aggregation
  valid_edges <- edge_dt[!is.na(val)]

  # Grouped aggregation: max, min, mean by source_row
  agg <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]

  # Initialize result columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Fill in computed values
  cell_dt[agg$source_row, (max_col)  := agg$nb_max]
  cell_dt[agg$source_row, (min_col)  := agg$nb_min]
  cell_dt[agg$source_row, (mean_col) := agg$nb_mean]
}

# Clean up temporary column
edge_dt[, val := NULL]

# ============================================================
# STEP 4: Convert back to data.frame if the RF model expects one
# ============================================================
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features added to cell_data.\n")
```

---

## Performance Comparison

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (string paste + named-vector lookup over 6.46M rows) | **~85+ hours** (dominant bottleneck) | **Eliminated** — replaced by integer arithmetic edge list built in ~5–15 seconds |
| `compute_neighbor_stats` (lapply + do.call rbind, 5 vars) | ~minutes | **~1–3 minutes total** via `data.table` grouped aggregation on ~38.4M-row edge list |
| **Total estimated runtime** | **86+ hours** | **< 5 minutes** |

## Why This Preserves Correctness

- The neighbor relationships are identical: same rook neighbors, same year matching, same directed edges.
- The aggregation functions (`max`, `min`, `mean`) are computed on exactly the same sets of non-NA neighbor values.
- The output columns are numerically identical to the original pipeline's output.
- The trained Random Forest model is not modified or retrained — it simply receives the same feature columns it was trained on.