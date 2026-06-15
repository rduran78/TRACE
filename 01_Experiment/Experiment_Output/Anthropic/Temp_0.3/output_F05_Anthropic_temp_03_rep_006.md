 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character conversions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **constructs ~4 string keys per row × 6.46M rows ≈ 25.8M string allocations** (for ~4 average rook neighbors).
4. Looks up those keys in `idx_lookup` — a **named character vector of length 6.46M**, so each lookup is an O(N) hash probe on a massive vector.

The total work is roughly **6.46M × (string construction + hash lookups into a 6.46M-entry table)**. This is the dominant bottleneck.

### The Deeper Structural Insight

The neighbor relationships are **time-invariant** (rook contiguity doesn't change across years), and the panel is **balanced** (every cell appears in every year). Therefore:

- The neighbor lookup for cell `c` in year `t` always returns the same set of neighbor cells in year `t`.
- You don't need string keys at all. You can compute a **row-offset scheme**: if data is sorted by `(id, year)` or `(year, id)`, then the row index of any `(cell, year)` pair is deterministic from the cell's position and the year's position.
- The neighbor index list can be built **once at the cell level** (344K entries, not 6.46M), then broadcast to all years via arithmetic.

### `compute_neighbor_stats` Is Also Suboptimal

It uses `lapply` over 6.46M elements, building small vectors and computing `max/min/mean` in R. This should be vectorized or pushed to `data.table` grouped operations.

---

## Optimization Strategy

1. **Sort data by `(id, year)`** so that each cell's 28 years are contiguous and year-offset arithmetic works.
2. **Build the neighbor index once at the cell level** (344K entries) using integer position, not string keys.
3. **Broadcast to cell-year level using vectorized arithmetic**: if cell `c` is at base position `b_c` (its first row), then its row in year-offset `t` is `b_c + t`. Neighbor cell `c'` in the same year is at `b_{c'} + t`.
4. **Vectorize `compute_neighbor_stats`** using `data.table` with pre-expanded neighbor-row indices, avoiding per-row `lapply`.

This reduces the lookup construction from **~6.46M string-key iterations** to **~344K integer iterations**, and the stats computation from **R-level lapply over 6.46M** to **vectorized grouped aggregation**.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 1: Convert to data.table and sort by (id, year)

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  setkey(dt, id, year)

  unique_ids   <- dt[, unique(id)]      # sorted because of setkey
  unique_years <- dt[, sort(unique(year))]
  n_ids   <- length(unique_ids)
  n_years <- length(unique_years)

  stopifnot(nrow(dt) == n_ids * n_years)  # balanced panel check

  # Map each id to its 0-based block position (each block = n_years rows)
  id_to_pos <- setNames(seq_along(unique_ids) - 1L, as.character(unique_ids))

  # ---------------------------------------------------------------
  # STEP 2: Build cell-level neighbor list (344K entries, integer)
  #
  # id_order is the vector of cell IDs in the order matching

  # rook_neighbors_unique (the nb object). id_order[k] is the cell
  # ID for the k-th element of the nb object.
  # ---------------------------------------------------------------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # For each unique cell in the sorted data, find its neighbor
  # cells' 0-based block positions.
  # This loop runs only 344K times (not 6.46M).
  cell_neighbor_pos <- vector("list", n_ids)

  for (j in seq_len(n_ids)) {
    cid     <- unique_ids[j]
    ref_idx <- id_to_ref[as.character(cid)]
    nb_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
    # Convert neighbor cell IDs to 0-based block positions
    nb_pos <- id_to_pos[as.character(nb_cell_ids)]
    nb_pos <- nb_pos[!is.na(nb_pos)]
    cell_neighbor_pos[[j]] <- as.integer(nb_pos)
  }

  # ---------------------------------------------------------------
  # STEP 3: Expand to cell-year row indices (vectorized)
  #
  # Data is sorted by (id, year). Cell j (0-based) occupies rows
  # (j * n_years + 1) through (j * n_years + n_years).
  # Year t (0-based within block) corresponds to offset t.
  # So row of cell j in year-offset t = j * n_years + t + 1 (1-based).
  #
  # For cell-year row i (1-based):
  #   cell index (0-based) = (i - 1) %/% n_years
  #   year offset (0-based) = (i - 1) %% n_years
  # ---------------------------------------------------------------

  # Pre-compute: for each cell-year row, the vector of neighbor ROW indices

  # We build this as two parallel vectors (from_row, to_row) for data.table

  # grouped aggregation — much faster than lapply over 6.46M.

  # Count total edges to pre-allocate
  n_neighbors_per_cell <- vapply(cell_neighbor_pos, length, integer(1))
  total_edges_per_year <- sum(n_neighbors_per_cell)
  total_edges <- total_edges_per_year * n_years
  cat(sprintf("Total directed neighbor-year edges: %s\n", format(total_edges, big.mark = ",")))

  # Pre-allocate edge table
  from_row <- integer(total_edges)
  to_row   <- integer(total_edges)

  idx <- 0L
  for (j in seq_len(n_ids)) {
    nb_pos <- cell_neighbor_pos[[j]]
    n_nb   <- length(nb_pos)
    if (n_nb == 0L) next

    cell_base <- (j - 1L) * n_years  # 0-based start of this cell's block

    for (t in seq_len(n_years)) {
      # 1-based row of this cell in year t
      this_row <- cell_base + t
      # 1-based rows of neighbor cells in the same year
      nb_rows  <- nb_pos * n_years + t

      rng <- idx + seq_len(n_nb)
      from_row[rng] <- this_row
      to_row[rng]   <- nb_rows
      idx <- idx + n_nb
    }
  }

  # Trim if any cells had no neighbors (over-allocated)
  if (idx < total_edges) {
    from_row <- from_row[seq_len(idx)]
    to_row   <- to_row[seq_len(idx)]
  }

  edge_dt <- data.table(from_row = from_row, to_row = to_row)

  cat("Edge table built. Computing neighbor stats...\n")

  # ---------------------------------------------------------------
  # STEP 4: Compute neighbor stats vectorized via data.table
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    # Extract the neighbor values by indexing into the data
    edge_dt[, nb_val := dt[[var_name]][to_row]]

    # Remove NAs from neighbor values, then aggregate
    stats <- edge_dt[!is.na(nb_val),
                     .(nb_max  = max(nb_val),
                       nb_min  = min(nb_val),
                       nb_mean = mean(nb_val)),
                     by = from_row]

    # Initialize columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]

    # Clean up temp column
    edge_dt[, nb_val := NULL]
  }

  cat("Done.\n")

  # ---------------------------------------------------------------
  # STEP 5: Return as data.frame in original row order if needed
  # ---------------------------------------------------------------
  # If the original cell_data had a different row order, restore it.
  # We add a restore key before sorting.
  as.data.frame(dt)
}

# ===================== USAGE =====================

# Preserve original row order by adding a sequence column before calling:
cell_data$.orig_order <- seq_len(nrow(cell_data))

cell_data_optimized <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Restore original row order (important for the trained RF model's alignment)
cell_data_optimized <- cell_data_optimized[order(cell_data_optimized$.orig_order), ]
cell_data_optimized$.orig_order <- NULL

# The trained Random Forest model is untouched — only the feature
# construction pipeline changed. The numerical values are identical
# because max/min/mean over the same neighbor sets yield the same results.
```

---

## If the 344K × 28 Edge-Table Loop Is Still Slow (Memory-Friendly Vectorized Alternative)

The inner double loop above (344K cells × 28 years) can itself be vectorized to avoid R-level looping entirely:

```r
# ---------------------------------------------------------------
# FULLY VECTORIZED edge-table construction (replaces the double loop)
# ---------------------------------------------------------------
build_edge_table_vectorized <- function(cell_neighbor_pos, n_ids, n_years) {

  n_nb_per_cell <- vapply(cell_neighbor_pos, length, integer(1))

  # Cells that have at least one neighbor
  has_nb    <- which(n_nb_per_cell > 0L)
  n_nb_sel  <- n_nb_per_cell[has_nb]

  # For each cell with neighbors, build (cell_0based, nb_0based) pairs
  cell_0    <- rep(has_nb - 1L, times = n_nb_sel)
  nb_0      <- unlist(cell_neighbor_pos[has_nb], use.names = FALSE)

  n_pairs   <- length(cell_0)

  # Now replicate across all n_years year-offsets
  # year offsets: 1, 2, ..., n_years (1-based for row indexing)
  year_offsets <- seq_len(n_years)

  # Each pair is repeated n_years times; each year offset is repeated n_pairs times
  from_row <- rep(cell_0, times = n_years) * n_years +
              rep(year_offsets, each = n_pairs)
  to_row   <- rep(nb_0, times = n_years) * n_years +
              rep(year_offsets, each = n_pairs)

  data.table(from_row = from_row, to_row = to_row)
}

# Use in place of the loop:
# edge_dt <- build_edge_table_vectorized(cell_neighbor_pos, n_ids, n_years)
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M iterations with string paste + hash lookup into 6.46M-entry named vector | 344K iterations with integer indexing, then vectorized arithmetic |
| **Edge table** | Implicit (rebuilt per-row) | Explicit, built once (~38.4M edges for 5 vars) |
| **Stats computation** | `lapply` over 6.46M rows, R-level `max/min/mean` | `data.table` grouped aggregation (C-level) |
| **String allocations** | ~25.8M+ per call | **Zero** |
| **Estimated time** | 86+ hours | **~2–10 minutes** (depending on RAM pressure) |
| **Peak RAM** | Low but slow | ~2–4 GB for edge table (fits in 16 GB) |
| **Numerical output** | Baseline | **Identical** (same neighbor sets, same aggregation functions) |
| **Trained RF model** | — | **Untouched** |

The fundamental insight: **time-invariant spatial topology + balanced panel = pure integer arithmetic**, eliminating all string operations and reducing the problem dimensionality from 6.46M to 344K for the structural computation.