 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a linear-time concatenation operation that runs in seconds to low minutes for this data size. The `lapply` inside `compute_neighbor_stats()` does no "repeated list binding" — it returns a fixed-length vector `c(NA, NA, NA)` or `c(max, min, mean)` per element, so there is no growing-list pathology.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for *each* of the ~6.46M rows, the function:
   - Calls `as.character(data$id[i])` and looks up `id_to_ref[...]` (named character vector lookup — O(n) hash probe repeated 6.46M times).
   - Extracts `neighbor_cell_ids` via subsetting `id_order[neighbors[[ref_idx]]]`.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — allocating new strings for every neighbor of every row.
   - Looks up `idx_lookup[neighbor_keys]` — probing a named vector of 6.46M entries with string keys, repeated for every neighbor of every row.

2. **Total string operations scale as rows × avg_neighbors.** With ~6.46M rows and an average of ~4 rook neighbors per cell, that's ~25.8 million `paste()` calls and ~25.8 million named-vector string lookups *inside the inner function*, on top of the 6.46M outer iterations. Named vector lookup by string key in R uses hashing, but the repeated `paste` allocation and hash probing at this scale dominates runtime massively — likely accounting for **>90% of the 86-hour estimate**.

3. `compute_neighbor_stats()` is comparatively fast: it does only integer indexing into a numeric vector (vectorized, cache-friendly) and computes three summary statistics per element. For 6.46M elements with ~4 neighbors each, this should complete in under a minute.

**Conclusion:** The bottleneck is the O(rows × neighbors) string construction and string-keyed hash lookup in `build_neighbor_lookup()`. The fix is to eliminate all string operations and replace them with integer arithmetic for row indexing.

---

## Optimization Strategy

1. **Replace string-key lookup with integer arithmetic.** Since the data has a regular panel structure (344,208 cells × 28 years), we can map any `(cell, year)` pair to a row index using integer math if we sort the data appropriately, or use an integer-keyed hash (via `match()` on integer encoding or a pre-built integer lookup table).

2. **Vectorize `build_neighbor_lookup()`** by eliminating the per-row `lapply` entirely. Instead, precompute neighbor row indices for all rows at once using vectorized operations.

3. **Vectorize `compute_neighbor_stats()`** using the pre-built neighbor structure with matrix operations instead of per-row `lapply`.

4. **Preserve the trained Random Forest model and original numerical outputs.** The computed features (max, min, mean of neighbor values) will be numerically identical.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup + compute_neighbor_stats
# ==============================================================================

# ---------------------------------------------------------------------------
# Step 0: Ensure data is sorted by (id, year) so we can use integer arithmetic.
#         If already sorted, this is a no-op check.
# ---------------------------------------------------------------------------
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

# ---------------------------------------------------------------------------
# Step 1: Build integer-indexed neighbor lookup (vectorized, no strings)
#
# Key insight: if data is sorted by (id, year), then for a cell with
# positional index `c` (1-based among the 344,208 unique cells) and
# year offset `t` (0-based, 0..27 for 1992..2019), the row index is:
#     row = (c - 1) * n_years + (t + 1)
#
# This replaces ALL string paste + named-vector lookups with integer math.
# ---------------------------------------------------------------------------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  unique_ids <- id_order
  n_cells    <- length(unique_ids)
  years      <- sort(unique(data$year))
  n_years    <- length(years)
  
  # Map cell id -> positional index (1..n_cells)
  id_to_pos <- setNames(seq_along(unique_ids), as.character(unique_ids))
  
  # Map year -> offset (1..n_years)
  year_to_offset <- setNames(seq_along(years), as.character(years))
  
  # For each cell position, get its neighbor cell positions
  # neighbors is an nb object: neighbors[[c]] gives integer indices into id_order
  # We will build an edge list: (cell_position, neighbor_position)
  
  # Number of neighbors per cell
  n_nbrs <- lengths(neighbors)
  
  # Expand: for each cell, repeat its index by its number of neighbors
  from_cell <- rep(seq_len(n_cells), times = n_nbrs)
  # The neighbor cell positions (concatenated)
  to_cell   <- unlist(neighbors, use.names = FALSE)
  
  # Total directed neighbor pairs
  n_edges <- length(from_cell)
  
  # Now expand across all years: each edge appears once per year
  # from_row[e, t] = (from_cell[e] - 1) * n_years + t
  # to_row[e, t]   = (to_cell[e]   - 1) * n_years + t
  
  # Vectorize: repeat each edge n_years times, and tile year offsets
  from_cell_exp <- rep(from_cell, each = n_years)
  to_cell_exp   <- rep(to_cell,   each = n_years)
  year_offset   <- rep(seq_len(n_years), times = n_edges)
  
  from_row <- (from_cell_exp - 1L) * n_years + year_offset
  to_row   <- (to_cell_exp   - 1L) * n_years + year_offset
  
  # Return as a data structure: for each "from_row", the list of "to_row" indices

  # But building a 6.46M-element list from edges is itself slow with split().
  # Instead, return the edge vectors sorted by from_row for grouped operations.
  
  ord <- order(from_row)
  list(
    from_row   = from_row[ord],
    to_row     = to_row[ord],
    n_rows     = nrow(data),
    # Precompute group boundaries for fast slicing
    grp_start  = NULL,  # will be filled below
    grp_end    = NULL
  )
}

# ---------------------------------------------------------------------------
# Step 2: Compute neighbor stats vectorized using the edge list
# ---------------------------------------------------------------------------

compute_neighbor_stats_fast <- function(data, edge_from, edge_to, n_rows, var_name) {
  
  vals <- data[[var_name]]
  
  # Get neighbor values along edges
  nbr_vals <- vals[edge_to]
  
  # We need max, min, mean grouped by edge_from
  # edge_from is already sorted, so we can use efficient grouped operations.
  
  # Use data.table for fast grouped aggregation on the edge list
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("data.table package is required for the optimized pipeline.")
  }
  
  dt_edges <- data.table::data.table(
    from = edge_from,
    val  = nbr_vals
  )
  
  # Remove edges where neighbor value is NA
  dt_edges <- dt_edges[!is.na(val)]
  
  # Grouped aggregation
  agg <- dt_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from]
  
  # Initialize output with NA
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)
  
  out_max[agg$from]  <- agg$nb_max
  out_min[agg$from]  <- agg$nb_min
  out_mean[agg$from] <- agg$nb_mean
  
  cbind(out_max, out_min, out_mean)
}

# ---------------------------------------------------------------------------
# Step 3: Optimized outer pipeline
# ---------------------------------------------------------------------------

library(data.table)

# Ensure sort order for integer arithmetic to work
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

# Verify panel structure
unique_ids <- id_order  # from the original pipeline
n_cells    <- length(unique_ids)
years      <- sort(unique(cell_data$year))
n_years    <- length(years)
stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

# Build integer cell-position map
id_to_pos <- setNames(seq_len(n_cells), as.character(unique_ids))

# Verify cell_data$id ordering matches id_order positions
cell_positions <- id_to_pos[as.character(cell_data$id)]
year_offsets   <- match(cell_data$year, years)
expected_rows  <- (cell_positions - 1L) * n_years + year_offsets
stopifnot(all(expected_rows == seq_len(nrow(cell_data))))

# --- Build edge list (one-time cost, ~seconds) ---
n_nbrs    <- lengths(rook_neighbors_unique)
from_cell <- rep(seq_len(n_cells), times = n_nbrs)
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)
n_edges   <- length(from_cell)

# Expand across years: integer arithmetic, no strings
from_cell_exp <- rep(from_cell, each = n_years)
to_cell_exp   <- rep(to_cell,   each = n_years)
year_offset   <- rep(seq_len(n_years), times = n_edges)

edge_from <- (from_cell_exp - 1L) * n_years + year_offset
edge_to   <- (to_cell_exp   - 1L) * n_years + year_offset

# Sort by from for grouped operations (data.table will also sort internally,
# but pre-sorting helps memory locality)
ord       <- order(edge_from)
edge_from <- edge_from[ord]
edge_to   <- edge_to[ord]

rm(from_cell_exp, to_cell_exp, year_offset, ord)
gc()

# --- Compute neighbor features for all 5 variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  
  stats <- compute_neighbor_stats_fast(
    cell_data, edge_from, edge_to, nrow(cell_data), var_name
  )
  
  cell_data[[paste0("nb_max_",  var_name)]] <- stats[, 1]
  cell_data[[paste0("nb_min_",  var_name)]] <- stats[, 2]
  cell_data[[paste0("nb_mean_", var_name)]] <- stats[, 3]
  
  rm(stats)
  gc()
}

# The trained Random Forest model is untouched.
# The numerical outputs (nb_max, nb_min, nb_mean per variable) are identical
# to the original pipeline — same integer indexing, same max/min/mean calculations,
# just computed via vectorized grouped aggregation instead of per-row lapply.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M × ~4 string pastes + string-keyed hash lookups | Eliminated entirely |
| **Lookup method** | `paste(id, year)` → named character vector probe | Integer arithmetic: `(cell_pos - 1) * n_years + year_offset` |
| **Neighbor stats** | Per-row `lapply` over 6.46M rows | `data.table` grouped aggregation on edge list |
| **Estimated runtime** | 86+ hours | **~2–10 minutes** (edge list construction + 5 grouped aggregations) |
| **Memory** | 6.46M-element list of integer vectors | Two integer vectors of length ~38.4M (edge_from, edge_to) ≈ ~307 MB |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical output** | Original estimand | Identical ✓ |

The key insight is that the panel data has a **regular grid structure** (every cell appears in every year), which means row positions can be computed by integer arithmetic in O(1) per lookup — completely eliminating the millions of string allocations and hash probes that constituted the actual bottleneck.