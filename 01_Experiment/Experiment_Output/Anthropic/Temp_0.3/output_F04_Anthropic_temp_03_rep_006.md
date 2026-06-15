 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks up those keys in `idx_lookup` (a named character vector).

String construction (`paste`) and named-vector lookups over millions of iterations in R are extremely slow. `compute_neighbor_stats` then iterates over 6.46M entries again per variable (×5 variables), doing subsetting and summary stats in pure R. Combined, this yields the estimated 86+ hour runtime.

**Root causes:**
- **Row-level `lapply` with string operations** over 6.46M rows is the primary bottleneck.
- **Redundant per-variable looping** in `compute_neighbor_stats` with R-level list operations is the secondary bottleneck.
- The neighbor topology is **time-invariant** (same grid, same rook neighbors every year), but the code rebuilds string keys for every cell-year pair as if topology changes per year.

## Optimization Strategy

1. **Exploit time-invariance of topology.** The neighbor graph is defined over 344,208 cells, not 6.46M cell-years. Build the lookup at the cell level (344K entries), then broadcast across years using vectorized integer indexing.

2. **Replace string key lookups with integer arithmetic.** If data is sorted by `(id, year)`, each cell's rows occupy a contiguous block of 28 rows. The row index for cell `c` in year `y` is `(c-1)*28 + (y - 1991)`. No `paste`, no named vector lookup needed.

3. **Vectorize `compute_neighbor_stats` using `data.table` grouping** or a single pre-allocated matrix operation instead of `lapply` over 6.46M elements.

4. **Pre-allocate output columns** and fill via vectorized assignment rather than `do.call(rbind, ...)` on a 6.46M-element list.

These changes reduce the problem from ~6.46M string-manipulation iterations to ~344K integer-index iterations (for the lookup) and fully vectorized column computation (for the stats), cutting runtime from 86+ hours to minutes.

## Optimized Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0 — Convert to data.table and sort deterministically
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Create integer cell index (1-based) aligned with id_order
cell_dt[, cell_idx := match(id, id_order)]

# Ensure year is integer
cell_dt[, year := as.integer(year)]

# Sort by (cell_idx, year) — this is critical for the arithmetic trick
setkey(cell_dt, cell_idx, year)

# Confirm contiguous year panel
years      <- sort(unique(cell_dt$year))
n_years    <- length(years)           # 28
n_cells    <- length(id_order)        # 344,208
year_min   <- min(years)              # 1992
stopifnot(nrow(cell_dt) == n_cells * n_years)

# Year offset: 1-based position of each year in the panel
cell_dt[, year_off := year - year_min + 1L]

# ---------------------------------------------------------------
# STEP 1 — Build neighbor lookup at the CELL level (344K, not 6.46M)
#           rook_neighbors_unique[[c]] gives neighbor cell indices
#           into id_order (already 1-based integer vectors).
# ---------------------------------------------------------------
# Nothing to change: rook_neighbors_unique is already a list of
# integer vectors indexed by cell position in id_order.

# ---------------------------------------------------------------
# STEP 2 — Expand cell-level neighbors to row-level indices
#           using integer arithmetic (fully vectorized).
#
#   Row index of cell c (1-based) in year_off t:
#       row = (c - 1) * n_years + t
# ---------------------------------------------------------------

# For every cell c, its neighbors are rook_neighbors_unique[[c]].
# For a given year_off t, the row indices of those neighbors are:
#   (neighbor_cell_idx - 1) * n_years + t
#
# We build three long vectors: (source_row, neighbor_row) pairs.

message("Building vectorized neighbor edge list...")

# Number of neighbors per cell
n_nbrs <- lengths(rook_neighbors_unique)  # length = n_cells

# Cell indices repeated by their neighbor count
source_cells <- rep(seq_len(n_cells), times = n_nbrs)
# Corresponding neighbor cell indices (unlisted)
target_cells <- unlist(rook_neighbors_unique, use.names = FALSE)

# Now expand across all years: each (source_cell, target_cell) pair
# appears once per year.
n_edges_per_year <- length(source_cells)  # ~1.37M

# Replicate for each year offset (1..28)
year_offsets <- rep(seq_len(n_years), each = n_edges_per_year)
source_rows  <- rep((source_cells - 1L) * n_years, times = n_years) + year_offsets
target_rows  <- rep((target_cells - 1L) * n_years, times = n_years) + year_offsets

# edge_dt: every row is one directed (source_row -> neighbor_row) edge
edge_dt <- data.table(src = source_rows, tgt = target_rows)
rm(source_rows, target_rows, year_offsets); gc()

message(sprintf("Edge list: %s edges", format(nrow(edge_dt), big.mark = ",")))

# ---------------------------------------------------------------
# STEP 3 — Compute neighbor stats for each variable (vectorized)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))

  # Attach the neighbor's value to each edge
  edge_dt[, val := cell_dt[[var_name]][tgt]]

  # Compute grouped stats: max, min, mean per source row
  stats <- edge_dt[!is.na(val),
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   keyby = src]

  # Initialize columns with NA
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Fill in computed values by integer index
  cell_dt[stats$src, (max_col)  := stats$nb_max]
  cell_dt[stats$src, (min_col)  := stats$nb_min]
  cell_dt[stats$src, (mean_col) := stats$nb_mean]
}

# ---------------------------------------------------------------
# STEP 4 — Restore original row order and convert back
# ---------------------------------------------------------------
# If the original cell_data had a specific row order, restore it.
# We kept cell_idx and year, so we can match back.
setkey(cell_dt, cell_idx, year)

# Drop helper columns
cell_dt[, c("cell_idx", "year_off") := NULL]

# Convert back to data.frame for downstream Random Forest predict()
cell_data <- as.data.frame(cell_dt)
rm(cell_dt, edge_dt, stats); gc()

message("Neighbor feature construction complete.")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values as the original code. No approximation. |
| **Trained RF model** | No model retraining. The code only constructs predictor columns; `predict()` is called downstream unchanged. |
| **Column names** | Output columns follow the same `nb_max_<var>`, `nb_min_<var>`, `nb_mean_<var>` naming convention (adjust prefixes to match your `compute_and_add_neighbor_features` wrapper if they differ). |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M `paste` + named-vector lookups | Integer arithmetic, no strings |
| Stats computation | 5 × `lapply` over 6.46M R lists | 5 × `data.table` grouped aggregation on ~38.4M-row edge table |
| Memory peak | Millions of small R list elements (high overhead) | ~38.4M-row `data.table` with 3 columns (~1.2 GB, fits in 16 GB) |
| **Estimated runtime** | **86+ hours** | **~5–15 minutes** |

The key insight is that the rook-neighbor topology is **time-invariant over the 28-year panel**, so the 6.46M-row problem decomposes into a 344K-cell spatial problem broadcast across 28 years via pure integer arithmetic—eliminating all string operations entirely.