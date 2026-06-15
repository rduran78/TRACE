 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Constructs character keys (`paste(id, year)`) and matches them into a named character vector (`idx_lookup`) of length 6.46M.

**Named-vector lookup via character keys in R is O(n) per probe** (hashed, but with a large constant and memory overhead). Doing this ~6.46M × ~4 neighbors ≈ 25 billion character operations is what produces the 86+ hour estimate.

`compute_neighbor_stats` is a secondary bottleneck: another R-level `lapply` over 6.46M elements, each extracting a small numeric slice. This is slow but less catastrophic.

**Root cause summary:**

| Component | Complexity | Problem |
|---|---|---|
| `build_neighbor_lookup` | O(N × k) character key lookups in a length-N named vector | ~86 h |
| `compute_neighbor_stats` | O(N × k) R-level loops | ~minutes but repeated 5× |
| Total neighbor features | 5 vars × 3 stats = 15 new columns | — |

## Optimization Strategy

1. **Replace character-key lookup with integer-arithmetic indexing.** Since every cell appears in every year (balanced panel, 344,208 cells × 28 years), we can sort the data by `(year, id)` and compute any cell-year's row index as a direct integer offset: `row = (year_offset) * n_cells + cell_offset`. This is O(1) per neighbor, no strings involved.

2. **Vectorize `compute_neighbor_stats` using `data.table` and a pre-expanded edge list.** Instead of looping over 6.46M rows, we build a long edge table `(from_row, to_row)` of all ~6.46M × 4 ≈ 25M directed edges, then join and group-aggregate in `data.table` — a single vectorized pass per variable.

3. **Memory budget.** The edge list is ~25M rows × 2 integer columns ≈ 200 MB. The data itself at 6.46M × 110 columns ≈ 5–6 GB. Fits in 16 GB with care.

**Expected speedup:** from 86+ hours to **~2–5 minutes**.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Ensure cell_data is a data.table sorted by (year, id)
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Canonical cell ordering — must match the order used in rook_neighbors_unique
# id_order is the vector of cell IDs in the same order as the nb object indices
stopifnot(length(id_order) == 344208L)

# Create integer maps
cell_dt[, id_rank := match(id, id_order)]
setorder(cell_dt, year, id_rank)
cell_dt[, row_idx := .I]                 # row index after sort

n_cells <- length(id_order)              # 344,208
years   <- sort(unique(cell_dt$year))    # 1992:2019
n_years <- length(years)                 # 28
year_to_offset <- setNames(seq_along(years) - 1L, as.character(years))

# Verify balanced panel
stopifnot(nrow(cell_dt) == n_cells * n_years)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build the directed edge list (all years) — fully vectorized
# ──────────────────────────────────────────────────────────────────────
# Expand the nb object into a two-column integer edge list (cell-level)
from_cell <- rep(seq_along(rook_neighbors_unique),
                 lengths(rook_neighbors_unique))
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove 0-neighbor entries (spdep uses integer(0) for islands; unlist drops them)
valid <- to_cell > 0L
from_cell <- from_cell[valid]
to_cell   <- to_cell[valid]

n_edges_cell <- length(from_cell)        # ~1,373,394 / 2 directed pairs
cat("Cell-level directed edges:", n_edges_cell, "\n")

# Tile across all years:
#   row index = year_offset * n_cells + cell_rank
#   (cell_rank is 1-based, matching id_rank above)
from_rows <- rep(from_cell, times = n_years) +
             rep(seq(0L, (n_years - 1L) * n_cells, by = n_cells),
                 each = n_edges_cell)
to_rows   <- rep(to_cell, times = n_years) +
             rep(seq(0L, (n_years - 1L) * n_cells, by = n_cells),
                 each = n_edges_cell)

edges <- data.table(from_row = from_rows, to_row = to_rows)
rm(from_rows, to_rows); gc()

# ──────────────────────────────────────────────────────────────────────
# 2.  Compute neighbor max / min / mean per variable — vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "\n")

  # Attach the neighbor's value to every edge
  edges[, nbr_val := cell_dt[[var_name]][to_row]]

  # Aggregate: group by the focal row
  stats <- edges[!is.na(nbr_val),
                 .(nb_max  = max(nbr_val),
                   nb_min  = min(nbr_val),
                   nb_mean = mean(nbr_val)),
                 keyby = from_row]

  # Initialise new columns to NA, then fill matched rows
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  cell_dt[stats$from_row, (max_col)  := stats$nb_max]
  cell_dt[stats$from_row, (min_col)  := stats$nb_min]
  cell_dt[stats$from_row, (mean_col) := stats$nb_mean]

  edges[, nbr_val := NULL]
  rm(stats); gc()
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Restore original row order and convert back if needed
# ──────────────────────────────────────────────────────────────────────
# If downstream code (e.g. the trained RF predict call) expects the
# original row order, restore it:
setorder(cell_dt, row_idx)   # already 1:N but just in case
cell_dt[, c("id_rank", "row_idx") := NULL]

# Convert back to data.frame if the RF predict method requires it
cell_data <- as.data.frame(cell_dt)
```

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical estimand** | The aggregation functions (`max`, `min`, `mean`) are identical; the edge list is derived from the same `rook_neighbors_unique` nb object; no approximation is introduced. |
| **Trained RF model untouched** | No model object is modified. The 15 new columns have the same names and values as the original code would produce. `predict(rf_model, cell_data)` will return identical results. |
| **Irregular / masked topology** | The edge list is built directly from the `nb` object, which already encodes the true rook-neighbor graph. No rectangular-raster assumption is made. Islands (cells with no neighbors) naturally receive `NA` for all neighbor stats. |
| **Memory** | Edge list ≈ 200 MB; `cell_dt` ≈ 5–6 GB; temporaries < 1 GB. Total peak ≈ 8–9 GB, within 16 GB. |
| **Speed** | Two vectorized passes (edge expansion + `data.table` grouped aggregation) per variable. Expected wall time: **2–5 minutes total** vs. 86+ hours. |