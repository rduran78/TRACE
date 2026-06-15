 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Retrieves neighbor cell IDs from the `nb` object,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks up those keys in a named character vector (`idx_lookup`).

String allocation, `paste()`, and named-vector lookups are extremely slow in R when repeated millions of times. The `compute_neighbor_stats` function is comparatively lighter but still uses an R-level `lapply` over 6.46M elements, each calling `max`, `min`, `mean` on small vectors.

**Root causes (ranked by impact):**

1. **`build_neighbor_lookup`**: ~6.46M iterations × multiple `paste()` and named-vector lookups per iteration. This is O(N × avg_neighbors) string operations — roughly 50+ billion character operations.
2. **`compute_neighbor_stats`**: R-level loop over 6.46M elements, called 5 times (once per variable). Slow but secondary.
3. **No vectorization or use of `data.table`** — everything is scalar R.

## Optimization Strategy

1. **Replace the character-key lookup with integer-arithmetic indexing.** Since years are contiguous (1992–2019, 28 years) and cell IDs can be mapped to integers 1–344,208, every (cell, year) pair maps to a unique row via `(cell_index - 1) * 28 + (year - 1992 + 1)`. This eliminates all `paste()` and named-vector lookups.

2. **Pre-expand the neighbor list from cell-level to row-level using vectorized operations** with `data.table` and `rep()`/arithmetic, avoiding any per-row `lapply`.

3. **Compute neighbor stats via vectorized `data.table` grouped aggregation** instead of R-level `lapply`.

This reduces estimated runtime from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Ensure cell_data is a data.table sorted by (id, year)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Create integer cell index (1-based) and ensure year ordering
#   id_order is the vector of cell IDs matching the nb object
id_to_int <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_idx := id_to_int[as.character(id)]]

# Sort so that row number = (cell_idx - 1) * n_years + (year - min_year + 1)
min_year <- min(cell_dt$year)
max_year <- max(cell_dt$year)
n_years  <- max_year - min_year + 1L  # 28

setorder(cell_dt, cell_idx, year)

# Verify contiguous panel (every cell has every year)
stopifnot(nrow(cell_dt) == length(id_order) * n_years)

# Add a row_id that matches the sort order
cell_dt[, row_id := .I]

# ---------------------------------------------------------------
# 1.  Build edge list (cell-level) from the nb object
#     rook_neighbors_unique is a list of length n_cells;
#     element i contains integer indices of neighbors of cell i.
# ---------------------------------------------------------------
n_cells <- length(id_order)

# Expand nb object to an edge-list data.table: (from_cell, to_cell)
from_cell <- rep(seq_len(n_cells),
                 times = lengths(rook_neighbors_unique))
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove 0-neighbor entries (spdep uses 0L for no-neighbor)
valid <- to_cell != 0L
from_cell <- from_cell[valid]
to_cell   <- to_cell[valid]

edges <- data.table(from_cell = from_cell, to_cell = to_cell)

# ---------------------------------------------------------------
# 2.  Expand to row-level edges by crossing with years
#     row_id of cell c in year y = (c - 1) * n_years + (y - min_year + 1)
# ---------------------------------------------------------------
years_vec <- seq.int(min_year, max_year)

# Cross join edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
edge_rows <- edges[, .(year = years_vec), by = .(from_cell, to_cell)]

edge_rows[, from_row := (from_cell - 1L) * n_years + (year - min_year + 1L)]
edge_rows[, to_row   := (to_cell   - 1L) * n_years + (year - min_year + 1L)]

# ---------------------------------------------------------------
# 3.  Compute neighbor stats for each source variable (vectorized)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value to each edge
  edge_rows[, nbr_val := cell_dt[[var_name]][to_row]]

  # Aggregate: for each from_row, compute max/min/mean of non-NA neighbor values
  agg <- edge_rows[!is.na(nbr_val),
                    .(nbr_max  = max(nbr_val),
                      nbr_min  = min(nbr_val),
                      nbr_mean = mean(nbr_val)),
                    by = from_row]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign aggregated values back by row index
  cell_dt[agg$from_row, (max_col)  := agg$nbr_max]
  cell_dt[agg$from_row, (min_col)  := agg$nbr_min]
  cell_dt[agg$from_row, (mean_col) := agg$nbr_mean]
}

# Clean up helper column
edge_rows[, nbr_val := NULL]

# ---------------------------------------------------------------
# 4.  Convert back to data.frame if needed for predict()
# ---------------------------------------------------------------
cell_dt[, cell_idx := NULL]
cell_dt[, row_id   := NULL]
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# 5.  Predict with the already-trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` on the same non-NA neighbor values — the estimand is unchanged. |
| **Trained RF model** | Untouched; only the feature-construction step is replaced. Column names are identical. |
| **Memory** | The largest object is `edge_rows` (~38.5M rows × 5 integer/numeric cols ≈ 1.5 GB). Fits in 16 GB RAM. |
| **Speed** | All hot loops replaced by `data.table` vectorized joins and grouped aggregation. Expected runtime: **2–10 minutes** on a standard laptop (vs. 86+ hours). |

**Key insight:** The original code performed ~6.46 million R-level iterations each involving string construction and named-vector lookup. The optimized version eliminates all per-row R interpretation by (a) replacing string keys with integer arithmetic, (b) pre-expanding the cell-level neighbor graph to a row-level edge list via vectorized cross-join, and (c) computing grouped aggregations entirely within `data.table`'s C backend.