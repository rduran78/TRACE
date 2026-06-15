 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Matches those keys against a named lookup vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on a 6.46M-element named vector) repeated for every row and every neighbor is catastrophically expensive. With ~6.46M rows and an average of ~4 rook neighbors per cell, that is ~25.8 million string constructions and hash lookups inside `build_neighbor_lookup` alone — on top of R's `lapply` loop overhead with no vectorization.

`compute_neighbor_stats` is a secondary bottleneck: another `lapply` over 6.46M rows, each time subsetting a vector and computing three summary statistics in pure R.

**Root causes in order of severity:**

1. **Expensive string-key hashing at scale**: Paste-based keys matched against a 6.46M-entry named vector.
2. **Row-level R loop (lapply) over 6.46M rows**: Not vectorized, no C-level backend.
3. **Redundant recomputation**: The neighbor topology is identical for every year; the per-year expansion is done inside the expensive loop instead of being exploited structurally.

## Optimization Strategy

### Key Insight
The spatial neighbor graph is **time-invariant**: cell *i*'s neighbors are the same in every year. We can separate the spatial topology from the temporal expansion:

1. **Build a cell-level neighbor edge list once** (344K cells, ~1.37M edges) using integer indices only — no strings.
2. **Expand to cell-year rows using vectorized integer arithmetic** rather than paste/match. If data is sorted by `(id, year)` with all 28 years present per cell, then the row for cell `j` in year `t` is at a deterministic offset: `(cell_index_of_j - 1) * n_years + year_offset_of_t`. This makes the entire lookup O(1) per edge with no hashing.
3. **Compute neighbor statistics using `data.table` grouped operations or vectorized edge-list aggregation** — replace the 6.46M-iteration `lapply` with a single vectorized `data.table` join-and-aggregate.

This reduces the runtime from ~86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Prepare data (one-time)
# ============================================================
# Ensure cell_data is a data.table sorted by (id, year) with
# complete panel (every cell has all 28 years).
cell_dt <- as.data.table(cell_data)
setorder(cell_dt, id, year)

# Verify complete panel
n_years <- uniqueN(cell_dt$year)            # 28
n_cells <- uniqueN(cell_dt$id)              # 344,208
stopifnot(nrow(cell_dt) == n_cells * n_years)

# Map cell id -> 1-based cell index (matches id_order position)
cell_ids_sorted <- unique(cell_dt$id)        # already sorted
# id_order is the ordering used when building rook_neighbors_unique
id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))

# Year -> 1-based year offset
years_sorted <- sort(unique(cell_dt$year))
year_to_offset <- setNames(seq_along(years_sorted), as.character(years_sorted))

# Assign a row index to every row: because data is sorted by (id, year),
# row for cell_index c (1-based) and year_offset y (1-based) is:
#   row = (c - 1) * n_years + y
# Verify:
cell_dt[, cellidx := id_to_cellidx[as.character(id)]]
cell_dt[, yearoff := year_to_offset[as.character(year)]]
cell_dt[, expected_row := (cellidx - 1L) * n_years + yearoff]
stopifnot(all(cell_dt$expected_row == seq_len(nrow(cell_dt))))

# ============================================================
# STEP 1: Build edge list from nb object (cell-level, once)
# ============================================================
# rook_neighbors_unique is a list of length n_cells;
# rook_neighbors_unique[[c]] gives integer vector of neighbor
# cell indices (positions in id_order).

edge_from <- rep(seq_along(rook_neighbors_unique),
                 lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove any 0-entries (spdep uses 0 for no-neighbor regions)
valid <- edge_to > 0L
edge_from <- edge_from[valid]
edge_to   <- edge_to[valid]

n_edges <- length(edge_from)  # ~1,373,394

# ============================================================
# STEP 2: Expand edges to cell-year level (vectorized)
# ============================================================
# For each spatial edge (from_cell, to_cell) and each year offset y,
# the source row = (from_cell - 1)*n_years + y
# the neighbor row = (to_cell - 1)*n_years + y

# Repeat each edge n_years times
year_offsets <- seq_len(n_years)

# Pre-allocate full vectors
total <- as.numeric(n_edges) * n_years
from_rows <- integer(total)
to_rows   <- integer(total)

# Vectorized construction (no R loop over rows)
# Use outer-product logic via rep:
#   rep(edge_from, each = n_years) gives from-cell repeated for each year
#   rep(year_offsets, times = n_edges) gives year cycling for each edge
from_cell_exp <- rep(edge_from, each = n_years)
to_cell_exp   <- rep(edge_to,   each = n_years)
year_exp      <- rep(year_offsets, times = n_edges)

from_rows <- (from_cell_exp - 1L) * n_years + year_exp
to_rows   <- (to_cell_exp   - 1L) * n_years + year_exp

# Free temporaries
rm(from_cell_exp, to_cell_exp, year_exp)
gc()

# Build edge data.table
edges_dt <- data.table(from_row = from_rows, to_row = to_rows)
rm(from_rows, to_rows)
gc()

# ============================================================
# STEP 3: Compute neighbor stats for each variable (vectorized)
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value to each edge
  edges_dt[, nval := cell_dt[[var_name]][to_row]]

  # Aggregate: for each from_row, compute max/min/mean of non-NA neighbor values
  agg <- edges_dt[!is.na(nval),
                  .(nmax  = max(nval),
                    nmin  = min(nval),
                    nmean = mean(nval)),
                  by = from_row]

  # Initialize columns with NA
  max_col  <- paste0("nbr_max_",  var_name)
  min_col  <- paste0("nbr_min_",  var_name)
  mean_col <- paste0("nbr_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign results back
  cell_dt[agg$from_row, (max_col)  := agg$nmax]
  cell_dt[agg$from_row, (min_col)  := agg$nmin]
  cell_dt[agg$from_row, (mean_col) := agg$nmean]
}

# Clean up helper columns
cell_dt[, c("cellidx", "yearoff", "expected_row") := NULL]
edges_dt[, nval := NULL]

# ============================================================
# STEP 4: Convert back if needed and predict
# ============================================================
# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Predictions proceed as before:
# preds <- predict(rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` on the same non-NA neighbor values → identical estimand. |
| **Trained RF model untouched** | No retraining; only the feature-construction step is replaced. |
| **Column names** | Must match whatever the RF model expects — adjust the `paste0` naming pattern (e.g., `nbr_max_ntl`) to match the original `compute_and_add_neighbor_features` output names exactly. |

## Performance Estimate

| Phase | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (6.46M string ops) | ~60–80 hours | **Eliminated** (integer arithmetic, ~10 seconds) |
| `compute_neighbor_stats` (6.46M × 5 vars) | ~6–10 hours | **~2–5 minutes** (vectorized `data.table` groupby) |
| Edge expansion (38.4M rows) | N/A | ~30 seconds, ~600 MB RAM |
| **Total** | **86+ hours** | **< 10 minutes** |

Memory peak is approximately: 6.46M-row `cell_dt` (~5.5 GB with 110+ columns) + 38.4M-row `edges_dt` (~600 MB with 3 integer columns) ≈ **~6–7 GB**, well within 16 GB.