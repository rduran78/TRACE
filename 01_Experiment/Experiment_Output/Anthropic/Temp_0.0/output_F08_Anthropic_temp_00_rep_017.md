 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this redundantly for every year. This means:

1. **Redundant topology computation**: The neighbor graph is spatial and static. There are only 344,208 unique cells, but the lookup is built for 6,460,000+ rows. That's a ~19× blowup.
2. **String-key hashing at scale**: `paste(id, year)` key construction and named-vector lookup for ~6.46M rows × ~4 neighbors each ≈ 25.8M string operations.
3. **Per-row R-level `lapply`**: 6.46M iterations of an R closure is inherently slow.
4. **`compute_neighbor_stats` also uses per-row `lapply`**: Another 6.46M R-level iterations per variable, times 5 variables = 32.3M iterations.

The fundamental insight: **neighbor relationships are between cells, not between cell-years**. The topology needs to be computed only once over 344,208 cells. Then for each year, we simply slice the variable values by cell and apply the static topology.

## Optimization Strategy

1. **Build the neighbor lookup once over cells, not cell-years.** Convert `rook_neighbors_unique` (an `nb` object) into a simple integer-index mapping from cell position → neighbor positions. This is O(344K) and trivial.

2. **Organize data so that for each year, variable values are in a vector indexed by cell position.** Use a matrix (cells × years) or split-by-year approach.

3. **Vectorize neighbor stat computation using `data.table` and matrix operations.** For each variable, build a cell×year matrix, then compute neighbor max/min/mean using the static adjacency list — iterating over 344K cells (not 6.46M rows) and leveraging vectorized column operations.

4. **Use `data.table` for fast joins** to merge results back.

5. **Preserve the trained RF model and numerical outputs exactly** — we only change how neighbor features are computed, not what they are.

Expected speedup: from ~86 hours to **minutes**.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table if not already
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build STATIC neighbor lookup (once, over cells only)
#
# rook_neighbors_unique is an nb object of length = length(id_order).
# id_order[i] is the cell id for position i.
# rook_neighbors_unique[[i]] gives integer positions of neighbors of cell i.
# This is ALREADY the static lookup we need — no transformation required
# beyond ensuring 0-neighbor entries are handled.
# ──────────────────────────────────────────────────────────────────────

n_cells <- length(id_order)

# Precompute: for each cell position, which positions are its neighbors?
# nb objects store 0L for no-neighbor cases; normalise to integer(0).
static_neighbors <- lapply(seq_len(n_cells), function(i) {

  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 1L && nb_i[1] == 0L) integer(0) else as.integer(nb_i)
})

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Establish a consistent cell-position index in the data
#
# We need each row's cell to map to a position in 1..n_cells matching
# the order in id_order (which matches the nb object).
# ──────────────────────────────────────────────────────────────────────

id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_idx := year_to_col[as.character(year)]]

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Function to compute neighbor stats for one variable
#
# Strategy: build a matrix [n_cells x n_years] of variable values,
# then for each cell, pull neighbor rows and compute column-wise
# (i.e., year-wise) max, min, mean across neighbors.
#
# To avoid a slow R loop over 344K cells, we use a "sparse expansion"
# approach: create a long table of (cell_pos, neighbor_pos), join
# variable values, and aggregate with data.table.
# ──────────────────────────────────────────────────────────────────────

compute_neighbor_features_fast <- function(dt, var_name, static_neighbors, years) {

  cat("Computing neighbor features for:", var_name, "\n")

  # --- Build edge list (static, computed once but passed in; we build here

  #     for clarity; in practice, factor this out) ---
  # Edge list: data.table with columns (cell_pos, neighbor_pos)
  # We'll build this once outside and reuse — see below.


  # --- Build cell_pos × year_idx value table ---
  val_dt <- dt[, .(cell_pos, year_idx, val = get(var_name))]
  setkey(val_dt, cell_pos, year_idx)

  # --- Join neighbor values via edge list ---
  # For each (cell_pos, neighbor_pos) pair and each year_idx,
  # get the neighbor's value, then aggregate.
  # edge_dt is (cell_pos, neighbor_pos) — see below, we use the
  # pre-built one.

  # Join: for each edge (cell_pos, neighbor_pos), for each year,
  # get neighbor's value.
  neighbor_vals <- edge_dt[val_dt,
    on = .(neighbor_pos = cell_pos),
    .(cell_pos = x.cell_pos, year_idx = i.year_idx, val = i.val),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Aggregate by (cell_pos, year_idx)
  stats <- neighbor_vals[,
    .(
      nb_max  = max(val, na.rm = TRUE),
      nb_min  = min(val, na.rm = TRUE),
      nb_mean = mean(val, na.rm = TRUE)
    ),
    by = .(cell_pos, year_idx)
  ]

  # Fix Inf/-Inf from all-NA groups
  stats[is.infinite(nb_max), nb_max := NA_real_]
  stats[is.infinite(nb_min), nb_min := NA_real_]
  stats[is.nan(nb_mean), nb_mean := NA_real_]

  # Rename columns to match original feature names
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  return(stats)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Build the STATIC edge list ONCE (reused for all variables)
# ──────────────────────────────────────────────────────────────────────

cat("Building static edge list...\n")

edge_list <- rbindlist(lapply(seq_len(n_cells), function(i) {
  nb <- static_neighbors[[i]]
  if (length(nb) == 0L) return(NULL)
  data.table(cell_pos = i, neighbor_pos = nb)
}))

setkey(edge_list, neighbor_pos)

# Make it available to the function (or pass explicitly)
edge_dt <- edge_list

cat("Edge list built:", nrow(edge_dt), "directed edges\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Compute and attach neighbor features for all variables
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a join key in cell_data
setkey(cell_data, cell_pos, year_idx)

for (var_name in neighbor_source_vars) {

  # --- Build cell_pos × year_idx value table ---
  val_dt <- cell_data[, .(cell_pos, year_idx, val = get(var_name))]
  setkey(val_dt, cell_pos, year_idx)

  # --- For each edge, look up the NEIGHBOR's value in each year ---
  # edge_dt has (cell_pos, neighbor_pos). We want, for each year,
  # the value of the neighbor cell.
  # Join val_dt onto edge_dt by neighbor_pos == cell_pos
  neighbor_vals <- val_dt[edge_dt,
    on = .(cell_pos = neighbor_pos),
    .(cell_pos = i.cell_pos,   # the focal cell
      year_idx = x.year_idx,
      val      = x.val),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Aggregate by (focal cell, year)
  stats <- neighbor_vals[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(cell_pos, year_idx)
  ]

  # Name columns to match original pipeline
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setkey(stats, cell_pos, year_idx)

  # Merge into cell_data
  # Remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(cell_pos, year_idx)]

  cat("Done:", var_name, "\n")
}

# ──────────────────────────────────────────────────────────────────────
# STEP 6: Clean up helper columns
# ──────────────────────────────────────────────────────────────────────
cell_data[, c("cell_pos", "year_idx") := NULL]

# ──────────────────────────────────────────────────────────────────────
# STEP 7: Predict with the pre-trained Random Forest (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────
# predictions <- predict(rf_model, newdata = cell_data)
# (This step remains exactly as in the original pipeline.)

cat("Neighbor feature computation complete.\n")
```

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup granularity** | Per cell-year (~6.46M entries) | Per cell (~344K entries, static) |
| **String key construction** | ~25.8M `paste()` + hash lookups | Zero — integer positions only |
| **Stat computation loop** | `lapply` over 6.46M rows × 5 vars | `data.table` grouped aggregation |
| **Edge list** | Rebuilt implicitly per row | Built once (1.37M edges), reused |
| **Numerical result** | max/min/mean of neighbor values | Identical — same neighbors, same values, same aggregation functions |
| **RF model** | Not touched | Not touched — only `newdata` is prepared |
| **Expected runtime** | ~86+ hours | ~5–15 minutes |

**Key invariant preserved**: For every (cell, year) row, the neighbor max, min, and mean of each variable are computed over exactly the same set of rook-contiguous neighbors with exactly the same values as the original code. The static topology is separated from the year-varying values, eliminating all redundant computation.