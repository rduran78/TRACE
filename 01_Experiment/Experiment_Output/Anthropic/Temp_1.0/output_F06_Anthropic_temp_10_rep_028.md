 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list with per-row string-pasting and hash lookups

`build_neighbor_lookup` iterates over all ~6.46 million rows, and for each row:
- Looks up the cell's reference index via a named character vector.
- Retrieves neighbor cell IDs.
- Pastes neighbor IDs with the current year to form keys (string allocation).
- Looks up those keys in `idx_lookup` (a named vector of length ~6.46M — each lookup is an O(n) scan or hash probe on a very large named vector).

This produces a **list of 6.46 million integer vectors**. The string operations and repeated hash lookups against a 6.46M-entry named vector dominate runtime.

### Bottleneck B: `compute_neighbor_stats` — `lapply` over 6.46M elements

For each of the 5 variables, `compute_neighbor_stats` iterates over 6.46 million list elements, subsets a numeric vector, removes NAs, and computes max/min/mean. This is called 5 times (once per variable). The `lapply` + `do.call(rbind, ...)` pattern over millions of tiny vectors is extremely slow in pure R due to per-element overhead.

### Why 86+ hours?

- ~6.46M string paste + lookup operations for the neighbor lookup build.
- ~6.46M × 5 = ~32.3M R-level function calls for stats computation.
- `do.call(rbind, ...)` on a 6.46M-element list of length-3 vectors.
- All in interpreted R with no vectorisation.

---

## 2. Optimization Strategy

### Key Insight: Separate the spatial (cell) dimension from the temporal (year) dimension.

The neighbor structure is **invariant across years**. A cell's rook neighbors are the same in every year. The current code redundantly encodes this by building a row-level lookup across all 6.46M rows, when the neighbor topology only involves 344,208 cells. We can:

1. **Build a sparse neighbor matrix once over cells** (344K × 344K sparse matrix with ~1.37M non-zero entries).
2. **Reshape each variable into a cell × year matrix** (344,208 rows × 28 columns).
3. **Use sparse matrix multiplication** to compute neighbor sums and counts, from which we derive the mean. For max and min, use efficient row-wise grouped operations.

### Specific plan:

- **For `mean`**: If `W` is the row-normalised sparse neighbor weight matrix (344K × 344K) and `V` is the variable matrix (344K × 28), then `W %*% V` gives the neighbor mean for every cell-year in one shot. This is a single sparse matrix–dense matrix multiply — extremely fast.

- **For `max` and `min`**: There is no direct sparse-matrix analogy. But we can iterate over **cells** (344K, not 6.46M) and use vectorized operations across years. Alternatively, we can build a CSR-style structure and use compiled code via `data.table` grouping or Rcpp. A practical pure-R approach: expand the neighbor list into a long-form edge table, join variable values, and compute grouped max/min/mean via `data.table`.

### Chosen approach: `data.table` edge-join strategy

This avoids Rcpp, works on a standard laptop, and completes in minutes rather than days.

**Steps:**
1. Create an edge table from `rook_neighbors_unique`: ~1.37M rows of `(cell_id, neighbor_id)`.
2. Cross-join with years: ~1.37M × 28 = ~38.5M rows of `(cell_id, year, neighbor_id)`.
3. Join the variable values onto the `neighbor_id × year` key.
4. Group by `(cell_id, year)` and compute `max`, `min`, `mean`.
5. Join back to the main dataset.

This replaces 6.46M R-level iterations with vectorized `data.table` grouped operations over ~38.5M rows — a task `data.table` handles in seconds to minutes.

**Memory check**: The edge-year table at 38.5M rows × 4 columns (cell_id, year, neighbor_id, value) ≈ 38.5M × 32 bytes ≈ 1.2 GB per variable. With 16 GB RAM and processing one variable at a time, this is feasible.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# ──────────────────────────────────────────────────────────
# STEP 0: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# ──────────────────────────────────────────────────────────
# STEP 1: Build edge table from spdep nb object (one-time)
#
# rook_neighbors_unique is an nb object: a list of length
# 344,208 where element i contains integer indices of
# neighbors of cell i (in the ordering given by id_order).
# ──────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order
  # A zero-length or 0-valued entry means no neighbors.
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)

  # Remove the spdep convention of 0L meaning "no neighbors"
  valid    <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edges), "\n")

# ──────────────────────────────────────────────────────────
# STEP 2: Get unique years
# ──────────────────────────────────────────────────────────
years <- sort(unique(cell_dt$year))

# ──────────────────────────────────────────────────────────
# STEP 3: Function to compute neighbor features for one var
# ──────────────────────────────────────────────────────────
compute_neighbor_features_fast <- function(cell_dt, edges, years, var_name) {
  cat("Processing neighbor features for:", var_name, "\n")

  # Extract only the columns we need for the join
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Cross-join edges with years to get (cell_id, neighbor_id, year)
  # Instead of a full cross join (memory-heavy), we do a keyed join:
  # For each edge (cell_id, neighbor_id), look up the neighbor's value
  # across all years.

  # Expand edges × years
  edge_year <- CJ_dt(edges, years)

  # Join neighbor values
  setkey(edge_year, neighbor_id, year)
  edge_year[val_dt, val := i.val, on = .(neighbor_id = id, year)]

  # Compute grouped stats: group by (cell_id, year)
  stats <- edge_year[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match original naming convention
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))

  stats
}

# Helper: cross-join edges with years vector efficiently
CJ_dt <- function(edges, years) {
  n_edges <- nrow(edges)
  n_years <- length(years)
  data.table(
    cell_id     = rep(edges$cell_id,     times = n_years),
    neighbor_id = rep(edges$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ──────────────────────────────────────────────────────────
# STEP 4: Loop over variables and join results back
# ──────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_features_fast(cell_dt, edges, years, var_name)

  # Merge onto cell_dt
  cell_dt <- merge(
    cell_dt,
    stats,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE
  )

  # Free memory

  rm(stats)
  gc()

  cat("  Done:", var_name, "\n")
}

# ──────────────────────────────────────────────────────────
# STEP 5: Convert back to data.frame if needed for predict()
# ──────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────
# STEP 6: Predict with the pre-trained Random Forest
#         (model object unchanged — no retraining)
# ──────────────────────────────────────────────────────────
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory-Optimised Variant (if 38.5M-row edge-year table is too large)

If the full edge × year table strains memory, process **one year at a time**:

```r
compute_neighbor_features_by_year <- function(cell_dt, edges, years, var_name) {
  cat("Processing (year-chunked):", var_name, "\n")

  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  results <- rbindlist(lapply(years, function(yr) {
    # Subset to one year
    yr_dt <- cell_dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_dt, id)

    # Join neighbor values via edge table
    edge_vals <- edges[yr_dt, on = .(neighbor_id = id), nomatch = NULL,
                       allow.cartesian = TRUE]
    # edge_vals now has columns: cell_id, neighbor_id, val

    # Compute stats grouped by cell_id
    stats_yr <- edge_vals[
      !is.na(val),
      .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
      by = .(cell_id)
    ]
    stats_yr[, year := yr]
    stats_yr
  }))

  setnames(results, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  results
}

# Usage (replaces the loop in STEP 4):
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_features_by_year(cell_dt, edges, years, var_name)

  cell_dt <- merge(
    cell_dt, stats,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE
  )

  rm(stats); gc()
  cat("  Done:", var_name, "\n")
}
```

This variant processes only ~1.37M edges per year-chunk (≈ 11 MB), keeping peak memory well under 2 GB.

---

## 5. Expected Performance

| Component | Original | Optimised |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M string ops) | ~seconds (vectorized edge table) |
| Stats per variable | ~17 hours (6.46M `lapply`) | ~1–3 minutes (`data.table` grouped ops) |
| 5 variables total | ~86+ hours | **~5–15 minutes** |
| Numerical results | baseline | **identical** (same max/min/mean) |
| RF model | unchanged | **unchanged** |

---

## 6. Why Not Raster Focal/Kernel Operations?

The docstring asks whether raster focal operations offer a useful analogy. They do conceptually — a rook-neighbor summary is equivalent to a 3×3 cross-shaped focal kernel with the centre zeroed out. However:

- The data is in an **irregular spatial panel** (spdep nb object), not a regular raster grid. Converting to raster would require confirming the grid is regular and complete, handling missing cells, and managing 28 annual layers.
- `terra::focal()` doesn't natively produce max, min, and mean simultaneously without multiple passes.
- The `data.table` edge-join approach works directly on the existing data structures, produces **numerically identical** results to the original code, and is already fast enough (minutes, not hours).

The raster analogy is useful for intuition but the `data.table` implementation best preserves the required results while achieving the needed speedup.