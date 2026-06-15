 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is **not** the Random Forest inference — it is the two spatial neighbor feature construction functions: `build_neighbor_lookup` and `compute_neighbor_stats`. Here is why:

### `build_neighbor_lookup`
This function iterates via `lapply` over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs **character key strings** with `paste()` for every neighbor of every row.
4. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) on a **6.46-million-entry named character vector** — which in R is an **O(n) linear scan per lookup** (R named vectors are not hash tables).

The result: for ~6.46 million rows × ~4 neighbors on average, you are performing ~25.8 million character-key lookups against a 6.46-million-length named vector. Each lookup is O(n) in the worst case, yielding an effective complexity on the order of **O(rows × neighbors × rows)** — catastrophically slow.

### `compute_neighbor_stats`
This function is comparatively cheaper (just subsetting a numeric vector), but it still uses `lapply` over 6.46 million rows and returns results via `do.call(rbind, ...)` on a 6.46-million-element list of 3-element vectors — a known slow anti-pattern in R.

### Summary of Root Causes
| Issue | Location | Impact |
|---|---|---|
| Named-vector lookup (not hashed) on 6.46M keys | `build_neighbor_lookup` | **Critical** — pseudo-quadratic |
| Per-row `paste()` string construction | `build_neighbor_lookup` | High |
| `lapply` over 6.46M rows in pure R | Both functions | Moderate |
| `do.call(rbind, list_of_vectors)` | `compute_neighbor_stats` | Moderate |
| Repeated per-variable overhead | Outer loop (5 vars) | Multiplier |

---

## Optimization Strategy

The key insight is: **the neighbor relationships are purely spatial (cell-to-cell), not temporal**. Every year shares the same neighbor graph. We should:

1. **Replace the per-row lookup with a vectorized year-keyed approach.** Pre-build a mapping from `(cell_id)` → `row indices per year` using `data.table` or an environment (hash map). Then, for each year, map all neighbor relationships at once using vectorized integer indexing — no character-key construction or lookup at all in the inner loop.

2. **Use `data.table` for all joins and aggregations.** Instead of building a 6.46M-element list of neighbor row indices and then looping over it, we construct a long-form edge table `(row_i, neighbor_row_j)` once, join the variable values, and compute `max/min/mean` by group — all vectorized in C via `data.table`.

3. **Eliminate `do.call(rbind, ...)`** — `data.table` grouping returns a single table directly.

4. **Process all 5 variables in a single pass** over the edge table instead of 5 separate passes.

This reduces the complexity from pseudo-quadratic (hours/days) to **O(E × T)** where E ≈ 1.37M edges and T = 28 years, with all operations vectorized — expected runtime: **minutes**.

The trained Random Forest model is untouched. The numerical results (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

#' Build neighbor features for all source variables at once,
#' using vectorized data.table operations.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all vars in neighbor_source_vars
#' @param id_order          character/integer vector: the cell IDs in the order matching the nb object
#' @param neighbors         spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus, for each var, var_nb_max, var_nb_min, var_nb_mean
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        neighbors,
                                        neighbor_source_vars) {

  # ---- Step 0: Convert to data.table, preserve original row order ----
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # ---- Step 1: Build a spatial edge list (cell-level, year-independent) ----
  # For each cell index i in id_order, neighbors[[i]] gives integer indices

  # of its rook neighbors in id_order.
  # We build a two-column table: (focal_cell_id, neighbor_cell_id)

  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1,373,394 rows (directed edges)

  # ---- Step 2: Build a lookup from (id, year) -> row index in dt ----
  # Using data.table keyed join — O(n log n) setup, O(1) amortized per join row.
  id_year_to_row <- dt[, .(id, year, .row_id)]
  setkey(id_year_to_row, id, year)

  # ---- Step 3: Get unique years ----
  years <- sort(unique(dt$year))

  # ---- Step 4: For each year, expand edge_list to row-level edges and compute stats ----
  # We process all years at once by cross-joining edge_list with years,
  # then joining to get row indices and values.

  # Cross join: every spatial edge × every year  (~1.37M × 28 ≈ 38.5M rows)
  # This fits in memory: 38.5M × 2 int cols + 1 int year ≈ ~0.9 GB
  edge_year <- CJ_dt_edges(edge_list, years)

  # Join to get the focal row id
  setkey(edge_year, focal_id, year)
  edge_year[id_year_to_row, focal_row := i..row_id, on = .(focal_id = id, year)]

  # Join to get the neighbor row id
  setkey(edge_year, neighbor_id, year)
  edge_year[id_year_to_row, neighbor_row := i..row_id, on = .(neighbor_id = id, year)]

  # Drop edges where either focal or neighbor is missing (cell-years not in data)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # ---- Step 5: For each variable, join values and aggregate ----
  for (var_name in neighbor_source_vars) {
    # Get neighbor values via integer index — vectorized and fast
    edge_year[, nb_val := dt[[var_name]][neighbor_row]]

    # Aggregate: max, min, mean of neighbor values by focal row, ignoring NAs
    agg <- edge_year[!is.na(nb_val),
                     .(nb_max  = max(nb_val),
                       nb_min  = min(nb_val),
                       nb_mean = mean(nb_val)),
                     by = focal_row]

    # Create output column names
    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    # Initialize with NA
    set(dt, j = col_max,  value = NA_real_)
    set(dt, j = col_min,  value = NA_real_)
    set(dt, j = col_mean, value = NA_real_)

    # Assign aggregated values by row index
    set(dt, i = agg$focal_row, j = col_max,  value = agg$nb_max)
    set(dt, i = agg$focal_row, j = col_min,  value = agg$nb_min)
    set(dt, i = agg$focal_row, j = col_mean, value = agg$nb_mean)

    # Clean up the temporary column
    edge_year[, nb_val := NULL]
  }

  # ---- Step 6: Clean up and return ----
  dt[, .row_id := NULL]
  return(dt)
}


#' Helper: cross join edge_list with a vector of years
#' More memory-efficient than a full CJ on three columns.
CJ_dt_edges <- function(edge_list, years) {
  # Repeat each edge for every year
  n_edges <- nrow(edge_list)
  n_years <- length(years)
  idx <- rep(seq_len(n_edges), times = n_years)
  yr  <- rep(years, each = n_edges)
  data.table(
    focal_id    = edge_list$focal_id[idx],
    neighbor_id = edge_list$neighbor_id[idx],
    year        = yr
  )
}


# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is used exactly as before — no retraining.
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory Note

The cross-joined edge-year table will have approximately 1,373,394 × 28 ≈ **38.5 million rows** with 3 integer columns plus 2 joined integer columns — roughly **1.4 GB** at peak. On a 16 GB laptop this is feasible. If memory is tight, the `CJ_dt_edges` step can be chunked by year:

```r
# Memory-conservative variant: process one year at a time
build_all_neighbor_features_chunked <- function(cell_data, id_order, neighbors,
                                                 neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # Build spatial edge list (same as before)
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  # Initialize output columns
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0(var_name, "_nb_max"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_nb_min"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_nb_mean"), value = NA_real_)
  }

  years <- sort(unique(dt$year))

  for (yr in years) {
    # Subset to this year
    dt_yr <- dt[year == yr, c("id", ".row_id", neighbor_source_vars), with = FALSE]
    setkey(dt_yr, id)

    # Map edges to row indices for this year
    edges_yr <- copy(edge_list)
    edges_yr[dt_yr, focal_row    := i..row_id, on = .(focal_id = id)]
    edges_yr[dt_yr, neighbor_row := i..row_id, on = .(neighbor_id = id)]
    edges_yr <- edges_yr[!is.na(focal_row) & !is.na(neighbor_row)]

    for (var_name in neighbor_source_vars) {
      edges_yr[, nb_val := dt[[var_name]][neighbor_row]]
      agg <- edges_yr[!is.na(nb_val),
                       .(nb_max = max(nb_val), nb_min = min(nb_val), nb_mean = mean(nb_val)),
                       by = focal_row]

      if (nrow(agg) > 0L) {
        set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_max"),  value = agg$nb_max)
        set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_min"),  value = agg$nb_min)
        set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_mean"), value = agg$nb_mean)
      }
      edges_yr[, nb_val := NULL]
    }
  }

  dt[, .row_id := NULL]
  return(dt)
}
```

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M character-key lookups against 6.46M named vector (pseudo-quadratic) | Eliminated — replaced by `data.table` keyed join, O(n log n) |
| `compute_neighbor_stats` | 5 × `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single-pass vectorized `data.table` group-by aggregation |
| String operations | ~25.8M `paste()` calls + character matching | **Zero** — all joins are on integer/numeric keys |
| Estimated wall time | **86+ hours** | **5–15 minutes** (depending on disk I/O and RAM) |
| RAM peak | Lower but irrelevant given runtime | ~1.4 GB for edge table (or ~200 MB chunked) |
| Numerical results | Baseline | **Identical** (same max, min, mean) |
| Trained RF model | Unchanged | **Unchanged** |