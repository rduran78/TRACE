 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character matching (`id_to_ref[as.character(...)]`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character key strings by pasting cell ID and year (`paste(..., sep="_")`).
4. Looks up row indices in a named character vector (`idx_lookup[neighbor_keys]`).

This means ~6.46 million iterations, each performing **character coercion, string pasting, and named-vector lookup** (which is O(n) hash-table probing on character keys). With ~1.37M neighbor relationships spread across 28 years, that's tens of millions of string operations. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M entries five times (once per variable), each time extracting, filtering NAs, and computing summary stats in pure R.

**Root causes:**
- Row-level `lapply` in R (interpreted loop over millions of rows).
- Repeated `paste()`/character key construction and named-vector lookups (slow hashing).
- `compute_neighbor_stats` uses per-row `lapply` with R-level `max/min/mean` calls instead of vectorized operations.
- The lookup is rebuilt monolithically instead of exploiting the panel structure (same neighbor topology repeats every year).

## Optimization Strategy

**Key insight:** The spatial neighbor graph is *time-invariant*. Cell `i`'s neighbors are the same in every year. Therefore, we should:

1. **Separate spatial topology from temporal indexing.** Build a cell-to-cell neighbor edge list once (~1.37M edges), then join it to the panel by year using `data.table` equi-joins — fully vectorized, no per-row loop.
2. **Replace `build_neighbor_lookup` entirely** with a vectorized `data.table` merge approach that expands the neighbor edge list across years.
3. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation (vectorized C-level `max`, `min`, `mean`) — no `lapply`.
4. **Process all 5 variables in one pass** per aggregation to avoid redundant joins.

This reduces the runtime from ~86+ hours to minutes.

## Optimized R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a time-invariant edge list from the nb object (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ---------------------------------------------------------------
# 2. Compute all neighbor features in a vectorized fashion
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  # Convert to data.table if not already; work on a copy to be safe
  dt <- as.data.table(copy(cell_data))

  # Ensure a row-order key so we can restore original order later
  dt[, .row_order := .I]

  # Step 1: build the edge list (time-invariant, ~1.37M rows)
  edges <- build_edge_list(id_order, neighbors)

  # Step 2: expand edges across years via a merge with the panel.
  #
  # We need, for every (focal_id, year) row, the values of each
  # source variable at every (neighbor_id, year) row.
  #
  # Strategy:

  #   a) Create a slim table: id, year, + source vars.
  #   b) Join edges to that table on neighbor_id == id to get
  #      neighbor values; this is keyed by (focal_id, year).
  #   c) Aggregate (max, min, mean) grouped by (focal_id, year).
  #   d) Join aggregated stats back to dt.

  keep_cols <- c("id", "year", neighbor_source_vars)
  slim <- dt[, ..keep_cols]

  # Keyed join: for every edge, attach neighbor values per year
  # Result has one row per (focal_id, neighbor_id, year) combination
  setkey(slim, id, year)
  neighbor_vals <- edges[slim,
                         on = .(neighbor_id = id),
                         allow.cartesian = TRUE,
                         nomatch = NULL]
  # neighbor_vals now has columns: focal_id, neighbor_id, year, + source vars

  # Step 3: aggregate by (focal_id, year)
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <-
      bquote(fifelse(all(is.na(.(v_sym))), NA_real_, max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_min_", v)]]  <-
      bquote(fifelse(all(is.na(.(v_sym))), NA_real_, min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_mean_", v)]] <-
      bquote(fifelse(all(is.na(.(v_sym))), NA_real_, mean(.(v_sym), na.rm = TRUE)))
  }
  # Build the call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  agg <- neighbor_vals[, eval(agg_call), by = .(focal_id, year)]

  # Step 4: merge aggregated stats back onto the main table
  setkey(agg, focal_id, year)
  setkey(dt, id, year)
  dt <- agg[dt, on = .(focal_id = id, year = year)]

  # Restore original row order and clean up
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  # Rename focal_id back to id (the join put it as focal_id)
  setnames(dt, "focal_id", "id")

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# 3. Drop-in replacement for the outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains nb_max_ntl, nb_min_ntl, nb_mean_ntl, … for all 5 vars.
# The trained Random Forest model can be applied directly to cell_data as before,
# since the column names and numerical values are identical to the original pipeline.
```

## Why This Works and What Changes

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M interpreted R iterations with string ops | Eliminated; replaced by vectorized edge expansion |
| Neighbor stats | 5 × 6.46M `lapply` iterations | Single `data.table` grouped aggregation (C-level) |
| String key creation | ~30M+ `paste()` calls | Zero; uses integer/numeric join keys |
| Passes over data | 5 (one per variable) | 1 (all variables aggregated together) |
| Memory pattern | Named character vector of 6.46M entries | Integer-keyed `data.table` joins (more cache-friendly) |
| Expected runtime | ~86+ hours | ~2–10 minutes on 16 GB laptop |
| Numerical output | Original max/min/mean | Identical max/min/mean (preserves estimand) |
| Trained RF model | N/A | Untouched; column names match |

**Memory note:** The Cartesian expansion `edges × years` produces ~1.37M edges × 28 years ≈ 38.4M rows, each carrying 5 numeric columns (~1.5 GB). This fits within 16 GB RAM. If memory is tight, process years in batches:

```r
# Optional: batch by year to reduce peak memory
year_list <- unique(dt$year)
agg_list <- lapply(year_list, function(yr) {
  slim_yr <- slim[year == yr]
  setkey(slim_yr, id)
  nv <- edges[slim_yr, on = .(neighbor_id = id),
              allow.cartesian = TRUE, nomatch = NULL]
  nv[, eval(agg_call), by = .(focal_id, year)]
})
agg <- rbindlist(agg_list)
```

This caps peak memory at ~1/28th of the full expansion (~55 MB per batch) while remaining fully vectorized within each year.