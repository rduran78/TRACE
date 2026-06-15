 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Pastes together character keys (`id_year`) for those neighbors.
3. Matches them against a named character vector (`idx_lookup`) of length 6.46M.

This means ~6.46 million iterations of string concatenation and named-vector lookup (which is O(n) or at best O(log n) per call in base R). The result is a list of ~6.46M integer vectors, consuming enormous memory and taking dozens of hours.

`compute_neighbor_stats` then loops over that 6.46M-element list again, once per variable (×5 variables = ~32M iterations).

**Root causes:**

| Problem | Impact |
|---|---|
| Row-level `lapply` in R (not vectorized) | ~6.46M R-level iterations |
| `paste` + named-vector lookup per row | String allocation + linear scan per row |
| Storing a 6.46M-element list of integer vectors | ~10+ GB RAM, GC pressure |
| Repeating the stats loop 5× over that list | Multiplies the cost |

## Optimization Strategy

**Key insight:** The neighbor graph is *time-invariant*. A cell's neighbors in 1992 are the same cells in 2019. So we can:

1. **Expand the cell-level adjacency list into a directed edge list once** (source_cell → neighbor_cell). This has ~1.37M edges.
2. **Join the edge list to the panel data by cell ID and year** using `data.table` equi-joins — fully vectorized, no per-row R loop.
3. **Compute grouped aggregations** (max, min, mean) with `data.table`'s `by=` — one pass per variable, all in C.

This replaces 6.46M R-level iterations with a single vectorized merge + grouped aggregation. Expected runtime: **minutes, not days**.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable) is identical to the original.

## Working R Code

```r
library(data.table)

# ── Step 0: Convert panel to data.table (if not already) ──────────────────────
setDT(cell_data)

# Ensure there is a row index we can group on later
cell_data[, .row_id := .I]

# ── Step 1: Build a directed edge table from the nb object (once) ─────────────
#
# rook_neighbors_unique is an nb object: a list of length 344,208
# where element [[i]] is an integer vector of neighbor indices into id_order.
# id_order is the vector of cell IDs in the same order.

build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))          # ~1.37M
  src <- integer(n_edges)
  tgt <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nb_i <- nb_obj[[i]]
    # spdep nb objects use 0L to signal "no neighbors"
    nb_i <- nb_i[nb_i != 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      src[pos:(pos + n_i - 1L)] <- id_order[i]
      tgt[pos:(pos + n_i - 1L)] <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  data.table(source_id = src[1:(pos - 1L)],
             neighbor_id = tgt[1:(pos - 1L)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges has columns: source_id, neighbor_id   (~1.37M rows)

# ── Step 2: For each source variable, compute neighbor stats vectorized ───────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need to join edges × years to the panel.
# Strategy:
#   1. Cross-join edges with the unique years → ~1.37M × 28 ≈ 38.4M rows
#      (fits comfortably in RAM as a 3-column integer/numeric table).
#   2. Join to cell_data on (neighbor_id, year) to pull the neighbor's value.
#   3. Aggregate by (source_id, year) → max, min, mean.
#   4. Join back to cell_data on (source_id = id, year).

years <- sort(unique(cell_data$year))

# Expand edges × years
edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
edge_years[, `:=`(source_id   = edges$source_id[edge_idx],
                   neighbor_id = edges$neighbor_id[edge_idx])]
edge_years[, edge_idx := NULL]
# edge_years: ~38.4M rows, columns: source_id, neighbor_id, year

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  message("Processing neighbor stats for: ", var_name)

  # Subset the value column we need from cell_data
  val_dt <- cell_data[, .(id, year, .val = get(var_name))]
  setkey(val_dt, id, year)

  # Join: for each (source_id, neighbor_id, year), get the neighbor's value
  work <- merge(edge_years, val_dt,
                by.x = c("neighbor_id", "year"),
                by.y = c("id", "year"),
                all.x = FALSE,   # inner join: drop if neighbor-year missing
                allow.cartesian = FALSE)

  # Aggregate by (source_id, year)
  agg <- work[!is.na(.val),
              .(nb_max  = max(.val),
                nb_min  = min(.val),
                nb_mean = mean(.val)),
              by = .(source_id, year)]

  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # Join aggregated stats back to cell_data
  setkey(agg, source_id, year)
  # Remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  cell_data <- merge(cell_data, agg,
                     by.x = c("id", "year"),
                     by.y = c("source_id", "year"),
                     all.x = TRUE)

  rm(val_dt, work, agg)
  gc()
}

# Clean up helper column
cell_data[, .row_id := NULL]

message("Done. All neighbor features added.")
```

### Memory-Constrained Variant

If the ~38.4M-row `edge_years` table is too large to hold alongside `cell_data` (on a 16 GB laptop), process years in batches:

```r
for (var_name in neighbor_source_vars) {

  message("Processing neighbor stats for: ", var_name)
  agg_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]

    # Subset cell_data to this year only
    val_yr <- cell_data[year == yr, .(id, .val = get(var_name))]
    setkey(val_yr, id)

    # Join edges to neighbor values for this year
    work <- copy(edges)
    work[val_yr, .val := i..val, on = .(neighbor_id = id)]

    # Aggregate
    agg_list[[yi]] <- work[!is.na(.val),
                           .(nb_max  = max(.val),
                             nb_min  = min(.val),
                             nb_mean = mean(.val),
                             year    = yr),
                           by = .(source_id)]
    rm(work, val_yr)
  }

  agg <- rbindlist(agg_list)
  rm(agg_list); gc()

  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  setkey(agg, source_id, year)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  cell_data <- merge(cell_data, agg,
                     by.x = c("id", "year"),
                     by.y = c("source_id", "year"),
                     all.x = TRUE)
  rm(agg); gc()
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations, string keys | Eliminated; replaced by 1.37M-row edge table |
| `compute_neighbor_stats` | 6.46M list traversals × 5 vars | Vectorized `data.table` grouped aggregation |
| Memory for lookup list | ~10+ GB (list of 6.46M int vectors) | ~22 MB (3-column edge table) |
| Estimated wall time | 86+ hours | **5–15 minutes** |
| Numerical result | max / min / mean of rook-neighbor values | **Identical** |
| Trained RF model | Untouched | **Untouched** |