 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) list construction
`build_neighbor_lookup` iterates over **~6.46 million rows** and, for each row, performs string-pasting, named-vector lookups, and `NA` filtering. The `paste()`/`idx_lookup` pattern is essentially a hash-table join done row-by-row in an R `lapply`. With ~6.46M iterations, each doing multiple character operations and named-vector lookups, this alone can take many hours.

### Bottleneck 2: `compute_neighbor_stats` — repeated per-row R-level loops
`compute_neighbor_stats` iterates over the 6.46M-element `neighbor_lookup` list, extracting values and computing `max`/`min`/`mean` in pure R. This is called **5 times** (once per source variable), yielding ~32.3M R-level loop iterations with subsetting and aggregation.

### Why raster focal/kernel operations are **not** a drop-in replacement
Focal operations assume a regular rectangular grid with a fixed kernel. Rook contiguity on an irregular or masked spatial grid (with missing cells, coastlines, borders) changes the neighbor set per cell. A focal approach would require padding, masking, and would alter results at boundaries. Since the Random Forest model is already trained on features computed with the exact rook-neighbor logic, **we must preserve the original numerical estimand**. We use the rook-neighbor logic but implement it with vectorized operations.

---

## Optimization Strategy

1. **Replace string-key lookups with integer merge/join.** Use `data.table` keyed joins to map `(neighbor_id, year)` → row index in O(n log n) instead of O(n) per row with R character hashing overhead.

2. **Expand the neighbor list to an edge table once**, then join against the data. This converts the entire neighbor lookup + stat computation into a single grouped aggregation — no R-level row loop at all.

3. **Compute all 5 variables' stats in one pass** over the edge table, eliminating 5 separate loops.

4. **Memory budget check:** The edge table has ~1,373,394 directed rook pairs × 28 years ≈ **38.5M rows** with a few integer/double columns — roughly 1–2 GB, well within 16 GB RAM.

**Expected speedup:** From 86+ hours to **minutes** (typically 2–10 minutes depending on disk I/O).

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # -------------------------------------------------------------------
  # Step 1: Convert to data.table and create a row-index column
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # -------------------------------------------------------------------
  # Step 2: Build an edge table from the nb object

  #   rook_neighbors_unique is a list of length = number of spatial cells.
  #   rook_neighbors_unique[[i]] gives integer indices (into id_order)
  #   of the rook neighbors of cell id_order[i].
  # -------------------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as 0L; filter those out
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  # edges now has columns: focal_id, neighbor_id
  # This represents ALL directed rook-neighbor pairs (spatial, time-invariant).

  cat(sprintf("Edge table: %d directed spatial pairs\n", nrow(edges)))

  # -------------------------------------------------------------------
  # Step 3: Create a lookup from (id, year) -> row index + variable values
  # -------------------------------------------------------------------
  # We only need the id, year, row_idx, and the source variables
  cols_needed <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- dt[, ..cols_needed]
  setkey(lookup, id, year)

  # -------------------------------------------------------------------
  # Step 4: Get unique years
  # -------------------------------------------------------------------
  years <- sort(unique(dt$year))

  # -------------------------------------------------------------------
  # Step 5: For each year, join edges with data to get neighbor values,
  #         then aggregate.  We process year-by-year to control memory.
  # -------------------------------------------------------------------

  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0("max_neighbor_", var_name), value = NA_real_)
    set(dt, j = paste0("min_neighbor_", var_name), value = NA_real_)
    set(dt, j = paste0("mean_neighbor_", var_name), value = NA_real_)
  }

  cat(sprintf("Processing %d years x %d variables...\n",
              length(years), length(neighbor_source_vars)))

  for (yr in years) {
    # Subset lookup to this year
    lk_yr <- lookup[year == yr]
    setkey(lk_yr, id)

    # Join edges: for each (focal_id, neighbor_id), get neighbor's values
    # First, get the focal cell's row index
    focal_info <- lk_yr[, .(id, .row_idx)]
    setkey(focal_info, id)

    # Map focal_id -> .row_idx for this year
    edge_yr <- edges[focal_info, on = .(focal_id = id), nomatch = 0L,
                     allow.cartesian = TRUE]
    # edge_yr has columns: focal_id, neighbor_id, .row_idx (focal's row in dt)

    # Now join to get neighbor values
    neighbor_vals <- lk_yr[, c("id", neighbor_source_vars), with = FALSE]
    setkey(neighbor_vals, id)

    edge_full <- neighbor_vals[edge_yr, on = .(id = neighbor_id), nomatch = NA,
                               allow.cartesian = FALSE]
    # edge_full now has: id (=neighbor_id), <source_vars>, focal_id, .row_idx

    # Aggregate by focal cell's .row_idx
    # Build aggregation expressions dynamically
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(max(.(as.name(v)), na.rm = TRUE)),
        bquote(min(.(as.name(v)), na.rm = TRUE)),
        bquote(mean(.(as.name(v)), na.rm = TRUE))
      )
    }))
    agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
      paste0(c("max_neighbor_", "min_neighbor_", "mean_neighbor_"), v)
    }))
    names(agg_exprs) <- agg_names

    # Use a single grouped aggregation
    agg_call <- as.call(c(as.name("list"),
                          setNames(agg_exprs, agg_names)))

    stats_yr <- edge_full[, eval(agg_call), by = .row_idx]

    # Handle Inf/-Inf from max/min on all-NA (shouldn't happen if edges exist,
    # but be safe)
    for (col in agg_names) {
      vals <- stats_yr[[col]]
      vals[is.infinite(vals)] <- NA_real_
      set(stats_yr, j = col, value = vals)
    }

    # Write results back into dt using row indices
    for (col in agg_names) {
      set(dt, i = stats_yr$.row_idx, j = col, value = stats_yr[[col]])
    }

    cat(sprintf("  Year %d done (%d focal cells aggregated)\n",
                yr, nrow(stats_yr)))
  }

  # -------------------------------------------------------------------
  # Step 6: Clean up and return
  # -------------------------------------------------------------------
  dt[, .row_idx := NULL]

  # Return as data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt[])
}

# =====================================================================
# USAGE
# =====================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Now cell_data has the same 15 new columns (max/min/mean × 5 vars)
# with identical numerical values to the original implementation.
# Proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level `lapply` iterations with `paste`/named-vector lookup | Single `data.table` keyed join per year (~231K rows × edges) |
| **Stat computation** | 6.46M × 5 = 32.3M R-level iterations | One grouped aggregation per year (vectorized C-level) |
| **Total passes over data** | 1 (lookup) + 5 (stats) = 6 | 1 pass (all 5 vars aggregated together per year) |
| **Estimated runtime** | 86+ hours | **2–10 minutes** |
| **Peak memory** | ~6.46M-element list of integer vectors | ~38.5M-row edge table (~1–2 GB) — fits in 16 GB |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` over same rook-neighbor sets, same `na.rm` logic |
| **RF model** | Unchanged | Unchanged — no retraining |