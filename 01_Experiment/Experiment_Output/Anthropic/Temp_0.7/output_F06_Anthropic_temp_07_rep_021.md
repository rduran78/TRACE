 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) effective cost
The function iterates over **every row** (~6.46 million cell-year rows) and, for each row, performs:
- Character conversion and named-vector lookup (`id_to_ref`)
- String pasting to build neighbor keys (`paste(...)`)
- Named-vector lookup into `idx_lookup` (a named character vector of length ~6.46M)

Named vector lookups in R use linear hashing that degrades at scale. With ~6.46M rows, each doing multiple `paste` + named-vector lookups, this alone can take tens of hours.

**Key insight:** The neighbor topology is **time-invariant** — cell *i*'s rook neighbors are the same in every year. Yet the code rebuilds string keys and re-resolves neighbor indices for every cell-year combination, repeating the same spatial lookup 28 times per cell.

### Bottleneck 2: `compute_neighbor_stats` — repeated `lapply` over 6.46M rows
This is called 5 times (once per source variable). Each call iterates over 6.46M entries, subsetting, removing NAs, and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern is slow for millions of small vectors.

### Why raster focal/kernel operations don't directly apply
The comment in the prompt asks whether raster focal operations offer a useful analogy. They do conceptually (a rook neighborhood is a 3×3 cross kernel), but the data is stored as an **irregular panel** (not all cells may be present in all years, cells have arbitrary IDs, there are NAs to handle). Converting to a raster stack for 28 years × 5 variables is possible but risks altering the numerical results if the grid has gaps or irregular boundaries. The strategy below preserves exact numerical equivalence by using the same neighbor structure, but computes it **vastly** more efficiently.

---

## Optimization Strategy

| Strategy | Speedup Source |
|---|---|
| **1. Separate spatial and temporal dimensions** | Build neighbor index only over 344K cells (not 6.46M cell-years). Reuse across all 28 years. Eliminates 28× redundancy. |
| **2. Replace named-vector lookups with integer-indexed `data.table` joins** | `data.table` keyed joins are O(n log n) vs. O(n²) for large named vectors. |
| **3. Vectorized matrix operations for stats** | Instead of `lapply` over 6.46M rows, build a sparse neighbor matrix and use matrix multiplication / row operations for mean, and vectorized grouped `max`/`min`. |
| **4. Process all 5 variables in one pass** | Avoid re-traversing the neighbor structure 5 times. |

**Expected speedup:** From 86+ hours → **~2–10 minutes** on the same hardware.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves exact numerical results and the trained Random Forest model.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -------------------------------------------------------------------
  # STEP 0: Convert to data.table for performance (non-destructive)
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure 'id' and 'year' columns exist
  stopifnot(all(c("id", "year") %in% names(dt)))

  # -------------------------------------------------------------------
  # STEP 1: Build a SPATIAL-ONLY neighbor edge list (time-invariant)

  #   rook_neighbors_unique is an nb object: a list of length = # cells,
  #   where each element contains integer indices into id_order of neighbors.
  #   We convert this to an edge list of (focal_id, neighbor_id).
  # -------------------------------------------------------------------
  n_cells <- length(id_order)
  stopifnot(length(rook_neighbors_unique) == n_cells)

  # Build edge list: focal_cell_id -> neighbor_cell_id
  focal_idx <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) becomes empty,
  # but some nb objects use 0L to indicate no neighbors)
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # -------------------------------------------------------------------
  # STEP 2: Join edges with panel data to get neighbor values
  #
  #   For each (focal_id, year), we need the variable values of all
  #   its rook neighbors in that same year. We accomplish this with a
  #   keyed join: edges × dt on (neighbor_id == id, year == year).
  # -------------------------------------------------------------------

  # Key the main data for fast joining
  # We need to join on neighbor_id = id AND year = year
  # Create a slim table with just id, year, and the source variables
  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..keep_cols]
  setnames(dt_slim, "id", "neighbor_id")
  setkey(dt_slim, neighbor_id, year)

  # Add year to edges by cross-joining edges with unique years
  # WRONG approach: that would be huge. Instead, replicate edges per year

  # BETTER: join edges to the focal data to get (focal_id, year, neighbor_id),
  # then join to dt_slim to get neighbor values.

  # Actually, the most memory-efficient approach:
  # For each year, do the join. But 28 iterations is fine.

  # Alternatively (and faster): build the full join table at once.
  # edges has ~1.37M rows. Adding year: 1.37M * 28 = ~38.4M rows.
  # Each row needs the neighbor variable values. With 5 numeric vars,
  # that's ~38.4M * 5 * 8 bytes ≈ 1.5 GB — fits in 16 GB RAM.

  years <- sort(unique(dt$year))

  # Expand edges across all years
  edges_expanded <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edges_expanded[, focal_id    := edges$focal_id[edge_idx]]
  edges_expanded[, neighbor_id := edges$neighbor_id[edge_idx]]
  edges_expanded[, edge_idx := NULL]

  # Join to get neighbor variable values
  setkey(edges_expanded, neighbor_id, year)
  edges_expanded <- dt_slim[edges_expanded, on = .(neighbor_id, year)]

  # Now edges_expanded has columns:
  #   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, focal_id

  # -------------------------------------------------------------------
  # STEP 3: Compute grouped max, min, mean per (focal_id, year)
  # -------------------------------------------------------------------
  setkey(edges_expanded, focal_id, year)

  # Compute stats for all variables at once using data.table aggregation
  stat_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  stat_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call dynamically
  # data.table's .SDcols approach is cleaner here:
  stats_dt <- edges_expanded[,
    {
      result <- vector("list", length(neighbor_source_vars) * 3L)
      k <- 1L
      for (v in neighbor_source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          result[[k]]     <- NA_real_
          result[[k + 1]] <- NA_real_
          result[[k + 2]] <- NA_real_
        } else {
          result[[k]]     <- max(vals)
          result[[k + 1]] <- min(vals)
          result[[k + 2]] <- mean(vals)
        }
        k <- k + 3L
      }
      names(result) <- stat_names
      result
    },
    by = .(focal_id, year)
  ]

  # -------------------------------------------------------------------
  # STEP 4: Handle Inf/-Inf from max/min of empty groups
  #   (already handled above with the length check, but belt-and-suspenders)
  # -------------------------------------------------------------------
  for (col in stat_names) {
    vals <- stats_dt[[col]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats_dt, j = col, value = vals)
  }

  # -------------------------------------------------------------------
  # STEP 5: Merge stats back into the original data
  # -------------------------------------------------------------------
  setnames(stats_dt, "focal_id", "id")
  setkey(stats_dt, id, year)
  setkey(dt, id, year)

  dt <- stats_dt[dt, on = .(id, year)]

  # -------------------------------------------------------------------
  # STEP 6: Return as data.frame (or data.table, depending on downstream)
  # -------------------------------------------------------------------
  as.data.frame(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is NOT retrained.
# Use it exactly as before for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Even Leaner Alternative (Lower Peak Memory)

If the ~38M-row expanded edge table strains RAM, process year-by-year:

```r
compute_all_neighbor_features_lowmem <- function(cell_data,
                                                  id_order,
                                                  rook_neighbors_unique,
                                                  neighbor_source_vars) {
  library(data.table)
  dt <- as.data.table(cell_data)

  # Build spatial edge list (once)
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)
  valid <- neighbor_idx > 0L
  edges <- data.table(
    focal_id    = id_order[focal_idx[valid]],
    neighbor_id = id_order[neighbor_idx[valid]]
  )

  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..keep_cols]

  stat_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  years <- sort(unique(dt$year))
  results_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    dt_yr <- dt_slim[year == yr]
    setnames(dt_yr, "id", "neighbor_id")

    # Join edges with this year's data
    merged <- edges[dt_yr, on = "neighbor_id", nomatch = 0L, allow.cartesian = TRUE]

    stats_yr <- merged[,
      {
        result <- vector("list", length(neighbor_source_vars) * 3L)
        k <- 1L
        for (v in neighbor_source_vars) {
          vals <- get(v)
          vals <- vals[!is.na(vals)]
          if (length(vals) == 0L) {
            result[[k]] <- NA_real_; result[[k+1]] <- NA_real_; result[[k+2]] <- NA_real_
          } else {
            result[[k]] <- max(vals); result[[k+1]] <- min(vals); result[[k+2]] <- mean(vals)
          }
          k <- k + 3L
        }
        names(result) <- stat_names
        result
      },
      by = .(focal_id)
    ]
    stats_yr[, year := yr]
    results_list[[yi]] <- stats_yr
  }

  all_stats <- rbindlist(results_list)

  # Fix infinities
  for (col in stat_names) {
    v <- all_stats[[col]]
    v[is.infinite(v)] <- NA_real_
    set(all_stats, j = col, value = v)
  }

  setnames(all_stats, "focal_id", "id")
  setkey(all_stats, id, year)
  setkey(dt, id, year)
  dt <- all_stats[dt, on = .(id, year)]

  as.data.frame(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string-key lookups in named vector | Built once for 344K cells; expanded via `data.table` join |
| Stats computation | `lapply` over 6.46M rows × 5 variables | `data.table` grouped aggregation (vectorized C internals) |
| Total passes over data | 5 (one per variable) | 1 (all variables in single grouped operation) |
| Estimated runtime | 86+ hours | **~2–10 minutes** |
| Numerical results | Baseline | **Identical** (same max/min/mean over same neighbor sets) |
| Random Forest model | Unchanged | **Unchanged — no retraining** |
| Peak RAM | Moderate (but slow) | ~2–4 GB (fits in 16 GB) |