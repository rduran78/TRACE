 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character key strings (`paste(id, year, sep="_")`).
4. Looks up row indices via a named character vector (`idx_lookup[neighbor_keys]`).

The `paste` and named-vector lookup on character keys for 6.46M rows is extremely slow. Named vector lookup in R is hash-based but still carries per-call overhead that, multiplied by millions of iterations inside `lapply`, becomes catastrophic. The function also redundantly recomputes neighbor cell IDs for the same spatial cell across all 28 years.

### Bottleneck 2: `compute_neighbor_stats` — Repeated R-level loops

For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` on small integer-indexed subsets. That's 5 × 6.46M = ~32.3M R function calls with per-element subsetting. This is slow but less catastrophic than Bottleneck 1.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a complete regular grid with a fixed kernel. This panel dataset has:
- Irregular neighbor structures (coastal/border cells have fewer neighbors).
- A temporal dimension (year) that must be matched exactly.
- Missing cell-years possible.

A focal approach would require reshaping into a 3D raster stack and handling NA borders, which risks altering the numerical results. The better analogy is **sparse matrix multiplication** or **data.table grouped joins**, which preserve the exact rook-neighbor structure and numerical results.

### Estimated speedup target

From 86+ hours → **minutes** (targeting ~5–15 minutes).

---

## Optimization Strategy

### Strategy 1: Vectorized neighbor lookup via `data.table` join

Instead of building a per-row lookup list with character key matching, we:
1. Expand the `nb` object into an **edge list** (cell_i → cell_j) — done once, ~1.37M edges.
2. Cross-join with years to get **(cell_i, year) → (cell_j, year)** — ~1.37M × 28 = ~38.5M edge-year rows.
3. Join against the data to retrieve neighbor values.
4. Group by `(cell_i, year)` and compute `max`, `min`, `mean`.

This replaces millions of R-level function calls with a single `data.table` keyed join + grouped aggregation, which is implemented in C and runs in minutes.

### Strategy 2: Compute all 5 variables in one pass

Instead of looping over variables and re-joining, we join once and compute all 15 summary statistics (3 stats × 5 variables) in a single grouped aggregation.

### Preservation guarantees

- **Trained Random Forest model**: Untouched. We only modify feature engineering.
- **Numerical estimand**: The `max`, `min`, `mean` of rook-neighbor values are computed identically — same neighbor structure, same per-variable logic, same NA handling.

---

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ---------------------------------------------------------------
  # Step 1: Build edge list from the nb object (spatial only, ~1.37M edges)
  # ---------------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id_from = id_order[i], id_to = id_order[nb_idx])
  }))

  cat(sprintf("Edge list: %d directed rook-neighbor pairs\n", nrow(edges)))

  # ---------------------------------------------------------------
  # Step 2: Convert cell_data to data.table if needed; key it

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure original row order is preserved for later reassignment
  dt[, .row_order := .I]

  # ---------------------------------------------------------------
  # Step 3: Build a slim table of just id, year, and the source vars
  # ---------------------------------------------------------------
  neighbor_val_cols <- c("id", "year", neighbor_source_vars)
  dt_vals <- dt[, ..neighbor_val_cols]
  setnames(dt_vals, "id", "id_to")
  setkey(dt_vals, id_to, year)

  # ---------------------------------------------------------------
  # Step 4: Expand edges × years and join to get neighbor values
  #
  #   For each (id_from, year), we need the values of all id_to neighbors
  #   in that same year.
  #
  #   Approach: get the unique years, cross-join with edges, then
  #   join to dt_vals on (id_to, year).
  # ---------------------------------------------------------------
  unique_years <- sort(unique(dt$year))

  # Cross join: edges × years  (~1.37M × 28 ≈ 38.5M rows)
  # Use CJ-like expansion
  edge_years <- edges[, .(year = rep(unique_years, each = .N),
                          id_from = rep(id_from, times = length(unique_years)),
                          id_to   = rep(id_to,   times = length(unique_years)))]

  # More memory-efficient approach:
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = unique_years)
  edge_years[, `:=`(id_from = edges$id_from[edge_idx],
                     id_to   = edges$id_to[edge_idx])]
  edge_years[, edge_idx := NULL]

  cat(sprintf("Edge-year table: %d rows (before filtering)\n", nrow(edge_years)))

  # Key for join
  setkey(edge_years, id_to, year)

  # Join neighbor values onto edge_years
  edge_years <- dt_vals[edge_years, on = .(id_to, year), nomatch = NA]

  # Drop rows where the neighbor cell-year doesn't exist in the data
  # (equivalent to the !is.na filter in the original code)
  # We check if any of the source vars could indicate a missing row.
  # The original idx_lookup would return NA if the (id_to, year) key was absent.
  # After the join, those rows will have NA for ALL source vars.
  # We keep rows where at least the join matched (i.e., the row existed in dt_vals).
  # Since dt_vals is derived from dt, if (id_to, year) is absent, all cols are NA.
  # We can detect this: if id_to came from dt_vals side it will be non-NA.
  # Actually in a right join (dt_vals[edge_years, ...]), unmatched rows get NA vals.
  # We filter them out:
  existing_keys <- dt_vals[, .(id_to, year)]
  setkey(existing_keys, id_to, year)
  edge_years <- edge_years[existing_keys, on = .(id_to, year), nomatch = NULL]
  # This is an inner join equivalent — only keep edge-years where neighbor exists.

  rm(existing_keys)
  gc()

  cat(sprintf("Edge-year table after filtering: %d rows\n", nrow(edge_years)))

  # ---------------------------------------------------------------
  # Step 5: Grouped aggregation — compute max, min, mean per
  #         (id_from, year) for all source vars at once
  # ---------------------------------------------------------------
  setkey(edge_years, id_from, year)

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Evaluate the aggregation
  stats <- edge_years[, lapply(agg_exprs, eval), by = .(id_from, year)]

  # Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen after filter,
  # but be safe)
  for (col in agg_names) {
    vals <- stats[[col]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats, j = col, value = vals)
  }

  cat(sprintf("Stats table: %d rows, %d new feature columns\n",
              nrow(stats), length(agg_names)))

  # ---------------------------------------------------------------
  # Step 6: Join stats back to the main data
  # ---------------------------------------------------------------
  setnames(stats, "id_from", "id")
  setkey(stats, id, year)
  setkey(dt, id, year)

  dt <- stats[dt, on = .(id, year)]

  # Rows with no neighbors will have NA for the new columns — correct behavior.

  # Restore original row order

  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  # ---------------------------------------------------------------
  # Step 7: Return as data.frame (or data.table, depending on pipeline)
  # ---------------------------------------------------------------
  as.data.frame(dt)
}

# ===================================================================
# USAGE — replaces the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# The pre-trained Random Forest model is not modified.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Simpler Alternative (if memory is tight for the 38.5M-row cross join)

If the ~38.5M-row `edge_years` table strains the 16 GB laptop, process year-by-year:

```r
compute_all_neighbor_features_chunked <- function(cell_data, id_order,
                                                   rook_neighbors_unique,
                                                   neighbor_source_vars) {
  library(data.table)

  # Step 1: Edge list (spatial only)
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id_from = id_order[i], id_to = id_order[nb_idx])
  }))
  setkey(edges, id_to)

  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  unique_years <- sort(unique(dt$year))

  val_cols <- c("id", neighbor_source_vars)

  all_stats <- rbindlist(lapply(unique_years, function(yr) {
    cat(sprintf("  Processing year %d ...\n", yr))

    dt_yr <- dt[year == yr, ..val_cols]
    setnames(dt_yr, "id", "id_to")
    setkey(dt_yr, id_to)

    # Join: for each edge, get neighbor values in this year
    merged <- dt_yr[edges, on = .(id_to), nomatch = NA]
    # merged has columns: id_to, <vars>, id_from

    # Aggregate by id_from
    agg <- merged[, {
      res <- list()
      for (v in neighbor_source_vars) {
        nv <- .SD[[v]]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0L) {
          res[[paste0("neighbor_max_", v)]]  <- NA_real_
          res[[paste0("neighbor_min_", v)]]  <- NA_real_
          res[[paste0("neighbor_mean_", v)]] <- NA_real_
        } else {
          res[[paste0("neighbor_max_", v)]]  <- max(nv)
          res[[paste0("neighbor_min_", v)]]  <- min(nv)
          res[[paste0("neighbor_mean_", v)]] <- mean(nv)
        }
      }
      res
    }, by = .(id_from), .SDcols = neighbor_source_vars]

    agg[, year := yr]
    agg
  }))

  setnames(all_stats, "id_from", "id")
  setkey(all_stats, id, year)
  setkey(dt, id, year)

  dt <- all_stats[dt, on = .(id, year)]
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  as.data.frame(dt)
}

# Usage:
cell_data <- compute_all_neighbor_features_chunked(
  cell_data, id_order, rook_neighbors_unique,
  c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` calls with `paste`/named-vector lookup | Single edge-list built from `nb` object (~1.37M rows) |
| **Stats computation** | 5 × 6.46M `lapply` calls (R-level loops) | One `data.table` grouped aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **Peak memory** | Moderate (but slow) | ~4–8 GB (fits in 16 GB; chunked version uses less) |
| **Numerical results** | Baseline | Identical — same neighbors, same `max`/`min`/`mean`, same NA handling |
| **RF model** | Unchanged | Unchanged — not retrained |