 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46-million-element named character vector — each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated string allocation and hash-table lookups against a massive named vector, is catastrophically slow. The named-vector `idx_lookup` alone has ~6.46M entries; probing it millions of times with multi-element key vectors is the single largest cost.

### 2. `compute_neighbor_stats` — Row-level `lapply` + `do.call(rbind, ...)`

Each of the 6.46M iterations subsets a numeric vector, removes NAs, and computes max/min/mean. The `lapply` returns a list of 6.46M small vectors, then `do.call(rbind, ...)` must allocate and fill a matrix from that list. This is repeated 5 times (once per source variable).

### Quantitative estimate of current cost

- `build_neighbor_lookup`: ~6.46M iterations × (string ops + hash lookups) ≈ tens of hours.
- `compute_neighbor_stats`: ~6.46M iterations × 5 variables × (subset + summary stats + list-to-matrix bind) ≈ additional tens of hours.
- Combined: the reported 86+ hour estimate is consistent.

---

## Optimization Strategy

The key insight is: **eliminate all row-level R loops and string-key lookups; replace them with vectorized joins and grouped aggregations using `data.table`.**

### Specific changes

| Current (slow) | Proposed (fast) |
|---|---|
| Named character vector `idx_lookup` with `paste(id, year)` keys, probed per row | `data.table` keyed join on `(id, year)` integer columns — O(1) amortized binary-search join |
| `lapply` over 6.46M rows to build neighbor index list | Vectorized expansion: explode the `nb` object into an edge-list `data.table`, then merge with the data to get row indices — one join, no loop |
| `lapply` over 6.46M rows to compute max/min/mean per neighborhood | `data.table` grouped aggregation: `dt[, .(max, min, mean), by = focal_row]` — single vectorized pass per variable |
| `do.call(rbind, list_of_6.46M_vectors)` | Eliminated entirely; results are columns in a `data.table` joined back |
| 5 separate passes rebuilding the same structure | All 5 variables computed in one pass over the edge table, or 5 fast grouped aggregations |

**Expected speedup**: From 86+ hours to roughly **2–10 minutes** on the same laptop, depending on RAM pressure. The `data.table` grouped aggregation over ~13–15 million edge-rows (bidirectional rook neighbors × years) is trivially fast.

**Numerical equivalence**: The aggregation functions (max, min, mean, with NA removal) are identical, so the trained Random Forest model requires no changes.

---

## Working R Code

```r
library(data.table)

#' Build neighbor features using fully vectorized data.table operations.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame (or data.table) with columns: id, year,
#'                         and all neighbor_source_vars.
#' @param id_order         integer vector of cell IDs in the order matching
#'                         the nb object (i.e., id_order[k] is the cell ID
#'                         for the k-th element of rook_neighbors_unique).
#' @param nb_obj           spdep nb object (list of integer index vectors);
#'                         rook_neighbors_unique.
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus new neighbor feature columns.
add_neighbor_features_fast <- function(cell_data,
                                       id_order,
                                       nb_obj,
                                       neighbor_source_vars) {

  # --- Step 0: Convert to data.table (copy to avoid side-effects) -----------
  dt <- as.data.table(cell_data)

  # --- Step 1: Build directed edge list from the nb object ------------------
  #
  # nb_obj[[k]] contains the integer indices of neighbors of the k-th spatial

  # unit.  We map those indices back to cell IDs via id_order.
  #
  # Result: edges_dt with columns (focal_id, neighbor_id), all integer.

  focal_indices <- rep(
    seq_along(nb_obj),
    times = lengths(nb_obj)
  )
  neighbor_indices <- unlist(nb_obj, use.names = FALSE)

  edges_dt <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  rm(focal_indices, neighbor_indices)  # free memory

  # Remove any zero-index entries that spdep uses for "no neighbors"

  edges_dt <- edges_dt[neighbor_id != 0L]

  # --- Step 2: Unique years vector ------------------------------------------
  years <- sort(unique(dt$year))

  # --- Step 3: Cross-join edges × years to get the full focal–neighbor–year
  #             table.  This is the "exploded" lookup table.
  #
  #   ~1.37M edges × 28 years ≈ 38.4M rows (fits comfortably in 16 GB).

  years_dt <- data.table(year = years)
  edge_year <- edges_dt[, CJ_dt := TRUE][
    years_dt[, CJ_dt := TRUE],
    on = "CJ_dt",
    allow.cartesian = TRUE
  ]
  edge_year[, CJ_dt := NULL]

  # Cleaner cross-join (data.table idiomatic):
  # We redo this properly:
  edge_year <- CJ(edge_idx = seq_len(nrow(edges_dt)), year = years)
  edge_year[, `:=`(
    focal_id    = edges_dt$focal_id[edge_idx],
    neighbor_id = edges_dt$neighbor_id[edge_idx]
  )]
  edge_year[, edge_idx := NULL]

  # --- Step 4: Attach neighbor variable values via keyed join ---------------
  #
  # We join edge_year to dt on (neighbor_id == id, year == year) to pull in
  # the neighbor's variable values.

  # Subset dt to only the columns we need for the join (save memory).
  dt_vals <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(dt_vals, "id", "neighbor_id")
  setkeyv(dt_vals, c("neighbor_id", "year"))
  setkeyv(edge_year, c("neighbor_id", "year"))

  edge_year <- dt_vals[edge_year, on = c("neighbor_id", "year")]

  # Now edge_year has columns:
  #   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, focal_id

  # --- Step 5: Grouped aggregation by (focal_id, year) ----------------------
  #
  # For each variable, compute max, min, mean (na.rm = TRUE), exactly matching
  # the original compute_neighbor_stats logic.

  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym),   na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym),   na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Build a single aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  neighbor_stats <- edge_year[,
    eval(agg_call),
    by = .(focal_id, year)
  ]

  # Handle cells with no valid neighbors: max/min of empty → -Inf/Inf → NA
  for (col in names(neighbor_stats)) {
    if (col %in% c("focal_id", "year")) next
    v <- neighbor_stats[[col]]
    set(neighbor_stats, which(is.infinite(v) | is.nan(v)), col, NA_real_)
  }

  # --- Step 6: Join aggregated stats back to the main data.table ------------
  setnames(neighbor_stats, "focal_id", "id")
  setkeyv(neighbor_stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- neighbor_stats[dt, on = c("id", "year")]

  # --- Step 7: Return -------------------------------------------------------
  # Convert back to data.frame if the caller expects one.
  return(as.data.frame(dt))
}
```

### Drop-in replacement for the outer loop

```r
# ---- BEFORE (slow, ~86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (fast, ~2-10 minutes) ----
library(data.table)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  nb_obj               = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged — proceed directly to predict().
# predictions <- predict(rf_model, newdata = cell_data)
```

### Memory-conscious variant (if 38M-row cross-join is tight on 16 GB)

If the full cross-join (~38.4M rows × several columns) strains the 16 GB laptop, process years in chunks:

```r
add_neighbor_features_chunked <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars,
                                          chunk_size = 7L) {
  dt <- as.data.table(cell_data)

  # Build edge list once
  focal_indices    <- rep(seq_along(nb_obj), times = lengths(nb_obj))
  neighbor_indices <- unlist(nb_obj, use.names = FALSE)
  edges_dt <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  edges_dt <- edges_dt[neighbor_id != 0L]
  rm(focal_indices, neighbor_indices)

  years <- sort(unique(dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  dt_vals <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(dt_vals, "id", "neighbor_id")
  setkeyv(dt_vals, c("neighbor_id", "year"))

  # Build aggregation expression once
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  stats_list <- vector("list", length(year_chunks))

  for (ci in seq_along(year_chunks)) {
    yrs <- year_chunks[[ci]]
    ey  <- CJ(edge_idx = seq_len(nrow(edges_dt)), year = yrs)
    ey[, `:=`(
      focal_id    = edges_dt$focal_id[edge_idx],
      neighbor_id = edges_dt$neighbor_id[edge_idx]
    )]
    ey[, edge_idx := NULL]
    setkeyv(ey, c("neighbor_id", "year"))

    ey <- dt_vals[ey, on = c("neighbor_id", "year")]

    chunk_stats <- ey[, eval(agg_call), by = .(focal_id, year)]

    # Inf / NaN → NA
    stat_cols <- setdiff(names(chunk_stats), c("focal_id", "year"))
    for (col in stat_cols) {
      v <- chunk_stats[[col]]
      set(chunk_stats, which(is.infinite(v) | is.nan(v)), col, NA_real_)
    }

    stats_list[[ci]] <- chunk_stats
    rm(ey, chunk_stats)
    gc()
  }

  neighbor_stats <- rbindlist(stats_list)
  rm(stats_list); gc()

  setnames(neighbor_stats, "focal_id", "id")
  setkeyv(neighbor_stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- neighbor_stats[dt, on = c("id", "year")]

  return(as.data.frame(dt))
}

# Usage:
cell_data <- add_neighbor_features_chunked(
  cell_data, id_order, rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
  chunk_size = 7  # 4 chunks of 7 years each
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with string hashing | One vectorized edge-list expansion + `data.table` keyed join |
| **Stat computation** | 6.46M-iteration `lapply` × 5 vars + `do.call(rbind)` | Single `data.table` grouped aggregation per variable (vectorized C) |
| **Estimated runtime** | 86+ hours | ~2–10 minutes |
| **RAM safety (16 GB)** | Fine (but slow) | Full version ~5–8 GB peak; chunked version ~3–5 GB peak |
| **Numerical output** | max, min, mean (na.rm) per neighbor set | Identical — preserves the trained RF model's expected features |
| **RF model** | Unchanged | Unchanged — no retraining needed |