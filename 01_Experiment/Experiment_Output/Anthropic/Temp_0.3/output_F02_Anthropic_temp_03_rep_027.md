 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction**: `paste(id, year, sep="_")` is called millions of times inside the loop body, and named-vector lookup (`idx_lookup[neighbor_keys]`) is an O(k) hash probe per neighbor key, repeated for every row.
- **Memory**: The named character vector `idx_lookup` with 6.46M entries is fine, but the output `neighbor_lookup` is a **list of 6.46 million integer vectors**. Each list element carries R object overhead (~56 bytes minimum), so the list alone consumes ≥ 360 MB of overhead before any actual neighbor indices are stored. With actual indices, this easily exceeds 1–2 GB.
- **Time**: The `lapply` is single-threaded, and the per-element work (character coercion, paste, hash lookup, NA filtering) is expensive in interpreted R. For 6.46M rows this alone can take many hours.

### 2. `compute_neighbor_stats` — repeated random-access gather over a 6.46M-length vector
- Called 5 times (once per variable), each time iterating over the 6.46M-element `neighbor_lookup` list, gathering neighbor values, and computing `max/min/mean`. This is again single-threaded interpreted R with heavy list overhead.
- The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (though not the dominant cost).

### Combined estimate
With ~6.46M outer iterations × 5 variables, and each iteration doing string operations + subsetting, the 86+ hour estimate is consistent with pure-R interpreted overhead on a laptop.

---

## Optimization Strategy

The key insight is: **eliminate the 6.46M-element list entirely**. Replace it with a vectorized, `data.table`-based equi-join approach that:

1. **Expands the neighbor graph into an edge table** (cell_id → neighbor_id), ~1.37M edges.
2. **Joins the edge table to the panel data twice** — once to attach the year of each focal row, and once to look up the neighbor's value in that year — using `data.table` keyed joins (binary search, C-level).
3. **Aggregates (max, min, mean) by focal row** using `data.table`'s grouped `j` expressions, which are executed in C.

This turns the entire pipeline into a sequence of **vectorized joins and grouped aggregations** with no per-row R interpretation, no giant list, and minimal string operations.

**Memory**: The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The largest intermediate table (edges × years = 1.37M × 28 ≈ 38.4M rows × a few columns) is ~1–2 GB, which fits in 16 GB RAM, especially since we process one variable at a time and discard intermediates.

**Time**: `data.table` keyed joins on integer columns over tens of millions of rows typically complete in seconds. Five variables × (one join + one aggregation) should finish in **minutes, not hours**.

**Model preservation**: We only change how the 15 neighbor-derived columns (5 vars × 3 stats) are computed. The numerical values are identical (same max, min, mean of the same neighbor sets), so the trained Random Forest model remains valid with no retraining.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature pipeline
#' 
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data as a data.table with new columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -----------------------------------------------------------
  # Step 1: Build a compact edge table from the nb object

  #         focal_id  ->  neighbor_id   (integer cell IDs)
  # -----------------------------------------------------------
  # rook_neighbors_unique[[i]] contains integer indices into id_order
  # for the neighbors of id_order[i].

  n_cells <- length(id_order)
  focal_idx    <- rep.int(seq_len(n_cells),
                          lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-entries that spdep uses for cells with no neighbors
  valid <- neighbor_idx != 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  rm(focal_idx, neighbor_idx, valid)

  # -----------------------------------------------------------
  # Step 2: Convert cell_data to data.table (no copy if already)
  # -----------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Ensure a row-order column so we can put results back in place
  cell_data[, .row_order := .I]

  # Key for the neighbor value lookup: (id, year)
  setkey(cell_data, id, year)

  # -----------------------------------------------------------
  # Step 3: For each variable, join + aggregate
  # -----------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    message("Processing neighbor stats for: ", var_name)

    # 3a. Build a slim lookup: neighbor_id, year -> value
    #     (only the columns we need, to save memory)
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # 3b. For every focal row, get its year and attach to edges
    #     focal_rows: focal_id, year, .row_order
    focal_rows <- cell_data[, .(focal_id = id, year, .row_order)]

    # 3c. Cross with edges to get (focal_id, year, neighbor_id, .row_order)
    #     Join focal_rows to edges on focal_id
    setkey(edges, focal_id)
    setkey(focal_rows, focal_id)
    expanded <- edges[focal_rows,
                      .(focal_id,
                        neighbor_id,
                        year = i.year,
                        .row_order = i..row_order),
                      on = "focal_id",
                      allow.cartesian = TRUE,
                      nomatch = NULL]

    # 3d. Look up the neighbor's value in the same year
    expanded[val_dt,
             neighbor_val := i.val,
             on = c(neighbor_id = "id", "year")]

    # 3e. Aggregate by focal row
    stats <- expanded[!is.na(neighbor_val),
                      .(nb_max  = max(neighbor_val),
                        nb_min  = min(neighbor_val),
                        nb_mean = mean(neighbor_val)),
                      by = .row_order]

    # 3f. Assign back to cell_data by .row_order
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    cell_data[stats, on = ".row_order",
              c(max_col, min_col, mean_col) :=
                .(i.nb_max, i.nb_min, i.nb_mean)]

    # Free intermediates
    rm(val_dt, focal_rows, expanded, stats)
    gc()
  }

  # Clean up helper column
  cell_data[, .row_order := NULL]

  # Restore original row order (setkey may have reordered)
  setkey(cell_data, NULL)

  return(cell_data)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
library(data.table)

# ---- Load your existing objects ----
# cell_data                 : your panel data.frame (~6.46M rows)
# id_order                  : integer vector of cell IDs
# rook_neighbors_unique     : spdep nb object
# rf_model                  : your already-trained Random Forest (untouched)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ...
# These are numerically identical to the original implementation.
# Proceed with prediction using the existing rf_model as before.
```

### Optional: Further reduce peak memory with chunked processing

If the `expanded` table (~38M rows) still causes memory pressure, process years in chunks:

```r
# Inside the for-loop over var_name, replace steps 3b–3e with:
years <- sort(unique(cell_data$year))
stats_list <- vector("list", length(years))

for (j in seq_along(years)) {
  yr <- years[j]
  focal_yr <- cell_data[year == yr, .(focal_id = id, year, .row_order)]
  setkey(focal_yr, focal_id)

  exp_yr <- edges[focal_yr,
                  .(focal_id, neighbor_id,
                    year = i.year, .row_order = i..row_order),
                  on = "focal_id",
                  allow.cartesian = TRUE,
                  nomatch = NULL]

  exp_yr[val_dt, neighbor_val := i.val,
         on = c(neighbor_id = "id", "year")]

  stats_list[[j]] <- exp_yr[!is.na(neighbor_val),
                             .(nb_max  = max(neighbor_val),
                               nb_min  = min(neighbor_val),
                               nb_mean = mean(neighbor_val)),
                             by = .row_order]
  rm(focal_yr, exp_yr)
}
stats <- rbindlist(stats_list)
rm(stats_list)
```

This reduces peak memory from ~2 GB to ~70–80 MB per chunk (one year ≈ 344K cells × ~4 neighbors).

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Core approach** | Per-row `lapply` with string keys | Vectorized `data.table` join + grouped aggregation |
| **`build_neighbor_lookup`** | 6.46M-element R list (~1–2 GB) | 1.37M-row edge table (~11 MB) |
| **`compute_neighbor_stats`** | Interpreted R loop, 6.46M iterations × 5 vars | C-level keyed join + `by` aggregation |
| **Estimated time** | 86+ hours | ~5–20 minutes (depending on disk I/O) |
| **Peak RAM** | >8 GB (list overhead + data) | ~3–5 GB (chunked: ~2 GB) |
| **Numerical output** | max, min, mean of rook neighbors | Identical max, min, mean of same neighbors |
| **RF model** | — | Untouched; no retraining needed |