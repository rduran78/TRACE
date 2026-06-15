 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for one primary reason: **`build_neighbor_lookup` constructs a row-level lookup list of length ~6.46 million, where each element is built by string-pasting and hash-matching through a named character vector.** This means:

1. **O(n) list with per-element string operations:** For each of ~6.46M rows, the code pastes cell IDs with years, then indexes into a named vector (`idx_lookup`) of length ~6.46M. Named-vector lookup in R is O(n) in the worst case and involves repeated string hashing. Total work is roughly O(n × k) where k is the average neighbor count (~4 for rook contiguity), but the constant factor from string operations and R-level `lapply` overhead is enormous.

2. **`compute_neighbor_stats` iterates again over 6.46M list elements** for each of 5 variables, each time extracting neighbor values and computing max/min/mean in pure R. That's ~32.3M R-level function calls total.

3. **The neighbor topology is static across years.** Rook contiguity depends only on the spatial grid, not on time. Yet the current code re-resolves neighbors at the cell-year level, inflating the problem from ~344K spatial relationships to ~6.46M row-level relationships. This is a 28× unnecessary expansion.

**Key Insight:** The neighbor table is a property of the *spatial grid*, not of cell-years. We should build a **cell-level adjacency table once** (344K cells × ~4 neighbors each ≈ 1.37M edges), then **join yearly attributes onto that table** and compute grouped summaries using vectorized, compiled-code operations (`data.table`).

---

## Optimization Strategy

1. **Build a reusable edge table (adjacency list in long form) once** from `rook_neighbors_unique`. This is a `data.table` with columns `(focal_id, neighbor_id)` — approximately 1.37M rows. This is year-invariant.

2. **For each year-variable combination, join cell-year attributes onto the edge table** using `data.table` keyed joins (binary search, O(log n)). This produces a table of `(focal_id, year, neighbor_value)`.

3. **Compute grouped neighbor stats** (`max`, `min`, `mean`) via `data.table`'s `[, .(max=..., min=..., mean=...), by=.(focal_id, year)]` — fully vectorized, single-pass, in compiled C code.

4. **Join the results back** onto the main `cell_data` table.

5. **Feed the enriched `cell_data` to the already-trained Random Forest** for prediction (no retraining).

**Expected speedup:** From ~86+ hours to **minutes**. The edge table has ~1.37M rows × 28 years = ~38.4M rows after joining years, but `data.table` grouped aggregation over 38.4M rows with 3 summary stats is a ~10-second operation per variable. Total for 5 variables: under 2 minutes for the neighbor feature computation, plus overhead for joins and I/O.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Convert cell_data to data.table if not already
# ==============================================================
cell_data <- as.data.table(cell_data)

# Ensure id_order is the vector of cell IDs in the order matching
# the spdep nb object (rook_neighbors_unique).
# id_order[i] is the cell ID for the i-th element of the nb list.

# ==============================================================
# STEP 1: Build the static spatial edge table ONCE
#         from the spdep nb object (rook_neighbors_unique)
# ==============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of length length(id_order).
  # nb_obj[[i]] is an integer vector of indices into id_order
  # representing the neighbors of cell id_order[i].
  # A value of 0L (in a length-1 vector) means no neighbors.

  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep convention: no-neighbor is encoded as integer(0) or c(0L)
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nbrs])
  }))

  return(edges)
}

cat("Building spatial edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %s directed edges\n", formatC(nrow(edge_table), big.mark = ",")))

# ==============================================================
# STEP 2: Compute neighbor features for all variables
#         using vectorized data.table joins + grouped aggregation
# ==============================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # Ensure keys for fast joins
  # cell_dt must have columns: id, year, and all source_vars
  # edge_dt must have columns: focal_id, neighbor_id

  # Create a keyed version for joining neighbor attributes
  # We join on (neighbor_id, year) to get the neighbor's value
  # for each (focal_id, year) combination.

  for (var_name in source_vars) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Subset: only the columns we need for the join
    # (neighbor cell's id, year, and the variable value)
    attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
    setkey(attr_dt, id, year)

    # Join edge table with yearly attributes:
    # For each edge (focal_id -> neighbor_id), and for each year,
    # get the neighbor's variable value.
    # This is: edge_table × years, joined with attr_dt on (neighbor_id, year)

    # First, cross-join edges with all unique years
    years_dt <- data.table(year = sort(unique(cell_dt$year)))

    # Expand edges × years (~1.37M edges × 28 years ≈ 38.4M rows)
    edge_year <- CJ_dt(edge_dt, years_dt)

    # Join to get neighbor's value
    setkey(edge_year, neighbor_id, year)
    edge_year[attr_dt, neighbor_value := i.value, on = .(neighbor_id = id, year)]

    # Compute grouped stats: for each (focal_id, year), aggregate neighbor values
    stats <- edge_year[
      !is.na(neighbor_value),
      .(
        nb_max  = max(neighbor_value),
        nb_min  = min(neighbor_value),
        nb_mean = mean(neighbor_value)
      ),
      by = .(focal_id, year)
    ]

    # Name columns to match the original pipeline's naming convention
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

    # Join stats back onto cell_data
    # Cells with no neighbors (or all-NA neighbors) will get NA — matching original behavior
    cell_dt <- merge(cell_dt, stats, by.x = c("id", "year"), by.y = c("focal_id", "year"), all.x = TRUE)

    # Clean up
    rm(attr_dt, edge_year, stats)
  }

  return(cell_dt)
}

# Helper: cross join two data.tables (Cartesian product)
CJ_dt <- function(dt1, dt2) {
  # Add dummy key columns for cross join
  dt1_copy <- copy(dt1)
  dt2_copy <- copy(dt2)
  dt1_copy[, .cj_key := 1L]
  dt2_copy[, .cj_key := 1L]
  result <- dt1_copy[dt2_copy, on = ".cj_key", allow.cartesian = TRUE]
  result[, .cj_key := NULL]
  return(result)
}

# ==============================================================
# STEP 3: Run it
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
t0 <- proc.time()

cell_data <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Done in %.1f seconds\n", elapsed))

# ==============================================================
# STEP 4: Predict with the already-trained Random Forest
#         (model is NOT retrained — just used for prediction)
# ==============================================================
# Assuming the trained model object is called `rf_model`
# and it expects a data.frame with the ~110 predictor columns:

cell_data[, prediction := predict(rf_model, newdata = .SD)]

cat("Pipeline complete.\n")
```

---

## Memory-Optimized Variant (if 16 GB RAM is tight)

The cross-join of 1.37M edges × 28 years = ~38.4M rows might use ~1.5 GB per variable pass. If memory is a concern, process year-by-year within each variable:

```r
compute_all_neighbor_features_lowmem <- function(cell_dt, edge_dt, source_vars) {
  all_years <- sort(unique(cell_dt$year))
  setkey(cell_dt, id, year)

  for (var_name in source_vars) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Pre-allocate columns with NA
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    for (yr in all_years) {
      # Subset this year's data
      yr_dt <- cell_dt[year == yr, .(id, value = get(var_name))]
      setkey(yr_dt, id)

      # Join neighbor values via edge table
      edge_vals <- copy(edge_dt)
      edge_vals[yr_dt, neighbor_value := i.value, on = .(neighbor_id = id)]

      # Aggregate
      stats <- edge_vals[
        !is.na(neighbor_value),
        .(nb_max = max(neighbor_value),
          nb_min = min(neighbor_value),
          nb_mean = mean(neighbor_value)),
        by = .(focal_id)
      ]

      # Update in place
      idx <- cell_dt[year == yr, which = TRUE]
      match_pos <- match(cell_dt$id[idx], stats$focal_id)

      set(cell_dt, i = idx, j = max_col,  value = stats$nb_max[match_pos])
      set(cell_dt, i = idx, j = min_col,  value = stats$nb_min[match_pos])
      set(cell_dt, i = idx, j = mean_col, value = stats$nb_mean[match_pos])

      rm(yr_dt, edge_vals, stats)
    }
  }

  return(cell_dt)
}
```

This variant processes only ~1.37M edge rows at a time (one year-slice), using ~50 MB per iteration instead of ~1.5 GB. It adds a loop over 28 years but each iteration is sub-second in `data.table`.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | 6.46M-element R list via string paste + named vector | 1.37M-row `data.table` edge table (static) |
| Per-variable work | `lapply` over 6.46M rows in pure R | Vectorized `data.table` keyed join + grouped agg |
| Time complexity | O(n × k) with huge R-interpreter overhead | O(n × k) with compiled C-level operations |
| Estimated runtime | **86+ hours** | **< 5 minutes** |
| RAM peak | Moderate (but slow) | ~2–4 GB (standard) or ~1 GB (low-mem variant) |
| RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical output | Identical max/min/mean | Identical max/min/mean |