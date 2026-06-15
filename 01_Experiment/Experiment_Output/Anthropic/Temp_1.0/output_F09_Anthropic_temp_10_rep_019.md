 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all 6.46M cell-year rows with expensive per-row operations
The function iterates over every cell-year row, performs character-based key lookups (`paste`, named-vector indexing) for each row's neighbors for each year. Because the spatial neighbor topology is **time-invariant** (a cell's rook neighbors don't change from year to year), this recomputes the same spatial relationships 28 times (once per year). That's ~6.46M iterations with string manipulation and named-vector lookups inside each iteration.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call iterates over every row, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is called 5 times (once per source variable), totaling ~32.3M individual R-level function calls.

### 3. The fundamental design flaw: the lookup conflates spatial topology with temporal panel structure
The neighbor lookup embeds *both* the spatial adjacency *and* the year-matching into one monolithic list of length 6.46M. This means you cannot exploit the factored structure: **neighbors are spatial, attributes are spatiotemporal**. A join-based strategy on a static adjacency table is dramatically faster.

---

## Optimization Strategy

**Core insight:** Build the adjacency table **once** as a two-column `data.table` (`id`, `neighbor_id`) with ~1.37M rows. Then for each year, join the cell attributes onto both sides of this table and compute grouped summary statistics. This replaces 6.46M R-level iterations with vectorized `data.table` grouped joins, reducing runtime from ~86 hours to **minutes**.

Steps:
1. Convert `rook_neighbors_unique` (spdep nb object) into a static `data.table` edge list: `(id, neighbor_id)`.
2. Convert `cell_data` to `data.table`.
3. For each neighbor source variable, join `cell_data` onto the edge list by `(neighbor_id, year)` to attach the neighbor's value, then compute `max`, `min`, `mean` grouped by `(id, year)`.
4. Join results back onto `cell_data`.
5. Feed augmented `cell_data` into the existing trained Random Forest via `predict()` — no retraining.

This preserves the original numerical estimand because the same neighbor topology and the same summary statistics (max, min, mean of rook neighbors' attribute values) are computed; only the *method of computation* changes.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial adjacency edge-list ONCE
# ──────────────────────────────────────────────────────────────────────────────
# rook_neighbors_unique : spdep nb object (list of integer neighbor indices)
# id_order              : vector of cell IDs in the same order as the nb object

build_adjacency_table <- function(id_order, neighbors) {
  # neighbors is a list of length N; each element is an integer vector of
  # neighbor indices (referencing positions in id_order), or 0L for no neighbors.
  edges <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    # spdep encodes "no neighbors" as a single 0
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  setkey(edges, id)
  edges
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# adj_table has ~1,373,394 rows: (id, neighbor_id)
# This is built ONCE and can be serialized for future runs:
# fst::write_fst(adj_table, "adj_table.fst")

cat("Adjacency table rows:", nrow(adj_table), "\n")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor stats for all source variables via vectorized joins
# ──────────────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-select only the columns we need for neighbor lookups (keep it lean)
# We will process one variable at a time to limit peak memory.

# Add a year column to the adjacency table via a cross-join approach:
#   For each year, join neighbor attributes.
# But more efficient: expand adj_table × years only implicitly via keyed join.

compute_all_neighbor_features <- function(cell_dt, adj_table, source_vars) {
  
  # We need: for each (id, year), the values of source_vars at all (neighbor_id, year).
  # Strategy: join adj_table with cell_dt on neighbor_id == id, then group by (id, year).
  
  # Prepare a slim lookup: just id, year, and the source variables
  lookup_cols <- c("id", "year", source_vars)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id)
  
  # Join: for every edge, attach the neighbor's year-specific values
  # adj_table has (id, neighbor_id); lookup has (neighbor_id, year, var1, var2, ...)
  # This is a many-to-many join: each edge appears for each year the neighbor_id has data.
  # Result: (id, neighbor_id, year, var1, var2, ...)
  
  # Use merge for clarity; data.table makes this fast with keys
  setkey(adj_table, neighbor_id)
  setkey(lookup, neighbor_id)
  
  # This join produces nrow(adj_table) * 28 ≈ 38.5M rows — fits in 16GB RAM
  # (1.37M edges × 28 years × ~7 columns × 8 bytes ≈ 2.4 GB)
  
  cat("Performing adjacency-attribute join...\n")
  joined <- adj_table[lookup, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # joined columns: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
  
  cat("Join complete. Rows:", nrow(joined), "\n")
  
  # Now group by (id, year) and compute max, min, mean for each source variable
  cat("Computing grouped neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Evaluate the aggregation
  stats <- joined[, 
    lapply(agg_exprs, eval, envir = .SD), 
    by = .(id, year)
  ]
  
  # The above dynamic approach can be tricky; here's an explicit, robust version:
  stats <- joined[, {
    out <- list()
    for (v in source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = .(id, year)]
  
  cat("Stats computed. Rows:", nrow(stats), "\n")
  
  setkey(stats, id, year)
  stats
}

neighbor_stats <- compute_all_neighbor_features(cell_dt, adj_table, neighbor_source_vars)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Join neighbor stats back onto the main data
# ──────────────────────────────────────────────────────────────────────────────
# Remove old neighbor columns if they exist (to avoid duplicates)
old_neighbor_cols <- grep("^neighbor_(max|min|mean)_", names(cell_dt), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_dt[, (old_neighbor_cols) := NULL]
}

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

cat("Final cell_dt rows:", nrow(cell_dt), " cols:", ncol(cell_dt), "\n")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest (NO retraining)
# ──────────────────────────────────────────────────────────────────────────────
# rf_model is the pre-trained randomForest / ranger object already in memory.
# Ensure column names match the training feature set.

cell_dt[, prediction := predict(rf_model, newdata = cell_dt)]

# Convert back to data.frame if downstream code expects it:
cell_data <- as.data.frame(cell_dt)
```

---

## Memory-Constrained Variant (if 38.5M-row join is too large)

If the single join exceeds available RAM, process year-by-year in a loop — still vastly faster than the original because each iteration is a vectorized `data.table` operation over ~1.37M edges rather than an `lapply` over 6.46M rows:

```r
compute_neighbor_features_by_year <- function(cell_dt, adj_table, source_vars) {
  
  years <- sort(unique(cell_dt$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    cat("Processing year", yr, "(", yi, "/", length(years), ")\n")
    
    # Subset to this year
    dt_yr <- cell_dt[year == yr, c("id", "year", source_vars), with = FALSE]
    
    # Create neighbor lookup: rename id -> neighbor_id for join
    nb_lookup <- copy(dt_yr)
    setnames(nb_lookup, "id", "neighbor_id")
    nb_lookup[, year := NULL]
    setkey(nb_lookup, neighbor_id)
    
    # Join: attach neighbor attributes to each edge
    setkey(adj_table, neighbor_id)
    joined <- adj_table[nb_lookup, on = "neighbor_id", nomatch = NULL]
    # joined: (id, neighbor_id, ntl, ec, pop_density, def, usd_est_n2)
    
    # Aggregate by id
    stats_yr <- joined[, {
      out <- list()
      for (v in source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[paste0("neighbor_max_", v)]]  <- NA_real_
          out[[paste0("neighbor_min_", v)]]  <- NA_real_
          out[[paste0("neighbor_mean_", v)]] <- NA_real_
        } else {
          out[[paste0("neighbor_max_", v)]]  <- max(vals)
          out[[paste0("neighbor_min_", v)]]  <- min(vals)
          out[[paste0("neighbor_mean_", v)]] <- mean(vals)
        }
      }
      out
    }, by = .(id)]
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
  }
  
  rbindlist(result_list)
}

neighbor_stats <- compute_neighbor_features_by_year(cell_dt, adj_table, neighbor_source_vars)
setkey(neighbor_stats, id, year)

# Join and predict as before (Steps 4–5 above)
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor topology | Rebuilt per cell-year (6.46M iterations) | Built once (344K cells → 1.37M edges) |
| Key lookup method | Character paste + named vector | Integer-keyed `data.table` join |
| Stats computation | `lapply` over 6.46M rows × 5 vars | Vectorized `data.table` group-by |
| Estimated runtime | ~86+ hours | **~5–15 minutes** |
| RAM peak | Moderate (but slow) | ~2–4 GB (full join) or ~0.5 GB (year-by-year) |
| Numerical output | ✅ Identical max/min/mean | ✅ Identical max/min/mean |
| RF model | Untouched | Untouched — `predict()` only |