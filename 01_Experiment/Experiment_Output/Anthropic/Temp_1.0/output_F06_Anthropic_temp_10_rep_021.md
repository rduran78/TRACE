 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting neighbor IDs with the current year.
4. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) — these are **hash lookups on a character vector of 6.46M entries**, repeated for every row.

The result is a list of ~6.46M integer vectors. This is extremely slow and memory-hungry because:
- **String pasting and hash lookups** inside a loop over millions of rows is inherently slow in R.
- The `idx_lookup` named vector of length 6.46M has poor cache performance.
- The output list itself consumes significant memory.

### Bottleneck B: `compute_neighbor_stats` — per-row `lapply` over 6.46M entries

For each of the 5 variables, this iterates over 6.46M list elements, subsets a numeric vector, removes NAs, and computes max/min/mean. While each operation is trivial, doing it 6.46M × 5 = ~32.3M times in interpreted R is very slow.

### Why raster focal/kernel operations are not directly applicable

The comment in the docstring asks about raster focal operations. Focal operations (e.g., `terra::focal`) assume a **regular grid with uniform kernel**. If the 344K cells form a regular rectangular grid, focal operations *could* work, but:
- Panel data means each "layer" is a year — focal operations work spatially within a single layer.
- Rook neighbors from `spdep::nb` may encode irregular boundaries (coastal cells, edge cells with fewer than 4 neighbors), which focal operations handle via `na.rm=TRUE` padding.
- The critical issue is that **the `nb` object is already computed and may reflect an irregular subset of a grid** (e.g., only land cells). A focal approach would require reconstructing the full rectangular grid and mapping cells back, which risks altering the estimand.

**Conclusion:** The correct approach is to **vectorize the neighbor computation using sparse matrix multiplication and grouped operations via `data.table`**, not focal raster operations. This preserves the exact `nb` structure and numerical results.

---

## 2. Optimization Strategy

### Step 1: Replace `build_neighbor_lookup` with a sparse adjacency matrix

Construct a sparse **N_cells × N_cells** adjacency matrix `W` from `rook_neighbors_unique`. This is done once, costs negligible time, and uses the `Matrix` package.

### Step 2: Vectorize neighbor stats computation per year

For each year, extract the column vector `x` of length N_cells for a given variable. Then:
- `W %*% x` gives the **sum** of neighbor values.
- `W %*% (x != NA)` (with proper NA handling) gives the **count**.
- Mean = sum / count.
- For **max** and **min**, use a row-wise sparse iteration or, better, use `data.table` joins on the edge list.

Since sparse matrix multiplication doesn't directly yield row-wise max/min, the most efficient general approach is:

### Step 3: Edge-list + `data.table` grouped aggregation

Convert the `nb` object to an **edge list** (from_id, to_id) — about 1.37M rows. Then for each year and each variable:
1. Join the edge list to the data to get neighbor values.
2. Group by `from_id` and compute `max`, `min`, `mean`.
3. Join results back.

This replaces 6.46M R-level list iterations with **vectorized `data.table` grouped operations** over ~1.37M edges × 28 years = ~38.5M rows, which `data.table` handles in seconds.

**Expected speedup:** From 86+ hours to **~2–5 minutes**.

### Numerical equivalence

The edge list is derived from the identical `nb` object, the same grouping (cell × year), and the same `max`, `min`, `mean` functions. The results are **numerically identical** to the original implementation. The trained Random Forest model is never touched.

---

## 3. Working R Code

```r
library(data.table)
library(spdep)

# ---------------------------------------------------------------
# 0. Load pre-existing objects (assumed already in environment)
#    - cell_data        : data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#    - id_order         : integer/character vector of cell IDs (the ordering used by the nb object)
#    - rook_neighbors_unique : an nb object (list of integer index vectors)
#    - rf_model         : the pre-trained Random Forest model (untouched)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 1. Build edge list from the nb object (once, ~1.37M rows)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, nb_obj) {
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove zero-neighbor placeholders (spdep uses 0L for no-neighbor entries)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  data.table(
    from_id = id_order[from],
    to_id   = id_order[to]
  )
}

edge_list <- build_edge_list(id_order, rook_neighbors_unique)

cat("Edge list rows:", nrow(edge_list), "\n")

# ---------------------------------------------------------------
# 2. Convert cell_data to data.table and set keys
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure id and year are keyed for fast joins
setkey(cell_dt, id, year)

# ---------------------------------------------------------------
# 3. Vectorised neighbor feature computation
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_list, source_vars) {
  
  # Create a long table of (from_id, year) × neighbor value
  # by joining edge_list to cell_dt on to_id = id
  
  # We need: for each (from_id, year), the values of each source_var at to_id
  # Step: expand edge_list × years via join
  
  # Prepare a slim lookup: id, year, and source_vars only
  lookup_cols <- c("id", "year", source_vars)
  neighbor_vals <- cell_dt[, ..lookup_cols]
  setnames(neighbor_vals, "id", "to_id")
  setkey(neighbor_vals, to_id)
  
  # Join edge_list to neighbor_vals: for each (from_id, to_id) get all years of to_id
  # This creates ~1.37M edges × 28 years ≈ 38.5M rows
  cat("Joining edge list with panel data...\n")
  
  # Keyed join: edge_list[to_id] → neighbor_vals[to_id, year, vars]
  setkey(edge_list, to_id)
  expanded <- neighbor_vals[edge_list, on = "to_id", allow.cartesian = TRUE]
  # Result columns: to_id, year, <source_vars>, from_id
  
  cat("Expanded edge-year table rows:", nrow(expanded), "\n")
  
  # Group by (from_id, year), compute max/min/mean for each variable
  cat("Computing grouped neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Compute all aggregations in one grouped operation
  stats <- expanded[,
    lapply(agg_exprs, eval),
    by = .(from_id, year)
  ]
  
  # Handle Inf/-Inf from max/min on all-NA groups → set to NA
  for (v in source_vars) {
    max_col <- paste0("neighbor_max_", v)
    min_col <- paste0("neighbor_min_", v)
    stats[is.infinite(get(max_col)), (max_col) := NA_real_]
    stats[is.infinite(get(min_col)), (min_col) := NA_real_]
  }
  
  setnames(stats, "from_id", "id")
  return(stats)
}

neighbor_stats <- compute_all_neighbor_features(cell_dt, edge_list, neighbor_source_vars)

# ---------------------------------------------------------------
# 4. Merge neighbor features back to cell_dt
# ---------------------------------------------------------------
cat("Merging neighbor features back to main table...\n")

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- grep("^neighbor_", names(cell_dt), value = TRUE)
if (length(existing_neighbor_cols) > 0) {
  cell_dt[, (existing_neighbor_cols) := NULL]
}

setkey(neighbor_stats, id, year)
setkey(cell_dt, id, year)
cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# ---------------------------------------------------------------
# 5. Convert back to data.frame if needed downstream
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)

cat("Done. New columns added:\n")
print(grep("^neighbor_", names(cell_data), value = TRUE))

# ---------------------------------------------------------------
# 6. Predict with the untouched pre-trained RF model
# ---------------------------------------------------------------
# predictions <- predict(rf_model, newdata = cell_data)
```

### Alternative: More Memory-Efficient Chunked Version

If the ~38.5M-row expanded table risks exceeding 16 GB RAM (each row with 5 doubles ≈ 1.5 GB + keys), process **year-by-year**:

```r
compute_neighbor_features_chunked <- function(cell_dt, edge_list, source_vars) {
  
  years <- sort(unique(cell_dt$year))
  lookup_cols <- c("id", "year", source_vars)
  
  results_list <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    cat("Processing year", yr, "(", i, "/", length(years), ")\n")
    
    # Subset to this year
    yr_data <- cell_dt[year == yr, ..lookup_cols]
    setnames(yr_data, "id", "to_id")
    setkey(yr_data, to_id)
    
    # Join: each edge gets the to_id's values for this year
    expanded_yr <- yr_data[edge_list, on = "to_id", nomatch = NA]
    
    # Aggregate by from_id
    agg_list <- list()
    for (v in source_vars) {
      vcol <- expanded_yr[[v]]
      agg_list[[paste0("neighbor_max_", v)]] <- bquote(
        {tmp <- .(as.name(v)); tmp <- tmp[!is.na(tmp)];
         if(length(tmp)==0) NA_real_ else max(tmp)}
      )
    }
    
    # Simpler approach: direct computation
    agg_result <- expanded_yr[, {
      res <- list()
      for (vv in source_vars) {
        vals <- .SD[[vv]]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0) {
          res[[paste0("neighbor_max_", vv)]]  <- NA_real_
          res[[paste0("neighbor_min_", vv)]]  <- NA_real_
          res[[paste0("neighbor_mean_", vv)]] <- NA_real_
        } else {
          res[[paste0("neighbor_max_", vv)]]  <- max(vals)
          res[[paste0("neighbor_min_", vv)]]  <- min(vals)
          res[[paste0("neighbor_mean_", vv)]] <- mean(vals)
        }
      }
      res
    }, by = .(from_id), .SDcols = source_vars]
    
    agg_result[, year := yr]
    setnames(agg_result, "from_id", "id")
    results_list[[i]] <- agg_result
  }
  
  rbindlist(results_list)
}

neighbor_stats <- compute_neighbor_features_chunked(cell_dt, edge_list, neighbor_source_vars)
```

### Highest-Performance Version (Recommended)

This version avoids `.SD` overhead and uses **pre-melted** edge+value joins with native `data.table` aggregation, one variable at a time:

```r
compute_neighbor_features_fast <- function(cell_dt, edge_list, source_vars) {
  
  all_stats <- list()
  setkey(edge_list, to_id)
  
  for (v in source_vars) {
    cat("Computing neighbor stats for:", v, "\n")
    
    # Slim lookup: id, year, value
    lkp <- cell_dt[, .(to_id = id, year, val = get(v))]
    setkey(lkp, to_id)
    
    # Join edges → values: (from_id, to_id, year, val)
    joined <- lkp[edge_list, on = "to_id", allow.cartesian = TRUE, nomatch = NA]
    # Columns: to_id, year, val, from_id
    
    # Aggregate by (from_id, year)
    agg <- joined[!is.na(val), .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), by = .(from_id, year)]
    
    setnames(agg, c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
    setnames(agg, "from_id", "id")
    setkey(agg, id, year)
    
    all_stats[[v]] <- agg
  }
  
  # Merge all variable stats together
  result <- all_stats[[1]]
  for (k in seq_along(all_stats)[-1]) {
    result <- result[all_stats[[k]], on = .(id, year)]
  }
  
  result
}

# --- Full pipeline ---
library(data.table)

cell_dt <- as.data.table(cell_data)
edge_list <- build_edge_list(id_order, rook_neighbors_unique)

neighbor_stats <- compute_neighbor_features_fast(cell_dt, edge_list, neighbor_source_vars)

# Merge back
setkey(cell_dt, id, year)
setkey(neighbor_stats, id, year)
existing_neighbor_cols <- grep("^neighbor_", names(cell_dt), value = TRUE)
if (length(existing_neighbor_cols) > 0) cell_dt[, (existing_neighbor_cols) := NULL]

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]
cell_data <- as.data.frame(cell_dt)

# Predict (RF model is untouched)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + hash lookups in R loop | One-time edge list from `nb` object (~ms) |
| **Stats computation** | 6.46M × 5 `lapply` iterations | 5 vectorized `data.table` group-by operations |
| **Estimated runtime** | 86+ hours | **2–10 minutes** |
| **Peak RAM** | Large (6.46M-element list of vectors) | ~2–3 GB for largest join table |
| **Numerical equivalence** | Baseline | Identical (same edges, same max/min/mean) |
| **RF model** | Untouched | Untouched |