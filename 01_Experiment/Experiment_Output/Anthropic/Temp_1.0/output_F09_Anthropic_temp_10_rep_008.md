 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all ~6.46 million cell-year rows
For every single cell-year row, the function:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them into a named index vector of 6.46M entries.
- `Named vector lookup by character key` in base R is effectively a repeated hash-table probe over a 6.46M-element named vector — **for each of 6.46M rows**. This is ~41.7 trillion character-comparison operations in the worst case.

The fundamental mistake: **the neighbor topology is time-invariant, but the lookup table is built as if it were time-varying.** Every cell has the same neighbors in 1992 as in 2019. Yet the code re-discovers and re-encodes this for every year, multiplying work by 28×.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call extracts a small vector of neighbor values and computes `max/min/mean`. The per-element overhead of `lapply` + anonymous function + subsetting for 6.46M iterations is enormous. This is done 5 times (once per source variable), totaling ~32.3 million R-level function calls.

### 3. Memory pressure from 6.46M-element list
`neighbor_lookup` is a list of 6.46M integer vectors. The list overhead alone (~50 bytes/element for the list spine, plus each integer vector's header) consumes multiple gigabytes before the actual index data, which is dangerous on a 16 GB laptop.

---

## Optimization Strategy

**Core insight:** Separate the **time-invariant spatial topology** from the **time-varying cell attributes**, then use vectorized joins.

### Step-by-step plan:

1. **Build a spatial neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_cell_id)` with ~1.37M rows (directed rook edges). This is built from `rook_neighbors_unique` and `id_order` and **never touches year**.

2. **Join yearly attributes onto the edge table** — for each year, the edge table is joined to the cell attributes by `neighbor_cell_id`, giving each edge the neighbor's variable values. This is a keyed `data.table` equi-join: extremely fast.

3. **Aggregate by `(cell_id, year)`** — group the joined edge table by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass per variable (or all variables at once).

4. **Join aggregated neighbor stats back** onto the main `cell_data` table.

This replaces:
- 6.46M-element `lapply` → **vectorized `data.table` join + group-by**
- 6.46M character-key lookups → **integer-keyed joins**
- 28× redundant topology work → **1× topology table reused for all years**

**Expected speedup:** From ~86 hours to **minutes** (typically 2–10 minutes depending on disk I/O and RAM).

**Preserves:** The trained Random Forest model (no retraining), and the original numerical estimand (same `max`, `min`, `mean` neighbor statistics, same column names, same NA behavior).

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build time-invariant spatial neighbor edge table (run once, reuse)
# ===========================================================================
build_neighbor_edge_table <- function(id_order, nb_object) {
  # nb_object: spdep::nb list — nb_object[[i]] gives integer indices into

  # id_order for the neighbors of id_order[i].
  # Returns a data.table with columns: cell_id, neighbor_cell_id
  
  n <- length(id_order)
  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(nb_object, function(x) {
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  
  for (i in seq_len(n)) {
    nb_idx <- nb_object[[i]]
    if (length(nb_idx) == 1L && nb_idx[1L] == 0L) next
    k <- length(nb_idx)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nb_idx]
    pos <- pos + k
  }
  
  data.table(cell_id = from_id, neighbor_cell_id = to_id)
}

# Build it once
edge_dt <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# ===========================================================================
# STEP 2-4: Compute neighbor stats for all variables, join back
# ===========================================================================
compute_all_neighbor_features <- function(cell_data_df, edge_dt,
                                          neighbor_source_vars) {
  # Convert to data.table if needed (by reference if already data.table)
  if (!is.data.table(cell_data_df)) {
    dt <- as.data.table(cell_data_df)
  } else {
    dt <- copy(cell_data_df)
  }
  
  # Ensure key columns are present
  stopifnot(all(c("id", "year") %in% names(dt)))
  
  # Columns we need from the main table for the neighbor join
  # (neighbor_source_vars + id + year)
  attr_cols <- c("id", "year", neighbor_source_vars)
  neighbor_attrs <- dt[, ..attr_cols]
  
  # Rename 'id' to 'neighbor_cell_id' for join
  setnames(neighbor_attrs, "id", "neighbor_cell_id")
  
  # Key the attribute table for fast join
  setkey(neighbor_attrs, neighbor_cell_id, year)
  
  # -------------------------------------------------------------------
  # Cross-join edge table with all years to create (cell_id, year, neighbor_cell_id)
  # Then join neighbor attributes.
  # 
  # Memory note: edge_dt has ~1.37M rows × 28 years = ~38.4M rows.
  # With 5 numeric columns + 3 key columns, this is roughly:
  #   38.4M × 8 cols × 8 bytes ≈ 2.5 GB — fits in 16 GB.
  # -------------------------------------------------------------------
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Expand edges × years
  edge_year <- CJ_dt(edge_dt, years)
  
  # Join neighbor attributes onto edge_year
  setkey(edge_year, neighbor_cell_id, year)
  edge_year <- neighbor_attrs[edge_year, on = .(neighbor_cell_id, year)]
  
  # Now group by (cell_id, year) and compute max, min, mean for each variable
  setkey(edge_year, cell_id, year)
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Evaluate aggregation
  stats_dt <- edge_year[,
    eval(as.call(c(as.name("list"), agg_exprs))),
    by = .(cell_id, year)
  ]
  
  # Replace -Inf/Inf (from max/min of all-NA groups) with NA
  for (col_name in agg_names) {
    set(stats_dt, which(is.infinite(stats_dt[[col_name]])), col_name, NA_real_)
  }
  
  # Join back onto the main data
  setkey(dt, id, year)
  setkey(stats_dt, cell_id, year)
  
  # Remove any pre-existing neighbor columns to avoid duplication
  existing_neighbor_cols <- intersect(names(dt), agg_names)
  if (length(existing_neighbor_cols) > 0) {
    dt[, (existing_neighbor_cols) := NULL]
  }
  
  dt <- stats_dt[dt, on = .(cell_id = id, year = year)]
  
  # Restore 'id' column name (the join renames it to cell_id)
  setnames(dt, "cell_id", "id")
  
  return(dt)
}

# Helper: Cross-join an edge data.table with a vector of years
CJ_dt <- function(edge_dt, years) {
  # Efficient: replicate edge_dt for each year
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  
  result <- data.table(
    cell_id         = rep(edge_dt$cell_id,         times = n_years),
    neighbor_cell_id = rep(edge_dt$neighbor_cell_id, times = n_years),
    year            = rep(years, each = n_edges)
  )
  result
}

# ===========================================================================
# EXECUTION
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
t0 <- proc.time()

cell_data <- compute_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ===========================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ===========================================================================
# The trained RF model object is assumed to be already in memory.
# Column names match because we used the same naming convention:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.
#
# If the original code used a different naming convention, adjust the
# agg_names construction above to match exactly.

# Example (adjust to your actual model object and column requirements):
# cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Constrained Variant (if 2.5 GB edge-year table is too large)

If RAM is tight, process year-by-year without ever building the full 38.4M-row table:

```r
compute_neighbor_features_chunked <- function(cell_data_df, edge_dt,
                                               neighbor_source_vars) {
  if (!is.data.table(cell_data_df)) {
    dt <- as.data.table(cell_data_df)
  } else {
    dt <- copy(cell_data_df)
  }
  
  years <- sort(unique(dt$year))
  attr_cols <- c("id", neighbor_source_vars)
  
  # Prepare aggregation names
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Process one year at a time → peak memory ≈ 1.37M rows × ~8 cols
  results_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Subset this year's attributes
    yr_attrs <- dt[year == yr, ..attr_cols]
    setnames(yr_attrs, "id", "neighbor_cell_id")
    setkey(yr_attrs, neighbor_cell_id)
    
    # Join neighbor attributes onto edge table
    edges_with_vals <- yr_attrs[edge_dt, on = .(neighbor_cell_id)]
    
    # Aggregate
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
      )
    }))
    names(agg_exprs) <- agg_names
    
    yr_stats <- edges_with_vals[,
      eval(as.call(c(as.name("list"), agg_exprs))),
      by = .(cell_id)
    ]
    
    # Fix Inf → NA
    for (col_name in agg_names) {
      set(yr_stats, which(is.infinite(yr_stats[[col_name]])), col_name, NA_real_)
    }
    
    yr_stats[, year := yr]
    results_list[[yi]] <- yr_stats
    
    if (yi %% 5 == 0) cat(sprintf("  Processed %d / %d years\n", yi, length(years)))
  }
  
  stats_dt <- rbindlist(results_list)
  
  # Join back
  setkey(dt, id, year)
  setkey(stats_dt, cell_id, year)
  
  existing <- intersect(names(dt), agg_names)
  if (length(existing) > 0) dt[, (existing) := NULL]
  
  dt <- stats_dt[dt, on = .(cell_id = id, year = year)]
  setnames(dt, "cell_id", "id")
  
  return(dt)
}

# Usage:
cell_data <- compute_neighbor_features_chunked(
  cell_data, edge_dt, neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology representation** | Rebuilt per cell-year (6.46M string-key lookups) | Built once as edge table (~1.37M rows) |
| **Neighbor stat computation** | `lapply` over 6.46M rows × 5 vars | Vectorized `data.table` join + group-by |
| **Estimated runtime** | ~86+ hours | ~2–10 minutes |
| **Peak RAM** | ~8–12 GB (6.46M-element list) | ~2.5 GB (full) or ~0.5 GB (chunked) |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical output** | Same `max`, `min`, `mean` per cell-year | Identical values |