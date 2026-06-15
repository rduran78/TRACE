 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups via `[` on character keys are O(n) in the worst case per call due to R's internal hashing overhead at scale.

### 2. Redundant recomputation of the same spatial topology per year
The rook-neighbor relationships are **purely spatial** — they don't change across years. Yet `build_neighbor_lookup` rebuilds the neighbor mapping for every cell-year combination, effectively duplicating the same spatial adjacency 28 times (once per year). This is the core architectural flaw.

### 3. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level `lapply` over 6.46M list elements is inherently slow. Each iteration has R function-call overhead, subsetting, `is.na` filtering, and three summary function calls.

**Summary:** The pipeline does ~6.46M × (string ops + list traversal) × 2 passes (build + compute) × 5 variables ≈ billions of interpreted R operations.

---

## Optimization Strategy

### Core Insight: Separate Spatial Topology from Temporal Attributes

The neighbor relationships are a property of **cells**, not cell-years. Build the adjacency table **once** at the cell level (344,208 cells × ~4 neighbors each ≈ 1.37M directed edges), then **join** yearly attributes onto that edge table and compute grouped summaries using vectorized `data.table` operations.

### Steps:

1. **Build a static edge table** (`from_id`, `to_id`) from the `spdep::nb` object — done once, ~1.37M rows.
2. **Join yearly cell attributes** onto the edge table by (`to_id`, `year`) — this replicates the attribute of each neighbor onto the edge, giving ~1.37M × 28 ≈ ~38.5M rows (fits in RAM).
3. **Group-by aggregate** (`from_id`, `year`) to compute `max`, `min`, `mean` — fully vectorized in `data.table`, no R-level loops.
4. **Join results back** to the main cell-year panel.
5. **Predict** with the existing trained Random Forest model (unchanged).

### Expected Speedup:
- Eliminates all `lapply` loops over 6.46M rows.
- Replaces string-key lookups with integer joins.
- `data.table` grouped aggregation on ~38.5M rows completes in seconds.
- **Total estimated time: 1–5 minutes** (vs. 86+ hours).

### Memory Check:
- Edge table: ~1.37M rows × 2 int cols ≈ 11 MB
- Edge-year-attribute table: ~38.5M rows × 4 cols ≈ 1.2 GB per variable (done one variable at a time, then discarded)
- Well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert main data to data.table (if not already)
# ============================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# plus all other predictor columns needed for RF prediction.
# rook_neighbors_unique: spdep::nb object (list of integer index vectors)
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# ============================================================
# STEP 1: Build static spatial edge table ONCE
# ============================================================
# rook_neighbors_unique[[i]] gives the index positions (into id_order)
# of the neighbors of cell id_order[i].

build_edge_table <- function(id_order, neighbors) {
  from_list <- vector("list", length(id_order))
  to_list   <- vector("list", length(id_order))
  
  for (i in seq_along(id_order)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) next
    # Remove self-references and zero entries (spdep convention)
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) next
    from_list[[i]] <- rep(id_order[i], length(nb_idx))
    to_list[[i]]   <- id_order[nb_idx]
  }
  
  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: one row per directed neighbor relationship
cat("Edge table rows:", nrow(edge_dt), "\n")

# ============================================================
# STEP 2: Get unique years
# ============================================================
all_years <- sort(unique(cell_dt$year))

# ============================================================
# STEP 3: For each neighbor source variable, compute neighbor
#          max, min, mean via vectorized join + group-by
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name, all_years) {
  
  # Extract only the columns we need for the neighbor attribute lookup
  # Columns: id, year, <var_name>
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Cross-join edges with all years to get edge-year combinations
  # edge_dt has (from_id, to_id); we need one copy per year
  edge_year_dt <- CJ_edge_year(edge_dt, all_years)
  
  # Join neighbor attribute: for each edge-year, get the neighbor's value
  # Key the attribute table on (id, year) and join on (to_id, year)
  setkey(attr_dt, id, year)
  edge_year_dt[attr_dt, neighbor_val := i.val, on = .(to_id = id, year)]
  
  # Group by (from_id, year) and compute stats, excluding NAs
  stats_dt <- edge_year_dt[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(from_id, year)
  ]
  
  # Rename columns to match original pipeline naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats_dt, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # Join back onto cell_dt by (id = from_id, year)
  cell_dt[stats_dt, (c(max_col, min_col, mean_col)) :=
            mget(c(max_col, min_col, mean_col)),
          on = .(id = from_id, year)]
  
  invisible(cell_dt)
}

# Helper: cross-join edge table with years (memory-efficient)
CJ_edge_year <- function(edge_dt, all_years) {
  # Repeat each edge for every year
  n_edges <- nrow(edge_dt)
  n_years <- length(all_years)
  
  result <- data.table(
    from_id = rep(edge_dt$from_id, times = n_years),
    to_id   = rep(edge_dt$to_id,   times = n_years),
    year    = rep(all_years, each = n_edges)
  )
  result
}

# ============================================================
# STEP 4: Run for all 5 neighbor source variables
# ============================================================
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- Sys.time()
  compute_neighbor_features_fast(cell_dt, edge_dt, var_name, all_years)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "sec\n")
}

# ============================================================
# STEP 5: Predict with the existing trained Random Forest
# ============================================================
# The RF model object (e.g., rf_model) is already in memory and must NOT
# be retrained. We simply score the enriched data.

# Convert back to data.frame if the RF predict method requires it:
cell_df <- as.data.frame(cell_dt)

# Generate predictions (preserves original numerical estimand)
cell_df$rf_prediction <- predict(rf_model, newdata = cell_df)

cat("Pipeline complete. Predictions generated for",
    nrow(cell_df), "cell-year rows.\n")
```

---

## Memory-Optimized Variant (if 16 GB is tight)

If the full cross-join (~38.5M rows) for all years at once strains memory, process **one year at a time**:

```r
compute_neighbor_features_by_year <- function(cell_dt, edge_dt, var_name) {
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  for (yr in sort(unique(cell_dt$year))) {
    # Subset attributes for this year
    attr_yr <- attr_dt[year == yr]
    setkey(attr_yr, id)
    
    # Join neighbor values onto edge table
    edge_yr <- copy(edge_dt)
    edge_yr[attr_yr, neighbor_val := i.val, on = .(to_id = id)]
    
    # Aggregate
    stats_yr <- edge_yr[
      !is.na(neighbor_val),
      .(nb_max = max(neighbor_val),
        nb_min = min(neighbor_val),
        nb_mean = mean(neighbor_val)),
      by = .(from_id)
    ]
    
    # Write back into cell_dt for this year
    cell_dt[stats_yr,
            on = .(id = from_id, year = yr),
            `:=`(
              (max_col)  = i.nb_max,
              (min_col)  = i.nb_min,
              (mean_col) = i.nb_mean
            )]
  }
  invisible(cell_dt)
}

# Usage:
for (var_name in neighbor_source_vars) {
  cat("Computing (year-by-year):", var_name, "\n")
  compute_neighbor_features_by_year(cell_dt, edge_dt, var_name)
}
```

This variant processes only ~1.37M edge rows per year-variable combination, using minimal memory.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element `lapply` with string keys | 1.37M-row static integer edge table (built once) |
| **Stat computation** | 6.46M-element `lapply` per variable | Vectorized `data.table` group-by join |
| **Topology reuse** | Rebuilt implicitly per cell-year | Built once, joined per year |
| **Estimated runtime** | ~86+ hours | ~1–5 minutes |
| **RAM peak** | Moderate (but slow) | ~2–4 GB (or ~1 GB with year-by-year variant) |
| **RF model** | Unchanged ✓ | Unchanged ✓ |
| **Numerical estimand** | Preserved ✓ | Preserved ✓ |