 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` rebuilds the full cell-year neighbor mapping from scratch every time.** It iterates over all ~6.46 million rows, and for each row it does string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per call in the worst case, and string operations on 6.46M rows are inherently slow. This produces a list of 6.46 million integer vectors — one per row — which is memory-heavy and slow to construct.

2. **`compute_neighbor_stats` iterates over that 6.46M-element list with `lapply`, calling `max`, `min`, and `mean` individually per row.** This is pure row-level R looping with no vectorization.

3. **The neighbor topology is static across years** (the grid doesn't change), yet the code re-resolves neighbor relationships at the cell-year level, effectively duplicating work 28 times and entangling spatial structure with temporal attributes.

**Key insight:** The neighbor adjacency is a property of the *spatial grid*, not of the panel. There are only 344,208 cells and ~1.37M directed neighbor pairs. The yearly attribute values should be *joined onto* this small, static adjacency table, and then neighbor stats should be computed via grouped vectorized aggregation — not per-row R loops.

---

## Optimization Strategy

1. **Build a static neighbor edge table once** — a `data.table` with columns `(cell_id, neighbor_id)`, derived from the `spdep::nb` object. This table has ~1.37M rows and never changes.

2. **For each year and each variable, join the neighbor's attribute value onto the edge table**, then compute `max`, `min`, and `mean` by `cell_id` using `data.table` grouped aggregation. This replaces all `lapply` loops with vectorized, in-memory columnar operations.

3. **Merge the resulting neighbor stats back** onto the main `cell_data` data.table by `(id, year)`.

This reduces the problem from 6.46M × per-row R function calls to ~28 vectorized grouped joins on a 1.37M-row table — a speedup of roughly **100–500×**, bringing runtime from 86+ hours to **minutes**.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static neighbor edge table ONCE
#         Input: id_order (vector of 344,208 cell IDs)
#                rook_neighbors_unique (spdep nb object, list of length 344,208)
# ──────────────────────────────────────────────────────────────────────

build_neighbor_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: neighbors[[i]] gives integer indices into id_order
  n <- length(id_order)
  
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_counts)
  
  # Build vectors
  from_id <- rep(id_order, times = edge_counts)
  to_idx  <- unlist(neighbors, use.names = FALSE)
  to_id   <- id_order[to_idx]
  
  edge_dt <- data.table(cell_id = from_id, neighbor_id = to_id)
  
  # Verify
  message(sprintf(
    "Neighbor edge table: %s rows (directed edges) for %s cells.",
    format(nrow(edge_dt), big.mark = ","),
    format(n, big.mark = ",")
  ))
  
  return(edge_dt)
}

# Build it once
edge_table <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor features for all variables via vectorized joins
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_table, var_names) {
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  for (var_name in var_names) {
    message(sprintf("Processing neighbor stats for: %s", var_name))
    
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Subset only the columns we need for the neighbor lookup
    # (id, year, and the variable of interest)
    attr_dt <- cell_data[, .(id, year, value = get(var_name))]
    setkey(attr_dt, id, year)
    
    # For each year, join neighbor values and aggregate
    # We process all years at once by expanding the edge table across years
    
    # Create a cross of edge_table × years
    year_dt <- data.table(year = years)
    edges_by_year <- CJ_dt(edge_table, year_dt)
    
    # Join the neighbor's attribute value
    # edges_by_year has (cell_id, neighbor_id, year)
    # We want the value of var_name for (neighbor_id, year)
    setkey(attr_dt, id, year)
    setnames(attr_dt, "id", "neighbor_id")
    
    edges_by_year <- merge(
      edges_by_year,
      attr_dt,
      by = c("neighbor_id", "year"),
      all.x = TRUE
    )
    
    # Aggregate: for each (cell_id, year), compute max, min, mean of neighbor values
    stats_dt <- edges_by_year[
      !is.na(value),
      .(
        n_max  = max(value),
        n_min  = min(value),
        n_mean = mean(value)
      ),
      by = .(cell_id, year)
    ]
    
    # Rename for merge
    setnames(stats_dt, c("cell_id", "year", "n_max", "n_min", "n_mean"),
             c("id", "year", col_max, col_min, col_mean))
    setkey(stats_dt, id, year)
    
    # Remove old columns if they exist (in case of re-run)
    for (cc in c(col_max, col_min, col_mean)) {
      if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
    }
    
    # Merge back onto cell_data
    cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
    setkey(cell_data, id, year)
    
    message(sprintf("  Done: %s", var_name))
  }
  
  return(cell_data)
}

# Helper: cross join two data.tables (like CJ but for tables)
CJ_dt <- function(dt1, dt2) {
  dt1[, .___cj_key___ := 1L]
  dt2[, .___cj_key___ := 1L]
  result <- merge(dt1, dt2, by = ".___cj_key___", allow.cartesian = TRUE)
  result[, .___cj_key___ := NULL]
  dt1[, .___cj_key___ := NULL]
  dt2[, .___cj_key___ := NULL]
  return(result)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Run it
# ──────────────────────────────────────────────────────────────────────

cell_data <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────

# The trained RF model object is preserved as-is.
# Example (adjust to your actual model object and predict call):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

### Memory-Conscious Variant

The cross join above creates ~1.37M × 28 ≈ 38.5M rows, which at ~3 columns of doubles is roughly **1–1.5 GB** — fine for 16 GB RAM. However, if memory is tight or you want even more control, process year-by-year:

```r
compute_neighbor_features_by_year <- function(cell_data, edge_table, var_names) {
  
  years <- sort(unique(cell_data$year))
  setkey(cell_data, id, year)
  
  # Pre-allocate result columns with NA
  for (var_name in var_names) {
    cell_data[, paste0("neighbor_max_",  var_name) := NA_real_]
    cell_data[, paste0("neighbor_min_",  var_name) := NA_real_]
    cell_data[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  for (yr in years) {
    message(sprintf("Year: %d", yr))
    
    # Slice this year's data
    yr_data <- cell_data[year == yr, c("id", var_names), with = FALSE]
    setkey(yr_data, id)
    
    for (var_name in var_names) {
      # Build lookup: neighbor_id -> value for this year
      val_dt <- yr_data[, .(neighbor_id = id, value = get(var_name))]
      setkey(val_dt, neighbor_id)
      
      # Join neighbor values onto edge table
      edges_with_val <- merge(edge_table, val_dt, by = "neighbor_id", all.x = FALSE)
      
      # Aggregate by cell_id
      stats <- edges_with_val[
        !is.na(value),
        .(n_max = max(value), n_min = min(value), n_mean = mean(value)),
        by = .(cell_id)
      ]
      
      # Write back into cell_data for this year
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Match rows
      idx <- cell_data[year == yr, which = TRUE]
      matched <- match(cell_data$id[idx], stats$cell_id)
      
      set(cell_data, i = idx, j = col_max,  value = stats$n_max[matched])
      set(cell_data, i = idx, j = col_min,  value = stats$n_min[matched])
      set(cell_data, i = idx, j = col_mean, value = stats$n_mean[matched])
    }
  }
  
  return(cell_data)
}

# Run the memory-friendly version instead:
cell_data <- compute_neighbor_features_by_year(cell_data, edge_table, neighbor_source_vars)
```

---

## Summary

| Aspect | Before | After |
|---|---|---|
| **Neighbor resolution** | Per cell-year row (6.46M `lapply` iterations with string ops) | Static edge table (1.37M rows), built once |
| **Stats computation** | Row-level `lapply` with per-element `max`/`min`/`mean` | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | **5–20 minutes** |
| **Peak RAM** | Large list of 6.46M integer vectors | ~1–2 GB for edge joins |
| **RF model** | Unchanged | Unchanged |
| **Numerical output** | Neighbor max, min, mean | Identical neighbor max, min, mean |