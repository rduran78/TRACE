 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-ish in practice
The function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor × that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup — R's named vector lookup is O(n) per query in the worst case, not hash-based like an environment or `data.table` key).

This means the function performs **~6.46M × k string constructions and lookups** (where k ≈ average neighbor count ~4 for rook contiguity), each against a 6.46M-length named vector. The string-key approach turns what should be a simple integer-index join into a massive string-matching problem.

### 2. The lookup is rebuilt monolithically across all years
The neighbor topology is **purely spatial** — it doesn't change across years. Yet the lookup fuses spatial adjacency with temporal indexing in a single 6.46M-element list, recomputing neighbor row indices for every cell-year combination even though the spatial structure is identical every year.

### 3. `compute_neighbor_stats` is fine algorithmically but bottlenecked by the lookup
Once the lookup list exists, the stats computation is a simple O(N×k) pass. The bottleneck is building the lookup.

---

## Optimization Strategy

**Core insight:** Separate the *spatial* adjacency structure (which is static) from the *temporal* attribute join (which varies by year).

### Step-by-step plan:

1. **Build a spatial neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_cell_id)` derived from the `nb` object. This has ~1.37M rows and never changes.

2. **For each year, join cell attributes onto the edge table** — use `data.table` keyed joins. For each year-slice, join the variable values for the neighbor cells onto the edge table, then compute `max`, `min`, `mean` grouped by `cell_id`.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table by `(cell_id, year)`.

This replaces 6.46M string lookups with ~28 vectorized `data.table` joins (one per year), each operating on ~1.37M edges. Expected runtime: **minutes, not hours**.

### Why this preserves correctness:
- The spatial neighbor set per cell is identical.
- The variable values used are identical (same cell-year attribute values).
- `max`, `min`, `mean` are computed over the same neighbor sets.
- The trained Random Forest model is never touched — we only recompute the input features identically.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a static spatial neighbor edge table (run once, reuse forever)
# ==============================================================================
build_neighbor_edge_table <- function(id_order, nb_object) {
  # nb_object: a list of length length(id_order), each element is an integer

#              vector of neighbor indices into id_order (spdep::nb format).
  # Returns a data.table with columns: cell_id, neighbor_id
  
  edges <- rbindlist(lapply(seq_along(nb_object), function(i) {
    nb_idx <- nb_object[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  return(edges)
}

# Build it once
neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns — tiny in memory

cat(sprintf("Neighbor edge table: %d rows\n", nrow(neighbor_edges)))

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure 'id' and 'year' columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: Compute neighbor stats for all variables, all years — vectorized
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, neighbor_edges, var_names) {
  # Pre-allocate output columns with NA
  for (var_name in var_names) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  # Key the edge table for fast joins
  setkey(neighbor_edges, neighbor_id)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  cat(sprintf("Processing %d years × %d variables...\n", 
              length(years), length(var_names)))
  
  for (yr in years) {
    t0 <- proc.time()[3]
    
    # Extract this year's cell attributes (only id + the variables we need)
    yr_attrs <- cell_data[year == yr, c("id", var_names), with = FALSE]
    setnames(yr_attrs, "id", "neighbor_id")
    setkey(yr_attrs, neighbor_id)
    
    # Join neighbor attributes onto the edge table
    # After this join, each row has: cell_id, neighbor_id, ntl, ec, ...
    edges_with_vals <- neighbor_edges[yr_attrs, on = "neighbor_id", nomatch = NULL]
    # edges_with_vals now has columns: cell_id, neighbor_id, <var_names...>
    
    # Compute grouped stats for each variable
    # Group by cell_id to get the neighbor summary for each cell
    agg_exprs <- list()
    for (var_name in var_names) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      agg_exprs[[col_max]]  <- call("max",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[col_min]]  <- call("min",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[col_mean]] <- call("mean", as.name(var_name), na.rm = TRUE)
    }
    
    # Build and evaluate the aggregation
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    neighbor_stats <- edges_with_vals[, eval(agg_call), by = cell_id]
    
    # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen with 
    # nomatch=NULL but defensive)
    stat_cols <- setdiff(names(neighbor_stats), "cell_id")
    for (sc in stat_cols) {
      neighbor_stats[is.infinite(get(sc)), (sc) := NA_real_]
    }
    
    # Join back into cell_data for this year
    # We need to match on (id == cell_id) AND (year == yr)
    setkey(neighbor_stats, cell_id)
    
    # Get row indices in cell_data for this year
    yr_row_idx <- cell_data[, which(year == yr)]
    yr_cell_ids <- cell_data$id[yr_row_idx]
    
    # Create a mapping from cell_id to the stats
    match_idx <- match(yr_cell_ids, neighbor_stats$cell_id)
    
    for (sc in stat_cols) {
      set(cell_data, i = yr_row_idx, j = sc, value = neighbor_stats[[sc]][match_idx])
    }
    
    elapsed <- proc.time()[3] - t0
    cat(sprintf("  Year %d done in %.1f seconds\n", yr, elapsed))
  }
  
  return(cell_data)
}

# ==============================================================================
# STEP 4: Run it
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  neighbor_edges, 
  neighbor_source_vars
)

# ==============================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now has the same neighbor feature columns as before:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, 
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec, ... etc.
#
# Predict:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)$predictions
# (or however your predict call is structured)

cat("Done. Neighbor features computed. RF model is unchanged.\n")
```

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| **Lookup construction** | ~6.46M string-key lookups against 6.46M named vector | One-time `rbindlist` over 344K cells (~1.37M edge rows) |
| **Per-variable stats** | 6.46M `lapply` iterations with subsetting | 28 keyed `data.table` joins + grouped aggregation |
| **Total estimated time** | **86+ hours** | **~5–15 minutes** |
| **Peak RAM** | ~6.46M-element list of integer vectors + named vector | ~1.37M-row edge table + year slices (~350K rows) |
| **RF model** | Unchanged ✓ | Unchanged ✓ |
| **Numerical output** | Original `max`/`min`/`mean` | Identical `max`/`min`/`mean` ✓ |

The speedup comes from:
1. **Eliminating string-key construction and lookup** — replaced by integer-keyed `data.table` joins.
2. **Vectorized grouped aggregation** — `data.table`'s `[, ..., by=]` replaces 6.46M individual `lapply` calls.
3. **Year-sliced processing** — keeps working memory small (~50MB per year-slice join) while still being fully vectorized within each year.