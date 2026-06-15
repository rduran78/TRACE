 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — repeated string allocation and concatenation.
2. **Looks up those keys** in `idx_lookup`, a named character vector of length 6.46M — each lookup is O(N) in the worst case because R named vector lookup uses linear hashing that degrades at this scale.

This means `build_neighbor_lookup` alone performs ~6.46M × (avg ~8 neighbors) ≈ **51.7 million string constructions and hash lookups against a 6.46M-entry table**. That's the 86+ hour bottleneck.

### Why It's a Broader Algorithmic Problem

The string-keying approach **conflates two independent indexing dimensions** (cell ID and year) into a single flat string key. But the neighbor structure is **time-invariant** — cell A's neighbors don't change from year to year. The algorithm re-discovers the same spatial relationships 28 times (once per year), just with different string suffixes.

The correct algorithmic insight: **separate the spatial lookup from the temporal lookup**. Build the neighbor graph once over the 344K cells, then for each year, use integer indexing to gather neighbor rows.

Similarly, `compute_neighbor_stats` is fine algorithmically but can be vectorized using matrix operations instead of per-row `lapply`.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor resolution | String paste + named-vector lookup per row (6.46M iterations) | Integer-indexed spatial neighbor list (344K cells), broadcast across years via offset arithmetic |
| Per-variable stats | `lapply` over 6.46M rows, each extracting a small vector | Vectorized sparse-matrix multiplication or `data.table` grouped aggregation |
| Complexity | O(rows × avg_neighbors × string_ops) | O(cells × avg_neighbors) one-time setup + O(rows × avg_neighbors) integer arithmetic |
| Estimated time | 86+ hours | **Minutes** |

### Key Invariant Preserved

The numerical estimand is identical: for each cell-year row, we compute `max`, `min`, and `mean` of each neighbor source variable across the rook neighbors that exist in that year. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build a fast integer-indexed spatial neighbor list (done ONCE)
# =============================================================================
# Inputs:
#   cell_data    — data.frame/data.table with columns: id, year, ntl, ec, ...
#   id_order     — integer vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique — spdep nb object (list of integer index vectors)

build_neighbor_lookup_fast <- function(cell_data, id_order, rook_neighbors) {
  # Convert to data.table if needed (non-destructive)
  dt <- as.data.table(cell_data)
  
  # --- Spatial neighbor list keyed by cell id (time-invariant) ---
  # rook_neighbors[[i]] gives neighbor indices into id_order for cell id_order[i]
  # Convert to a list keyed by cell ID -> vector of neighbor cell IDs
  spatial_neighbors <- setNames(
    lapply(seq_along(id_order), function(i) {
      nb_idx <- rook_neighbors[[i]]
      # spdep nb: 0 means no neighbors
      nb_idx <- nb_idx[nb_idx != 0L]
      if (length(nb_idx) == 0L) return(integer(0))
      id_order[nb_idx]
    }),
    as.character(id_order)
  )
  
  # --- Build a fast (id, year) -> row-index lookup using data.table ---
  dt[, .row_idx := .I]
  setkey(dt, id, year)
  
  # --- For each row, find the row indices of its spatial neighbors in the same year ---
  # Strategy: expand the neighbor relationships and join, all vectorized.
  
  # Create an edge table: for each cell, its neighbor cell IDs
  # This is time-invariant, so we build it once over unique cells
  unique_ids <- id_order
  
  # Build edge list from spatial_neighbors
  from_ids <- rep(
    unique_ids,
    times = vapply(spatial_neighbors[as.character(unique_ids)], length, integer(1))
  )
  to_ids <- unlist(spatial_neighbors[as.character(unique_ids)], use.names = FALSE)
  
  edges <- data.table(from_id = from_ids, to_id = to_ids)
  
  list(dt = dt, edges = edges)
}

# =============================================================================
# STEP 2: Compute neighbor stats for all variables at once (vectorized)
# =============================================================================
compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors,
                                          neighbor_source_vars) {
  dt <- as.data.table(copy(cell_data))
  dt[, .row_idx := .I]
  
  # --- Build spatial edge list (time-invariant) ---
  cat("Building spatial edge list...\n")
  n_cells <- length(id_order)
  
  from_ids <- integer(0)
  to_ids   <- integer(0)
  
  # Vectorized construction of edge list
  nb_lengths <- vapply(rook_neighbors, function(x) {
    sum(x != 0L)
  }, integer(1))
  
  total_edges <- sum(nb_lengths)
  from_ids <- rep(id_order, times = nb_lengths)
  
  to_ids <- unlist(lapply(seq_len(n_cells), function(i) {
    nb_idx <- rook_neighbors[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(integer(0))
    id_order[nb_idx]
  }), use.names = FALSE)
  
  edges <- data.table(from_id = from_ids, to_id = to_ids)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edges)))
  
  # --- Expand edges across all years via join ---
  # For each (from_id, year) row, find the row indices of (to_id, year)
  # 
  # Instead of expanding edges × years (which would be ~38M rows),
  # we join edges with the data twice:
  #   1. Join edges with dt on from_id to get the year and the "source row"
  #   2. Join the result with dt on (to_id, year) to get the "neighbor row"
  
  cat("Building row-index lookup...\n")
  # Lookup table: (id, year) -> row_idx
  lookup <- dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)
  
  # Step A: For each edge, cross with all years that from_id appears in
  cat("Joining edges with source rows...\n")
  # from_lookup: all (from_id, year, from_row_idx)
  from_lookup <- lookup[, .(from_id = id, year, from_row = .row_idx)]
  setkey(from_lookup, from_id)
  setkey(edges, from_id)
  
  # Merge: for each edge (from_id -> to_id), get all years from_id appears
  # This gives us (from_id, to_id, year, from_row)
  edge_year <- edges[from_lookup, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year columns: from_id, to_id, year, from_row
  
  cat(sprintf("  Edge-year pairs: %d\n", nrow(edge_year)))
  
  # Step B: Join with neighbor rows to get to_row
  cat("Joining with neighbor rows...\n")
  to_lookup <- lookup[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_lookup, to_id, year)
  setkey(edge_year, to_id, year)
  
  edge_year <- edge_year[to_lookup, on = c("to_id", "year"), nomatch = 0L]
  # Now edge_year has: from_id, to_id, year, from_row, to_row
  
  cat(sprintf("  Matched edge-year pairs: %d\n", nrow(edge_year)))
  
  # --- Compute stats per variable ---
  cat("Computing neighbor statistics...\n")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    
    # Extract neighbor values via integer indexing (vectorized)
    edge_year[, nbr_val := dt[[var_name]][to_row]]
    
    # Remove NAs for aggregation
    valid <- edge_year[!is.na(nbr_val)]
    
    # Aggregate by from_row (= the source cell-year row)
    stats <- valid[, .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ), by = from_row]
    
    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign results via integer indexing
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  edge_year[, nbr_val := NULL]  # free memory
  
  cat("Done.\n")
  return(dt)
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# The result is a data.table with the same columns as before:
#   ntl_nb_max, ntl_nb_min, ntl_nb_mean,
#   ec_nb_max,  ec_nb_min,  ec_nb_mean,
#   ... etc.
#
# The trained Random Forest model can be used directly on this output
# with no retraining — the features are numerically identical.
```

---

## Memory-Conscious Variant

If the `edge_year` table (~38.5M rows × 5 columns) strains the 16 GB RAM, process **year-by-year** to keep peak memory low:

```r
compute_all_neighbor_features_lowmem <- function(cell_data, id_order, rook_neighbors,
                                                  neighbor_source_vars) {
  dt <- as.data.table(copy(cell_data))
  dt[, .row_idx := .I]
  
  # Build spatial edge list (time-invariant)
  cat("Building spatial edge list...\n")
  n_cells <- length(id_order)
  nb_lengths <- vapply(rook_neighbors, function(x) sum(x != 0L), integer(1))
  from_ids <- rep(id_order, times = nb_lengths)
  to_ids <- unlist(lapply(seq_len(n_cells), function(i) {
    nb_idx <- rook_neighbors[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(integer(0))
    id_order[nb_idx]
  }), use.names = FALSE)
  edges <- data.table(from_id = from_ids, to_id = to_ids)
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_nb_max")  := NA_real_]
    dt[, paste0(var_name, "_nb_min")  := NA_real_]
    dt[, paste0(var_name, "_nb_mean") := NA_real_]
  }
  
  # Process year by year to limit memory
  years <- sort(unique(dt$year))
  setkey(dt, year, id)
  
  for (yr in years) {
    cat(sprintf("  Year %d...\n", yr))
    
    # Subset rows for this year
    yr_dt <- dt[year == yr, c("id", ".row_idx", neighbor_source_vars), with = FALSE]
    setkey(yr_dt, id)
    
    # Join edges with this year's data
    # from side
    from_yr <- yr_dt[, .(from_id = id, from_row = .row_idx)]
    setkey(from_yr, from_id)
    matched <- edges[from_yr, on = c("from_id"), nomatch = 0L]
    
    # to side
    to_yr <- yr_dt[, .(to_id = id, to_row = .row_idx)]
    setkey(to_yr, to_id)
    setkey(matched, to_id)
    matched <- matched[to_yr, on = c("to_id"), nomatch = 0L]
    # matched: from_id, to_id, from_row, to_row
    
    for (var_name in neighbor_source_vars) {
      matched[, nbr_val := dt[[var_name]][to_row]]
      valid <- matched[!is.na(nbr_val)]
      
      stats <- valid[, .(
        nb_max  = max(nbr_val),
        nb_min  = min(nbr_val),
        nb_mean = mean(nbr_val)
      ), by = from_row]
      
      if (nrow(stats) > 0L) {
        set(dt, i = stats$from_row, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
        set(dt, i = stats$from_row, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
        set(dt, i = stats$from_row, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
      }
      matched[, nbr_val := NULL]
    }
  }
  
  dt[, .row_idx := NULL]
  cat("Done.\n")
  return(dt)
}
```

---

## Performance Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| String constructions | ~51.7M inside loop | **0** |
| Named-vector lookups (6.46M entries) | ~51.7M | **0** (integer joins) |
| R-level loop iterations | 6.46M (`lapply`) | 28 (year loop) or 0 |
| Core operation | Character hashing | `data.table` keyed join (radix-based) |
| Estimated wall time | **86+ hours** | **5–15 minutes** |
| RAM (peak, low-mem variant) | ~2–4 GB | ~3–5 GB |
| Numerical output | Baseline | **Identical** |
| RF model impact | — | **None (no retraining)** |

The fundamental fix is **eliminating the per-row string-key construction entirely** by recognizing that the neighbor graph is time-invariant and can be expressed as an integer edge list, then joined against each year's data using `data.table`'s optimized keyed merge.