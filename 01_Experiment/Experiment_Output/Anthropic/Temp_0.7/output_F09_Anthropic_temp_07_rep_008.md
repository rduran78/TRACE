 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### A. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every one of the ~6.46M cell-year rows, the function:
1. Looks up the cell's spatial reference index.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes together character keys (`"cellid_year"`).
4. Matches those keys against a named character vector of length 6.46M.

Character-key matching via named vectors is O(n) per lookup in the worst case. Doing this 6.46M times produces roughly **O(n²)** behavior. This single step likely accounts for the majority of the 86-hour runtime.

### B. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Even after the lookup is built, `compute_neighbor_stats` iterates row-by-row in R, extracting neighbor values, removing NAs, and computing `max/min/mean`. This is called 5 times (once per source variable), so ~32.3M R-level iterations with small-vector operations.

### C. Redundant recomputation of spatial topology per cell-year
The neighbor relationships are **purely spatial** — they do not change across years. Yet the current code embeds the year into the lookup, effectively rebuilding the spatial graph 28 times (once per year) inside a single flat structure. This is wasteful.

---

## 2. Optimization Strategy

### Core Insight: Separate Spatial Topology from Temporal Attributes

The neighbor graph is **time-invariant**. Build it once as a **cell-to-cell adjacency table** (a two-column data.table of `id → neighbor_id`), then for each year, join the yearly attribute values onto this table and compute grouped aggregates. This converts the problem from 6.46M row-level R loops into a small number of **vectorized data.table joins and grouped aggregations**.

### Specific Steps

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `nb` object → two-column `data.table(id, neighbor_id)` | One-time, ~1.37M rows |
| 2 | For each year, subset `cell_data` to that year's attributes | 28 iterations |
| 3 | Join attributes onto the edge table by `neighbor_id` | Vectorized, keyed join |
| 4 | Group by `id`, compute `max`, `min`, `mean` per variable | Vectorized aggregation |
| 5 | Join results back onto `cell_data` | Vectorized |

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes the data.table joins and aggregations over ~1.37M × 28 = ~38.5M edge-year rows, which data.table handles in seconds per variable).

**Memory:** The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. Yearly attribute joins peak at ~38.5M rows × a few columns ≈ hundreds of MB. Well within 16 GB.

**Preserves:** The trained Random Forest model is untouched. The numerical outputs (neighbor max, min, mean) are identical because the same neighbor relationships and the same aggregation functions are used.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the time-invariant cell-to-cell adjacency table
#         (run once; can be serialized to disk for reuse)
# ==============================================================

build_adjacency_table <- function(id_order, nb_object) {
  # id_order: vector of cell IDs in the same order as the nb object
  # nb_object: spdep::nb list (rook_neighbors_unique)
  #
  # Returns a data.table with columns: id, neighbor_id
  # Each row is one directed neighbor relationship.
  
  n <- length(nb_object)
  
  # Pre-count total edges to pre-allocate
  edge_counts <- vapply(nb_object, length, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id     <- integer(total_edges)
  to_id       <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- nb_object[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    k <- length(nb_idx)
    if (k > 0L) {
      from_id[pos:(pos + k - 1L)] <- id_order[i]
      to_id[pos:(pos + k - 1L)]   <- id_order[nb_idx]
      pos <- pos + k
    }
  }
  
  # Trim in case some nb entries were 0-length
  adj <- data.table(id = from_id[1:(pos - 1L)],
                    neighbor_id = to_id[1:(pos - 1L)])
  return(adj)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)

# Optional: save for future reuse
# fwrite(adj_table, "adjacency_table.csv")
# or: saveRDS(adj_table, "adjacency_table.rds")


# ==============================================================
# STEP 2: Compute neighbor stats via vectorized joins
# ==============================================================

compute_all_neighbor_features <- function(cell_data, adj_table, source_vars) {
  # cell_data:   data.frame or data.table with columns: id, year, and all source_vars
  # adj_table:   data.table with columns: id, neighbor_id
  # source_vars: character vector of variable names for which to compute neighbor stats
  #
  # Returns cell_data (data.table) with new columns:

  #   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
  #   for each var in source_vars.
  
  dt <- as.data.table(cell_data)
  adj <- copy(adj_table)  # avoid modifying the original
  
  # Ensure key columns are of consistent type
  dt[, id := as.integer(id)]
  dt[, year := as.integer(year)]
  adj[, id := as.integer(id)]
  adj[, neighbor_id := as.integer(neighbor_id)]
  
  # Key the main data for fast joins
  setkey(dt, id, year)
  
  # For each source variable, compute neighbor max, min, mean
  for (var in source_vars) {
    message("Computing neighbor stats for: ", var)
    
    # Extract only the columns we need for the join (id, year, value)
    # This keeps memory usage minimal.
    attr_cols <- dt[, .(id, year, value = get(var))]
    setnames(attr_cols, "id", "neighbor_id")
    setkey(attr_cols, neighbor_id)
    
    # Expand adjacency table by year:
    # For each year, every edge id->neighbor_id gets the neighbor's attribute value.
    # We do this by joining adj_table with the attribute table on neighbor_id,
    # but we also need to match on year. Strategy:
    #   1. Cross-join adj_table with unique years? No — too large and wasteful.
    #   2. Better: join dt's (id, year) with adj_table on id, then join neighbor
    #      attributes on (neighbor_id, year).
    
    # Get the (id, year) pairs that exist in the data
    id_year <- dt[, .(id, year)]
    setkey(id_year, id)
    
    # For each (id, year), attach all neighbors
    # This creates ~1.37M * 28 ≈ 38.5M rows if every cell appears every year
    edges_by_year <- adj[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
    # edges_by_year has columns: id, neighbor_id, year
    
    # Now join the neighbor's attribute value for that year
    setkey(edges_by_year, neighbor_id, year)
    setkey(attr_cols, neighbor_id, year)  # re-key with year
    
    # Perform the join: attach neighbor's value
    edges_by_year[attr_cols, value := i.value, on = .(neighbor_id, year)]
    
    # Compute grouped stats: for each (id, year), aggregate over neighbors
    stats <- edges_by_year[!is.na(value),
                           .(nb_max  = max(value),
                             nb_min  = min(value),
                             nb_mean = mean(value)),
                           by = .(id, year)]
    
    # Name the output columns
    max_col  <- paste0(var, "_neighbor_max")
    min_col  <- paste0(var, "_neighbor_min")
    mean_col <- paste0(var, "_neighbor_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    
    # Remove any pre-existing columns of the same name in dt (for idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(dt)) dt[, (col) := NULL]
    }
    
    # Join stats back onto the main data
    setkey(stats, id, year)
    dt[stats, (c(max_col, min_col, mean_col)) :=
         mget(paste0("i.", c(max_col, min_col, mean_col))),
       on = .(id, year)]
    
    # Clean up to free memory
    rm(attr_cols, edges_by_year, stats)
    gc()
  }
  
  return(dt)
}


# ==============================================================
# STEP 3: Run the pipeline
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data   = cell_data,
  adj_table   = adj_table,
  source_vars = neighbor_source_vars
)

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
#          (model object is unchanged)
# ==============================================================

# cell_data now contains the same neighbor feature columns as before:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ... etc.
#
# These are numerically identical to the original implementation's output
# because the same neighbor relationships and aggregation functions are used.

# Example prediction (adjust to your actual model object and column setup):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Summary of Why This Works

| Aspect | Old Approach | New Approach |
|--------|-------------|--------------|
| **Neighbor lookup** | 6.46M character-key matches via named vector | One-time construction of ~1.37M-row integer edge table |
| **Per-variable computation** | `lapply` over 6.46M rows in R | Vectorized `data.table` join + `group by` aggregation |
| **Year handling** | Year baked into character keys, re-matched per row | Natural join on `(neighbor_id, year)` — no string operations |
| **Total R-level iterations** | ~32.3M (6.46M × 5 vars) | 5 (one per variable; inner work is C-level data.table) |
| **Expected runtime** | ~86+ hours | **~5–15 minutes** on a 16 GB laptop |
| **Numerical result** | max, min, mean of rook neighbors | Identical max, min, mean of same rook neighbors |
| **RF model** | — | **Untouched; no retraining** |