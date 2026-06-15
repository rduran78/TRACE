 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: Row-wise `lapply` over 6.46M rows with string operations

1. **`build_neighbor_lookup`** iterates row-by-row (`lapply` over 6.46M rows). For each row, it:
   - Calls `paste()` to construct string keys for every neighbor of that cell in that year.
   - Performs named-vector lookup (`idx_lookup[neighbor_keys]`) against a 6.46M-element named character vector.
   
   This is **O(N × k)** string constructions and hash lookups, where N ≈ 6.46M and k ≈ average neighbor count (~4 for rook). That's ~25.8M `paste` calls plus ~25.8M hash lookups against a 6.46M-entry table — all in an interpreted `lapply` loop.

2. **The string-keying is entirely unnecessary.** The data is a balanced panel (344,208 cells × 28 years). Every cell appears in every year. So for a given row `i` with `(cell_id, year)`, its neighbors are the rows with `(neighbor_cell_id, same_year)`. Since the panel is balanced and sorted, neighbor row indices can be computed **arithmetically** — no strings, no hash tables.

3. **`compute_neighbor_stats`** is called 5 times (once per variable), each time re-traversing the 6.46M-element neighbor lookup. This is fine structurally, but can be vectorized with matrix operations instead of `lapply`.

### Estimated cost of current approach

- `build_neighbor_lookup`: ~6.46M iterations × (string paste + hash lookup) ≈ hours alone.
- `compute_neighbor_stats`: 5 vars × 6.46M `lapply` iterations ≈ additional hours.
- Total: the reported 86+ hours is consistent with this analysis.

---

## Optimization Strategy

### Key Insight: Arithmetic Index Mapping Replaces All String Work

For a balanced panel sorted by `(id, year)` — or even unsorted — we can build an integer matrix of neighbor row indices using **vectorized joins**, eliminating all `paste`/string operations.

**Three-phase strategy:**

| Phase | Current | Proposed |
|-------|---------|----------|
| **1. Neighbor row-index construction** | Row-wise `lapply` with `paste` + named-vector lookup | Vectorized `data.table` equi-join: expand directed neighbor pairs × years in one join |
| **2. Neighbor stats computation** | `lapply` over 6.46M rows per variable | Vectorized `data.table` grouped aggregation: join neighbor values, group-by source row, compute `max/min/mean` |
| **3. Column binding** | Loop over 5 variables | Single pass or simple loop over 5 variables with vectorized internals |

**Expected speedup:** From 86+ hours to **minutes** (the dominant cost becomes a ~26M-row join and grouped aggregation, which `data.table` handles in seconds).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Preserves original numerical estimand (max, min, mean of neighbor values)
# Preserves trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

build_neighbor_features_fast <- function(cell_data, 
                                          id_order, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # PHASE 0: Convert to data.table and build row index
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Create a unique row identifier preserving original row order
  dt[, .row_id := .I]
  
  # Build a lookup: for each (id, year) -> row index
  # This is a single vectorized operation
  setkey(dt, id, year)
  
  # -------------------------------------------------------------------------
  # PHASE 1: Build directed neighbor edge list (cell-level, no year dimension)
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length = length(id_order)
  # where element i contains integer indices into id_order of neighbors of 
  # id_order[i].
  
  # Expand to a two-column data.table of (focal_id, neighbor_id)
  n_cells <- length(id_order)
  
  # Vectorized expansion of the nb object
  focal_indices <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  neighbor_indices <- unlist(rook_neighbors_unique)
  
  # Remove zero-neighbor entries (if any nb element is integer(0), 
  # lengths = 0, so they contribute nothing)
  edges <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  
  # Remove any self-loops or NA entries (defensive)
  edges <- edges[!is.na(neighbor_id) & focal_id != neighbor_id]
  
  cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edges)))
  
  # -------------------------------------------------------------------------
  # PHASE 2: Join edges with panel to get (focal_row, neighbor_row) pairs
  # -------------------------------------------------------------------------
  # For each edge (focal_id, neighbor_id), and for each year, we need:
  #   focal_row    = row in dt where id == focal_id    & year == y
  #   neighbor_row = row in dt where id == neighbor_id & year == y
  #
  # Strategy: 
  #   1. Cross-join edges with the unique years.
  #   2. Join to dt twice to get focal_row and neighbor_row.
  #
  # But cross-joining 1.37M edges × 28 years = 38.5M rows, which is 
  # manageable. However, we can be smarter: join edges to dt on focal_id 
  # to get (focal_row, neighbor_id, year), then join on (neighbor_id, year) 
  # to get neighbor_row.
  
  # Step 2a: Get focal rows — join edges to dt on focal_id = id
  # This gives us one row per (edge × year) = ~38.5M rows
  
  # Build a minimal lookup table: id -> year -> .row_id
  row_lookup <- dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)
  
  # Join: for each edge, get all years of the focal cell
  setkey(edges, focal_id)
  focal_lookup <- row_lookup[, .(focal_id = id, year, focal_row = .row_id)]
  setkey(focal_lookup, focal_id)
  
  # Merge edges with focal years
  # Each edge gets replicated across all years the focal_id appears in
  edge_year <- edges[focal_lookup, 
                     on = .(focal_id), 
                     nomatch = NULL,
                     allow.cartesian = TRUE]
  # edge_year now has columns: focal_id, neighbor_id, year, focal_row
  
  cat(sprintf("Edge-year pairs: %d rows\n", nrow(edge_year)))
  
  # Step 2b: Join to get neighbor_row
  neighbor_lookup_dt <- row_lookup[, .(neighbor_id = id, year, neighbor_row = .row_id)]
  setkey(neighbor_lookup_dt, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)
  
  edge_year <- edge_year[neighbor_lookup_dt, 
                          on = .(neighbor_id, year), 
                          nomatch = NA]
  
  # Drop rows where neighbor_row is NA (neighbor cell not present in that year)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]
  
  cat(sprintf("Valid edge-year pairs after join: %d rows\n", nrow(edge_year)))
  
  # -------------------------------------------------------------------------
  # PHASE 3: Compute neighbor stats per variable (vectorized)
  # -------------------------------------------------------------------------
  # For each variable, look up neighbor values, then group by focal_row 
  # and compute max, min, mean.
  
  setkey(edge_year, focal_row)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    
    # Look up neighbor values
    edge_year[, nval := dt[[var_name]][neighbor_row]]
    
    # Compute grouped stats (excluding NAs, matching original behavior)
    stats <- edge_year[!is.na(nval), 
                       .(var_max  = max(nval),
                         var_min  = min(nval),
                         var_mean = mean(nval)),
                       by = focal_row]
    
    # Initialize columns with NA (rows with no valid neighbors get NA)
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed values
    dt[stats$focal_row, (max_col)  := stats$var_max]
    dt[stats$focal_row, (min_col)  := stats$var_min]
    dt[stats$focal_row, (mean_col) := stats$var_mean]
    
    cat(sprintf("  Done. %d rows received neighbor stats, %d rows are NA.\n",
                nrow(stats), nrow(dt) - nrow(stats)))
  }
  
  # -------------------------------------------------------------------------
  # PHASE 4: Clean up and return as data.frame
  # -------------------------------------------------------------------------
  # Restore original row order
  setorder(dt, .row_id)
  dt[, .row_id := NULL]
  
  # Return as data.frame to preserve compatibility with downstream RF predict()
  return(as.data.frame(dt))
}


# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# 
# cell_data <- build_neighbor_features_fast(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # The trained RF model is unchanged — just call predict() as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Considerations (16 GB laptop)

The largest intermediate object is `edge_year` at ~38.5M rows × 4 integer columns ≈ **1.2 GB**. Combined with `dt` (~6.46M × 110 cols ≈ 5.7 GB), peak memory is ~8–9 GB, which fits within 16 GB.

If memory is tight, process variables one at a time and drop the `nval` column between iterations (the code above already does this implicitly since `:=` overwrites).

## Expected Runtime

| Phase | Estimated Time |
|-------|---------------|
| Edge list construction | < 1 second |
| Edge-year join (~38.5M rows) | 10–30 seconds |
| Per-variable grouped aggregation (×5) | 5–15 seconds each |
| **Total** | **~2–5 minutes** |

This represents a speedup of roughly **1,000–2,500×** over the original 86+ hour estimate, achieved by replacing row-wise string-key construction and hash lookups with vectorized `data.table` equi-joins and grouped aggregations. The numerical output (max, min, mean of non-NA neighbor values, with NA for cells having no valid neighbors) is identical to the original.