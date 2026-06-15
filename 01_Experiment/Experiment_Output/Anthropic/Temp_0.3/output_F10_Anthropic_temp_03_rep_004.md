 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The string-keyed lookup `idx_lookup` is a named character vector of length ~6.46M — each lookup into it is O(n) in the worst case because R named vectors use linear hashing that degrades at scale.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements**, subsetting a numeric vector and computing `max/min/mean` per element. The `lapply` + `do.call(rbind, ...)` pattern over millions of tiny vectors is extremely slow due to R interpreter overhead and memory allocation churn.

3. **The neighbor lookup is row-based (cell×year), but the graph topology is year-invariant.** The rook-neighbor structure is purely spatial — it doesn't change across years. Yet the current code expands it to the full panel, multiplying work by 28×. For 344,208 cells × 28 years, the lookup list has ~9.6M entries instead of ~344K.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~6.46M string paste + named vector lookups → hours.
- `compute_neighbor_stats` called 5 times: 5 × 6.46M list iterations → hours each.
- `do.call(rbind, list_of_6.46M_vectors)` → massive memory pressure and GC pauses.

---

## Optimization Strategy

### Core Insight: Separate Topology from Attributes

The rook-neighbor graph is **static across years**. We should:

1. **Build the sparse adjacency structure once at the cell level (344K nodes, ~1.37M edges)** as a CSR (Compressed Sparse Row) representation using integer vectors — no strings, no named vectors, no lists-of-lists.

2. **Reshape attribute data into a cell × year matrix** (344,208 rows × 28 columns), so that for each variable, neighbor aggregation becomes **sparse matrix–vector multiplication** or equivalent vectorized operation — one operation per year, not per cell-year.

3. **Use `data.table` for fast grouping and joining**, and **sparse matrix algebra (`Matrix` package)** for neighbor aggregation:
   - `mean(neighbors)` = sparse adjacency matrix (row-normalized) × attribute vector
   - `max(neighbors)` and `min(neighbors)` = row-wise max/min over sparse matrix with attribute-filled entries

4. **For max and min**, since sparse matrix algebra doesn't directly support these, we use the CSR structure with vectorized C-level operations via `data.table` grouping on the edge list.

### Complexity Reduction

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | O(6.46M) string ops | O(1.37M) integer edge list, built once |
| Stats per variable | O(6.46M) R-level list iterations × 5 | O(1.37M × 28) vectorized group-by × 5 |
| Memory pattern | 6.46M R list elements + string keys | Integer edge list + dense matrices |
| Expected runtime | 86+ hours | ~5–15 minutes |

---

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original max, min, mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)

# ---- Step 0: Ensure cell_data is a data.table keyed properly ----
# Assumes: cell_data has columns 'id', 'year', and the neighbor_source_vars.
# Assumes: id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
# Assumes: rook_neighbors_unique is an spdep::nb object (list of integer index vectors).

cell_dt <- as.data.table(cell_data)

# ---- Step 1: Build the directed edge list ONCE (topology only) ----
# rook_neighbors_unique[[i]] gives the indices (into id_order) of neighbors of cell i.
# We build a two-column integer edge list: (from_cell_idx, to_cell_idx)
# where indices refer to positions in id_order.

build_edge_list <- function(nb_obj) {
  # nb_obj: list of integer vectors (spdep::nb), 0L means no neighbors
  n <- length(nb_obj)
  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    if (length(nb) == 1L && nb[1] == 0L) next
    k <- length(nb)
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- nb
    pos <- pos + k
  }
  
  data.table(from_cell_idx = from_idx, to_cell_idx = to_idx)
}

cat("Building edge list...\n")
edge_dt <- build_edge_list(rook_neighbors_unique)
# edge_dt has ~1,373,394 rows — small and fast.

cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# ---- Step 2: Map cell IDs to cell indices ----
# id_order[i] is the cell ID for cell index i.
# We need to go from cell_dt$id to cell index.

id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_idx := id_to_cellidx[as.character(id)]]

# Verify no NAs (all cell IDs should be in id_order)
stopifnot(!anyNA(cell_dt$cell_idx))

# ---- Step 3: Key the data for fast joins ----
setkey(cell_dt, cell_idx, year)

# Get sorted unique years
all_years <- sort(unique(cell_dt$year))
n_years   <- length(all_years)
year_to_int <- setNames(seq_along(all_years), as.character(all_years))

cat(sprintf("Panel: %d cells × %d years = %d cell-years\n",
            length(id_order), n_years, nrow(cell_dt)))

# ---- Step 4: Vectorized neighbor aggregation per variable ----
# Strategy: For each variable, join edge_dt with cell_dt to get neighbor values,
# then group-by (from_cell_idx, year) to compute max, min, mean.
# This is done entirely in data.table with vectorized C-level operations.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
  
  # Extract the relevant columns for the "to" (neighbor) side:
  # We need (cell_idx, year, value) for the neighbor cells.
  neighbor_vals <- cell_dt[, .(to_cell_idx = cell_idx, year, val = get(var_name))]
  setkey(neighbor_vals, to_cell_idx, year)
  
  # Expand edges × years: for each edge (from, to), for each year,
  # look up the neighbor's (to) value.
  # Instead of a full cross join (expensive), we join edge_dt with neighbor_vals.
  
  # Create edge_dt with the "to" key for joining
  # For each (from_cell_idx, to_cell_idx) pair, we need all years.
  # But we can do this efficiently: join neighbor_vals onto edges.
  
  # Approach: create (from_cell_idx, to_cell_idx, year, val) by joining
  # edge_dt[, .(from_cell_idx, to_cell_idx)] with neighbor_vals on to_cell_idx and year.
  # We need to replicate each edge across all years the "to" cell has data for.
  
  # Efficient: merge edge_dt with neighbor_vals on to_cell_idx
  # This gives us (from_cell_idx, to_cell_idx, year, val) for all edge-year combos
  # where the neighbor has data.
  
  edge_vals <- merge(edge_dt, neighbor_vals, by = "to_cell_idx", allow.cartesian = TRUE)
  # edge_vals columns: to_cell_idx, from_cell_idx, year, val
  
  # Now aggregate: for each (from_cell_idx, year), compute max, min, mean of val
  # excluding NAs.
  stats <- edge_vals[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     by = .(from_cell_idx, year)]
  
  # Rename columns to match original naming convention
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(stats, "from_cell_idx", "cell_idx")
  
  # Join back to cell_dt
  setkeyv(stats, c("cell_idx", "year"))
  
  # Remove old columns if they exist (for idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt <- merge(cell_dt, stats, by = c("cell_idx", "year"), all.x = TRUE)
  setkey(cell_dt, cell_idx, year)
  
  cell_dt
}

cat("Computing neighbor features...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_dt <- compute_neighbor_features_fast(cell_dt, edge_dt, var_name)
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor feature computation complete in %.1f seconds.\n", elapsed))

# ---- Step 5: Clean up helper column and convert back if needed ----
cell_dt[, cell_idx := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_dt)

# If downstream code expects a data.table, just reassign:
cell_data <- cell_dt

# ---- Step 6: Apply the pre-trained Random Forest model ----
# The model object (e.g., `rf_model`) is already in memory.
# Predict using the updated cell_data with new neighbor features.
# Example:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete. Ready for Random Forest prediction.\n")
```

---

## Memory-Optimized Variant (If 16 GB RAM Is Tight)

The `merge(edge_dt, neighbor_vals, ..., allow.cartesian = TRUE)` in the above creates a table of ~1.37M edges × 28 years ≈ **38.4M rows** per variable. At ~32 bytes/row, that's ~1.2 GB per variable — feasible but snug on 16 GB with a 6.46M-row main table already in memory.

If memory is a concern, process **one year at a time**:

```r
compute_neighbor_features_by_year <- function(cell_dt, edge_dt, var_name) {
  cat(sprintf("  Computing neighbor stats for: %s (year-by-year)\n", var_name))
  
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  # Pre-allocate result columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  all_years <- sort(unique(cell_dt$year))
  
  for (yr in all_years) {
    # Subset to this year
    yr_dt <- cell_dt[year == yr, .(cell_idx, val = get(var_name))]
    setkey(yr_dt, cell_idx)
    
    # Join edges with neighbor values for this year
    # edge_dt: (from_cell_idx, to_cell_idx)
    # We want val of to_cell_idx
    edge_yr <- merge(edge_dt, yr_dt, by.x = "to_cell_idx", by.y = "cell_idx")
    # edge_yr: (to_cell_idx, from_cell_idx, val)
    
    # Aggregate per from_cell_idx
    stats_yr <- edge_yr[!is.na(val),
                        .(nb_max  = max(val),
                          nb_min  = min(val),
                          nb_mean = mean(val)),
                        by = .(from_cell_idx)]
    
    if (nrow(stats_yr) > 0) {
      # Update cell_dt in place for this year
      # Create a join key
      setkey(stats_yr, from_cell_idx)
      
      # Get row indices in cell_dt for this year
      yr_rows <- cell_dt[year == yr, which = TRUE]
      yr_cellidx <- cell_dt$cell_idx[yr_rows]
      
      # Match
      m <- match(yr_cellidx, stats_yr$from_cell_idx)
      matched <- !is.na(m)
      
      set(cell_dt, i = yr_rows[matched], j = max_col,  value = stats_yr$nb_max[m[matched]])
      set(cell_dt, i = yr_rows[matched], j = min_col,  value = stats_yr$nb_min[m[matched]])
      set(cell_dt, i = yr_rows[matched], j = mean_col, value = stats_yr$nb_mean[m[matched]])
    }
  }
  
  cell_dt
}

# Usage:
cat("Computing neighbor features (memory-optimized)...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_dt <- compute_neighbor_features_by_year(cell_dt, edge_dt, var_name)
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds.\n", elapsed))
```

**Per-year memory:** ~1.37M edge rows × ~24 bytes ≈ 33 MB per iteration — negligible.

---

## Why This Preserves Numerical Equivalence

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` where `idx` = row indices of neighbor cell-years | `merge(edge_dt, yr_dt, ...)` joining on `to_cell_idx` for same year | Same neighbor values selected |
| `neighbor_vals[!is.na(neighbor_vals)]` | `edge_yr[!is.na(val), ...]` | Same NA filtering |
| `max(neighbor_vals)` | `max(val)` in `data.table` group-by | Identical IEEE 754 result |
| `min(neighbor_vals)` | `min(val)` in `data.table` group-by | Identical |
| `mean(neighbor_vals)` | `mean(val)` in `data.table` group-by | Identical (same sum/count) |
| Cells with no valid neighbors → `NA, NA, NA` | Left join → `NA` for unmatched rows | Identical |

The pre-trained Random Forest model is never modified — only the feature engineering pipeline is optimized. The model receives identically-valued predictor columns and produces identical predictions.

---

## Expected Performance

| Step | Time Estimate |
|---|---|
| Build edge list (1.37M edges) | < 2 seconds |
| Cell index mapping | < 1 second |
| Neighbor stats per variable per year (~1.37M edge joins + group-by on ~344K groups) | ~1–3 seconds |
| Total: 5 variables × 28 years | **~2–7 minutes** |
| **Speedup vs. original 86+ hours** | **~700–2,500×** |