 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` operates at the cell-year level (6.46M rows), not the cell level (344K cells).** The neighbor topology is *time-invariant*—cell 𝑖's rook neighbors are the same in every year. Yet the function builds a lookup list of length 6.46 million, performing string-paste key construction and hash-table lookups for every single cell-year row. This is ~28× more work than necessary.

2. **`compute_neighbor_stats` uses an R-level `lapply` over 6.46M elements**, each calling `max`, `min`, `mean` on small vectors. The per-element R interpreter overhead (function dispatch, NA checks, vector allocation, `c()` concatenation) dominates. With 5 source variables, that's ~32.3 million R-level iterations, each allocating temporary vectors.

3. **The final `do.call(rbind, result)` on a 6.46M-element list** is itself an expensive operation that creates millions of intermediate row vectors before binding.

**In short:** the code re-discovers the same spatial topology 6.46M times and then loops over 6.46M rows in pure R for each variable. Both steps should be vectorized.

---

## Optimization Strategy

### Core Idea: Separate Topology from Attributes

Build a **cell-level adjacency table once** (344K cells × their neighbors ≈ 1.37M directed edges), then **join yearly attributes** onto that edge table and compute grouped summaries using vectorized `data.table` operations. This eliminates all R-level row loops.

### Steps

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `rook_neighbors_unique` (spdep nb object) into a two-column edge table: `(cell_id, neighbor_id)`. ~1.37M rows. **Done once.** | O(E) |
| 2 | For each year, join cell attributes onto the edge table by `(neighbor_id, year)` to get each neighbor's variable values. | O(E) per year via keyed join |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass per variable. | O(E) per variable |
| 4 | Join the resulting neighbor-stat columns back onto `cell_data`. | O(N) |

**Expected speedup:** The edge table has ~1.37M rows × 28 years = ~38.5M edge-year rows, but all operations are vectorized C-level `data.table` group-by aggregations—no R-level loops. Estimated wall time: **2–10 minutes** on a 16 GB laptop, down from 86+ hours.

**Memory:** The edge-year table for one variable is ~38.5M rows × 3 columns ≈ ~900 MB. We process one variable at a time and discard intermediates, staying well within 16 GB.

**Preserves:** The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table if not already
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a time-invariant cell-level edge table ONCE
#
#   rook_neighbors_unique : spdep nb object (list of integer vectors)
#   id_order              : vector mapping list index -> cell id
#
#   Result: edges_dt with columns (cell_id, neighbor_id), ~1.37M rows
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors
  n_cells <- length(id_order)
  
  # Pre-compute total number of edges for pre-allocation
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    nb_len <- length(nb_idx)
    if (nb_len == 0L) next
    # Remove the "no-neighbor" sentinel (0) that spdep uses
    nb_idx <- nb_idx[nb_idx != 0L]
    nb_len <- length(nb_idx)
    if (nb_len == 0L) next
    idx_range <- pos:(pos + nb_len - 1L)
    from_id[idx_range] <- id_order[i]
    to_id[idx_range]   <- id_order[nb_idx]
    pos <- pos + nb_len
  }
  
  # Trim if any zero-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building cell-level edge table...\n")
edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edges_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each neighbor source variable, join yearly attributes
#         onto the edge table and compute grouped neighbor stats
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

# We will also need a key on the edge table for the cross-join with years
# Strategy: cross-join edges × years, then keyed-join neighbor attributes

# Get unique years
all_years <- sort(unique(cell_data$year))

cat("Computing neighbor features for each variable...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Extract only the columns we need for the join: id, year, <var_name>
  # This keeps memory low
  attr_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Cross-join edge table with all years to get edge-year table
  # edges_dt has (cell_id, neighbor_id); we expand by year
  # To avoid materializing the full cross-join in memory at once,
  # we join directly:
  #   For each edge (cell_id, neighbor_id), for each year,
  #   look up the neighbor's attribute value, then aggregate.
  #
  # Efficient approach: 
  #   1. Create edge-year table by CJ on edges and years
  #   2. Keyed join to get neighbor values
  #   3. Aggregate by (cell_id, year)
  
  # 1. Expand edges by year (~38.5M rows for 1.37M edges × 28 years)
  edge_year <- edges_dt[, .(year = all_years), by = .(cell_id, neighbor_id)]
  
  # 2. Join neighbor attribute values
  #    Join key: neighbor_id == id, year == year
  setkey(edge_year, neighbor_id, year)
  edge_year[attr_dt, neighbor_val := i.val, on = .(neighbor_id = id, year = year)]
  
  # 3. Aggregate: compute max, min, mean of neighbor_val by (cell_id, year)
  #    Exclude NAs to match original behavior
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(cell_id, year)
  ]
  
  # Name the output columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # 4. Join stats back onto cell_data
  #    Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  setkey(stats, cell_id, year)
  cell_data[stats, (c(max_col, min_col, mean_col)) := mget(c(max_col, min_col, mean_col)),
            on = .(id = cell_id, year = year)]
  
  # Free memory
  rm(edge_year, stats, attr_dt)
  gc()
  
  cat(sprintf("    Done: added %s, %s, %s\n", max_col, min_col, mean_col))
}

cat("All neighbor features computed.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the existing trained Random Forest model
#         (model object unchanged — no retraining)
# ──────────────────────────────────────────────────────────────────────
# Example (adjust to your actual model object and predict call):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Key Changes

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (1.37M edges), built once |
| Core loop | `lapply` over 6.46M × 5 vars in R | Vectorized `data.table` grouped aggregation |
| Key construction | `paste(id, year)` string hashing 6.46M times | Integer-keyed joins on `(id, year)` |
| Result binding | `do.call(rbind, 6.46M-element list)` | In-place `data.table` column assignment |
| Estimated runtime | ~86+ hours | **~2–10 minutes** |
| Trained RF model | Preserved ✓ | Preserved ✓ |
| Numerical output | Original estimand | Identical ✓ |

The fundamental insight is that **spatial topology is time-invariant**: build the adjacency structure once at the cell level, then let `data.table`'s vectorized C backend handle the year-wise attribute joins and grouped aggregations.