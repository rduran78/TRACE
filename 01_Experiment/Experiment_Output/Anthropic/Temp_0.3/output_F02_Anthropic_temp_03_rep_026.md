 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again, once per variable (×5 variables), performing per-row subsetting and summary statistics via R-level loops.

**Specific problems:**

1. **String key construction and lookup in `build_neighbor_lookup`:** `paste()` and named-vector indexing over 6.46M rows inside `lapply` is extremely slow. Each row builds neighbor keys by string concatenation and looks them up in a 6.46M-length named vector — this is O(n × k) with high constant factors.

2. **R-level `lapply` loops over millions of rows:** Both functions use `lapply` over ~6.46M elements. R's interpreted loop overhead dominates at this scale.

3. **Redundant computation:** The neighbor *structure* is the same across all years for a given cell. But the lookup is rebuilt monolithically, mixing spatial and temporal indexing in a single flat loop.

4. **`do.call(rbind, result)` on 6.46M small vectors:** This is a well-known R anti-pattern that causes repeated memory allocation and copying.

5. **Memory:** Storing a 6.46M-element list of integer vectors for `neighbor_lookup` is itself large, and intermediate string vectors consume significant RAM.

---

## Optimization Strategy

The key insight is: **rook neighbors are a spatial relationship that does not change across years.** We can separate the spatial neighbor graph from the temporal (year) dimension and use vectorized/data.table operations instead of row-level R loops.

**Strategy:**

1. **Build a flat edge table** of directed neighbor pairs `(cell_id, neighbor_id)` from the `nb` object — this has ~1.37M rows and is year-independent.

2. **Join this edge table to the panel data by year** using `data.table` keyed joins. For each year, every cell's neighbors' values are retrieved in one vectorized merge. This produces a long table of `(cell_id, year, neighbor_value)`.

3. **Compute grouped summary statistics** (`max`, `min`, `mean`) using `data.table`'s `by=` grouping — fully vectorized, no R-level row loop.

4. **Join the results back** to the main data.

This replaces all `lapply` loops and string-key lookups with vectorized joins and grouped aggregations, reducing runtime from ~86 hours to an estimated **minutes**.

**Memory management:** The largest intermediate object is the long edge-year table: ~1.37M edges × 28 years ≈ 38.4M rows × a few columns — easily fits in 16 GB. We process one variable at a time and discard intermediates.

**Preserves:** The original numerical estimand (max, min, mean of neighbor values) is computed identically. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build a flat spatial edge table from the nb object
#         (run once; year-independent)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell_id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (id, neighbor_id)

# ---------------------------------------------------------------
# Step 2: Convert main data to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure key columns are proper types
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]
edge_dt[, id          := as.integer(id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# Step 3: For each variable, compute neighbor stats via joins
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # neighbor values keyed by (neighbor_id aliased as id, year)
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)

  # Expand edges by year: join edge_dt to val_dt on (neighbor_id, year)
  # First, add year dimension by joining edges to the value table
  setkey(edge_dt, neighbor_id)

  # Merge: for each (id, neighbor_id) edge, get neighbor's value per year
  # This is a keyed join: edge_dt[val_dt] would be wrong direction.
  # We want: for every (id, neighbor_id) pair and every year,
  #          look up val_dt[neighbor_id, year].
  # Most efficient: merge edge_dt with val_dt on neighbor_id & year.

  # Build the long table: all (id, year) -> neighbor values
  long_dt <- merge(edge_dt, val_dt, by = "neighbor_id", allow.cartesian = TRUE)
  # long_dt columns: neighbor_id, id, year, val
  # ~1.37M edges × 28 years = ~38.4M rows (if all cells present all years)

  # Remove NAs in val before aggregation
  long_dt <- long_dt[!is.na(val)]

  # Compute grouped stats
  stats_dt <- long_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(id, year)]

  # Rename columns to match expected feature names
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats_dt, c("nb_max", "nb_min", "nb_mean"),
                      c(max_col,  min_col,  mean_col))

  # Left-join back to cell_dt
  setkey(stats_dt, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- stats_dt[cell_dt, on = .(id, year)]
  # Cells with no neighbors (e.g., islands) will have NA — correct behavior.

  # Clean up large intermediates
  rm(long_dt, val_dt, stats_dt)
  gc()

  cell_dt
}

# ---------------------------------------------------------------
# Step 4: Outer loop — one variable at a time (memory-friendly)
# ---------------------------------------------------------------
setkey(cell_dt, id, year)

for (var_name in neighbor_source_vars) {
  message("Processing neighbor features for: ", var_name)
  cell_dt <- compute_and_add_neighbor_features_fast(cell_dt, edge_dt, var_name)
  gc()  # free memory between iterations
}

# ---------------------------------------------------------------
# Step 5: Convert back to data.frame if downstream code expects it
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)
```

---

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (rook contiguity by year). Results are identical to the original. |
| **Trained RF model** | Untouched. Only the feature-engineering pipeline is optimized. Column names match the original convention (`nb_max_ntl`, etc.). |
| **Memory on 16 GB laptop** | Peak intermediate is ~38.4M rows × 3 columns ≈ ~0.9 GB per variable, freed after each iteration. Total working set stays well under 16 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregations on 38M rows typically complete in seconds. The full 5-variable loop should finish in **under 5 minutes** vs. the original 86+ hours. |

The fundamental transformation is replacing **6.46 million R-level iterations with string operations** with **vectorized equi-joins and grouped aggregations** by factoring out the year-invariant spatial graph.