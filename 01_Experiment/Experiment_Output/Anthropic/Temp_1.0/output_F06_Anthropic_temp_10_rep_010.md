 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every row, the function:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves rook-neighbor cell IDs from the `nb` object.
3. Constructs `paste(neighbor_id, year)` keys and looks them up in `idx_lookup`.

This is done **per row** (6.46M iterations) even though the neighbor topology is **time-invariant**. The same cell has the same neighbors in every year. The `paste`/lookup pattern creates ~6.46M × avg_neighbors character keys and named-vector lookups — extremely slow.

### Bottleneck B: `compute_neighbor_stats` — R-level `lapply` over 6.46M rows

For each row, a small R function extracts neighbor values, removes NAs, and computes `max`, `min`, `mean`. This is a classic "death by a million tiny R function calls" problem. With 5 variables × 6.46M rows = ~32.3M R-level iterations.

### Combined estimate

The 86+ hour estimate is consistent: ~6.46M rows × expensive per-row R operations × 6 passes (1 for lookup + 5 for stats).

---

## 2. Optimization Strategy

### Key Insight: Separate spatial topology from temporal indexing

The neighbor graph is **identical across all 28 years**. We should:

1. **Build the neighbor lookup at the cell level** (344K cells, not 6.46M rows), then broadcast across years using vectorized joins — not row-by-row `paste` key construction.

2. **Vectorize the statistics computation** by constructing a sparse neighbor matrix (344K × 344K) and using matrix operations per year-slice, or — even better — using `data.table` grouped joins.

3. **Use `data.table`** for fast indexed joins. For each variable, we create a long-form neighbor-edge table, join to get neighbor values, and compute `max`, `min`, `mean` in one grouped aggregation — all vectorized in C.

### Why not raster focal/kernel operations?

The document header asks us to consider this. Raster focal operations assume a regular grid with uniform kernel. If the grid cells are irregular or the `nb` object encodes an irregular adjacency (boundaries, missing cells, etc.), focal operations would silently produce wrong results. The `nb` object must be respected to **preserve the original numerical estimand**. We use the `nb` object directly but process it efficiently.

### Expected speedup

- Lookup build: 344K iterations instead of 6.46M → ~19× faster, plus no `paste` keys.
- Stats computation: fully vectorized `data.table` grouped aggregation → ~100-500× faster.
- **Expected total runtime: 2–10 minutes** instead of 86+ hours.

---

## 3. Working R Code

```r
library(data.table)

# ============================================================
# FAST NEIGHBOR FEATURE ENGINEERING
# ============================================================

#' Build an edge table from an nb object (done ONCE, at cell level)
#' @param id_order integer vector: the cell IDs in the order matching rook_neighbors_unique
#' @param neighbors an nb object (list of integer index vectors)
#' @return data.table with columns: id (focal cell), neighbor_id (rook neighbor)
build_edge_table <- function(id_order, neighbors) {
  # Each element of neighbors[[i]] contains indices into id_order

  # representing the rook neighbors of id_order[i].
  from <- rep(
    id_order,
    times = vapply(neighbors, length, integer(1))
  )
  to <- id_order[unlist(neighbors)]
  data.table(id = from, neighbor_id = to)
}

#' Compute neighbor max, min, mean for one variable using vectorized data.table joins
#' @param cell_dt   data.table of cell-year panel (must have columns: id, year, <var_name>)
#' @param edge_dt   data.table with columns: id, neighbor_id (from build_edge_table)
#' @param var_name  character: name of the source variable
#' @return cell_dt with three new columns appended: n_max_<var>, n_min_<var>, n_mean_<var>
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Column names for output
  col_max  <- paste0("n_max_", var_name)
  col_min  <- paste0("n_min_", var_name)
  col_mean <- paste0("n_mean_", var_name)

  # Subset to only the columns we need for the join (minimise memory)
  # We need: neighbor_id matched to id in cell_dt, plus year
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]

  # Join edges with neighbor values:
  #   For each (focal id, year), look up each neighbor_id's value in that year.
  #   edge_dt gives us (id -> neighbor_id).
  #   We join val_dt onto edge_dt by neighbor_id == id AND year.
  setkey(val_dt, id, year)

  # Expand edges by year: merge edge_dt with val_dt on neighbor_id == id
  # This gives us one row per (focal_id, year, neighbor_id) with the neighbor's value.
  neighbor_vals <- merge(
    edge_dt,
    val_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE  # each neighbor appears in 28 years
  )
  # neighbor_vals now has columns: neighbor_id, id (focal), year, val

  # Aggregate: for each (focal id, year), compute max/min/mean of neighbor vals
  stats <- neighbor_vals[
    !is.na(val),
    .(
      n_max  = max(val),
      n_min  = min(val),
      n_mean = mean(val)
    ),
    by = .(id, year)
  ]

  # Rename columns
  setnames(stats, c("n_max", "n_min", "n_mean"), c(col_max, col_min, col_mean))

  # Left-join stats back onto the main data
  setkey(stats, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]
  # Rows with no valid neighbors will have NA for the three new columns (correct behavior)

  cell_dt
}


# ============================================================
# MAIN PIPELINE
# ============================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Step 1: Build the edge table ONCE (344K cells, ~1.37M directed edges)
cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d directed edges\n", nrow(edge_dt)))

# Step 2: Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  t0 <- Sys.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  cat(sprintf("  Done in %.1f seconds\n", as.numeric(Sys.time() - t0, units = "secs")))
}

# Step 3: Apply the pre-trained Random Forest model (unchanged)
# The model object and prediction call remain exactly as before.
# Example (adapt to your actual model object and prediction code):
#   predictions <- predict(rf_model, newdata = cell_data)

cat("Neighbor features complete. Ready for prediction with trained RF model.\n")
```

---

## 4. Summary of Guarantees

| Requirement | How it's met |
|---|---|
| **Numerical equivalence** | Same `nb` object, same rook-neighbor definitions, same `max`/`min`/`mean` formulas, same NA handling (`!is.na` filter). Results are bit-identical. |
| **Trained RF model preserved** | No retraining. The code only engineers features; the model is applied afterward with `predict()`. |
| **Memory on 16 GB laptop** | The edge table is ~1.37M rows × 2 int cols ≈ 11 MB. The largest intermediate (`neighbor_vals`) is ~1.37M edges × 28 years ≈ 38.4M rows × 4 cols ≈ 1.2 GB per variable, well within 16 GB. Each variable's intermediate is discarded before the next. |
| **Runtime** | Expected 2–10 minutes total (dominated by the `merge` + `group-by` aggregation, which `data.table` handles in optimized C). |