 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates an O(n × k) interpreted-loop problem (n ≈ 6.46M rows, k ≈ average neighbor count ≈ 4). The secondary bottleneck is `compute_neighbor_stats`, which again loops over 6.46M entries in R, extracting and summarizing neighbor values. Together, these two functions dominate runtime because:

1. **String key construction per row**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called millions of times inside `lapply`, creating enormous GC pressure from transient character allocations.
2. **Named-vector hash lookups at scale**: `idx_lookup[neighbor_keys]` on a 6.46M-element named vector is slow per call due to R's internal hashing on character vectors.
3. **`lapply` returns a list of 6.46M elements**: The neighbor lookup list alone consumes several GB, and `do.call(rbind, ...)` on 6.46M 3-element vectors is extremely slow (repeated memory reallocation).
4. **The loop runs 5 times** (once per neighbor source variable), amplifying the `compute_neighbor_stats` cost.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized joins and grouped aggregations using `data.table`.**

The key insight: the neighbor lookup is a **join problem**. Each row `(id, year)` needs to be joined to its neighbors' rows `(neighbor_id, same year)` via a precomputed edge list. Once joined, `max`, `min`, and `mean` are grouped aggregations — exactly what `data.table` is optimized for.

**Steps:**

1. **Flatten the `nb` object into an edge-list `data.table`** with columns `(id, neighbor_id)` — done once, O(total edges ≈ 1.37M).
2. **Join the edge list to the panel data by `(neighbor_id, year)`** to get neighbor values — a single keyed merge, fully vectorized in C.
3. **Group by the focal row and compute `max`, `min`, `mean`** — a single `data.table` grouped aggregation.
4. **Process all 5 variables in one pass** over the joined table to avoid repeated joins.

This eliminates all `lapply` loops, all string-key construction, and all list-of-vectors overhead. Expected runtime: **minutes, not hours**. Memory: the join temporarily expands rows by ~4× (average neighbors), so peak ≈ 6.46M × 4 × (few columns) which fits comfortably in 16 GB if we process variables in batches.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert the spdep nb object to a data.table edge list (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# 2. Prepare the panel data as a data.table
# ---------------------------------------------------------------
# Assume cell_data is a data.frame or data.table with columns:
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (110 vars)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Create a stable row identifier to merge results back
cell_data[, .row_id := .I]

# ---------------------------------------------------------------
# 3. Vectorized neighbor feature computation
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edge_dt,
                                          neighbor_source_vars) {
  # Subset only the columns we need for the join
  join_cols <- c("id", "year", neighbor_source_vars)
  dt_slim   <- cell_data[, ..join_cols]

  # Key the slim table for fast join on (id, year)
  # We join edge_dt$neighbor_id == dt_slim$id AND same year
  # So rename for clarity:
  setnames(dt_slim, "id", "neighbor_id")
  setkey(dt_slim, neighbor_id, year)

  # Build the focal side: each row's (id, year, .row_id)
  focal <- cell_data[, .(id, year, .row_id)]

  # Merge focal with edge list to get (focal .row_id, year, neighbor_id)
  # This gives one row per (focal_row, neighbor) pair
  focal_edges <- merge(focal, edge_dt, by = "id", allow.cartesian = TRUE)
  #   columns: id, year, .row_id, neighbor_id

  # Now join to get neighbor variable values
  setkey(focal_edges, neighbor_id, year)
  focal_edges <- dt_slim[focal_edges, on = .(neighbor_id, year), nomatch = NA]
  #   columns: neighbor_id, year, <vars>, id (focal), .row_id

  # Compute grouped stats for each variable
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)),  list(v_sym = v_sym))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)),  list(v_sym = v_sym))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(as.numeric(mean(.(v_sym), na.rm = TRUE)), list(v_sym = v_sym))
  }

  # Build a single aggregation call
  # data.table allows multiple expressions in j via a list
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  stats_dt <- focal_edges[, eval(agg_call), by = .row_id]

  # Replace Inf / -Inf (from max/min on all-NA) with NA
  for (col in names(stats_dt)[-1]) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  return(stats_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats_dt <- compute_all_neighbor_features(cell_data, edge_dt,
                                          neighbor_source_vars)

# ---------------------------------------------------------------
# 4. Merge results back into cell_data
# ---------------------------------------------------------------
setkey(stats_dt, .row_id)
setkey(cell_data, .row_id)

cell_data <- stats_dt[cell_data, on = ".row_id"]

# Clean up helper column
cell_data[, .row_id := NULL]

# ---------------------------------------------------------------
# 5. Predict with the existing trained Random Forest (unchanged)
# ---------------------------------------------------------------
# The trained model object (e.g., rf_model) is used as-is.
# Ensure column names match what the model expects.
# Example:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` with `na.rm = TRUE` are identical to the original `compute_neighbor_stats` logic. The `Inf`→`NA` replacement mirrors the original `length(neighbor_vals) == 0 → NA` guard. |
| **Trained RF model** | No model retraining occurs. The code only reconstructs the same feature columns the model expects, then calls `predict()`. |
| **Neighbor definition** | The edge list is a lossless flattening of the same `rook_neighbors_unique` nb object. |

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `lapply` iterations + string ops) | ~seconds (vectorized `rep`/`unlist`) |
| Neighbor stats (5 vars) | ~hours (5 × 6.46M `lapply`) | ~minutes (one keyed join + one grouped aggregation) |
| Peak memory | ~10–14 GB (6.46M-element list + copies) | ~6–10 GB (expanded join table, freed after aggregation) |
| **Total estimated wall time** | **86+ hours** | **5–15 minutes** |

If memory is still tight on a 16 GB machine, process variables one at a time (loop over `neighbor_source_vars`, joining only one variable column per iteration) to reduce the width of the expanded join table. This trades a small amount of speed for a ~5× reduction in peak join-table size.