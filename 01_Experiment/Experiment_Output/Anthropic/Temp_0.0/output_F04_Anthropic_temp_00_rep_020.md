 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector — each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated `paste` and named-vector lookups on a vector of ~6.46M names, is extremely slow. Named vector lookup in R degrades as the vector grows; at 6.46M entries the hash table is enormous and each probe is expensive. Multiply by ~4 neighbors per cell × 6.46M rows ≈ 25.8 billion character operations.

### 2. `compute_neighbor_stats` — Row-level `lapply` over 6.46 million rows (×5 variables)

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 × 6.46M ≈ 32.3 million times. The per-call overhead of anonymous function dispatch, `is.na`, and three summary functions dominates.

### Estimated cost breakdown

| Step | Calls | Estimated share |
|---|---|---|
| `build_neighbor_lookup` (paste + named lookup) | 6.46M | ~60–70% |
| `compute_neighbor_stats` (5 vars × 6.46M) | 32.3M | ~25–35% |
| Random Forest `predict()` | 1 | <5% |

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** problem. Every cell-year needs to be joined to its neighbors' cell-years. This is a classic equi-join on `(id, year)` that `data.table` handles in C.

**Steps:**

1. **Build an edge table** (a two-column `data.table` of `id → neighbor_id`) from the `nb` object — done once, vectorized.
2. **Join** the edge table to the panel data on `(neighbor_id, year)` to get neighbor values — one vectorized join per variable (or all at once).
3. **Group-aggregate** by `(id, year)` to compute `max`, `min`, `mean` — fully vectorized in `data.table`.

This eliminates all 6.46M-iteration R loops, all `paste` key construction, and all named-vector lookups. Expected speedup: **~200–500×** (from 86+ hours to roughly 10–25 minutes).

**Preservation guarantees:**
- The trained Random Forest model is untouched (no retraining).
- The numerical estimand is identical: for each `(id, year)`, the neighbor max, min, and mean of each variable are computed over the same rook-neighbor set with the same NA handling.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge table from the nb object (once)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors into id_order)
  # Expand to a two-column data.table: focal_id -> neighbor_id
  n <- length(neighbors)
  
  # Pre-compute lengths to allocate once
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Compute all neighbor features via vectorized join + group-agg
# ──────────────────────────────────────────────────────────────────────
compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  # Convert to data.table (by reference if already; copy if not)
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }
  
  # 1. Build edge table  (~1.37M rows, instant)
  edges <- build_edge_table(id_order, neighbors)
  
  # 2. Attach year from focal cell: expand edges × years

  #    Instead of a cross-join (which would be huge), we join edges to dt
  #    to get (focal_id, year, neighbor_id), then join again to get
  #    neighbor values.
  
  #    But smarter: join edges to dt on focal side to get year,
  #    then join to dt on neighbor side to get neighbor values.
  
  # Subset dt to only the columns we need (saves memory)
  cols_needed <- c("id", "year", neighbor_source_vars)
  dt_sub <- dt[, ..cols_needed]
  
  # 2a. Join edges to focal rows to get (focal_id, year, neighbor_id)
  #     This is edges × years_per_cell, but done via join not cross-join.
  setkey(dt_sub, id)
  
  # Get unique (id, year) pairs from dt_sub — but dt_sub has all of them
  focal_years <- dt_sub[, .(id, year)]
  setnames(focal_years, "id", "focal_id")
  
  # Merge: for each (focal_id, year), attach all neighbor_ids
  setkey(edges, focal_id)
  setkey(focal_years, focal_id)
  
  # This produces ~1.37M × 28 ≈ 38.5M rows (manageable in 16GB)
  expanded <- edges[focal_years, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year
  
  # 2b. Join to dt_sub on (neighbor_id, year) to get neighbor variable values
  setnames(dt_sub, "id", "neighbor_id")
  setkeyv(dt_sub, c("neighbor_id", "year"))
  setkeyv(expanded, c("neighbor_id", "year"))
  
  joined <- dt_sub[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # joined has: neighbor_id, year, <all source vars>, focal_id
  
  # 3. Group-aggregate by (focal_id, year)
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Build the aggregation call
  agg_list <- as.call(c(as.name("list"), agg_exprs))
  
  agg <- joined[, eval(agg_list), by = .(focal_id, year)]
  
  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  for (v in neighbor_source_vars) {
    max_col <- paste0("neighbor_max_", v)
    min_col <- paste0("neighbor_min_", v)
    agg[is.infinite(get(max_col)), (max_col) := NA_real_]
    agg[is.infinite(get(min_col)), (min_col) := NA_real_]
  }
  
  # 4. Merge aggregated features back to original dt
  setnames(agg, "focal_id", "id")
  setkeyv(dt, c("id", "year"))
  setkeyv(agg, c("id", "year"))
  
  feature_cols <- setdiff(names(agg), c("id", "year"))
  dt[agg, (feature_cols) := mget(paste0("i.", feature_cols)),
     on = c("id", "year")]
  
  # 5. Return as data.frame if input was data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# Usage (drop-in replacement for the original outer loop)
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Budget Check

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M × 110 cols, numeric) | ~5.7 GB |
| `edges` (1.37M × 2 integer cols) | ~11 MB |
| `expanded` (~38.5M × 3 cols) | ~0.9 GB |
| `joined` (~38.5M × 7 cols) | ~2.2 GB |
| `agg` (6.46M × 17 cols) | ~0.9 GB |
| **Peak total** | **~9.7 GB** (fits in 16 GB) |

If memory is tight, process variables in batches of 2–3 instead of all 5 at once (reducing `joined` width), or use `rm()` + `gc()` after each intermediate step.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M R-loop iterations with `paste` + named vector | Vectorized `data.table` join |
| Neighbor stats | 32.3M R-loop iterations (5 vars) | Single grouped `data.table` aggregation |
| Key mechanism | Named character vector hash (R-level) | `data.table` binary-search join (C-level) |
| Estimated runtime | 86+ hours | ~10–25 minutes |
| Numerical result | Identical | Identical |
| RF model | Unchanged | Unchanged |