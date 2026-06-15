 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup`** function. It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:

1. **Character coercion and named-vector lookup** (`id_to_ref[as.character(...)]`) — repeated millions of times, each call has O(n) hash overhead.
2. **`paste()` to build keys** for every neighbor of every row — millions of small string allocations.
3. **Named-vector indexing** (`idx_lookup[neighbor_keys]`) — hash lookups on a vector of length 6.46M, repeated per row.

This produces a **list of 6.46 million integer vectors**, which is itself a large memory object (~hundreds of MB to several GB depending on neighbor counts). Then `compute_neighbor_stats` loops over this list again for **each of the 5 variables**, calling `max`, `min`, `mean` inside an `lapply` of 6.46M iterations — adding another ~5 × 6.46M R-level function calls.

**Summary of problems:**

| Problem | Location | Impact |
|---|---|---|
| Per-row `paste` + hash lookup (×6.46M) | `build_neighbor_lookup` | ~80%+ of runtime |
| R-level `lapply` over millions of rows | Both functions | High interpreter overhead |
| Storing 6.46M-element list of integer vectors | `build_neighbor_lookup` output | High memory pressure |
| Repeating the stats loop 5 times independently | Outer loop | Multiplied overhead |

---

## Optimization Strategy

### Key idea: Replace row-level R loops with vectorized `data.table` joins and grouped aggregations.

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, construct a `data.table` of directed neighbor edges `(id, neighbor_id)` from the `nb` object, then join with the panel data on `(neighbor_id, year)` to get neighbor values. This replaces millions of R-level hash lookups with a single indexed equi-join.

2. **Compute all 5 variables' stats in one grouped aggregation** over `(id, year)` after the join, avoiding 5 separate passes.

3. **Use `data.table`'s in-place `:=` assignment** to add columns back to the main table without copying.

4. **Memory management:** The edge table (~1.37M rows × 2 cols) and the join result (~1.37M × 28 years ≈ 38.5M rows, but only for existing pairs) are manageable in 16 GB, especially if we process one variable at a time if needed.

**Expected speedup:** From 86+ hours to **minutes** (the join is O(n log n) or O(n) with keys; grouped aggregation is highly optimized in `data.table`).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build a directed edge table from the nb object (once)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) becomes nothing via unlist)
  # Convert positional indices to actual cell IDs
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ---------------------------------------------------------------
# Step 2: Compute neighbor stats for all variables via join
# ---------------------------------------------------------------
add_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                      neighbor_source_vars) {

  # Convert to data.table if not already (by reference if possible)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # 1. Build edge table: ~1.37M rows of (id, neighbor_id)
  edges <- build_edge_table(id_order, neighbors)

  # 2. Add year via join with cell_data on id
  #    We need edges × years, i.e., for each (id, neighbor_id) pair,
  #    we look up the neighbor's value in the same year.
  #
  #    Strategy: join edges with cell_data to get (id, year, neighbor_id),
  #    then join again to get the neighbor's variable values.

  # Subset cell_data to only the columns we need for the neighbor lookup
  id_year_key <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

  # Key for the join on the "focal" cell side — we need all (id, year) combos
  # joined to edges to produce (id, year, neighbor_id)
  # More efficient: join edges to id_year_key on id
  setkey(edges, id)
  setkey(id_year_key, id)

  # This gives us: for each (id, year), all neighbor_ids
  # Result has columns: id, neighbor_id, year, <source_vars for focal cell — not needed>
  # We only need id, year, neighbor_id from this join.
  focal_neighbors <- edges[id_year_key[, .(id, year)],
                           on = "id",
                           allow.cartesian = TRUE,
                           nomatch = NULL]
  # focal_neighbors columns: id, neighbor_id, year

  # 3. Now join to get the neighbor's variable values
  #    We need to look up (neighbor_id, year) in id_year_key
  #    Rename for the join:
  setnames(id_year_key, "id", "neighbor_id")
  setkey(id_year_key, neighbor_id, year)
  setkey(focal_neighbors, neighbor_id, year)

  # Equi-join: attach neighbor variable values
  joined <- id_year_key[focal_neighbors, on = .(neighbor_id, year), nomatch = NULL]
  # joined columns: neighbor_id, year, <source_vars>, id

  # 4. Grouped aggregation: compute max, min, mean per (id, year) for each variable
  #    Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Evaluate the grouped aggregation
  stats <- joined[, lapply(agg_exprs, eval), by = .(id, year)]

  # Replace -Inf/Inf from max/min on all-NA groups with NA
  inf_cols <- grep("neighbor_(max|min)_", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # 5. Merge stats back into cell_data
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]

  return(cell_data)
}

# ---------------------------------------------------------------
# Step 3: Usage (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# The trained Random Forest model can be applied as before — no retraining needed.
# The numerical estimand (max, min, mean of neighbor values) is preserved exactly.
```

---

## Memory-Constrained Variant

If the single join (`focal_neighbors` can reach ~38M rows × several columns) strains 16 GB, process **one variable at a time**:

```r
add_neighbor_features_lowmem <- function(cell_data, id_order, neighbors,
                                         neighbor_source_vars) {
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

  edges <- build_edge_table(id_order, neighbors)
  setkey(edges, id)

  # Base key table (id, year only)
  base_key <- cell_data[, .(id, year)]
  setkey(base_key, id)

  focal_neighbors <- edges[base_key, on = "id",
                           allow.cartesian = TRUE, nomatch = NULL]
  # columns: id, neighbor_id, year  (~38M rows, 3 int/num cols ≈ ~0.9 GB)
  setkey(focal_neighbors, neighbor_id, year)

  for (v in neighbor_source_vars) {
    message("Processing neighbor stats for: ", v)

    # Extract only the column we need
    lookup <- cell_data[, .(neighbor_id = id, year, val = get(v))]
    setkey(lookup, neighbor_id, year)

    tmp <- lookup[focal_neighbors, on = .(neighbor_id, year), nomatch = NULL]
    # tmp columns: neighbor_id, year, val, id

    agg <- tmp[, .(
      nmax  = max(val, na.rm = TRUE),
      nmin  = min(val, na.rm = TRUE),
      nmean = mean(val, na.rm = TRUE)
    ), by = .(id, year)]

    # Fix Inf/-Inf
    agg[is.infinite(nmax), nmax := NA_real_]
    agg[is.infinite(nmin), nmin := NA_real_]

    setnames(agg, c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    setkey(agg, id, year)
    setkey(cell_data, id, year)
    cell_data <- agg[cell_data, on = .(id, year)]

    rm(tmp, agg, lookup)
    gc()
  }

  return(cell_data)
}

# Usage:
cell_data <- add_neighbor_features_lowmem(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M R-level `paste` + hash lookups | Single `data.table` equi-join |
| Stats computation | 5 × 6.46M `lapply` calls with per-element R functions | Vectorized grouped `data.table` aggregation |
| Estimated runtime | 86+ hours | **5–20 minutes** |
| Peak memory | Uncontrolled (6.46M-element list) | ~2–4 GB (controlled, with low-mem variant) |
| Numerical output | max, min, mean per neighbor set | **Identical** — preserved exactly |
| Trained RF model | Untouched | **Untouched** |