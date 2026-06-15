 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which performs ~6.46 million `lapply` iterations, each doing character-based key construction and named-vector lookups. Named vector lookups via character keys in R are hash-table operations, but doing millions of `paste()` calls and millions of named-vector subscripts inside a sequential `lapply` is extremely slow. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again, extracting and summarizing neighbor values per row. This is repeated **5 times** (once per variable), compounding the cost.

**Specific problems:**

1. **`build_neighbor_lookup`:** Character key construction (`paste`) and named-vector indexing (`idx_lookup[neighbor_keys]`) inside a per-row `lapply` over 6.46M rows. This is O(n × k) string operations where k is the average neighbor count, and R's overhead per iteration is high.

2. **`compute_neighbor_stats`:** Returns a list of 6.46M three-element vectors, then `do.call(rbind, ...)` on that list — a notoriously slow pattern in R for large lists.

3. **Memory:** Storing a list of 6.46M integer vectors (the neighbor lookup) plus intermediate character vectors consumes significant RAM on a 16 GB machine.

4. **Redundant structure:** The neighbor relationships are **time-invariant** (the same cell has the same rook neighbors every year), but the lookup is built by expanding neighbors into every cell-year combination, duplicating the spatial topology 28 times.

---

## Optimization Strategy

### Key Insight: Separate Spatial Topology from Temporal Expansion

Rook neighbors are a property of **cells**, not cell-years. There are only 344,208 cells. The neighbor graph should be stored as a sparse structure over cells, and temporal expansion should be handled via vectorized joins, not per-row iteration.

### Strategy Summary

| Step | Technique | Speedup Source |
|---|---|---|
| 1 | Represent the neighbor graph as a two-column **edge list** (integer cell indices), not a per-row list. | Eliminates 6.46M list elements; enables vectorized ops. |
| 2 | Use `data.table` keyed joins instead of named-vector lookups. | O(n log n) merge vs. O(n × k) hash lookups in a loop. |
| 3 | Compute all 5 variables' neighbor stats in a **single pass** via `data.table` grouped aggregation over the edge list. | Eliminates 5 separate `lapply` passes over 6.46M rows. |
| 4 | Avoid `do.call(rbind, list_of_vectors)`. | `data.table` returns a single data.table directly. |
| 5 | Use integer cell indices and integer year throughout. | Eliminates all `paste`/character operations. |

**Expected runtime:** Minutes instead of 86+ hours. Memory stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Convert spdep nb object to an integer edge-list (one-time, fast)
# ===========================================================================
# rook_neighbors_unique is a list of length 344,208.
# id_order is the vector of cell IDs in the same order as the nb object.

build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order of i's neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Map positional indices to actual cell IDs
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows: (from_id, to_id)

# ===========================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ===========================================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns are proper types for fast joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ===========================================================================
# STEP 3: Compute all neighbor features in one vectorized pass
# ===========================================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, var_names) {
  # We need: for every (id, year) row, find its neighbors' values and compute
  # max, min, mean for each variable.
  #
  # Approach:
  #   1. Join edge_dt × cell_data on (to_id = id, year) to get neighbor values.
  #   2. Group by (from_id, year) and compute stats.
  #   3. Join results back to cell_data on (id = from_id, year).

  # Subset cell_data to only the columns we need for the neighbor lookup
  # to minimise memory during the join.
  lookup_cols <- c("id", "year", var_names)
  neighbor_vals <- cell_data[, ..lookup_cols]
  setnames(neighbor_vals, "id", "to_id")
  setkey(neighbor_vals, to_id, year)

  # Expand edge list by year: every edge exists in every year.
  # Instead of a full cross-join (which would be 1.37M × 28 = 38.4M rows),
  # we do a keyed join which is efficient.
  # Join: for each edge (from_id, to_id) and each year that from_id appears,
  # pull the neighbor (to_id) values for that year.
  #
  # Efficient approach: join edge_dt with cell_data to get (from_id, year)
  # combinations, then join to get neighbor values.

  # Get all (from_id, year) combinations that exist in the data
  from_years <- cell_data[, .(from_id = id, year)]
  setkey(from_years, from_id)

  # Add neighbor IDs: (from_id, year, to_id)
  setkey(edge_dt, from_id)
  expanded <- edge_dt[from_years, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: from_id, to_id, year
  # Approx rows: 6.46M × avg_neighbors ≈ 6.46M × 4 ≈ 25.8M (manageable)

  # Join neighbor values: attach variable values for (to_id, year)
  setkey(expanded, to_id, year)
  expanded <- neighbor_vals[expanded, on = .(to_id, year), nomatch = NA]
  # Now expanded has: to_id, year, from_id, and all var_name columns

  # Compute grouped stats
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Construct the call
  agg_list <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))

  stats <- expanded[, eval(agg_list), by = .(from_id, year)]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  setnames(stats, "from_id", "id")
  setkey(stats, id, year)

  return(stats)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# ===========================================================================
# STEP 4: Join the neighbor features back onto cell_data
# ===========================================================================
setkey(cell_data, id, year)

# Identify new columns to add (avoid overwriting if re-running)
new_cols <- setdiff(names(neighbor_stats), c("id", "year"))
existing  <- intersect(new_cols, names(cell_data))
if (length(existing) > 0) {
  cell_data[, (existing) := NULL]
}

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# ===========================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ===========================================================================
# The model object (e.g., rf_model) is already in memory or loaded from disk.
# The feature names it expects now exist in cell_data.
# Predict exactly as before — the numerical estimand is preserved because
# we computed identical max/min/mean statistics, just more efficiently.
#
# Example (adjust to your actual model object and feature matrix):
# cell_data[, predicted_gdp := predict(rf_model, newdata = cell_data)]
```

---

## Why This Preserves the Original Numerical Estimand

The optimized code computes **exactly the same three summary statistics** (max, min, mean of non-NA neighbor values) for each cell-year and each variable. The only difference is the computational pathway:

- **Original:** Per-row `lapply` → extract neighbor indices → subset values → compute stats.
- **Optimized:** Vectorized edge-list join → grouped `data.table` aggregation → same stats.

The resulting feature columns are numerically identical (up to floating-point associativity, which is negligible here since `mean` is over small neighbor sets of typically 2–4 values). The trained Random Forest model is never modified; it simply receives the same feature matrix it was trained on.

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with `paste`/hash lookup | Single vectorized edge-list construction (<1 sec) |
| Temporal expansion + join | Implicit in per-row loop | `data.table` keyed join (~25.8M rows, ~10–30 sec) |
| Neighbor stats (5 vars) | 5 × 6.46M `lapply` + `do.call(rbind)` | Single grouped aggregation (~30–60 sec) |
| **Total estimated time** | **86+ hours** | **~2–5 minutes** |
| Peak memory | Multiple large intermediate lists | ~3–5 GB (edge expansion table) |