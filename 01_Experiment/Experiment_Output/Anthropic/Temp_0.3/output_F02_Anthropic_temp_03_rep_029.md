 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with millions of keys is extremely slow. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, once per variable. Combined:

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing string concatenation and named-vector subsetting against a 6.46M-length named vector. This is the dominant cost — estimated at tens of hours alone.
2. **`compute_neighbor_stats`**: Called 5 times (once per variable), each time iterating 6.46M rows with `lapply` and `do.call(rbind, ...)` on a list of 6.46M small vectors. The `do.call(rbind, ...)` on millions of 3-element vectors is also very slow.
3. **Memory**: Storing a 6.46M-element list of integer vectors (the neighbor lookup) plus intermediate string vectors consumes significant RAM on a 16 GB machine.

**Root causes in summary:**
- Row-level `lapply` loops over millions of rows in pure R.
- Repeated string construction (`paste`) and named-vector lookups at scale.
- `do.call(rbind, list_of_millions)` is notoriously slow.
- No vectorization or use of efficient join/merge operations.

---

## Optimization Strategy

### Key Insight
The neighbor lookup is **year-invariant**: the spatial neighbor structure is the same for every year. We should exploit this by separating the spatial topology from the temporal dimension. Instead of building a 6.46M-row lookup, we build a ~344K-cell spatial neighbor edge list once, then use **vectorized merge/join operations** via `data.table` to compute neighbor statistics across all years simultaneously.

### Steps

1. **Convert the `spdep::nb` neighbor list into a flat edge-list `data.table`** with columns `(id, neighbor_id)`. This has ~1.37M rows — tiny and fast.
2. **Join the edge list to the panel data by `(neighbor_id, year)`** to pull neighbor values for all cell-years in one vectorized merge. This produces a long table of ~(1.37M × 28) ≈ 38.5M rows.
3. **Group-by aggregate** `(id, year)` to compute `max`, `min`, `mean` in one pass per variable using `data.table`'s optimized grouped operations.
4. **Join the aggregated stats back** to the main data.
5. Repeat for each of the 5 variables (or batch them).

This replaces ~86 hours of row-wise R loops with a handful of vectorized `data.table` joins and group-by operations that should complete in **minutes**.

### Why This Preserves Correctness
- The neighbor relationships are identical (same `rook_neighbors_unique` nb object).
- The statistics computed (`max`, `min`, `mean` of non-NA neighbor values) are identical.
- The main data and the trained Random Forest model are untouched.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Convert spdep::nb object to a flat edge-list
# ============================================================
# id_order is the vector of cell IDs corresponding to positions
# in rook_neighbors_unique (the nb object).

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer vectors (positional indices)
  # id_order maps position -> cell id
  from_ids <- rep(id_order, times = lengths(neighbors))
  to_positions <- unlist(neighbors)
  to_ids <- id_order[to_positions]
  
  data.table(id = from_ids, neighbor_id = to_ids)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (id, neighbor_id)

# ============================================================
# STEP 2: Convert main data to data.table (in place if possible)
# ============================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist and are properly typed
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# ============================================================
# STEP 3: Compute neighbor features for all variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal lookup table: (id, year, var1, var2, ...) for neighbor values
# We only need the neighbor source variables plus id and year for the join.
neighbor_val_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..neighbor_val_cols]

# Rename 'id' to 'neighbor_id' so we can join on the neighbor side
setnames(neighbor_vals_dt, "id", "neighbor_id")

# ============================================================
# STEP 4: Join edge list with panel data to get all neighbor
#          observations across all years (vectorized)
# ============================================================
# For each (id, year), we want the values of all neighbors in that same year.
# Join: edge_dt[neighbor_vals_dt] on neighbor_id, then we get
#       (id, neighbor_id, year, ntl, ec, ...)

# Set keys for fast join
setkey(edge_dt, neighbor_id)
setkey(neighbor_vals_dt, neighbor_id)

# This is an inner join: for each edge (id, neighbor_id), attach all years
# of data for that neighbor_id.
# Result: one row per (id, neighbor_id, year) combination.
expanded <- edge_dt[neighbor_vals_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
# expanded has columns: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
# Approximate rows: ~1.37M edges × 28 years = ~38.4M rows

# ============================================================
# STEP 5: Aggregate neighbor stats per (id, year) for each var
# ============================================================
# We compute max, min, mean of non-NA neighbor values grouped by (id, year).

# Build aggregation expressions dynamically
agg_exprs <- lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
})

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

agg_calls <- unlist(agg_exprs, recursive = FALSE)

# Build a single aggregation call
# Using a simpler, robust approach:
agg_stats <- expanded[, {
  out <- list()
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
    } else {
      out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
      out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
      out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
    }
  }
  out
}, by = .(id, year)]

# ============================================================
# STEP 6: Faster alternative using pre-built expressions
#          (replaces STEP 5 if the loop-in-j is too slow)
# ============================================================
# For better data.table optimization, use explicit column expressions:

# Uncomment below and comment out STEP 5 if preferred:
#
# agg_stats <- expanded[, .(
#   neighbor_ntl_max         = ifelse(all(is.na(ntl)), NA_real_, max(ntl, na.rm = TRUE)),
#   neighbor_ntl_min         = ifelse(all(is.na(ntl)), NA_real_, min(ntl, na.rm = TRUE)),
#   neighbor_ntl_mean        = mean(ntl, na.rm = TRUE),
#   neighbor_ec_max          = ifelse(all(is.na(ec)), NA_real_, max(ec, na.rm = TRUE)),
#   neighbor_ec_min          = ifelse(all(is.na(ec)), NA_real_, min(ec, na.rm = TRUE)),
#   neighbor_ec_mean         = mean(ec, na.rm = TRUE),
#   neighbor_pop_density_max = ifelse(all(is.na(pop_density)), NA_real_, max(pop_density, na.rm = TRUE)),
#   neighbor_pop_density_min = ifelse(all(is.na(pop_density)), NA_real_, min(pop_density, na.rm = TRUE)),
#   neighbor_pop_density_mean= mean(pop_density, na.rm = TRUE),
#   neighbor_def_max         = ifelse(all(is.na(def)), NA_real_, max(def, na.rm = TRUE)),
#   neighbor_def_min         = ifelse(all(is.na(def)), NA_real_, min(def, na.rm = TRUE)),
#   neighbor_def_mean        = mean(def, na.rm = TRUE),
#   neighbor_usd_est_n2_max  = ifelse(all(is.na(usd_est_n2)), NA_real_, max(usd_est_n2, na.rm = TRUE)),
#   neighbor_usd_est_n2_min  = ifelse(all(is.na(usd_est_n2)), NA_real_, min(usd_est_n2, na.rm = TRUE)),
#   neighbor_usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
# ), by = .(id, year)]

# ============================================================
# STEP 7: Merge aggregated neighbor features back to cell_data
# ============================================================
setkey(agg_stats, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

cell_data <- agg_stats[cell_data, on = .(id, year)]
# Left join: all rows of cell_data are preserved.
# Cells with no neighbors will have NA for neighbor stats (correct behavior).

# ============================================================
# STEP 8: Clean up large intermediate objects to free RAM
# ============================================================
rm(expanded, neighbor_vals_dt, agg_stats, edge_dt)
gc()

# cell_data now has all 15 neighbor feature columns appended.
# The trained Random Forest model is untouched.
# The numerical values are identical to the original implementation.
```

---

## Expected Performance

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M row-wise `lapply` with string ops | Eliminated; replaced by ~1.37M-row edge list built in <1 sec |
| `compute_neighbor_stats` | 5 × 6.46M row-wise `lapply` + `do.call(rbind)` | 1 vectorized `data.table` grouped aggregation over ~38.4M rows |
| **Estimated wall time** | **86+ hours** | **5–15 minutes** |
| **Peak RAM** | High (6.46M-element list + string vectors) | ~3–5 GB for the expanded join table (fits in 16 GB) |

### Why It's Correct
- The edge list is derived from the same `rook_neighbors_unique` nb object and `id_order`.
- The join on `(neighbor_id, year)` replicates the exact same lookup the original code performed row by row.
- `max`, `min`, `mean` with `na.rm = TRUE` and the `all(is.na(...))` guard reproduce the original `c(NA, NA, NA)` fallback for cells with no valid neighbor values.
- The trained Random Forest model is never touched — only the input feature table is augmented.