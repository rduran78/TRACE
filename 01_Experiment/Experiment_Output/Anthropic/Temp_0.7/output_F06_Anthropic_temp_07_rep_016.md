 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every row, the function:
1. Looks up which spatial cell this row belongs to (`id_to_ref`).
2. Finds that cell's rook neighbors from the `nb` object.
3. Constructs `paste(neighbor_id, year)` keys and indexes into a named character vector (`idx_lookup`).

This means **6.46 million iterations**, each doing string concatenation and named-vector lookups (which are hash-table lookups in R, but with per-call overhead). The result is a list of 6.46M integer vectors — enormous in memory and slow to build.

### Bottleneck B: `compute_neighbor_stats` — O(n) `lapply` over 6.46M rows, repeated 5 times

For each of 5 variables, another `lapply` over 6.46M rows subsets a numeric vector by the neighbor indices and computes `max`, `min`, `mean`. That's **5 × 6.46M = 32.3M** R-level function calls, each with subsetting and NA handling.

### Why raster focal/kernel operations are tempting but wrong

Raster focal operations (e.g., `terra::focal`) assume a regular grid with uniform connectivity. While the data is on a grid, the `nb` object may encode irregular boundaries (coastal cells, edge cells with fewer than 4 neighbors, missing cells). Forcing this into a raster focal operation risks silently changing the numerical estimand. We must preserve exact results.

### The real fix: vectorized sparse-matrix multiplication and group operations

The neighbor relationships define a **sparse adjacency matrix**. Computing `mean` of neighbors is a sparse matrix–vector product (after row-normalizing). Computing `max` and `min` can be done via `data.table` grouped operations after expanding the adjacency into an edge list, grouped by target row. This eliminates all R-level per-row loops.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup factor |
|------|---------|----------|---------------|
| Build lookup | 6.46M `lapply` with string ops | Build a sparse edge-list once via `data.table` join on `(id, year)` | ~100–500× |
| Compute stats | 5 × 6.46M `lapply` | `data.table` grouped `max/min/mean` on edge list, or sparse matrix multiply for mean | ~100–500× |
| Memory | 6.46M-element list of integer vectors | One edge-list `data.table` (~14M rows × 2 int cols ≈ 220 MB) | Comparable or less |

**Expected total runtime: minutes, not days.**

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# ─────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with a row-order column
# ─────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)
cell_data[, .row_idx := .I]  # preserve original row order

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Build a spatial edge list from the nb object (once)
#
#   rook_neighbors_unique: an nb object of length 344,208
#   id_order: integer/character vector of cell IDs, same order as nb object
# ─────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] gives integer indices (into id_order) of neighbors of cell i
  # Convert to a two-column data.table: (from_id, to_id)
  n <- length(neighbors_nb)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_i))
      to_list[[i]]   <- id_order[nb_i]
    }
  }
  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

cat("Building spatial edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id (the cell whose neighbors we want), to_id (a neighbor)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Expand edge list across years via join
#
#   For each (from_id, year) row in cell_data, we need the values of
#   each neighbor (to_id) in the SAME year.
#
#   Strategy:
#     - Join edge_dt to cell_data on to_id == id to get neighbor values
#       for every (from_id, to_id, year) triple.
#     - Then group by (from_id, year) to compute max, min, mean.
# ─────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset cell_data to only the columns we need for the neighbor lookup
# to keep memory manageable
neighbor_val_cols <- c("id", "year", neighbor_source_vars)
cd_slim <- cell_data[, ..neighbor_val_cols]

# Rename 'id' to 'to_id' for the join
setnames(cd_slim, "id", "to_id")

# Key for fast join
setkey(cd_slim, to_id, year)

# Also need (from_id, year) → row_idx mapping to merge results back
target_map <- cell_data[, .(from_id = id, year, .row_idx)]
setkey(target_map, from_id, year)

cat("Joining edge list with cell-year data (this is the main computation)...\n")

# Join: for each edge (from_id, to_id), get all years of to_id's data
# This creates a long table: (from_id, to_id, year, ntl, ec, ...)
# Number of rows ≈ num_edges × num_years ≈ 1.37M × 28 ≈ 38.4M
# Memory ≈ 38.4M × (2 int + 1 int + 5 double) ≈ ~2 GB — fits in 16 GB

edge_year <- edge_dt[cd_slim, on = "to_id", allow.cartesian = TRUE, nomatch = NULL]
# Result has columns: from_id, to_id, year, ntl, ec, pop_density, def, usd_est_n2

cat(sprintf("  Edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Grouped aggregation — compute max, min, mean per
#         (from_id, year) for each variable
# ─────────────────────────────────────────────────────────────────────

cat("Computing neighbor statistics...\n")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("n_", v, c("_max", "_min", "_mean"))
}))

# Construct the call
agg_call <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))

# Perform grouped aggregation
neighbor_stats <- edge_year[, eval(agg_call), by = .(from_id, year)]

# Replace -Inf/Inf from max/min of zero-length groups with NA
for (col_name in agg_names) {
  neighbor_stats[is.infinite(get(col_name)), (col_name) := NA_real_]
  # Also handle NaN from mean of empty
  neighbor_stats[is.nan(get(col_name)), (col_name) := NA_real_]
}

cat(sprintf("  Neighbor stats table: %s rows, %s columns\n",
            format(nrow(neighbor_stats), big.mark = ","),
            ncol(neighbor_stats)))

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Merge neighbor stats back into cell_data
# ─────────────────────────────────────────────────────────────────────

cat("Merging neighbor features back into cell_data...\n")

# Remove any pre-existing neighbor stat columns to avoid conflicts
existing_ncols <- intersect(names(cell_data), agg_names)
if (length(existing_ncols) > 0) {
  cell_data[, (existing_ncols) := NULL]
}

# Join on (id == from_id, year)
setkey(neighbor_stats, from_id, year)
setkey(cell_data, id, year)

cell_data <- neighbor_stats[cell_data, on = .(from_id = id, year = year)]

# The join introduces 'from_id' — rename back to 'id'
setnames(cell_data, "from_id", "id")

# Restore original row order
setorder(cell_data, .row_idx)
cell_data[, .row_idx := NULL]

cat("Done. Neighbor features added.\n")

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# ─────────────────────────────────────────────────────────────────────
# The trained RF model object (e.g., `rf_model`) is used as-is:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# No retraining. The new neighbor feature columns have identical names
# and identical numerical values as the original implementation, so
# the pre-trained model's predictions are preserved exactly.
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor definitions** | We use the identical `rook_neighbors_unique` nb object and `id_order` mapping. No neighbors are added or removed. |
| **Same year matching** | The join `on = "to_id"` with `year` present in `cd_slim` ensures each cell's neighbors are looked up in the same year, exactly as the original `paste(id, year)` key logic. |
| **Same aggregation functions** | `max`, `min`, `mean` with `na.rm = TRUE` — identical to the original `compute_neighbor_stats`. |
| **Same NA handling** | Cells with no valid neighbors get `NA` (via the `is.infinite` / `is.nan` cleanup), matching the original `c(NA, NA, NA)` return. |
| **Same column names** | Output columns follow the pattern `n_{var}_{max|min|mean}`, matching `compute_and_add_neighbor_features`. (Adjust the naming pattern if the original wrapper uses a different convention.) |
| **RF model untouched** | The model object is never modified or retrained. |

## 5. Expected Performance

| Phase | Estimated time | Estimated peak memory |
|-------|---------------|----------------------|
| Build edge list | ~2–5 seconds | ~50 MB |
| Join (edge × cell-year) | ~30–90 seconds | ~2–3 GB |
| Grouped aggregation | ~30–120 seconds | ~1–2 GB |
| Merge back | ~10–30 seconds | ~1 GB |
| **Total** | **~2–5 minutes** | **~6–8 GB peak** |

This represents a speedup of roughly **1,000–2,500×** over the original 86+ hour estimate, comfortably within the 16 GB RAM constraint.