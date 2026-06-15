 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine — small constant per cell).
2. Constructs character keys by pasting cell ID and year (`paste(id, year, sep="_")`).
3. Performs a named-vector lookup (`idx_lookup[neighbor_keys]`) — this is a **hash lookup on a character vector of length 6.46M**, repeated 6.46M times.

The `idx_lookup` named vector has 6.46M entries. Each call indexes into it with ~4 keys (average rook neighbors ≈ 1,373,394 / 344,208 ≈ 4). That's ~25.8M character hash lookups total, but the overhead of `lapply` over 6.46M iterations, each allocating small vectors and calling `paste`, dominates. The result is a **list of length 6.46M**, each element a small integer vector — enormous memory overhead from list structure alone.

### Bottleneck B: `compute_neighbor_stats` — O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 variables, another `lapply` iterates over 6.46M list elements, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times = ~32.3M R-level loop iterations with per-element allocation.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular grid with a fixed rectangular kernel. Here the grid cells have an irregular neighbor structure (coastal cells, boundary cells have fewer neighbors), and the data is in long panel format (cell × year), not a raster stack. Converting to raster stacks for 28 years and 5 variables is possible but would require reshaping and would lose the irregular-boundary handling. The better approach is to **vectorize the neighbor computation using sparse matrix algebra**, which is the direct analogue of a focal operation on an irregular grid.

### Summary

| Component | Current complexity | Core issue |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations + character hashing | Slow loop, character key construction |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M R-level iterations | Slow loop, per-element allocation |
| Total estimated wall time | 86+ hours | Both bottlenecks compound |

---

## 2. Optimization Strategy

### Key insight: Express neighbor aggregation as sparse matrix multiplication

A rook-neighbor adjacency can be represented as a sparse matrix **W** of dimension 344,208 × 344,208. For a given year, the neighbor-max, neighbor-min, and neighbor-mean of a variable can be computed by operating on the sparse structure directly. But `max` and `min` are not linear, so we can't use a single matrix multiply for all three. However:

- **Neighbor mean**: `W_row_normalized %*% x` (one sparse matrix multiply per year-variable).
- **Neighbor max and min**: Iterate over the sparse structure, but do it in **vectorized C-level code** via `data.table` grouping or via the sparse matrix's row structure.

The overall strategy:

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, build a spatial-only adjacency edge list (source_cell, neighbor_cell) of ~1.37M rows. Then join on year to expand to ~1.37M × 28 ≈ 38.5M edge-year rows. This is a `data.table` cross-join — fast and memory-efficient (~38.5M rows × a few columns ≈ < 1 GB).

2. **Compute all neighbor stats via `data.table` grouped aggregation.** For each (source_cell, year) group, compute max, min, mean of the neighbor values. `data.table` does this in parallel C-level code — orders of magnitude faster than R-level `lapply`.

3. **Do all 5 variables in one pass** by joining once and aggregating all variables simultaneously.

### Expected speedup

- `data.table` grouped aggregation over 38.5M rows with ~6.46M groups: **seconds to low minutes**.
- Total pipeline: **under 5 minutes** on a 16 GB laptop, down from 86+ hours.
- Memory: edge table ~38.5M rows × ~8 columns × 8 bytes ≈ 2.5 GB peak, feasible on 16 GB.

### Numerical equivalence

The `max`, `min`, and `mean` operations are computed on exactly the same neighbor sets with the same values. The results are **numerically identical** (not approximate). The trained Random Forest model is untouched.

---

## 3. Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────────────
# STEP 0: Inputs assumed available
# ─────────────────────────────────────────────────────────────────────
# cell_data            : data.frame/data.table with columns id, year,
#                        ntl, ec, pop_density, def, usd_est_n2, ...
# rook_neighbors_unique: spdep nb object (list of length 344,208)
# id_order             : integer/character vector of cell IDs in the
#                        same order as rook_neighbors_unique
# rf_model             : pre-trained Random Forest model (untouched)

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Build spatial edge list from the nb object (once)
# ─────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order of neighbors of

  # the i-th cell. We expand this into a two-column edge list of cell IDs.
  n <- length(nb_obj)
  # Pre-compute total edges for pre-allocation
  n_edges <- sum(lengths(nb_obj))

  from_idx <- rep(seq_len(n), times = lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep convention where 0L means "no neighbors"
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    source_id   = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table and set keys
# ─────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim table with only the columns we need for the neighbor join
# to minimize memory during the large join
keep_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..keep_cols]
setkey(neighbor_vals_dt, id, year)

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Expand edge list × years and join neighbor values
# ─────────────────────────────────────────────────────────────────────
# Get unique years
years <- sort(unique(cell_data$year))

# Cross-join edges with years: each spatial edge exists in every year
# This produces ~1.37M × 28 ≈ 38.5M rows
edge_year_dt <- CJ_dt <- edge_dt[, .(source_id, neighbor_id, year = rep(list(years), .N))]
# More memory-efficient approach: use CJ inside a merge
edge_year_dt <- edge_dt[, .(year = years), by = .(source_id, neighbor_id)]
cat("Edge-year rows:", nrow(edge_year_dt), "\n")

# Join to get neighbor variable values
# We join on neighbor_id + year to get the variable values of each neighbor
setkey(edge_year_dt, neighbor_id, year)
setnames(neighbor_vals_dt, "id", "neighbor_id")

edge_year_dt <- neighbor_vals_dt[edge_year_dt, on = .(neighbor_id, year)]

# Now edge_year_dt has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, source_id

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Grouped aggregation — compute max, min, mean per
#         (source_id, year) for all 5 variables at once
# ─────────────────────────────────────────────────────────────────────
# Build aggregation expressions dynamically
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

# Construct the call
agg_call <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))

# Execute grouped aggregation
neighbor_stats <- edge_year_dt[, eval(agg_call), by = .(source_id, year)]

# Handle Inf/-Inf from max/min on all-NA groups (replace with NA)
inf_cols <- grep("neighbor_(max|min)_", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")
# Expected: ~6.46M (one row per cell-year)

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Merge neighbor stats back into cell_data
# ─────────────────────────────────────────────────────────────────────
# Rename source_id back to id for the merge
setnames(neighbor_stats, "source_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# ─────────────────────────────────────────────────────────────────────
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# ─────────────────────────────────────────────────────────────────────
# The RF model expects a data.frame with the original column names.
# cell_data now contains all ~110 predictors + the 15 new neighbor features.
# Predict exactly as before:
#
# cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
#
# (Uncomment and adapt to your actual prediction call.)

# ─────────────────────────────────────────────────────────────────────
# CLEANUP: Free large intermediate objects
# ─────────────────────────────────────────────────────────────────────
rm(edge_dt, edge_year_dt, neighbor_vals_dt, neighbor_stats)
gc()
```

---

## 4. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with character hashing | Eliminated; replaced by `data.table` keyed join |
| **Neighbor stats** | 5 × 6.46M-iteration `lapply` with per-element allocation | Single `data.table` grouped aggregation over all 5 vars |
| **Estimated runtime** | 86+ hours | **2–5 minutes** |
| **Peak memory** | ~6.46M-element R list (high overhead) | ~38.5M-row `data.table` (~2–3 GB, fits in 16 GB) |
| **Numerical results** | max/min/mean of rook-neighbor values | **Identical** — same neighbor sets, same operations |
| **RF model** | Pre-trained | **Untouched** — no retraining |

The core insight is that the neighbor aggregation is a **group-by operation on an edge list**, which `data.table` executes in optimized C code with radix-sort grouping — replacing millions of R-level loop iterations and character-key hash lookups with a single vectorized pass.