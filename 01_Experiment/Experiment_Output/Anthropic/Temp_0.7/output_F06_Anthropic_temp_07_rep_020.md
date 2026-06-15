 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row R closures

`build_neighbor_lookup` creates a list of **6.46 million elements**, one per cell-year row. For every row it does string-pasting, hash lookups, and NA filtering inside a base-R `lapply`. This alone can take many hours because:

- `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called ~6.46M times.
- Each call indexes into `idx_lookup` (a named character vector), which is O(n) hashing on a 6.46M-length vector.
- The output is a **list of integer vectors** consuming substantial RAM (~6.46M list elements × average ~4 neighbors each).

### Bottleneck B: `compute_neighbor_stats` — repeated per-variable column extraction in a loop

`compute_neighbor_stats` iterates over the 6.46M-element `neighbor_lookup` list **once per variable** (×5 variables). Each iteration extracts subsets of a numeric vector by index, computes `max/min/mean`, and returns a 3-element vector, then `do.call(rbind, ...)` binds 6.46M tiny vectors — extremely slow.

### Why raster focal/kernel operations don't directly apply

The comment in the docstring is a red herring for this case. Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. Here the neighbor structure is an **irregular spdep::nb object** (rook contiguity on an arbitrary polygon/grid layout), and the panel is indexed by `(id, year)`. Focal operations would only work if the grid is perfectly regular *and* the nb object exactly matches a 3×3 rook kernel on that grid. Even then, converting to/from raster for 28 years × 5 variables adds complexity with no guarantee of correctness. The safer and faster approach is **vectorized sparse-matrix multiplication**, which preserves the exact nb structure.

### Summary

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | 6.46M string-paste + named-vector lookups |
| `compute_neighbor_stats` | ~hours × 5 vars | 6.46M `lapply` iterations per variable, `do.call(rbind)` |
| Total | 86+ hours estimated | Pure-R loops on millions of rows |

---

## 2. Optimization Strategy

### Key Insight: Replace per-row loops with sparse matrix operations

A rook-neighbor adjacency can be represented as a **sparse matrix W** of dimension `N_cells × N_cells` (344,208 × 344,208). For each year, the neighbor statistics (max, min, mean) across neighbor cells can be computed using:

- **Mean**: sparse matrix–vector product `W %*% x / degree` (or row-normalized W).
- **Max / Min**: use the sparse structure to do grouped max/min via `data.table` grouping on the edge list — far faster than 6.46M `lapply` calls.

### Plan

1. **Convert `nb` to a sparse adjacency matrix** (once, ~344K × 344K, very sparse with ~1.37M entries). Use `spdep::nb2listw` → `as(listw, "CsparseMatrix")` or build directly.
2. **Convert `nb` to an edge-list data.table** with columns `(from, to)` for max/min computation.
3. **For each year and each variable**, extract the values vector, then:
   - **Mean**: sparse matrix–vector multiply (one operation).
   - **Max/Min**: join edge list to values, then group-by `from` to get `max` and `min`.
4. **Vectorize across years** by working on the full `data.table` keyed by `(id, year)`.

This reduces the work from ~6.46M R-level iterations per variable to a handful of vectorized/compiled operations.

### Expected speedup

- Sparse matrix multiply for mean: seconds per year, ~1 minute total for all years × 5 vars.
- `data.table` grouped max/min on edge list: seconds per year.
- **Total: minutes instead of 86+ hours.**

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Prerequisites:
#   cell_data        : data.frame/data.table with columns: id, year, ntl, ec,
#                      pop_density, def, usd_est_n2 (and others)
#   id_order         : character/integer vector of cell IDs matching the nb object
#   rook_neighbors_unique : spdep::nb object (list of integer index vectors)
#
# Output:
#   cell_data gains 15 new columns:
#     {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#     for var in (ntl, ec, pop_density, def, usd_est_n2)
#
# The trained Random Forest model is NOT touched.
# =============================================================================

library(data.table)
library(Matrix)

# ---------- Step 0: Convert cell_data to data.table if needed ----------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---------- Step 1: Build edge list from nb object (once) --------------------
# rook_neighbors_unique[[i]] gives the indices (into id_order) of neighbors of
# cell id_order[i].

build_edge_list <- function(nb_obj) {
  # Pre-allocate by counting total edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    len <- length(nbrs)
    from_idx[pos:(pos + len - 1L)] <- i
    to_idx[pos:(pos + len - 1L)]   <- nbrs
    pos <- pos + len
  }
  data.table(from_ref = from_idx[1:(pos - 1L)],
             to_ref   = to_idx[1:(pos - 1L)])
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(rook_neighbors_unique)

# Map ref indices to actual cell IDs
edge_dt[, from_id := id_order[from_ref]]
edge_dt[, to_id   := id_order[to_ref]]

# Compute degree (number of neighbors per cell) for mean calculation
degree_dt <- edge_dt[, .(degree = .N), by = from_id]

cat(sprintf("Edge list: %d directed edges, %d unique cells\n",
            nrow(edge_dt), length(id_order)))

# ---------- Step 2: Key cell_data for fast joins -----------------------------
# Ensure id column type matches id_order type
cell_data[, id := as.character(id)]
edge_dt[, from_id := as.character(from_id)]
edge_dt[, to_id   := as.character(to_id)]
degree_dt[, from_id := as.character(from_id)]

setkey(cell_data, id, year)

# ---------- Step 3: Vectorized neighbor stats computation --------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, degree_dt,
                                          var_names) {
  # Get unique years
  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d variables × %d years = %d tasks\n",
              length(var_names), length(years), length(var_names) * length(years)))

  for (var_name in var_names) {
    cat(sprintf("  Variable: %s\n", var_name))

    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    # Initialize result columns with NA
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    for (yr in years) {
      # Extract values for this year: a lookup table id -> value
      yr_vals <- cell_data[year == yr, .(id, val = get(var_name))]
      setkey(yr_vals, id)

      # Join neighbor values: for each edge (from_id, to_id), get val of to_id
      # This gives us, for each "from" cell, all its neighbor values
      edge_yr <- edge_dt[, .(from_id, to_id)]
      edge_yr[yr_vals, neighbor_val := i.val, on = .(to_id = id)]

      # Remove edges where neighbor value is NA
      edge_yr <- edge_yr[!is.na(neighbor_val)]

      if (nrow(edge_yr) == 0L) next

      # Grouped aggregation: max, min, sum by from_id
      agg <- edge_yr[, .(
        n_max = max(neighbor_val),
        n_min = min(neighbor_val),
        n_sum = sum(neighbor_val),
        n_cnt = .N
      ), by = from_id]

      agg[, n_mean := n_sum / n_cnt]

      # Write results back into cell_data
      # Build a join key
      setkey(agg, from_id)

      # Get row indices in cell_data for this year
      idx <- cell_data[year == yr, which = TRUE]
      ids_this_year <- cell_data$id[idx]

      # Match aggregated results to cell_data rows
      match_idx <- match(ids_this_year, agg$from_id)

      set(cell_data, i = idx, j = max_col,  value = agg$n_max[match_idx])
      set(cell_data, i = idx, j = min_col,  value = agg$n_min[match_idx])
      set(cell_data, i = idx, j = mean_col, value = agg$n_mean[match_idx])
    }

    cat(sprintf("    Done: %s — added %s, %s, %s\n",
                var_name, max_col, min_col, mean_col))
  }

  return(cell_data)
}

# ---------- Step 4: Run it ---------------------------------------------------
cat("Computing neighbor features (optimized)...\n")
t0 <- proc.time()

cell_data <- compute_all_neighbor_features(
  cell_data, edge_dt, degree_dt, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor feature computation completed in %.1f seconds (%.1f minutes)\n",
            elapsed, elapsed / 60))

# ---------- Step 5: Apply the pre-trained Random Forest model ----------------
# (Unchanged — the model is already trained and is applied as before)
# Example:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Is Correct and Preserves the Estimand

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | For each `(cell, year)` pair, the code computes `max`, `min`, and `mean` over the exact same set of rook neighbors defined by `rook_neighbors_unique`. The edge list is a lossless representation of the nb object. The `data.table` grouped `max`/`min`/`sum`/`.N` operations are mathematically identical to the original `lapply` code. |
| **NA handling** | Edges where the neighbor's value is `NA` are dropped before aggregation (`edge_yr[!is.na(neighbor_val)]`), exactly matching `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` in the original. Cells with zero valid neighbors get `NA` (from the `match` returning `NA`). |
| **Mean = sum/count of non-NA** | The original computes `mean(neighbor_vals)` after removing NAs. The optimized code computes `n_sum / n_cnt` where `n_cnt` counts only non-NA values — identical. |
| **Trained RF model** | Untouched. The 15 new columns have the same names and values as the original code would produce. |
| **Memory** | The edge list has ~1.37M rows × 4 columns — trivial. Per-year slices are ~344K rows. Peak memory is well within 16 GB. |

### Expected Runtime

- 5 variables × 28 years = 140 iterations.
- Each iteration: one `data.table` join (~1.37M rows) + one grouped aggregation (~344K groups) + one `match` + three `set` calls.
- Each iteration: ~0.5–2 seconds on a modern laptop.
- **Total: ~2–5 minutes** (vs. 86+ hours originally — roughly a **1,000× speedup**).