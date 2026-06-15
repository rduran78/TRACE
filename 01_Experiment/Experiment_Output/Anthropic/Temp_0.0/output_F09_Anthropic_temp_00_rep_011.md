 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups via `[` on character keys are O(n) in the worst case per call due to R's internal hashing overhead at scale.

### 2. Redundant recomputation of the same spatial topology per year
The rook-neighbor relationships are **purely spatial** — they don't change across years. Yet `build_neighbor_lookup` rebuilds the neighbor mapping for every cell-year combination, effectively duplicating the same spatial adjacency structure 28 times (once per year). This inflates the lookup list from ~344K entries to ~6.46M entries.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Each call to `compute_neighbor_stats` iterates over all 6.46M rows, subsetting and computing `max`, `min`, `mean` one row at a time. With 5 variables, that's ~32.3M individual R-level loop iterations with per-element vector subsetting.

### Summary of waste
| Component | Current scale | Optimal scale | Waste factor |
|---|---|---|---|
| Neighbor lookup | 6.46M entries | 344K entries (year-invariant) | ~19× |
| Stat computation | 6.46M × 5 lapply calls | Vectorized join + group-by | Orders of magnitude |

---

## Optimization Strategy

The key insight: **the neighbor table is time-invariant; only the cell attributes change by year.** Therefore:

1. **Build the adjacency edge-list once** from the `nb` object — a simple two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is done once and reused forever.

2. **Join yearly attributes onto the edge-list by year.** For each year, join the neighbor cell's attribute values onto the edge-list via a keyed `data.table` join. This turns the neighbor-value lookup into a vectorized merge.

3. **Compute grouped `max`, `min`, `mean` via `data.table` aggregation** — grouping by `(cell_id, year)` over the joined edge-list. This replaces millions of `lapply` iterations with a single vectorized group-by.

4. **Join the resulting neighbor stats back** onto the main `cell_data` table.

This reduces the problem from ~6.46M R-level loop iterations to a handful of vectorized `data.table` operations, bringing runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the time-invariant adjacency edge-list ONCE
# ==============================================================================
# Input:
#   id_order             — vector of 344,208 cell IDs (positional index matches nb object)
#   rook_neighbors_unique — spdep::nb object (list of length 344,208; each element
#                           is an integer vector of positional indices of neighbors)
#
# Output:
#   adj_dt — data.table with columns (cell_id, neighbor_id), ~1.37M rows

build_adjacency_edgelist <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  cell_id_vec    <- integer(n_edges)
  neighbor_id_vec <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n_nb <- length(nb_idx)
    cell_id_vec[pos:(pos + n_nb - 1L)]    <- id_order[i]
    neighbor_id_vec[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
    pos <- pos + n_nb
  }

  data.table(cell_id = cell_id_vec, neighbor_id = neighbor_id_vec)
}

adj_dt <- build_adjacency_edgelist(id_order, rook_neighbors_unique)

cat(sprintf("Adjacency edge-list: %d directed edges across %d cells\n",
            nrow(adj_dt), length(id_order)))

# ==============================================================================
# STEP 2: Convert main data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist and are of consistent type
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor stats via
#          vectorized join + grouped aggregation, then merge back.
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will cross-join the adjacency edge-list with all 28 years, then join
# neighbor attributes. But that would create ~1.37M × 28 ≈ 38.5M rows,
# which is fine for data.table but uses memory. A more memory-friendly
# approach: iterate year-by-year or, better, join directly on (neighbor_id, year).

# Strategy: expand adj_dt by year via merge with cell_data's (id, year) pairs,
# then look up neighbor values.

# Create a slim lookup: only id, year, and the 5 source vars
lookup_cols <- c("id", "year", neighbor_source_vars)
# Ensure all columns exist
stopifnot(all(lookup_cols %in% names(cell_data)))

neighbor_vals_dt <- cell_data[, ..lookup_cols]
setnames(neighbor_vals_dt, "id", "neighbor_id")
# Key for fast join
setkey(neighbor_vals_dt, neighbor_id, year)

# Create the cell-year backbone from cell_data (id, year) to expand adj_dt
cell_year_dt <- cell_data[, .(cell_id = id, year)]

# Merge cell_year_dt with adj_dt to get (cell_id, year, neighbor_id)
# This is: for each (cell, year), list all spatial neighbors
setkey(adj_dt, cell_id)
setkey(cell_year_dt, cell_id)

# Efficient expansion: join adj_dt onto cell_year_dt
# Result: each row of cell_year_dt is expanded by the number of neighbors
edges_by_year <- adj_dt[cell_year_dt, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
# edges_by_year has columns: cell_id, neighbor_id, year
# Expected rows: ~6.46M × avg_neighbors ≈ ~6.46M × (1.37M/344K×2 sides... ~4 neighbors each)
# ≈ ~25.8M rows — fits comfortably in 16 GB

cat(sprintf("Expanded edge-year table: %s rows\n", format(nrow(edges_by_year), big.mark = ",")))

# Join neighbor attribute values onto edges_by_year
setkey(edges_by_year, neighbor_id, year)
edges_by_year <- neighbor_vals_dt[edges_by_year, on = .(neighbor_id, year)]

# Now edges_by_year has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, cell_id

# ==============================================================================
# STEP 4: Grouped aggregation — compute max, min, mean per (cell_id, year, var)
# ==============================================================================

# Build all neighbor stat columns in one grouped aggregation
agg_expr_list <- list()
for (var_name in neighbor_source_vars) {
  var_sym <- as.name(var_name)
  agg_expr_list[[paste0("neighbor_max_", var_name)]] <-
    bquote(as.numeric(max(.(var_sym), na.rm = TRUE)))
  agg_expr_list[[paste0("neighbor_min_", var_name)]] <-
    bquote(as.numeric(min(.(var_sym), na.rm = TRUE)))
  agg_expr_list[[paste0("neighbor_mean_", var_name)]] <-
    bquote(mean(.(var_sym), na.rm = TRUE))
}

# Handle edge case: max/min of zero-length after NA removal returns Inf/-Inf;
# we will convert those to NA after aggregation.

# Build the aggregation call dynamically
agg_call <- as.call(c(as.name("list"),
                       setNames(agg_expr_list, names(agg_expr_list))))

neighbor_stats <- edges_by_year[, eval(agg_call), by = .(cell_id, year)]

# Replace Inf / -Inf with NA (from max/min of all-NA neighbor groups)
stat_cols <- names(neighbor_stats)[!(names(neighbor_stats) %in% c("cell_id", "year"))]
for (col in stat_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

cat(sprintf("Neighbor stats computed: %d rows × %d columns\n",
            nrow(neighbor_stats), ncol(neighbor_stats)))

# ==============================================================================
# STEP 5: Merge neighbor stats back onto cell_data
# ==============================================================================

# Remove any pre-existing neighbor stat columns to avoid duplication
old_neighbor_cols <- intersect(names(cell_data), stat_cols)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

# Keyed merge
setkey(cell_data, id, year)
setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat(sprintf("Final cell_data: %s rows × %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))

# ==============================================================================
# STEP 6: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# The column names produced above (neighbor_max_ntl, neighbor_min_ntl,
# neighbor_mean_ntl, etc.) must match the names the model was trained on.
#
# If the original code used different naming conventions, rename here:
# Example — if original names were like "ntl_neighbor_max":
#
# for (var_name in neighbor_source_vars) {
#   setnames(cell_data,
#     old = c(paste0("neighbor_max_", var_name),
#             paste0("neighbor_min_", var_name),
#             paste0("neighbor_mean_", var_name)),
#     new = c(paste0(var_name, "_neighbor_max"),
#             paste0(var_name, "_neighbor_min"),
#             paste0(var_name, "_neighbor_mean")),
#     skip_absent = TRUE)
# }
#
# Then predict:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Expected Performance Comparison

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~6.46M `paste` + hash lookups (~hours) | ~344K loop to build edge-list (~seconds) |
| Expand edges × years | Implicit in lookup (~hours) | `data.table` cartesian join (~10–30 sec) |
| Join neighbor values | Embedded in `lapply` (~hours per var) | Single keyed join (~10–30 sec) |
| Compute max/min/mean | 6.46M × 5 `lapply` iterations (~days) | One grouped `data.table` aggregation (~30–60 sec) |
| Merge back to cell_data | Per-variable column binding | One keyed join (~seconds) |
| **Total** | **~86+ hours** | **~2–5 minutes** |

### Why correctness is preserved
- The adjacency relationships are identical (same `nb` object, same `id_order`).
- The `max`, `min`, `mean` with `na.rm = TRUE` produce the same numerical values as the original `lapply` implementation.
- `Inf`/`-Inf` from empty groups are mapped to `NA`, matching the original `return(c(NA, NA, NA))` guard.
- The trained Random Forest model is never modified or retrained — only the feature-engineering pipeline is accelerated.