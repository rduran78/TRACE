 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is created once, but the `paste()` call to construct `neighbor_keys` and the named-vector lookup `idx_lookup[neighbor_keys]` are executed **once per row** — 6.46 million times. Each call to `paste()` allocates a new character vector, and each named-vector lookup performs repeated hash-table probes against a 6.46M-entry hash. This is O(N × K) string operations where K is the average neighbor count.

### Broader Algorithmic Problem
The real issue is that the entire neighbor-lookup architecture is **row-centric** when it should be **vectorized and join-based**. The pattern is:

1. For each cell-year row, find which other rows share the same year and are spatial neighbors.
2. For each such neighbor row, pull variable values and compute max/min/mean.

This is fundamentally a **merge-aggregate** operation that can be expressed as a single vectorized join + grouped aggregation — no per-row `lapply`, no string keys, no 6.46M iterations.

### Root Cause Summary

| Layer | Problem | Cost |
|-------|---------|------|
| String key construction | `paste()` called 6.46M × (1 + avg_neighbors) times | ~billions of string allocs |
| Named vector lookup | Hash probe into 6.46M-entry table, per row | O(N × K) hash lookups |
| Row-wise `lapply` | R-level loop over 6.46M rows | Interpreter overhead |
| Repeated per variable | `compute_neighbor_stats` re-traverses all 6.46M neighbor lists 5 times | 5× redundant traversal |
| Architecture | Row-centric instead of vectorized join-aggregate | Orders of magnitude slower |

## Optimization Strategy

1. **Build an integer edge list once** — expand the `nb` object into a two-column `data.table` of `(row_i, row_j)` where both rows share the same year. Use integer IDs throughout; no strings.
2. **Vectorized join** — for each variable, join the neighbor edge list to the data column, then aggregate with `data.table` grouped operations (`max`, `min`, `mean` by source row).
3. **Single pass for all variables** — optionally batch all 5 variables in one join.
4. **Memory-conscious** — the edge list is ~(6.46M × avg_neighbors) ≈ ~26M rows of two integers ≈ ~200 MB, feasible on 16 GB.

Expected speedup: from ~86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ============================================================
# 0. Assumptions about inputs already in the environment:
#    - cell_data       : data.frame/data.table with columns id, year, and the 5 vars
#    - id_order        : integer vector of cell IDs in the order matching rook_neighbors_unique
#    - rook_neighbors_unique : an nb object (list of integer index vectors)
#    - The trained RF model object is untouched.
# ============================================================

# Convert to data.table if not already (non-destructive copy)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build a spatial edge list (id_from, id_to) from the nb object
#         This is done ONCE and uses only integer cell IDs.
# ============================================================

build_spatial_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[k]] contains integer indices into id_order that are neighbors of id_order[k]
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses 0L for "no neighbors"
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )
}

cat("Building spatial edge list...\n")
spatial_edges <- build_spatial_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed neighbor pairs\n", nrow(spatial_edges)))

# ============================================================
# STEP 2: Build the full row-level edge list by joining on year.
#         For each row i in cell_data, find all rows j that share
#         the same year AND whose cell id is a spatial neighbor.
#
#         We do this via keyed joins — no string pasting.
# ============================================================

# Add a row index to cell_data
cell_data[, .row_idx := .I]

# Create a lean lookup: (id, year) -> row_idx
# This replaces the old paste-based idx_lookup
id_year_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_lookup, id, year)

# For each row, get its (id, year), then find neighbor ids via spatial_edges,
# then find neighbor rows via id_year_lookup.
# We do this as a chain of data.table joins — fully vectorized.

cat("Building row-level neighbor edge list...\n")

# Start from cell_data rows: get (row_idx_from, id_from, year)
row_info <- cell_data[, .(.row_idx, id, year)]
setnames(row_info, c("row_from", "id_from", "year"))

# Join to spatial_edges to get neighbor cell IDs
setkey(row_info, id_from)
setkey(spatial_edges, id_from)

# This is the big join: for each row, expand to its spatial neighbors
# Result: (row_from, id_from, year, id_to)
edges_with_year <- spatial_edges[row_info, on = "id_from", allow.cartesian = TRUE,
                                  nomatch = NULL]
# edges_with_year now has columns: id_from, id_to, row_from, year

# Now find the actual row index of each (id_to, year) pair
setkey(id_year_lookup, id, year)
setkey(edges_with_year, id_to, year)

neighbor_edges <- id_year_lookup[edges_with_year,
                                  on = c("id" = "id_to", "year" = "year"),
                                  nomatch = NULL]
# neighbor_edges has: .row_idx (= row_to), id, year, row_from, id_from

# Keep only what we need
neighbor_edges <- neighbor_edges[, .(row_from, row_to = .row_idx)]
setkey(neighbor_edges, row_from)

cat(sprintf("  Row-level edge list: %d edges\n", nrow(neighbor_edges)))

# ============================================================
# STEP 3: For each neighbor source variable, compute max/min/mean
#         via vectorized grouped aggregation.
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_data, neighbor_edges, var_name) {
  cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

  # Extract the variable values for the "to" (neighbor) rows
  vals <- cell_data[[var_name]]
  neighbor_edges[, val := vals[row_to]]

  # Grouped aggregation — one pass, fully vectorized

  stats <- neighbor_edges[!is.na(val),
                           .(nmax  = max(val),
                             nmin  = min(val),
                             nmean = mean(val)),
                           by = row_from]

  # Allocate result columns (NA by default for rows with no valid neighbors)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]

  cell_data[stats$row_from, (col_max)  := stats$nmax]
  cell_data[stats$row_from, (col_min)  := stats$nmin]
  cell_data[stats$row_from, (col_mean) := stats$nmean]

  # Clean up temporary column
  neighbor_edges[, val := NULL]

  invisible(cell_data)
}

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features_fast(cell_data, neighbor_edges, var_name)
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done.\n")

# ============================================================
# STEP 4: Verify column names match what the trained RF expects.
#         Adjust the column naming pattern above if your original
#         code used a different convention (e.g., "ntl_neighbor_max").
#         The numerical values are identical — same estimand,
#         same max/min/mean over the same neighbor sets.
# ============================================================
```

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor definition | `rook_neighbors_unique` via `spdep::nb` | Same `nb` object, same directed pairs |
| Year matching | `paste(id, year)` string match | Integer join on `(id, year)` — identical match |
| Aggregation | `max`, `min`, `mean` of non-NA neighbor values | Identical `max`, `min`, `mean` with `!is.na` filter |
| NA handling | Rows with no valid neighbors get `NA` | Same — `stats` only contains rows with ≥1 valid neighbor; rest stay `NA` |
| RF model | Untouched | Untouched — only the feature columns are populated |

## Expected Performance

| Step | Estimated Time | Estimated Memory |
|------|---------------|-----------------|
| Spatial edge list construction | < 1 second | ~20 MB |
| Row-level edge join (the big step) | 1–3 minutes | ~2–3 GB peak |
| Per-variable grouped aggregation (×5) | ~30 sec each | ~200 MB temp |
| **Total** | **~5–8 minutes** | **~4 GB peak** |

This replaces the estimated 86+ hours with a run time on the order of minutes, by eliminating all per-row string operations and R-level loops in favor of vectorized `data.table` keyed joins and grouped aggregations.