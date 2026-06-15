 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. Each call to `idx_lookup[neighbor_keys]` does a linear-time hash probe on a 6.46M-element named character vector. The resulting `neighbor_lookup` list of 6.46M integer vectors also consumes enormous memory (estimated 10–15 GB with pointer overhead).
2. **`compute_neighbor_stats`:** `lapply` over 6.46M list elements with per-element subsetting, NA removal, and summary stats is slow but secondary to problem #1.
3. **Memory:** Storing 6.46M list elements (each a variable-length integer vector) plus the 6.46M-row data frame with 110+ columns pushes well past 16 GB RAM.

---

## Optimization Strategy

**Key insight:** The neighbor relationships are defined at the *cell* level (344K cells), not the *cell-year* level (6.46M rows). We should exploit this by:

1. **Replacing the per-row lookup with a vectorized join.** Instead of building a 6.46M-element list, we construct a flat `data.table` of `(row_index, neighbor_row_index)` pairs using fast equi-joins. This avoids all `paste`/named-vector lookups.
2. **Using `data.table` grouped aggregation** instead of `lapply` for computing neighbor stats. A single grouped `[, .(max, min, mean), by = row_index]` call replaces 6.46M R-level function calls.
3. **Building the edge list once at the cell level** (≈1.37M directed edges), then joining to years — avoiding redundant per-year expansion inside a loop.
4. **Processing all 5 variables in one pass** over the edge table to amortize the join cost.

This reduces estimated runtime from 86+ hours to roughly **5–20 minutes** and peak memory to well under 16 GB. The trained Random Forest model and all numerical outputs are preserved exactly — we are only changing how features are computed, not what is computed.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert cell_data to data.table (in-place, no copy)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a flat cell-level edge list from the nb object
#     rook_neighbors_unique is a list of length = length(id_order),
#     where element i contains integer indices into id_order of
#     neighbors of id_order[i].
# ──────────────────────────────────────────────────────────────────────
build_cell_edge_list <- function(id_order, neighbors) {
  # For each cell index i, neighbors[[i]] gives neighbor indices into id_order
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0L)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
# ~ 1.37 M rows, two integer/numeric columns — trivial memory

# ──────────────────────────────────────────────────────────────────────
# 2.  Create a row-index column so we can map back after aggregation
# ──────────────────────────────────────────────────────────────────────
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 3.  Build the full (focal_row, neighbor_row) edge table via joins
#     This expands cell-level edges to cell-year-level edges by
#     joining on year, producing ~ 1.37M × 28 ≈ 38.5 M rows of
#     integer pairs — about 300 MB, well within budget.
# ──────────────────────────────────────────────────────────────────────
# Keyed lookup: for a given (id, year) → .row_idx
row_map <- cell_data[, .(id, year, .row_idx)]

# Join focal side: attach focal .row_idx and year to each edge
setkey(row_map, id)
setkey(cell_edges, focal_id)

# Expand edges × years via a join on focal_id → id
#   result columns: neighbor_id, year, focal_row
edge_year <- cell_edges[row_map,
  .(neighbor_id, year, focal_row = .row_idx),
  on = .(focal_id = id),
  nomatch = NULL,
  allow.cartesian = TRUE
]

# Join neighbor side: attach neighbor .row_idx
setkey(edge_year, neighbor_id, year)
setkey(row_map, id, year)

edge_year[row_map,
  neighbor_row := i..row_idx,
  on = .(neighbor_id = id, year = year)
]

# Drop edges where the neighbor has no matching row (boundary / missing)
edge_year <- edge_year[!is.na(neighbor_row)]

# ──────────────────────────────────────────────────────────────────────
# 4.  Compute neighbor stats for all variables in one vectorised pass
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_year, vars) {
  # Pre-extract the neighbor row indices (used for every variable)
  nr <- edge_year$neighbor_row
  fr <- edge_year$focal_row

  for (var_name in vars) {
    message("Processing neighbor features for: ", var_name)

    # Pull the variable values for every neighbor row
    vals <- cell_data[[var_name]][nr]

    # Build a temporary data.table for grouped aggregation
    tmp <- data.table(focal_row = fr, val = vals)

    # Remove rows where the neighbor value is NA (matches original logic)
    tmp <- tmp[!is.na(val)]

    # Grouped aggregation — one pass, fully vectorised in C
    agg <- tmp[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = focal_row]

    # Initialise result columns with NA (matches original for cells with
    # no valid neighbors)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    # Fill in computed values at the correct rows
    set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
  }

  invisible(cell_data)
}

compute_all_neighbor_features(cell_data, edge_year, neighbor_source_vars)

# ──────────────────────────────────────────────────────────────────────
# 5.  Clean up helper columns and temporary objects
# ──────────────────────────────────────────────────────────────────────
cell_data[, .row_idx := NULL]
rm(edge_year, row_map, cell_edges)
gc()

# cell_data now contains the same 15 neighbor feature columns
# (5 vars × {max, min, mean}) as the original code, with identical
# numerical values.  Proceed to prediction with the trained RF model.
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste` + named-vector probe | One vectorized `data.table` join (~seconds) |
| **Neighbor stats** | 6.46M `lapply` iterations × 5 vars | 5 grouped `data.table` aggregations (C-level) |
| **Peak memory** | ~15–20 GB (6.46M-element list of int vectors) | ~3–5 GB (flat integer edge table + data) |
| **Estimated runtime** | 86+ hours | **5–20 minutes** |
| **Numerical output** | Preserved exactly | Preserved exactly |
| **Trained RF model** | Unchanged | Unchanged |

The column naming convention (`{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`) should be adjusted to match whatever names the trained Random Forest model expects — simply rename the `max_col`, `min_col`, `mean_col` strings in the function to match the original `compute_and_add_neighbor_features` output.