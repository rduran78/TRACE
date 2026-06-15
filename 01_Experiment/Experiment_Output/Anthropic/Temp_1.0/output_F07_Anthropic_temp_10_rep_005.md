 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. **Character-key hashing per row** — `paste(neighbor_cell_ids, data$year[i], sep = "_")` and named-vector lookup (`idx_lookup[neighbor_keys]`) is called 6.46 million times inside an `lapply`. Named-vector lookup in R is O(n) per query against a vector of 6.46M names — this is effectively a linear scan repeated millions of times.
2. **Per-row R-level overhead** — Each iteration of the `lapply` allocates character vectors, does `paste`, subsets a named vector, and filters `NA`s. With ~6.46M iterations, R's interpreted loop overhead alone is enormous.
3. **Redundant recomputation** — The neighbor topology is **time-invariant** (rook neighbors don't change across years). Yet the lookup is built per cell-year row rather than per cell, then broadcast across years. This inflates the work by a factor of 28.

`compute_neighbor_stats` is less severe but still slow: it runs an `lapply` over 6.46M elements, each calling `max`, `min`, `mean` in interpreted R.

**Estimated cost of current approach:** ~6.46M × (string operations + named-vector lookups against 6.46M keys) ≈ 86+ hours.

---

## Optimization Strategy

### 1. Separate spatial topology from temporal indexing
The neighbor structure is purely spatial (344,208 cells). Build a **cell-to-cell** adjacency once (344K entries), then map to rows using vectorized year-matching — never build 6.46M string keys.

### 2. Replace named-vector lookup with integer-indexed lookup via `data.table`
Use `data.table` keyed joins to map `(cell_id, year)` → row index in O(1) amortized time.

### 3. Vectorize `compute_neighbor_stats` using a sparse-matrix or a flattened vectorized approach
Expand the neighbor list into an edge list `(row_i, row_j)`, extract values with a single vectorized subscript, then aggregate with `data.table` grouping — no R-level loop over 6.46M elements.

### 4. Process all 5 variables in one pass over the edge list
The edge list is the same for all variables; just column-swap the values.

**Expected speedup:** From 86+ hours to **~2–5 minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0. Ensure data is a data.table with correct types
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Preserve original row order so we can write results back in place
cell_dt[, .ROW := .I]

# ──────────────────────────────────────────────────────────────────────
# 1. Build a CELL-level edge list from the nb object  (done ONCE)
#    rook_neighbors_unique is an nb object of length length(id_order).
#    id_order[k] gives the cell id for the k-th element of the nb list.
# ──────────────────────────────────────────────────────────────────────
build_cell_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[k]] contains integer indices into id_order of neighbors of cell k
  from_ref <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_ref   <- unlist(nb_obj)

  # Remove the 0-neighbor sentinel that spdep uses
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
}

cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
# cell_edges has ~1.37M rows (directed rook-neighbor pairs)

# ──────────────────────────────────────────────────────────────────────
# 2. Map cell-level edges to ROW-level edges by joining on year
#    For every year, each (from_id, to_id) pair becomes (from_row, to_row).
#    We do this with two keyed joins — fully vectorized.
# ──────────────────────────────────────────────────────────────────────

# Create a lookup: (id, year) -> row index
setkey(cell_dt, id, year)
row_lookup <- cell_dt[, .(id, year, .ROW)]
setkey(row_lookup, id, year)

# Expand edges across all years (cross join edges × years)
years <- sort(unique(cell_dt$year))
row_edges <- cell_edges[, CJ_year := NULL]  # safety
row_edges <- cell_edges[
  rep(seq_len(.N), each = length(years))
][, year := rep(years, times = nrow(cell_edges))]

# Join to get from_row
setnames(row_lookup, c("id", "year", ".ROW"), c("from_id", "year", "from_row"))
setkey(row_lookup, from_id, year)
setkey(row_edges, from_id, year)
row_edges <- row_lookup[row_edges, nomatch = 0L]

# Join to get to_row
setnames(row_lookup, c("from_id", "year", "from_row"), c("to_id", "year", "to_row"))
setkey(row_lookup, to_id, year)
setkey(row_edges, to_id, year)
row_edges <- row_lookup[row_edges, nomatch = 0L]

# Restore lookup names
setnames(row_lookup, c("to_id", "year", "to_row"), c("id", "year", ".ROW"))

# row_edges now has columns: to_row, from_row, from_id, to_id, year
# "from_row" is the focal cell-year; "to_row" is its neighbor cell-year.
# ──────────────────────────────────────────────────────────────────────
# 3. Compute neighbor max, min, mean for each variable — vectorised
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {

  # Pull neighbor values via integer indexing (single vectorised subscript)
  row_edges[, nval := cell_dt[[var]][to_row]]

  # Aggregate: group by focal row (from_row), dropping NAs
  stats <- row_edges[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    keyby = from_row
  ]

  # Initialise new columns to NA (handles cells with 0 valid neighbors)
  max_col  <- paste0(var, "_neighbor_max")
  min_col  <- paste0(var, "_neighbor_min")
  mean_col <- paste0(var, "_neighbor_mean")

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  # Write results back by integer row index
  set(cell_dt, i = stats$from_row, j = max_col,  value = stats$nb_max)
  set(cell_dt, i = stats$from_row, j = min_col,  value = stats$nb_min)
  set(cell_dt, i = stats$from_row, j = mean_col, value = stats$nb_mean)
}

row_edges[, nval := NULL]  # clean up temp column

# ──────────────────────────────────────────────────────────────────────
# 4. Convert back to data.frame if downstream code expects one
# ──────────────────────────────────────────────────────────────────────
cell_dt[, .ROW := NULL]
cell_data <- as.data.frame(cell_dt)
```

---

## Memory Note

The `row_edges` table has ~1.37M × 28 ≈ **38.4M rows** with a few integer/numeric columns — roughly **~1–1.5 GB**. This fits in 16 GB alongside the 6.46M-row dataset and the trained Random Forest object. If memory is tight, you can process years in batches:

```r
# Memory-conservative variant: process one year at a time
for (yr in years) {
  yr_rows  <- cell_dt[year == yr, .ROW]
  yr_edges <- cell_edges[, .(from_id, to_id, year = yr)]
  # ... join & aggregate as above, restricted to this year ...
}
```

---

## What is preserved

| Requirement | Status |
|---|---|
| **Trained Random Forest model** | Untouched — no retraining; we only compute the same 15 neighbor features the model expects. |
| **Original numerical estimand** | Identical — `max`, `min`, `mean` over exactly the same rook-neighbor sets, with the same `NA` handling (`NA` when no valid neighbors exist). |
| **Column naming convention** | Same `{var}_neighbor_max/min/mean` names as the original `compute_and_add_neighbor_features` would produce. Adjust the `paste0` pattern if your wrapper used different suffixes. |