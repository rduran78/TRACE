 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction:** `paste(id, year, sep="_")` is called millions of times inside the loop body, and named-vector indexing (`idx_lookup[neighbor_keys]`) is an O(k) hash lookup repeated for every row.
- **Redundant work across years:** Every cell has the *same* neighbors in every year, yet the function re-discovers them for each of the 28 year-copies. This multiplies work by 28×.
- **Memory:** The `lapply` returns a list of 6.46 M integer vectors — a large, fragmented object that is hard on the garbage collector.

### 2. `compute_neighbor_stats` — Pure-R row-wise aggregation over 6.46 M list elements
- Each call iterates through the 6.46 M-element list in interpreted R, extracting subsets of a numeric vector and computing `max/min/mean`. This is repeated 5 times (once per variable), totaling ~32.3 M interpreted iterations.
- `do.call(rbind, result)` on a 6.46 M-element list of length-3 vectors is itself slow and memory-hungry.

### Combined effect
The estimated 86+ hours is dominated by these two interpreted-R loops over millions of rows with per-element allocation.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate the per-row loop in `build_neighbor_lookup`** | Exploit the fact that the neighbor graph is *time-invariant*. Build a sparse adjacency structure once over the 344 K cells, then join it to the panel via `data.table` keyed merge — no `lapply`, no string keys. |
| **Vectorize `compute_neighbor_stats`** | Represent the neighbor graph as a two-column edge table (`from_row`, `to_row`). Then for each variable, extract all neighbor values in one vectorized subscript, group by `from_row`, and compute `max/min/mean` with `data.table`'s `by=` — fully compiled C code under the hood. |
| **Minimize memory** | Use `data.table` in-place `:=` assignment. Never materialise the 6.46 M-element list. The edge table has ~1.37 M × 28 ≈ 38.5 M rows of two integers (~300 MB), which fits in 16 GB alongside the panel. |
| **Preserve the trained RF model** | Only the feature columns are being added; the model object is untouched. Column names and numerical values are identical to the original code. |

**Expected speedup:** From 86+ hours to roughly 5–15 minutes on the same laptop.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert panel to data.table (if not already) and create a row index
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)
cell_data[, .row_id := .I]                 # preserve original row order

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a time-invariant edge list from the nb object  (one-time, fast)
#     rook_neighbors_unique is an nb object: list of integer vectors
#     id_order is the vector that maps list position -> cell id
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains the *positions* in id_order of cell i's neighbors
  from_pos <- rep(seq_along(neighbors), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)

  data.table(
    from_id = id_order[from_pos],
    to_id   = id_order[to_pos]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1.37 M rows (directed rook edges, time-invariant)

# ──────────────────────────────────────────────────────────────────────
# 2.  Expand edges across years by merging with the panel
#     This creates a table:  (from_row, to_row) for every cell-year pair
# ──────────────────────────────────────────────────────────────────────

# Keyed lookup:  (id, year) -> .row_id
setkey(cell_data, id, year)

# Attach the "from" row id  (the cell whose feature we are computing)
edge_panel <- edge_dt[
  cell_data[, .(from_id = id, year, from_row = .row_id)],
  on = "from_id",
  allow.cartesian = TRUE,
  nomatch = NULL
]
# edge_panel now has columns: from_id, to_id, year, from_row

# Attach the "to" row id  (the neighbor whose value we need)
edge_panel <- cell_data[, .(to_id = id, year, to_row = .row_id)][
  edge_panel,
  on = c("to_id", "year"),
  nomatch = NULL
]
# edge_panel now has columns: to_id, year, to_row, from_id, from_row
# ~38.5 M rows  (1.37 M edges × 28 years, minus any missing combos)

# Keep only what we need
edge_panel <- edge_panel[, .(from_row, to_row)]
setkey(edge_panel, from_row)

# ──────────────────────────────────────────────────────────────────────
# 3.  Vectorised neighbor-stat computation
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(dt, edge, var_name) {
  # Pull neighbor values in one vectorised subscript
  edge[, val := dt[[var_name]][to_row]]

  # Drop NAs once
  valid <- edge[!is.na(val)]

  # Grouped aggregation — runs in compiled C inside data.table

  stats <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = from_row]

  # Column names matching the original pipeline
  col_max  <- paste0("nb_max_",  var_name)
  col_min  <- paste0("nb_min_",  var_name)
  col_mean <- paste0("nb_mean_", var_name)

  # Initialise with NA, then fill matched rows  (in-place, no copy)
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)

  set(dt, i = stats$from_row, j = col_max,  value = stats$nb_max)
  set(dt, i = stats$from_row, j = col_min,  value = stats$nb_min)
  set(dt, i = stats$from_row, j = col_mean, value = stats$nb_mean)

  invisible(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Outer loop — identical variable list, same column names
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, edge_panel, var_name)
}

# ──────────────────────────────────────────────────────────────────────
# 5.  Clean up helper column; restore original key / order if needed
# ──────────────────────────────────────────────────────────────────────
setorder(cell_data, .row_id)
cell_data[, .row_id := NULL]

# cell_data now contains the 15 new nb_* columns (3 stats × 5 vars)
# and is ready for predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per cell-year as the original code. The edge list encodes the identical rook adjacency, and the year-join ensures temporal alignment. |
| **Column naming** | `nb_max_ntl`, `nb_min_ntl`, `nb_mean_ntl`, … — identical to what the original `compute_and_add_neighbor_features` would produce (adjust the prefix pattern if your wrapper used a different convention). |
| **Trained RF model** | The model object is never touched. Only the prediction input (`cell_data`) is augmented with the same 15 feature columns the model expects. |
| **Rows with no valid neighbors** | They receive `NA` for all three stats, exactly as the original code does via the `if (length(idx)==0)` / `if (length(neighbor_vals)==0)` guards. |

---

## Memory Budget (approximate)

| Object | Size |
|---|---|
| `cell_data` (6.46 M × 125 cols, numeric) | ~6.5 GB |
| `edge_panel` (≤38.5 M × 2 int cols) | ~0.3 GB |
| Temporary `val` column + `stats` table | ~0.6 GB peak |
| **Total peak** | **~7.4 GB** (fits in 16 GB) |

The original list-of-vectors neighbor lookup alone consumed several GB of fragmented memory; this approach is both faster and leaner.