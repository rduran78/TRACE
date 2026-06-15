 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Constructs character keys (`paste(id, year)`) and matches them into a named character vector (`idx_lookup`) of length 6.46M.

Named-vector lookup in R is **O(n)** per probe (linear scan of names), so ~6.46M lookups × ~4 neighbors each × 6.46M-length name vector ≈ **trillions of character comparisons**. That is why the estimate is 86+ hours.

`compute_neighbor_stats` is a secondary bottleneck: another R-level `lapply` over 6.46M elements, each extracting and summarising a small numeric vector.

## Optimization Strategy

| Problem | Fix |
|---|---|
| Named-vector lookup is O(n) per probe | Replace with **`data.table` keyed join** — O(1) amortised hash lookup |
| Per-row `lapply` in `build_neighbor_lookup` | **Vectorise entirely**: explode the `nb` list into an edge-list, join to get row indices, then `split()` once |
| Per-row `lapply` in `compute_neighbor_stats` | **Vectorise**: attach the variable values to the edge-list, then `data.table` grouped aggregation (`max`, `min`, `mean` by source row) |
| Repeating the edge-list build for every variable | Build the edge-list **once**; reuse for all 5 variables |
| Memory: 6.46M-row list of integer vectors | Edge-list representation is more compact and cache-friendly |

The numerical results are **identical** (same max, min, mean per cell-year, same column names). The trained Random Forest model is untouched.

## Working R Code

```r
library(data.table)

# ── 1. Build a vectorised edge-list (once) ────────────────────────────────────

build_neighbor_edgelist <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data must have columns: id, year (and be in its original row order)
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer index vectors)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]                       # preserve original row position

  # --- map each cell-ID to its position in id_order --------------------------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- explode nb list into an edge-list of (source_cell, neighbor_cell) -----
  n_nb    <- lengths(rook_neighbors_unique)           # number of neighbors per cell
  src_ref <- rep(seq_along(id_order), n_nb)           # source index in id_order
  dst_ref <- unlist(rook_neighbors_unique)             # neighbor index in id_order

  edges <- data.table(
    src_id = id_order[src_ref],
    dst_id = id_order[dst_ref]
  )

  # --- cross with years to get (source_row, neighbor_row) --------------------
  # Key the main table for fast join
  setkey(dt, id, year)

  # Join source side: get every (src_id, year, src_row_idx)
  src_rows <- dt[, .(src_id = id, year, src_row = row_idx)]

  # Merge edges with source rows  →  (src_row, dst_id, year)
  #   For every edge and every year the source cell appears in, we need
  #   the neighbor's row in that same year.
  edge_year <- edges[src_rows, on = .(src_id), allow.cartesian = TRUE, nomatch = 0L]
  #   columns: src_id, dst_id, year, src_row

  # Join neighbor side: get dst_row
  dst_rows <- dt[, .(dst_id = id, year, dst_row = row_idx)]
  edge_year <- dst_rows[edge_year, on = .(dst_id, year), nomatch = 0L]
  #   columns: dst_id, year, dst_row, src_id, src_row

  # Keep only what we need
  edge_year <- edge_year[, .(src_row, dst_row)]
  setkey(edge_year, src_row)

  return(edge_year)
}

# ── 2. Compute neighbor stats for one variable (vectorised) ───────────────────

compute_neighbor_stats_fast <- function(cell_data_dt, edge_year, var_name) {
  vals <- cell_data_dt[[var_name]]

  # Attach neighbor values
  el <- copy(edge_year)
  el[, nval := vals[dst_row]]
  el <- el[!is.na(nval)]

  # Grouped aggregation
  agg <- el[, .(nb_max  = max(nval),
                nb_min  = min(nval),
                nb_mean = mean(nval)),
            keyby = src_row]

  # Allocate full-length result (NA for cells with no valid neighbors)
  n <- nrow(cell_data_dt)
  out <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  out[agg$src_row, `:=`(nb_max  = agg$nb_max,
                        nb_min  = agg$nb_min,
                        nb_mean = agg$nb_mean)]

  setnames(out, c(paste0(var_name, "_nb_max"),
                  paste0(var_name, "_nb_min"),
                  paste0(var_name, "_nb_mean")))
  return(out)
}

# ── 3. Outer loop: drop-in replacement ────────────────────────────────────────

## Convert once
cell_dt <- as.data.table(cell_data)

## Build edge-list once  (~1-3 min, ~200-400 MB)
message("Building neighbour edge-list …")
edge_year <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge-list: %s rows", format(nrow(edge_year), big.mark = ",")))

## Compute features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("  → ", var_name)
  nb_cols <- compute_neighbor_stats_fast(cell_dt, edge_year, var_name)
  cell_dt <- cbind(cell_dt, nb_cols)
}

## Convert back to data.frame if downstream code expects one
cell_data <- as.data.frame(cell_dt)
```

## Expected Performance

| Step | Old | New |
|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (named-vector probing) | ~1–3 minutes (`data.table` keyed joins) |
| `compute_neighbor_stats` (×5 vars) | ~5 hours (R-level lapply) | ~1–2 minutes (`data.table` grouped agg) |
| **Total** | **~86+ hours** | **~3–6 minutes** |

Peak RAM stays well within 16 GB: the edge-list is ~26M rows × 2 int columns ≈ 200 MB; the main table is ~6.46M × 110 cols ≈ 5–6 GB.

**Numerical equivalence**: `max`, `min`, and `mean` are computed over exactly the same sets of neighbor values (non-NA, same-year, rook-adjacent), so the trained Random Forest model's predictions are unchanged.