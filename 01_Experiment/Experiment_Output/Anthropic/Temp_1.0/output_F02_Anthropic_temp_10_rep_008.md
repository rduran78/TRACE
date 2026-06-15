 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows
For every row, the function performs character coercion, `paste` key construction, and named-vector lookups. Named-vector lookups in R are hash-table–based but the constant factor is large when called 6.46 million times inside `lapply`. The result is a **list of 6.46M integer vectors**, which is both slow to build and memory-heavy (each list element carries R object overhead ≈ 100–200 bytes, so ~1–1.3 GB just in list overhead alone, before the actual index data).

### 2. `compute_neighbor_stats` — O(n) `lapply` iterated 5 times
For each of the 5 source variables, another 6.46M-element `lapply` computes `max/min/mean` over small integer-index subsets. The `do.call(rbind, ...)` at the end materialises a 6.46M × 3 matrix from a list of 6.46M length-3 vectors — an extremely expensive pattern in R.

### Combined effect
- ~6.46M × (1 + 5) = ~38.8M R-level anonymous-function calls.
- `paste`/character matching repeated millions of times.
- `do.call(rbind, list_of_millions)` is notoriously slow.
- Peak RAM easily exceeds 16 GB once you account for intermediate copies.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Row-level `lapply` + `paste` key lookup | Replace with **vectorised `data.table` merge/join** — build a flat `data.table` of `(row_index, neighbor_row_index)` in one shot using integer keys, not character keys. |
| Per-row `max/min/mean` in a loop | Compute all three stats in **one grouped `data.table` aggregation** per variable — completely vectorised in C. |
| `do.call(rbind, list_of_millions)` | Eliminated; `data.table` returns a single result table directly. |
| 5 separate passes over the edge list | Melt or loop is fine (only 5 iterations over a vectorised operation). |
| Memory: 6.46M-element list | Replaced by a flat two-column integer `data.table` of edges (~1.37M × 28 ≈ 38.5M rows × 2 cols × 8 bytes ≈ 0.6 GB). |

**Expected speed-up**: from 86+ hours to roughly **5–20 minutes** on the same laptop, well within 16 GB RAM.

**Preservation guarantees**:
- The trained Random Forest model is not touched.
- The output columns are numerically identical (`max`, `min`, `mean` of the same neighbor values), preserving the original estimand.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a flat edge table:  (row_i, neighbor_row_i)
#     Completely replaces build_neighbor_lookup().
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edge_table <- function(cell_data_dt, id_order, neighbors) {
 
  # --- Map each cell id to its position in id_order (integer key) ----
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build a data.table that maps (id, year) → row index -----------
  #     Using integer keys avoids paste / character matching entirely.
  row_key <- cell_data_dt[, .(id, year, row_i = .I)]
  setkey(row_key, id, year)

  # --- Expand the nb object into a flat edge list of cell-id pairs ---
  #     neighbors[[k]] gives the *positions in id_order* of the
  #     neighbours of cell id_order[k].
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_cells <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  # --- Cross-join edges with every year present in the data ----------
  years <- sort(unique(cell_data_dt$year))
  edge_cells_years <- edge_cells[, .(year = years), by = .(from_id, to_id)]
  rm(edge_cells)

  # --- Attach the originating row index (the row that *needs* the
  #     neighbour feature) ------------------------------------------------
  setnames(edge_cells_years, c("from_id"), c("id"))
  edge_cells_years <- row_key[edge_cells_years, on = .(id, year), nomatch = 0L]
  setnames(edge_cells_years, c("row_i", "id"), c("focal_row", "focal_id"))

  # --- Attach the neighbour row index ------------------------------------
  setnames(edge_cells_years, c("to_id"), c("id"))
  edge_cells_years <- row_key[edge_cells_years, on = .(id, year), nomatch = 0L]
  setnames(edge_cells_years, c("row_i", "id"), c("nbr_row", "nbr_id"))

  # Keep only the two columns we need for aggregation
  edge_cells_years[, .(focal_row, nbr_row)]
}

# ──────────────────────────────────────────────────────────────────────
# 2.  Vectorised neighbour stats for one variable.
#     Replaces compute_neighbor_stats() + compute_and_add_neighbor_features().
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_features_vec <- function(cell_data_dt, var_name, edges) {
  # Attach the neighbour's value to every edge
  edges[, nbr_val := cell_data_dt[[var_name]][nbr_row]]

  # Grouped aggregation — all three stats in one pass
  stats <- edges[!is.na(nbr_val),
                 .(nb_max  = max(nbr_val),
                   nb_min  = min(nbr_val),
                   nb_mean = mean(nbr_val)),
                 keyby = .(focal_row)]

  # Initialise output columns with NA (rows that have no valid neighbours)
  max_col  <- rep(NA_real_, nrow(cell_data_dt))
  min_col  <- rep(NA_real_, nrow(cell_data_dt))
  mean_col <- rep(NA_real_, nrow(cell_data_dt))

  max_col [stats$focal_row] <- stats$nb_max
  min_col [stats$focal_row] <- stats$nb_min
  mean_col[stats$focal_row] <- stats$nb_mean

  # Clean up the temporary column on the shared edges table
  edges[, nbr_val := NULL]

  # Return a named list of three vectors
  setNames(
    list(max_col, min_col, mean_col),
    paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  )
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Main pipeline — drop-in replacement for the outer loop.
# ──────────────────────────────────────────────────────────────────────
run_neighbor_feature_pipeline <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building flat edge table …")
  edges <- build_neighbor_edge_table(cell_data_dt, id_order,
                                     rook_neighbors_unique)
  setkey(edges, focal_row)
  message(sprintf("  Edge table: %s rows (%.1f MB)",
                  formatC(nrow(edges), big.mark = ","),
                  object.size(edges) / 1e6))

  neighbor_source_vars <- c("ntl", "ec", "pop_density",
                            "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbour stats for '%s' …", var_name))
    new_cols <- compute_neighbor_features_vec(cell_data_dt, var_name, edges)
    for (nm in names(new_cols)) {
      set(cell_data_dt, j = nm, value = new_cols[[nm]])
    }
  }

  message("Done.")
  # Return a data.table (also a data.frame) — compatible downstream
 cell_data_dt[]
}

# ── Usage (preserves trained RF model; drop-in replacement) ──────────
cell_data <- run_neighbor_feature_pipeline(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique
)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbour values as before (same rook-neighbor list, same year matching, same NA removal). |
| **Column naming** | Output columns follow the same `{var}_nb_max / _nb_min / _nb_mean` convention — adjust the suffix pattern if your RF model expects a different naming scheme. |
| **Trained RF model** | The code only adds feature columns to `cell_data`. The model object is never loaded, modified, or retrained. |
| **No approximation** | No sampling, no tiling, no lossy compression — the computation is exact. |

---

## Summary of Expected Gains

| Metric | Original | Optimised |
|---|---|---|
| R-level function calls | ~38.8 M | ~5 (one `data.table` grouped aggregation per variable) |
| Key construction | `paste` on 6.46M rows × 6 | Integer join keys only |
| Memory pattern | 6.46M-element R list (~1.3 GB overhead) | Flat integer `data.table` (~0.6 GB) |
| Estimated wall time | 86+ hours | **5–20 minutes** |
| Peak RAM | >16 GB (swapping) | ~8–12 GB (fits in 16 GB) |