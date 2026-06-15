 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and named-vector lookups, which are extremely slow in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which again loops over 6.46 million entries with per-element subsetting and summary computation. Together, these two stages dominate the 86+ hour runtime.

**Specific problems:**

1. **String-key lookups in `build_neighbor_lookup`:** For every row, `paste()` constructs neighbor keys and `idx_lookup[neighbor_keys]` performs named-vector lookup. With ~6.46M rows × ~4 neighbors on average, this is ~25M string constructions and lookups — all in an interpreted `lapply`.
2. **List-of-vectors output:** The neighbor lookup is a list of 6.46M integer vectors. This is memory-heavy (each list element has R object overhead) and forces downstream `lapply` iteration.
3. **`compute_neighbor_stats` iterates row-by-row** over the 6.46M-element list, computing `max`, `min`, `mean` per element — no vectorization.
4. **Memory pressure:** 6.46M rows × 110 columns is already ~5–6 GB as doubles. The neighbor lookup list and intermediate copies can push past 16 GB.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** operation. If we have an edge list `(id, neighbor_id)` and a panel keyed by `(id, year)`, then for each `(id, year)` we can join to get all `(neighbor_id, year)` rows and compute grouped statistics — all vectorized in C via `data.table`.

**Steps:**

1. Convert the `spdep::nb` object into a flat edge-list `data.table` with columns `(id, neighbor_id)`.
2. Convert the panel data to a `data.table` keyed on `(id, year)`.
3. For each neighbor source variable, perform a keyed join of the edge list against the panel to retrieve neighbor values, then compute `max`, `min`, `mean` grouped by `(id, year)`.
4. Merge the results back into the main table.

This eliminates all `lapply` loops, all string-key construction, and all per-row R overhead. Expected speedup: **~100–500×** (minutes instead of days). Memory is also reduced because we never materialize a 6.46M-element list.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Convert spdep::nb neighbor list to a flat edge-list DT
# ---------------------------------------------------------------
# rook_neighbors_unique is a list where element i contains the
# integer indices (into id_order) of neighbors of id_order[i].
# id_order is the vector of cell IDs in the order matching the nb object.

build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
      n_i <- length(nb_i)
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  # Trim if any nb entries were empty / zero-neighbor sentinels
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# Step 2: Convert panel data to data.table (in-place if possible)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns are proper types for joining
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]
edge_dt[, id := as.integer(id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# Step 3: For each variable, compute neighbor max/min/mean via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Build a slim lookup table: (id, year, value)
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Join edge list with the focal row's year, then look up neighbor values
  # Step A: attach year to each edge via the focal cell
  #   For each (id, neighbor_id) edge, we need one copy per year that id appears in.
  #   Instead of exploding edges × years, we join edges onto the panel.
  
  # focal_edges: for every (id, year) row, get all neighbor_ids
  # This is: cell_dt[, .(id, year)] joined with edge_dt on id
  focal <- cell_dt[, .(id, year)]
  setkey(focal, id)
  setkey(edge_dt, id)
  
  # This produces one row per (id, year, neighbor_id) — ~6.46M × ~4 ≈ 26M rows
  expanded <- edge_dt[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Step B: look up the neighbor's value in that year
  expanded[lookup, val := i.val, on = .(neighbor_id = id, year)]
  
  # Step C: compute grouped stats, dropping NAs
  stats <- expanded[!is.na(val),
                    .(nb_max  = max(val),
                      nb_min  = min(val),
                      nb_mean = mean(val)),
                    by = .(id, year)]
  
  # Rename columns to match the variable
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # Step D: merge back into cell_dt (left join to preserve all rows)
  # Remove old columns if they exist (idempotent re-runs)
  for (nm in new_names) {
    if (nm %in% names(cell_dt)) cell_dt[, (nm) := NULL]
  }
  
  cell_dt[stats, on = .(id, year), (new_names) := mget(paste0("i.", new_names))]
  
  invisible(cell_dt)
}

# ---------------------------------------------------------------
# Step 4: Run for all neighbor source variables
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_neighbor_features_dt(cell_data, edge_dt, var_name)
  gc()  # free intermediate memory between variables
}

cat("Done. Neighbor features added.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code — the edge list encodes the same rook-neighbor relationships, and the join on `(neighbor_id, year)` replicates the original `paste(id, year)` key lookup. NA handling is identical (`!is.na(val)` before aggregation). |
| **Trained Random Forest** | The RF model is not touched. We only modify the feature-construction step that feeds *new* data into `predict()`. The column names and semantics are preserved (`{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`). |
| **Column name convention** | Adjust the `new_names` pattern above if your existing pipeline uses a different naming convention (e.g., `ntl_max_neighbor`). |

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M `lapply` + string ops) | ~seconds (vectorized edge-list construction) |
| `compute_neighbor_stats` (per var) | ~17 hours (6.46M `lapply`) | ~30–90 seconds (`data.table` grouped join) |
| **Total (5 vars)** | **86+ hours** | **~5–10 minutes** |
| Peak RAM | >16 GB (list overhead) | ~8–12 GB (flat DT intermediates, freed per var with `gc()`) |

The entire pipeline should complete in under 15 minutes on a 16 GB laptop.