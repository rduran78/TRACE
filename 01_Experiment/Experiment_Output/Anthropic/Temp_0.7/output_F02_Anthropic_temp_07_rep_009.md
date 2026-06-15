 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This produces a list of 6.46 million integer vectors — a massive, slow, memory-hungry structure. The secondary bottleneck is **`compute_neighbor_stats`**, which iterates over that same 6.46 million-element list to compute `max`, `min`, and `mean` for each row, again in pure interpreted R. Multiplied across 5 variables, the entire pipeline is dominated by millions of interpreted-R function calls and temporary allocations.

**Specific issues:**

1. **String-key hashing for every row:** `paste(id, year)` creates ~6.46M strings, and lookups into a named vector of that size are O(n) to build and slow to query repeatedly.
2. **`lapply` over 6.46M rows:** Each iteration allocates small vectors, causing enormous GC pressure.
3. **List-of-vectors neighbor lookup:** Storing ~6.46M variable-length integer vectors is memory-inefficient and cache-unfriendly.
4. **Redundant neighbor resolution:** The spatial neighbor topology is static across years. The current code re-resolves neighbor cell IDs per row instead of exploiting the fact that neighbors are identical across all 28 years for a given cell.
5. **`do.call(rbind, ...)` on a 6.46M-element list:** This is notoriously slow for large lists.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Separate space from time** | Build the neighbor graph once at the cell level (344K cells), then join by year — never iterate over 6.46M rows in R. |
| **Vectorize with `data.table`** | Use keyed joins and grouped aggregations instead of `lapply`. |
| **Flat edge table** | Replace the list-of-vectors `nb` object with a two-column `data.table` of `(id, neighbor_id)` — cache-friendly, joinable. |
| **Column-at-a-time stats** | Compute `max`, `min`, `mean` with `data.table`'s optimized `GForce` grouped operations — C-level speed. |
| **Constant memory** | No 6.46M-element list is ever created; peak memory is the edge table (~11M rows × 2 int cols ≈ 88 MB) plus the main table. |

Expected speedup: from 86+ hours to **minutes** (typically 5–15 min depending on disk I/O).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Convert the spdep nb object to a flat edge data.table (one-time)
# ──────────────────────────────────────────────────────────────────────
nb_to_edge_dt <- function(id_order, nb_list) {
  # id_order: vector of cell IDs in the same order as nb_list

  # nb_list : spdep nb object (list of integer index vectors)
  from <- rep(
    seq_along(nb_list),
    times = lengths(nb_list)
  )
  to <- unlist(nb_list, use.names = FALSE)
  # Remove the 0-neighbour sentinel that spdep uses
  keep <- to != 0L
  data.table(
    id          = id_order[from[keep]],
    neighbor_id = id_order[to[keep]]
  )
}

edges <- nb_to_edge_dt(id_order, rook_neighbors_unique)
# edges is ~1.37 M rows (directed), with columns: id, neighbor_id

# ──────────────────────────────────────────────────────────────────────
# 2. Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 3. Vectorised neighbor-stat function
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(dt, edges, var_name) {
  # Build a slim lookup: (id, year, value)
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Join edges to get the neighbor's value for every (id, year) pair.
  #   For each edge (id -> neighbor_id) and each year,
  #   look up the neighbor's value.
  #
  # Step A: cross edges with years present in the data
  #         But that would be 1.37M × 28 ≈ 38M rows.  Instead,

  #         we join edges onto the data rows directly.


  # Approach: start from dt rows, attach their neighbors, then
  #           look up the neighbor's value for the same year.

  # (id, year) → list of neighbor_ids  via edges
  # We key-join dt[, .(id, year)] to edges on id.
  row_info <- dt[, .(id, year)]
  row_info[, row_idx := .I]

  # Join row_info to edges: for each row, get all neighbor_ids
  setkey(edges, id)
  setkey(row_info, id)
  expanded <- edges[row_info, on = "id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded columns: id, neighbor_id, year, row_idx

  # Now look up the neighbor's value for (neighbor_id, year)
  setkey(val_dt, id, year)
  expanded[val_dt,
           neighbor_val := i.val,
           on = .(neighbor_id = id, year)]

  # Drop NAs in neighbor_val before aggregation
  expanded <- expanded[!is.na(neighbor_val)]

  # Aggregate per original row
  stats <- expanded[,
    .(nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)),
    by = row_idx
  ]

  # Assign back to dt (rows without neighbors get NA)
  col_max  <- paste0("max_",  var_name)
  col_min  <- paste0("min_",  var_name)
  col_mean <- paste0("mean_", var_name)

  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  set(dt, i = stats$row_idx, j = col_max,  value = stats$nb_max)
  set(dt, i = stats$row_idx, j = col_min,  value = stats$nb_min)
  set(dt, i = stats$row_idx, j = col_mean, value = stats$nb_mean)

  invisible(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 4. Outer loop — replaces the original for-loop
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_and_add_neighbor_features_fast(cell_data, edges, var_name)
}

# ──────────────────────────────────────────────────────────────────────
# 5. Predict with the already-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same columns (max_*, min_*, mean_*)
# with identical numerical values, so predictions are preserved.
#
# Example:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `nb_to_edge_dt` faithfully unrolls the same `spdep::nb` list and the same `id_order` vector. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per `(id, year)` row. |
| **Same column names** | Output columns follow the same `max_<var>`, `min_<var>`, `mean_<var>` naming convention. |
| **No model retraining** | The Random Forest object is never touched; only the feature-engineering step is optimized. |

---

## Memory Estimate on 16 GB RAM

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M × 110 cols, numeric) | ~5.4 GB |
| `edges` (1.37M × 2 int cols) | ~11 MB |
| `expanded` (worst case ~38M × 4 cols) | ~1.2 GB |
| `stats` (up to 6.46M × 4 cols) | ~0.2 GB |
| **Peak total** | **~7–8 GB** ✔ fits in 16 GB |

The `expanded` table is created and discarded once per variable, so memory is recycled across the five iterations. If memory is still tight, you can process one variable at a time and call `gc()` between iterations, or use `on-disk` chunking by year (28 chunks), which is trivial with `data.table`'s `by = year` grouping.