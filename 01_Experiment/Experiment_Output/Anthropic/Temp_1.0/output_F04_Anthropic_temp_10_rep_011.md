 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The suspicion is correct: **the spatial neighbor feature construction is the computational bottleneck**, not Random Forest inference. Here is why:

### `build_neighbor_lookup` — O(N) character-key lookups via `lapply`

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to character, looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes together `(neighbor_id, year)` string keys — one `paste()` call per neighbor per row.
4. Looks up each key in a named character vector (`idx_lookup`) — effectively a hash-table probe per key, but wrapped in R's slow single-threaded `lapply` over 6.46M iterations.

With ~1.37M directed neighbor relationships spread across 28 years, the total number of string constructions and lookups is roughly **1,373,394 × 28 ≈ 38.5 million paste + match operations**, all inside an interpreted R loop. The named-vector lookup `idx_lookup[neighbor_keys]` is an O(1)-amortised hash probe per key, but the per-element R interpreter overhead across millions of iterations dominates.

### `compute_neighbor_stats` — repeated R-level loops

For each of the 5 source variables, `compute_neighbor_stats` iterates over all 6.46M rows again in `lapply`, subsetting a numeric vector and computing `max/min/mean`. That is **5 × 6.46M = 32.3 million R-level function calls** with small-vector allocation overhead each time.

### Combined cost

The total is ~6.46M R-level iterations for the lookup build, plus ~32.3M R-level iterations for statistics — all sequential, all in interpreted R. On a laptop this easily reaches the estimated 86+ hours.

---

## Optimization Strategy

The core idea: **replace the row-level R `lapply` loops with vectorized `data.table` grouped joins and aggregations.** Specifically:

1. **Replace `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list of integer vectors, construct a long-form `data.table` edge table `(row_i, neighbor_row_j)` via a single vectorized merge. This eliminates millions of `paste` and named-vector lookups.

2. **Replace `compute_neighbor_stats` with a single grouped `data.table` aggregation.** Join the edge table to the source variable column, then `group by row_i` and compute `max`, `min`, `mean` in one vectorized pass — for all 5 variables together if desired.

3. **Memory is manageable.** The edge table has at most ~38.5M rows (1.37M neighbor pairs × 28 years), each row being two integers (8 bytes each) ≈ ~600 MB — fits within 16 GB alongside the 6.46M-row data.

Expected speedup: from 86+ hours to **minutes** (typically 5–20 minutes depending on disk/RAM speed), because `data.table` grouped operations are implemented in C and parallelised internally.

---

## Working R Code

```r
library(data.table)

#
# Step 0: Convert cell_data to data.table if not already;
#         assume columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#
cell_dt <- as.data.table(cell_data)

# Preserve original row order so the final result aligns with any downstream
# Random Forest predict() call that expects the same row order.
cell_dt[, .row_idx := .I]

# -----------------------------------------------------------------------
# Step 1: Build a vectorised edge table replacing build_neighbor_lookup
# -----------------------------------------------------------------------

# id_order is the vector whose positional index matches the nb object
# rook_neighbors_unique is the spdep nb list: rook_neighbors_unique[[k]]
# gives the positional indices (into id_order) of cell id_order[k]'s neighbors.

# 1a. Expand the nb list into a long-form edge list of (focal_id, neighbor_id).
#     This is ~1.37M rows — one per directed neighbor pair.
nb_lengths <- lengths(rook_neighbors_unique)
focal_pos  <- rep(seq_along(id_order), times = nb_lengths)
neigh_pos  <- unlist(rook_neighbors_unique)           # positional indices

edge_ids <- data.table(
  focal_id    = id_order[focal_pos],
  neighbor_id = id_order[neigh_pos]
)
rm(focal_pos, neigh_pos, nb_lengths)                  # free memory

# 1b. Cross-join with years to get (focal_id, year, neighbor_id) — the set of
#     all cell-year to neighbor-cell-year links.
#     Instead of an expensive explicit cross join (~38.5M rows built at once),
#     we merge through cell_dt which already has the (id, year) combinations.

# Create a keyed lookup: row index by (id, year)
row_lookup <- cell_dt[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# 1c. For every row in cell_dt, attach its neighbors via merge on focal_id == id.
#     Result: one row per (focal_row, neighbor_id, year).
focal_rows <- cell_dt[, .(focal_row = .row_idx, focal_id = id, year)]
setkey(edge_ids, focal_id)
setkey(focal_rows, focal_id)

# Merge: for each focal row, get all its neighbor cell IDs
edges <- edge_ids[focal_rows, on = "focal_id", allow.cartesian = TRUE,
                  nomatch = NULL]
# edges now has columns: focal_id, neighbor_id, focal_row, year

# 1d. Resolve neighbor_id + year → neighbor_row via the row_lookup
setkey(edges, neighbor_id, year)
setkey(row_lookup, id, year)
edges <- row_lookup[edges, on = c(id = "neighbor_id", "year"), nomatch = NA]
# After this join, .row_idx is the neighbor's row index.
# Rename for clarity:
setnames(edges, ".row_idx", "neighbor_row")

# Drop rows where the neighbor didn't exist in that year
edges <- edges[!is.na(neighbor_row)]

# Keep only what we need to save memory
edges <- edges[, .(focal_row, neighbor_row)]
setkey(edges, focal_row)

cat(sprintf("Edge table: %s rows (%.1f MB)\n",
            formatC(nrow(edges), big.mark = ","),
            object.size(edges) / 1e6))

# Free intermediate objects
rm(focal_rows, row_lookup, edge_ids)
gc()

# -----------------------------------------------------------------------
# Step 2: Compute neighbor statistics for all 5 variables — vectorised
# -----------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pull the neighbor row's values into the edge table, compute grouped stats.
# We do this one variable at a time to limit peak memory.

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Attach the variable's value at the neighbor row
  edges[, val := cell_dt[[var_name]][neighbor_row]]

  # Grouped aggregation: max, min, mean — excluding NAs
  stats <- edges[!is.na(val),
                 .(nmax  = max(val),
                   nmin  = min(val),
                   nmean = mean(val)),
                 by = focal_row]

  # Initialise new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Fill in computed values by row index
  cell_dt[stats$focal_row, (max_col)  := stats$nmax]
  cell_dt[stats$focal_row, (min_col)  := stats$nmin]
  cell_dt[stats$focal_row, (mean_col) := stats$nmean]

  rm(stats)
}

# Clean up helper column
edges[, val := NULL]

# -----------------------------------------------------------------------
# Step 3: Convert back to data.frame, preserving original row order, and
#         remove the helper column.
# -----------------------------------------------------------------------

cell_dt[, .row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# -----------------------------------------------------------------------
# Step 4: Predict with the EXISTING trained Random Forest (unchanged)
# -----------------------------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (The model object and predict call remain exactly as before.)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per cell-year as the original code. The `data.table` grouped aggregation is algebraically identical. |
| **Same row alignment** | The `.row_idx` bookkeeping ensures every computed statistic is written back to the correct row. The final `cell_data` data.frame has the same row order and column values as the original pipeline would produce. |
| **Trained RF untouched** | No model retraining occurs. The same `predict()` call on the same feature matrix yields identical predictions. |

---

## Expected Performance

| Stage | Original | Optimised |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M R-level iterations with string ops) | ~1–3 min (vectorised `data.table` keyed joins) |
| `compute_neighbor_stats` (×5 vars) | ~tens of hours (32.3M R-level `lapply` calls) | ~2–5 min (grouped C-level aggregation) |
| **Total neighbor feature construction** | **~86+ hours** | **~5–15 minutes** |
| Peak RAM | Moderate (6.46M-element list of int vectors) | ~1–3 GB (edge table + cell_dt) — fits in 16 GB |

The bottleneck is eliminated by moving from millions of interpreted R loop iterations with string allocation to a single vectorised join-then-aggregate pattern in `data.table`.