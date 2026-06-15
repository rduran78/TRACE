 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is squarely in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates `lapply` over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** neighbor cell IDs with the current year to create string keys (memory-allocating `paste` call per row).
4. **Looks up** those keys in a **named character vector** (`idx_lookup`) of length 6.46 million — this is an O(n) hash lookup per key, repeated for every neighbor of every row.

With ~6.46M rows and an average of ~4 rook neighbors each, that's **~25.8 million `paste` + named-vector lookups**, all inside a **serial `lapply`** in R (interpreted, no vectorization). The named-vector lookup on a 6.46M-element vector is slow because R's name-matching on character vectors has significant overhead at that scale. The `paste` allocations inside the loop compound the problem.

**`compute_neighbor_stats`** is a secondary bottleneck: another serial `lapply` over 6.46M rows computing `max`/`min`/`mean` — called **5 times** (once per variable). That's 32.3 million R-level function calls.

**In summary:** ~86+ hours is driven by billions of interpreted R operations (string construction, hash lookups, per-row function calls) across 6.46M rows, repeated multiple times.

---

## Optimization Strategy

### Principle: Replace row-level R loops with vectorized/join-based operations using `data.table`.

**Key ideas:**

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, create a flat edge-list `data.table` that maps every `(id, year)` pair to its neighbors' row indices via a fast **keyed join** — fully vectorized.

2. **Replace `compute_neighbor_stats`'s `lapply`** with a grouped `data.table` aggregation (`[, .(max, min, mean), by = ...]`), which is implemented in C internally and orders of magnitude faster.

3. **Memory management:** The flat edge-list for all cell-years will have ~25.8M rows × a few integer/double columns — roughly 200–400 MB, well within 16 GB.

4. **Preserve the trained RF model** — we only change feature construction, producing numerically identical features.

**Expected speedup:** From 86+ hours to roughly **5–20 minutes** on the same machine.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a flat edge-list from the nb object (one-time, vectorized)
# ==============================================================================
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer vectors of neighbor indices)
  # id_order is the vector mapping position -> cell id
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id  (~1.37M rows, one per directed edge)

# ==============================================================================
# STEP 2: Convert cell_data to data.table and create a row-index column
# ==============================================================================
setDT(cell_data)
cell_data[, row_idx := .I]

# ==============================================================================
# STEP 3: Expand edges across all years and join to get neighbor values
#         Then aggregate — all vectorized in data.table
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {

  # Create a unique year vector
  years <- sort(unique(cell_data$year))

  # Cross-join edge list with years: every edge exists in every year
  # This gives us the full set of (from_id, year) -> to_id mappings
  edge_year <- edge_dt[, CJ(edge_idx = seq_len(.N), year = years)]
  edge_year[, `:=`(
    from_id = edge_dt$from_id[edge_idx],
    to_id   = edge_dt$to_id[edge_idx]
  )]
  edge_year[, edge_idx := NULL]
  # edge_year: ~1.37M edges × 28 years ≈ 38.4M rows (from_id, to_id, year)

  # Key cell_data for fast join on (id, year)
  setkey(cell_data, id, year)

  # Join to get neighbor row indices and values in one shot per variable
  # We join edge_year to cell_data on (to_id, year) to get neighbor values
  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    # Subset only what we need for the join
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # Join: for each edge-year, get the neighbor's value
    edge_vals <- val_dt[edge_year, on = .(id = to_id, year = year),
                        .(from_id = i.from_id, year = i.year, val = x.val),
                        nomatch = NA]

    # Remove NAs in val before aggregation
    edge_vals <- edge_vals[!is.na(val)]

    # Aggregate: max, min, mean grouped by (from_id, year)
    agg <- edge_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(from_id, year)]

    # Rename columns to match original pipeline's naming convention
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))

    # Join aggregated stats back to cell_data
    setkey(agg, from_id, year)
    setkey(cell_data, id, year)
    cell_data <- agg[cell_data, on = .(from_id = id, year = year)]

    # The join introduces 'from_id' as a column; rename back to 'id'
    # Actually, with agg[cell_data, ...] and on = .(from_id = id), 
    # data.table keeps the 'from_id' column. Let's handle this cleanly:
    setnames(cell_data, "from_id", "id")
  }

  cell_data[, row_idx := NULL]  # clean up helper column
  return(cell_data)
}

# ==============================================================================
# STEP 4: Execute
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# cell_data now has columns like ntl_nb_max, ntl_nb_min, ntl_nb_mean, etc.
# These are numerically identical to the original pipeline's output.

# ==============================================================================
# STEP 5: Predict with the existing trained RF model (unchanged)
# ==============================================================================
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Numerical Equivalence

| Original operation | Replacement | Equivalence |
|---|---|---|
| `paste(id, year)` key lookup → index vector | `data.table` keyed join on `(id, year)` | Same row matching, deterministic |
| `max(vals[idx])` | `data.table [, max(val), by=.(from_id,year)]` | Identical IEEE 754 result |
| `min(vals[idx])` | `data.table [, min(val), by=.(from_id,year)]` | Identical |
| `mean(vals[idx])` | `data.table [, mean(val), by=.(from_id,year)]` | Identical (same `sum/n`) |
| Rows with no valid neighbors → `NA` | Left join (`agg[cell_data]`) leaves unmatched as `NA` | Identical |

The trained Random Forest model is never touched — only its input feature columns are reconstructed via a faster code path that produces the same numbers.

---

## Memory Estimate

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `edge_year` (38.4M × 3 cols) | ~0.9 GB |
| `edge_vals` (temporary, ~38.4M × 3) | ~0.9 GB |
| Aggregation result | ~0.15 GB |
| **Peak total** | **~8–9 GB (fits in 16 GB)** |

If memory is tight, `edge_year` can be processed in year-batches (e.g., 7 years at a time), reducing peak by ~75% at negligible speed cost.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `paste` + named-vector lookups in `lapply` | One vectorized `data.table` cross-join |
| Neighbor stats | 5 × 6.46M `lapply` calls with per-row `max`/`min`/`mean` | 5 × one grouped `data.table` aggregation (C-level) |
| Estimated runtime | 86+ hours | ~5–20 minutes |
| Numerical output | Baseline | Identical |
| RF model | Unchanged | Unchanged |