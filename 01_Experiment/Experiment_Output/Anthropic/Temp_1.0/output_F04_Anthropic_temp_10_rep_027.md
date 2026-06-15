 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Specifically:

1. **`build_neighbor_lookup` uses `lapply` over ~6.46 million rows.** Each iteration performs character coercion, `paste`-based key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash probing per call, and constructing ~6.46M × ~4 average neighbors ≈ ~26M `paste` and name-match operations inside an interpreted loop. This is the dominant cost.

2. **`compute_neighbor_stats` is a secondary bottleneck.** It loops over 6.46M entries, subsetting a numeric vector each time. While cheaper per iteration, `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself expensive (repeated memory allocation).

3. **The `paste`-based keying strategy is fundamentally slow.** Building string keys for every cell-year-neighbor combination creates massive temporary character vectors and relies on R's slow string hashing.

**Root cause summary:** An interpreted R loop over 6.46 million rows doing string operations and named-vector lookups for each row, repeated once for building the lookup and then once per variable for stats computation.

---

## Optimization Strategy

### Principle: Replace string keys and R-level loops with vectorized integer-indexed operations using `data.table`.

**Key ideas:**

1. **Replace the string-keyed lookup with integer join.** Instead of `paste(id, year)` keys, assign each row an integer row index and build a `data.table` with `(id, year) → row_index` for O(1) keyed joins.

2. **Vectorize `build_neighbor_lookup`.** Expand all neighbor relationships into a flat edge table `(row_i, neighbor_id, year)`, then batch-join to resolve `neighbor_id + year → row_j` in one vectorized `data.table` merge. This eliminates the 6.46M-iteration `lapply`.

3. **Vectorize `compute_neighbor_stats`.** Once we have a flat `(row_i, row_j)` edge list, computing `max/min/mean` of neighbor values is a single grouped aggregation: `edges[, .(max, min, mean), by = row_i]`.

4. **Memory check:** The flat edge list will have ~6.46M rows × ~4 neighbors ≈ ~26M rows of two integers ≈ ~200 MB. Feasible on 16 GB RAM.

5. **Preserve the trained RF model and numerical outputs exactly.** We compute identical `max`, `min`, `mean` statistics — just faster.

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────────────
# Step 0: Convert cell_data to data.table (if not already) and add row index
# ─────────────────────────────────────────────────────────────────────
setDT(cell_data)
cell_data[, .row_idx := .I]

# ─────────────────────────────────────────────────────────────────────
# Step 1: Build a flat directed edge table from the nb object
#         (one-time cost, fully vectorized)
# ─────────────────────────────────────────────────────────────────────
build_neighbor_edges_dt <- function(cell_data, id_order, neighbors) {
  # Map: position in id_order → cell id
  # neighbors[[k]] gives the positions (in id_order) that are neighbors of id_order[k]

  n_cells <- length(id_order)

  # Expand nb list into a flat edge table: (from_id, to_id)
  # from_pos = index in id_order, to_pos = neighbor indices in id_order
  from_pos <- rep(seq_len(n_cells), lengths(neighbors))
  to_pos   <- unlist(neighbors)

  # Remove zero-length / empty neighbor entries (spdep convention: 0L means no neighbors)
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]

  edges_cell <- data.table(
    from_id = id_order[from_pos],
    to_id   = id_order[to_pos]
  )

  # Get unique years
  years <- sort(unique(cell_data$year))

  # Cross-join edges with years to get all (from_id, year, to_id) combinations
  # This is the full set of potential neighbor lookups
  edges_full <- edges_cell[, .(year = years), by = .(from_id, to_id)]

  # Build keyed lookup: (id, year) → .row_idx
  row_key <- cell_data[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # Resolve from_id,year → row_i
  edges_full[, row_i := row_key[.(from_id, year), .row_idx, on = .(id, year)]]

  # Resolve to_id,year → row_j (the neighbor's row)
  edges_full[, row_j := row_key[.(to_id, year), .row_idx, on = .(id, year)]]

  # Drop unresolved (NA) — cells/years not present in the data
  edges_resolved <- edges_full[!is.na(row_i) & !is.na(row_j), .(row_i, row_j)]

  setkey(edges_resolved, row_i)
  return(edges_resolved)
}

cat("Building edge table...\n")
system.time({
  edge_dt <- build_neighbor_edges_dt(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~26M rows, completes in seconds to low minutes.

# ─────────────────────────────────────────────────────────────────────
# Step 2: Vectorized neighbor stat computation + feature attachment
# ─────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_dt) {
  # Extract the variable values indexed by row
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the edge table
  edge_dt[, nval := vals[row_j]]

  # Grouped aggregation — single vectorized pass
  stats <- edge_dt[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    by = row_i
  ]

  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign by reference using integer row indices
  cell_data[stats$row_i, (max_col)  := stats$nb_max]
  cell_data[stats$row_i, (min_col)  := stats$nb_min]
  cell_data[stats$row_i, (mean_col) := stats$nb_mean]

  # Clean up temporary column from edge_dt
  edge_dt[, nval := NULL]

  invisible(cell_data)
}

# ─────────────────────────────────────────────────────────────────────
# Step 3: Outer loop — compute neighbor features for all source vars
# ─────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  }
})

# ─────────────────────────────────────────────────────────────────────
# Step 4: Drop helper column, apply the existing trained RF model
# ─────────────────────────────────────────────────────────────────────
cell_data[, .row_idx := NULL]

# The trained Random Forest model is unchanged — predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Expected Performance Improvement

| Phase | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` (6.46M `lapply` + string ops) | ~70–80 hours | Replaced by `build_neighbor_edges_dt`: **2–5 minutes** |
| `compute_neighbor_stats` × 5 vars (6.46M `lapply` + `do.call(rbind)`) | ~6–10 hours | Replaced by 5 × `data.table` grouped aggregation: **1–3 minutes** |
| **Total neighbor feature construction** | **~86 hours** | **~5–8 minutes** |

The speedup comes from eliminating ~32 million interpreted R-level string operations and replacing them with `data.table`'s radix-key joins and grouped aggregations, which execute in compiled C code. The numerical outputs (max, min, mean of non-NA neighbor values per row) are identical, so the trained Random Forest model requires no changes.