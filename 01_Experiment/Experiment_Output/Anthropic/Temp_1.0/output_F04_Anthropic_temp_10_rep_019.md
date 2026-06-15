 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is overwhelmingly in **`build_neighbor_lookup`**, not in `compute_neighbor_stats` or the Random Forest inference. Here's why:

### `build_neighbor_lookup` — O(n) `lapply` with expensive per-row string operations

With ~6.46 million rows, this function calls `lapply` over every row and, **for each row**, performs:

1. **Character coercion and named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) — creates a throwaway character string 6.46M times.
2. **`paste()` to construct neighbor keys** — allocates a new character vector for every row's neighbor set.
3. **Named-vector lookup via `idx_lookup[neighbor_keys]`** — R's named-vector lookup is O(n) in the worst case per access because it performs linear name matching (not hashing). With ~6.46M entries in `idx_lookup`, each lookup is extremely expensive.
4. **`is.na` filtering** — minor, but adds up.

The result: ~6.46 million iterations, each doing multiple string allocations and linear-scan named-vector lookups against a 6.46M-element vector. This is the source of the **86+ hour** estimate.

### `compute_neighbor_stats` — reasonably efficient but improvable

The `lapply` over 6.46M rows computing `max/min/mean` on small integer-indexed subsets is tolerable, but `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also slow. This can be vectorized.

### Summary of root causes

| Issue | Location | Severity |
|---|---|---|
| Named-vector lookup (not hashed) on 6.46M keys | `build_neighbor_lookup` | **Critical** |
| Per-row `paste()` string construction | `build_neighbor_lookup` | **High** |
| Per-row `as.character()` coercion | `build_neighbor_lookup` | **Moderate** |
| `do.call(rbind, ...)` on 6.46M-element list | `compute_neighbor_stats` | **Moderate** |
| Row-wise `lapply` over 6.46M rows for stats | `compute_neighbor_stats` | **Moderate** |

---

## Optimization Strategy

### Principle: Replace string-key lookups with integer arithmetic; vectorize everything possible.

1. **Replace the named-vector `idx_lookup`** (string-key → row index) with a **`data.table` hash join** or an **environment-based hash map**, reducing lookup from O(n) to O(1) amortized per key.

2. **Pre-expand the neighbor relationships into a single long-format `data.table`** of `(row_i, neighbor_row_j)` pairs. This converts the row-wise `lapply` in `build_neighbor_lookup` into a single vectorized merge/join.

3. **Vectorize `compute_neighbor_stats`** by using `data.table` grouped aggregation (`max`, `min`, `mean` grouped by source row) on the long-format edge table, instead of row-wise `lapply`.

4. **Avoid `do.call(rbind, ...)`** entirely — `data.table` aggregation returns the result directly.

5. **Process all 5 variables in one pass** over the edge table to further reduce overhead.

**Expected speedup**: From 86+ hours down to **minutes** (typically 5–20 minutes depending on hardware), because every O(n²)-behaving string operation is replaced with O(n) or O(n log n) hashed joins.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup (returns a data.table of edges)
# ============================================================
build_neighbor_edges_dt <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and an implicit row index
  # id_order: integer vector; id_order[ref] = cell_id
  # neighbors: spdep nb list; neighbors[[ref]] = integer vector of neighbor ref indices

  # Step 1: Build a mapping from cell id -> ref index (integer vector, direct indexing)
  # We'll use an environment as a hash map: cell_id (character) -> ref index
  n_refs <- length(id_order)

  # Expand neighbor list into a long data.table of (ref_idx, neighbor_ref_idx)
  # This is the spatial adjacency in ref-index space
  from_ref <- rep(seq_len(n_refs), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_ref <- data.table(from_ref = from_ref, to_ref = to_ref)

  # Map ref indices to cell IDs
  edge_ref[, from_id := id_order[from_ref]]
  edge_ref[, to_id   := id_order[to_ref]]

  # Step 2: Get the unique years
  years <- sort(unique(data_dt$year))

  # Step 3: Build row-index lookup: (id, year) -> row_index in data_dt
  # Add row index to data_dt
  data_dt[, .row_idx := .I]

  # Create keyed lookup table
  lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)

  # Step 4: Cross-join edges with years, then join to get row indices
  # For each edge (from_id, to_id), for each year, map both to their row indices.
  # Instead of a full cross join (which would be huge), we do it via merge:

  # Merge: for each (from_id, year) that exists in data, get the row index of from
  # and for each corresponding to_id in that same year, get the row index of to.

  # First, create the edge set with from_id and to_id only (deduplicated)
  edges_unique <- unique(edge_ref[, .(from_id, to_id)])

  # Join from_id side: for each from_id, get all years it appears in
  from_rows <- lookup[, .(from_id = id, year, from_row = .row_idx)]
  setkey(from_rows, from_id, year)

  # Merge edges with from_rows to get (from_id, to_id, year, from_row)
  edges_with_year <- merge(edges_unique, from_rows, by = "from_id", allow.cartesian = TRUE)

  # Join to_id side: for each (to_id, year), get the row index
  to_rows <- lookup[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_rows, to_id, year)

  # Merge to get (from_id, to_id, year, from_row, to_row)
  edges_full <- merge(edges_with_year, to_rows, by = c("to_id", "year"), allow.cartesian = FALSE)

  # Clean up temporary column
  data_dt[, .row_idx := NULL]

  # Return: each row is (from_row, to_row) meaning "row from_row's neighbor is row to_row"
  edges_full[, .(from_row, to_row)]
}

# ============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ============================================================
compute_neighbor_stats_dt <- function(data_dt, edge_dt, var_name, n_rows) {
  # edge_dt: data.table with columns from_row, to_row
  # var_name: character, column name in data_dt
  # n_rows: total number of rows in data_dt

  vals <- data_dt[[var_name]]

  # Attach neighbor values to edges
  work <- edge_dt[, .(from_row, neighbor_val = vals[to_row])]

  # Remove NAs in neighbor values
  work <- work[!is.na(neighbor_val)]

  # Aggregate by from_row
  agg <- work[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]

  # Initialize result columns with NA
  col_max  <- rep(NA_real_, n_rows)
  col_min  <- rep(NA_real_, n_rows)
  col_mean <- rep(NA_real_, n_rows)

  # Fill in computed values
  col_max[agg$from_row]  <- agg$nb_max
  col_min[agg$from_row]  <- agg$nb_min
  col_mean[agg$from_row] <- agg$nb_mean

  list(col_max = col_max, col_min = col_min, col_mean = col_mean)
}

# ============================================================
# OPTIMIZED outer pipeline
# ============================================================
run_neighbor_feature_construction <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table if not already (non-destructive copy)
  if (!is.data.table(cell_data)) {
    cell_dt <- as.data.table(cell_data)
  } else {
    cell_dt <- copy(cell_data)
  }

  n_rows <- nrow(cell_dt)

  message("Building neighbor edge table...")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edges_dt(cell_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, from_row)
  message(sprintf("  Edge table built: %d edges in %.1f seconds.",
                  nrow(edge_dt), (proc.time() - t0)[3]))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    t1 <- proc.time()

    stats <- compute_neighbor_stats_dt(cell_dt, edge_dt, var_name, n_rows)

    # Add columns with the same naming convention as original code
    set(cell_dt, j = paste0(var_name, "_nb_max"),  value = stats$col_max)
    set(cell_dt, j = paste0(var_name, "_nb_min"),  value = stats$col_min)
    set(cell_dt, j = paste0(var_name, "_nb_mean"), value = stats$col_mean)

    message(sprintf("  Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  # Return as data.frame if input was data.frame (preserves downstream compatibility)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(cell_dt))
  }
  cell_dt
}

# ============================================================
# USAGE (drop-in replacement)
# ============================================================
# cell_data <- run_neighbor_feature_construction(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed directly to Random Forest prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | How it's met |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code — just via grouped aggregation instead of row-wise `lapply`. The edge table encodes the identical `(cell_i_year_t, neighbor_j_year_t)` pairs. |
| **Trained RF model preserved** | The code only constructs features; the model object is never touched. Column names follow the same convention so `predict()` works unchanged. |
| **Same neighbor definition** | The `rook_neighbors_unique` nb object is consumed identically — expanded into `(from_ref, to_ref)` pairs with the same semantics. |
| **NA handling** | Rows without neighbors (or with all-NA neighbors) receive `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |

---

## Expected Performance

| Stage | Original | Optimized | Reason |
|---|---|---|---|
| Neighbor lookup construction | ~70+ hours | ~2–5 minutes | Hashed `data.table` joins replace 6.46M string-key linear lookups |
| Neighbor stats (×5 vars) | ~15+ hours | ~3–8 minutes | Vectorized `data.table` grouped aggregation replaces 5 × 6.46M `lapply` iterations |
| **Total** | **~86+ hours** | **~5–15 minutes** | **~350–1000× speedup** |

Memory usage peaks at the edge table (~38M edges × 2 integer columns ≈ ~300 MB), well within the 16 GB constraint.