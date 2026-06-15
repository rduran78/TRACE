 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The `idx_lookup` named vector has 6.46M entries. R's named-vector lookup is **O(N)** per query (linear scan or hashing with overhead). Doing this 6.46M times with ~4 keys each yields roughly **25.8 billion character comparisons** in the worst case. This is why the pipeline takes 86+ hours.

### The Broader Pattern

The entire design conflates two independent structures:

- **Spatial topology** (which cells neighbor which) — fixed across years.
- **Temporal alignment** (same year) — trivially handled by grouping.

By encoding both into a single string key (`id_year`), the code forces an expensive join at every row. The neighbor relationships are **identical across all 28 years**, so the lookup should be built **once on cell IDs** and then broadcast across years.

### `compute_neighbor_stats` Is Efficient *Given* the Lookup

Once `neighbor_lookup` (a list of integer row indices) is built, `compute_neighbor_stats` is a simple vectorized operation. The bottleneck is entirely in `build_neighbor_lookup`.

---

## Optimization Strategy

1. **Separate spatial topology from temporal alignment.** Build a cell-ID → neighbor-cell-IDs mapping once (344K entries, not 6.46M).
2. **Use `data.table` for fast equi-joins.** Instead of string-key named-vector lookups, use integer-keyed joins.
3. **Vectorize the neighbor-stats computation.** Expand the neighbor relationships into an edge table, join variable values, and compute grouped aggregates — all vectorized, zero R-level loops.
4. **Process all 5 variables in one pass** over the edge table rather than 5 separate passes.

### Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(R × K × N) string ops (R=rows, K=avg neighbors, N=lookup size) | O(E) integer join (E=edges×years) |
| Compute stats | O(R × K) — already fine | O(E) vectorized grouped aggregation |
| Total string allocs | ~25.8M | 0 |

Expected speedup: **~1000×** or more (minutes instead of days).

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED PIPELINE — drop-in replacement
# Preserves the exact same numerical output columns.
# Preserves the trained Random Forest model (no retraining).
# ============================================================

build_neighbor_edge_table <- function(id_order, neighbors) {

  # Build a data.table of directed spatial edges: focal_id -> neighbor_id

  # This is year-independent (topology is fixed).
  # neighbors is an nb object (list of integer index vectors into id_order).

  focal_indices <- which(lengths(neighbors) > 0)

  focal_ids <- rep(id_order[focal_indices], times = lengths(neighbors[focal_indices]))
  neighbor_ids <- id_order[unlist(neighbors[focal_indices])]

  data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
}

compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  # cell_data: data.frame or data.table with columns id, year, and the source vars
  # id_order: integer vector of cell IDs matching the nb object indexing

  # neighbors: spdep nb object (list of integer neighbor indices)
  # neighbor_source_vars: character vector of variable names

  dt <- as.data.table(cell_data)

  # 1. Build spatial edge table (year-independent): ~1.37M rows
  edges <- build_neighbor_edge_table(id_order, neighbors)

  # 2. Cross with years to get full edge table: ~1.37M × 28 ≈ 38.5M rows
  #    But we only need edges where both focal and neighbor exist in the data.
  #    Instead of a cross-join, we join through the data itself.

  # Create a row-key table: (id, year) -> row index + variable values
  dt[, row_idx := .I]

  # Subset to only the columns we need for the join
  value_cols <- neighbor_source_vars
  key_cols <- c("id", "year", value_cols)
  dt_key <- dt[, ..key_cols]

  # 3. For each focal row, find its neighbors in the same year.
  #    Join edges to focal rows to get (focal_id, year, neighbor_id),
  #    then join to neighbor rows to get neighbor variable values.

  # Step A: Join focal rows to edges on focal_id
  #   Result: each focal (id, year) is expanded to its neighbors
  setnames(edges, c("focal_id", "neighbor_id"))

  # Focal side: get (focal_id, year) pairs
  focal_dt <- dt[, .(focal_id = id, year)]

  # Merge: focal_dt × edges on focal_id → (focal_id, year, neighbor_id)
  # This is the big expansion: ~6.46M rows × ~4 neighbors = ~25.8M rows
  setkey(edges, focal_id)
  setkey(focal_dt, focal_id)
  expanded <- edges[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year

  # Step B: Join neighbor values on (neighbor_id, year)
  setnames(dt_key, "id", "neighbor_id")
  setkey(dt_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  merged <- dt_key[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged has: neighbor_id, year, <value_cols>, focal_id

  # 4. Compute grouped stats: max, min, mean per (focal_id, year) per variable
  #    We do all variables in one grouped operation.

  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in value_cols) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]] <- substitute(
      as.numeric(max(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
    agg_exprs[[paste0("n_min_", v)]] <- substitute(
      as.numeric(min(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(
      as.numeric(mean(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
  }

  # Custom aggregation that handles all-NA groups correctly (return NA)
  # We use a single lapply-based aggregation for clarity and correctness.
  stat_fn <- function(vals) {
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) return(list(NA_real_, NA_real_, NA_real_))
    list(max(vals), min(vals), mean(vals))
  }

  # Aggregate all variables
  result_list <- vector("list", length(value_cols))
  names(result_list) <- value_cols

  for (v in value_cols) {
    cat("Computing neighbor stats for:", v, "\n")
    agg <- merged[, {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(n_max = NA_real_, n_min = NA_real_, n_mean = NA_real_)
      } else {
        list(n_max = max(vals), n_min = min(vals), n_mean = mean(vals))
      }
    }, by = .(focal_id, year)]

    setnames(agg, c("n_max", "n_min", "n_mean"),
             paste0(c("n_max_", "n_min_", "n_mean_"), v))
    result_list[[v]] <- agg
  }

  # 5. Merge all results back to the original data
  #    Successive joins on (focal_id = id, year)
  for (v in value_cols) {
    agg <- result_list[[v]]
    setnames(agg, "focal_id", "id")
    dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  }

  # 6. Handle rows with no neighbors (isolated cells): they get NA, which is correct.

  # Clean up helper column
  dt[, row_idx := NULL]

  return(dt)
}

# ============================================================
# USAGE — replaces the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains the same columns as before:
#   n_max_ntl, n_min_ntl, n_mean_ntl,
#   n_max_ec,  n_min_ec,  n_mean_ec,
#   ... etc.
#
# The trained Random Forest model can be used directly for prediction
# on this data — no retraining needed.
```

---

## Memory Considerations (16 GB Laptop)

| Object | Estimated Size |
|---|---|
| `dt` (cell_data as data.table) | ~5.7 GB (6.46M × 110 cols) |
| `edges` (spatial) | ~10.5 MB (1.37M × 2 int cols) |
| `expanded` (edges × years) | ~580 MB (38.4M × 3 cols) |
| `merged` (expanded + 5 value cols) | ~2.3 GB (38.4M × 8 cols) |
| Per-variable `agg` | ~150 MB each |

**Peak: ~10 GB** — fits in 16 GB with headroom. If memory is tight, process variables one at a time and drop the `merged` column after each aggregation:

```r
# Memory-conservative variant: process one variable at a time
for (v in value_cols) {
  cat("Processing:", v, "\n")

  # Minimal merge: only bring the one variable we need
  dt_one <- dt[, .(neighbor_id = id, year, val = get(v))]
  setkey(dt_one, neighbor_id, year)

  merged_one <- dt_one[expanded, on = c("neighbor_id", "year"), nomatch = NA]

  agg <- merged_one[, {
    vals <- val[!is.na(val)]
    if (length(vals) == 0L) {
      list(n_max = NA_real_, n_min = NA_real_, n_mean = NA_real_)
    } else {
      list(n_max = max(vals), n_min = min(vals), n_mean = vals))
    }
  }, by = .(focal_id, year)]

  setnames(agg, c("n_max", "n_min", "n_mean"),
           paste0(c("n_max_", "n_min_", "n_mean_"), v))
  setnames(agg, "focal_id", "id")

  dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  rm(merged_one, agg, dt_one); gc()
}
```

*(Note: there's a typo in the memory-conservative snippet above — `mean(vals))` has an extra paren. Corrected version below.)*

```r
# Corrected memory-conservative variant
for (v in value_cols) {
  cat("Processing:", v, "\n")

  dt_one <- dt[, .(neighbor_id = id, year, val = get(v))]
  setkey(dt_one, neighbor_id, year)

  merged_one <- dt_one[expanded, on = c("neighbor_id", "year"), nomatch = NA]

  agg <- merged_one[, {
    vals <- val[!is.na(val)]
    if (length(vals) == 0L) {
      list(n_max = NA_real_, n_min = NA_real_, n_mean = NA_real_)
    } else {
      list(n_max = max(vals), n_min = min(vals), n_mean = mean(vals))
    }
  }, by = .(focal_id, year)]

  setnames(agg, c("n_max", "n_min", "n_mean"),
           paste0(c("n_max_", "n_min_", "n_mean_"), v))
  setnames(agg, "focal_id", "id")

  dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  rm(merged_one, agg, dt_one); gc()
}
```

## Expected Runtime

| Phase | Estimated Time |
|---|---|
| Build edge table | < 1 second |
| Expand edges × focal rows | ~10 seconds |
| Merge neighbor values (per var) | ~20 seconds |
| Grouped aggregation (per var) | ~30 seconds |
| Final merge back (per var) | ~10 seconds |
| **Total (5 variables)** | **~5–10 minutes** |

**Down from 86+ hours to under 10 minutes** — roughly a **500–1000× speedup** — by eliminating the row-level string-key lookups and replacing them with vectorized `data.table` integer-keyed joins.