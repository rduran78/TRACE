 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, each constructed via an `lapply` call that performs per-row string pasting, hash lookups, and NA filtering. This is O(n) in rows × O(k) in neighbors, but with enormous R-interpreter overhead per iteration (string allocation, named-vector lookup, subsetting). The 86+ hour estimate comes almost entirely from:

1. **`build_neighbor_lookup`**: ~6.46M iterations of `lapply`, each doing `paste`, named-vector indexing (`id_to_ref`, `idx_lookup`), and NA removal. Named-vector lookup in R is O(n) in the worst case for long vectors, and `paste` allocates a new string each time. This alone can take tens of hours.

2. **`compute_neighbor_stats`**: Another 6.46M-iteration `lapply` per variable (×5 variables = ~32.3M iterations), each subsetting a numeric vector and computing max/min/mean. The per-call overhead is smaller but still substantial.

3. **Memory pressure**: Storing 6.46M list elements, each an integer vector, plus intermediate string vectors, can push past 16 GB and trigger garbage collection thrashing.

**Root cause**: The code solves a spatial-temporal join problem using R-level loops over millions of rows with string-key lookups, when this is fundamentally a vectorized join + grouped aggregation problem.

---

## Optimization Strategy

### Key Insight
The neighbor lookup can be expressed as a **join table** (an edge list in cell-year space), and the statistics as a **grouped aggregation** over that join — both of which `data.table` handles in vectorized C code.

**Steps:**

1. **Build a directed edge list** from the `nb` object: `(from_id, to_id)` for all rook-neighbor pairs. This is ~1.37M rows (spatial edges only, time-invariant).

2. **Cross-join with years** to get a cell-year edge list: `(from_id, year, to_id)`. This is ~1.37M × 28 ≈ ~38.5M rows — large but manageable in RAM as a 3-column integer `data.table` (~900 MB).

3. **Join** the edge list to the data to attach each neighbor's variable values.

4. **Group by `(from_id, year)`** and compute `max`, `min`, `mean` — a single vectorized `data.table` aggregation.

5. **Join results back** to the main data table.

This replaces ~86 hours of R-level looping with a few vectorized operations that should complete in **minutes**.

### Memory Management
- Process one variable at a time to avoid duplicating the full edge-list with all 5 variable columns simultaneously.
- The cross-joined edge list (~38.5M rows × 3 integer columns) uses ~900 MB. Each variable join adds one double column (~308 MB). Total peak is well under 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Prepare data  (assumed available: cell_data, id_order,
#     rook_neighbors_unique, the trained RF model)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table if not already (non-destructive; keeps all columns)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure there is a row key for final merge-back
cell_data[, .rowid := .I]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build the spatial edge list from the nb object  (~1.37M rows)
# ──────────────────────────────────────────────────────────────────────

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) → nothing,
  # but some nb objects store a 0L for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_spatial <- build_edge_list(id_order, rook_neighbors_unique)
# ~ 1.37M rows, two integer (or numeric) columns

cat(sprintf("Spatial edge list: %s rows\n", format(nrow(edge_spatial), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 2.  Cross-join with years to get the cell-year edge list  (~38.5M rows)
# ──────────────────────────────────────────────────────────────────────

years <- sort(unique(cell_data$year))   # 1992:2019, 28 values

edge_cy <- edge_spatial[, .(year = years), by = .(from_id, to_id)]
# This is a cross join: each spatial edge × each year
# Result: ~1.37M × 28 ≈ 38.5M rows

# Set keys for fast joins
setkey(edge_cy, to_id, year)

cat(sprintf("Cell-year edge list: %s rows\n", format(nrow(edge_cy), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 3.  For each neighbor source variable, compute max/min/mean and attach
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-build a minimal keyed lookup table: id, year, + all 5 vars
# (avoids re-keying inside the loop)
lookup_cols <- c("id", "year", neighbor_source_vars)
val_lookup  <- cell_data[, ..lookup_cols]
setnames(val_lookup, "id", "to_id")
setkey(val_lookup, to_id, year)

for (var_name in neighbor_source_vars) {

  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Join: attach the neighbor's value to each edge
  # (only the column we need, to save memory)
  edges_with_val <- val_lookup[edge_cy, .(from_id, year, val = get(var_name)), on = .(to_id, year)]

  # Remove NAs before aggregation (matches original logic)
  edges_with_val <- edges_with_val[!is.na(val)]

  # Grouped aggregation
  stats <- edges_with_val[,
    .(
      nbr_max  = max(val),
      nbr_min  = min(val),
      nbr_mean = mean(val)
    ),
    by = .(from_id, year)
  ]

  # Rename columns to match the original naming convention
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"), new_names)

  # Merge back to cell_data (left join so cells with no neighbors get NA)
  setnames(stats, "from_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_data, c("id", "year"))

  cell_data <- stats[cell_data, on = .(id, year)]

  # Clean up to free memory
  rm(edges_with_val, stats)
  gc()

  cat(sprintf("  Done. Columns added: %s\n", paste(new_names, collapse = ", ")))
}

# Restore original row order
setorder(cell_data, .rowid)
cell_data[, .rowid := NULL]

cat("All neighbor features computed.\n")

# ──────────────────────────────────────────────────────────────────────
# 4.  Prediction with the existing trained Random Forest
#     (No retraining — just call predict with the enriched data)
# ──────────────────────────────────────────────────────────────────────
# Example (adapt to your actual model object name):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string hashing | Vectorized edge-list construction + `data.table` cross-join |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M `lapply` iterations | 5 vectorized `data.table` grouped aggregations |
| Estimated runtime | 86+ hours | **~5–15 minutes** |
| Peak RAM | Potentially > 16 GB (list + strings + GC) | ~4–6 GB (edge list + one variable column at a time) |
| Numerical equivalence | baseline | **Identical** `max`, `min`, `mean` over non-NA rook neighbors |
| RF model | — | **Untouched; no retraining** |

The numerical estimand is preserved exactly: for each `(cell, year)`, the max, min, and mean are computed over the same set of non-NA rook-neighbor values as in the original code. The trained Random Forest model is never modified or retrained.