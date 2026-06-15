 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for three compounding reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function:
- Looks up the cell's reference index in `id_to_ref`.
- Extracts neighbor cell IDs from the `nb` object.
- Pastes together `neighbor_key` strings (character allocation for 6.46M × ~4 neighbors ≈ 25M+ string operations).
- Looks up those keys in `idx_lookup` (a named character vector — O(n) hashing overhead per call).

This produces a **list of 6.46 million integer vectors**. The string pasting and named-vector lookup inside an R-level loop is the primary bottleneck. The neighbor topology is **year-invariant** — every cell has the same neighbors in every year — yet the lookup is rebuilt per cell-year row, redundantly repeating the same neighbor resolution 28 times per cell.

### 2. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean by iterating an R `lapply` over 6.46M elements, each time subsetting a numeric vector and calling `max`/`min`/`mean`, is extremely slow due to R's interpreter overhead. This is repeated for each of the 5 source variables (5 × 6.46M = 32.3M R-level function calls).

### 3. Memory pressure from the 6.46M-element list
Storing `neighbor_lookup` as a list of 6.46 million integer vectors creates massive R object overhead (each list element has its own SEXP header). On a 16 GB laptop this risks swapping.

**Core insight:** The neighbor graph is a property of the spatial grid, not of time. There are only 344,208 cells, each with ~4 rook neighbors. The adjacency table has ~1.37M directed edges. This is tiny. The explosion to 6.46M rows happens only because the panel has 28 years. The correct approach is:

> Build a **cell-level** adjacency table once (344K × ~4 = ~1.37M edge rows), then **join** yearly attributes onto it, and compute grouped summaries using vectorized `data.table` operations.

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| **A** | Convert the `nb` object to a two-column `data.table` of directed edges (`cell_id`, `neighbor_id`). This is ~1.37M rows, built once. | Eliminates per-row string pasting and named-vector lookup entirely. |
| **B** | Convert `cell_data` to a `data.table` keyed on `(id, year)`. | Enables keyed joins in O(n log n) instead of character hashing. |
| **C** | Join the edge table to the panel data by `(neighbor_id, year)` to pull each neighbor's attribute values. This produces a ~1.37M × 28 ≈ ~38.5M row "long" table (edges × years) per variable — but we do it for all variables at once. | Vectorized, cache-friendly, no R-level loop. |
| **D** | Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one `data.table` aggregation. | `data.table`'s grouped aggregation is C-level and parallelized via OpenMP. |
| **E** | Join the resulting summary columns back onto `cell_data`. | Single keyed join, O(n log n). |
| **F** | Predict with the existing trained Random Forest model on the enriched `cell_data`. | Model is untouched; column names and numerical values are preserved exactly. |

**Expected speedup:** From ~86 hours to **minutes** (typically 2–10 minutes on a 16 GB laptop), because:
- The edge table join is ~38.5M rows, handled vectorially by `data.table`.
- Grouped aggregation on 38.5M rows with 4-key groups is a bread-and-butter `data.table` operation.
- No R-level loop over 6.46M or 32.3M iterations.

**Memory:** Peak usage is the 38.5M-row join table with ~7 columns ≈ ~2 GB, well within 16 GB.

**Numerical equivalence:** `max`, `min`, and `mean` over the same non-NA neighbor values produce identical results. The column names are constructed identically (`n_max_ntl`, `n_min_ntl`, `n_mean_ntl`, etc.), so the trained Random Forest model sees exactly the same feature matrix.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP A: Build a cell-level directed edge table from the nb object (once)
# ===========================================================================
build_edge_table <- function(id_order, nb_obj) {
  # id_order: vector of cell IDs in the same order as nb_obj

# nb_obj:   spdep::nb list — nb_obj[[i]] gives integer indices of neighbors of cell i
  from <- rep(
    seq_along(nb_obj),
    times = lengths(nb_obj)
  )
  to <- unlist(nb_obj, use.names = FALSE)

  # Remove zero-neighbor placeholders (spdep uses 0L for no-neighbor entries)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]

  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

# ===========================================================================
# STEP B–E: Compute all neighbor features in one vectorized pass
# ===========================================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {
  # Convert to data.table if not already (non-destructive copy)
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }

  # Columns to carry from the neighbor rows
  cols_to_fetch <- intersect(neighbor_source_vars, names(dt))

  # ---- Join edges × years to get neighbor attribute values ----
  # We need: for each (cell_id, year), the values of cols_to_fetch from each neighbor.
  # 1. Cross-join edges with years is wasteful. Instead, join edges onto the panel.


  # Build a slim table of neighbor attributes keyed on (id, year)
  neighbor_attrs <- dt[, c("id", "year", cols_to_fetch), with = FALSE]
  setnames(neighbor_attrs, "id", "neighbor_id")
  setkeyv(neighbor_attrs, c("neighbor_id", "year"))

  # Also need to know which (cell_id, year) pairs exist — use edges joined to years
  # Strategy: start from dt's (id, year), expand via edge_dt, then join attributes.

  # Create the cell-year backbone with just id and year
  cell_year <- dt[, .(cell_id = id, year = year)]
  setkeyv(cell_year, "cell_id")
  setkeyv(edge_dt, "cell_id")

  # Inner join: for each cell-year row, get all neighbor_ids
  # Result: one row per (cell_id, year, neighbor_id)
  expanded <- edge_dt[cell_year, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year

  # Now join neighbor attributes onto expanded
  setkeyv(expanded, c("neighbor_id", "year"))
  expanded <- neighbor_attrs[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # expanded now has: neighbor_id, year, <cols_to_fetch>, cell_id

  # ---- Grouped aggregation: max, min, mean per (cell_id, year) ----
  agg_exprs <- list()
  for (v in cols_to_fetch) {
    max_name  <- paste0("n_max_",  v)
    min_name  <- paste0("n_min_",  v)
    mean_name <- paste0("n_mean_", v)
    agg_exprs[[max_name]]  <- call("max",  as.name(v), na.rm = TRUE)
    agg_exprs[[min_name]]  <- call("min",  as.name(v), na.rm = TRUE)
    agg_exprs[[mean_name]] <- call("mean", as.name(v), na.rm = TRUE)
  }

  # Build the j-expression programmatically
  j_expr <- as.call(c(as.name("list"), agg_exprs))

  stats <- expanded[, eval(j_expr), by = .(cell_id, year)]

  # Replace Inf / -Inf from max/min of empty sets with NA (matches original behavior)
  for (col in names(stats)) {
    if (is.numeric(stats[[col]])) {
      set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
    }
  }

  # Also replace NaN from mean of empty sets
  for (col in names(stats)) {
    if (is.numeric(stats[[col]])) {
      set(stats, which(is.nan(stats[[col]])), col, NA_real_)
    }
  }

  # ---- Join stats back onto the main data table ----
  setnames(stats, "cell_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  # Remove any pre-existing neighbor columns to avoid duplication
  neighbor_cols <- setdiff(names(stats), c("id", "year"))
  existing <- intersect(neighbor_cols, names(dt))
  if (length(existing) > 0) {
    dt[, (existing) := NULL]
  }

  dt <- stats[dt, on = c("id", "year")]

  return(dt)
}

# ===========================================================================
# STEP F: Full pipeline — drop-in replacement
# ===========================================================================

# --- 1. Build edge table (once, ~1.37M rows, takes < 1 second) ---
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# --- 2. Compute all neighbor features (vectorized, ~2-10 min) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  edge_dt,
  neighbor_source_vars
)

# cell_data is now a data.table. Convert back to data.frame if the RF model expects one:
# cell_data <- as.data.frame(cell_data)

# --- 3. Predict with the existing trained Random Forest (unchanged) ---
# The trained model object (e.g., `rf_model`) is preserved as-is.
# Column names match the original: n_max_ntl, n_min_ntl, n_mean_ntl, ...
# Numerical values are identical (same max/min/mean over same non-NA neighbor sets).

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | R-level `lapply` over 6.46M rows with string pasting | One-time vectorized `data.table` edge table (1.37M rows) |
| **Stat computation** | R-level `lapply` over 6.46M rows × 5 vars = 32.3M calls | Single `data.table` grouped aggregation on ~38.5M rows |
| **Estimated runtime** | ~86+ hours | **~2–10 minutes** |
| **Peak RAM** | Massive list overhead (~6.46M SEXP headers) | ~2 GB for the expanded join table |
| **RF model** | Untouched ✓ | Untouched ✓ |
| **Numerical output** | `max`/`min`/`mean` over non-NA neighbor values | Identical `max`/`min`/`mean` over identical non-NA neighbor values ✓ |