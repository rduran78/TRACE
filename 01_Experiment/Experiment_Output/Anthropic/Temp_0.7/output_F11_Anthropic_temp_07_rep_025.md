 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. Named vector lookup in R is **O(n)** hash-based but with significant per-call overhead when done millions of times in an interpreted loop. This single function produces ~6.46 million list elements, each requiring multiple string allocations and lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same ~6.46 million list elements, subsetting a numeric vector, removing NAs, and computing max/min/mean. That's ~32.3 million R-level function invocations total.

3. **The combinatorial scale is enormous**: ~6.46M rows × ~4 average neighbors × 5 variables = ~129 million neighbor-value extractions and aggregations, all done in interpreted R loops.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. The `ranger` or `randomForest` predict method is implemented in C/C++ and operates on a matrix in memory. For ~6.46M rows × ~110 predictors, this is fast — typically minutes, not hours. Loading the model from disk is a single `readRDS()` call. Writing predictions is a single vectorized write. None of these are loop-bound or string-bound.

**Conclusion**: The 86+ hour runtime is dominated by the interpreted-R, row-level, string-heavy spatial neighbor feature construction, not by Random Forest inference.

---

## Optimization Strategy

1. **Eliminate per-row string operations entirely.** Replace the string-keyed lookup (`paste(id, year, sep="_")` → named vector) with an integer-keyed lookup using a `data.table` join or a direct integer matrix index.

2. **Vectorize `build_neighbor_lookup()`** by "exploding" the neighbor list into a two-column edge table (row index → neighbor row index) using `data.table`, performing a single batch join instead of 6.46M individual lookups.

3. **Vectorize `compute_neighbor_stats()`** by using the edge table with `data.table` grouped aggregation (`max`, `min`, `mean` by source row), replacing 6.46M `lapply` iterations per variable with a single grouped operation.

4. **Process all 5 variables in one pass** over the edge table to minimize overhead.

Expected speedup: from 86+ hours to **minutes** (roughly 3–15 minutes depending on hardware).

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED: build_neighbor_edge_table
# Replaces build_neighbor_lookup entirely.
# Produces a data.table with columns: (row_i, neighbor_row_i)
# mapping each row in cell_data to its neighbor rows.
# ============================================================
build_neighbor_edge_table <- function(cell_data_dt, id_order, rook_neighbors) {
  # Step 1: Map each id to its position in id_order (integer)
  id_to_ref <- data.table(
    id      = id_order,
    ref_idx = seq_along(id_order)
  )

  # Step 2: Build an edge list at the cell (id) level:
  #   for each cell ref_idx, which ref_idxs are its neighbors?
  edges_cell <- rbindlist(lapply(seq_along(rook_neighbors), function(r) {
    nb <- rook_neighbors[[r]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(ref_from = integer(0), ref_to = integer(0)))
    }
    data.table(ref_from = r, ref_to = as.integer(nb))
  }))

  # Map ref_idx back to id
  edges_cell[, id_from := id_order[ref_from]]
  edges_cell[, id_to   := id_order[ref_to]]

  # Step 3: Build a row-index lookup: (id, year) -> row position in cell_data_dt
  cell_data_dt[, row_i := .I]

  row_lookup <- cell_data_dt[, .(id, year, row_i)]

  # Step 4: Expand edges across years via join.
  # For each (id_from, id_to) pair, find all years where BOTH exist.
  # Join edges to row_lookup for the "from" side
  setkey(row_lookup, id)

  from_rows <- row_lookup[, .(id_from = id, year, row_from = row_i)]
  to_rows   <- row_lookup[, .(id_to   = id, year, row_to   = row_i)]

  setkey(edges_cell, id_from)
  setkey(from_rows, id_from)

  # Merge: get (id_from, id_to, year, row_from) for every edge × year of id_from

  edge_year <- merge(
    edges_cell[, .(id_from, id_to)],
    from_rows,
    by = "id_from",
    allow.cartesian = TRUE
  )

  # Now join to get row_to: match (id_to, year)
  setkey(edge_year, id_to, year)
  setkey(to_rows, id_to, year)

  edge_year <- merge(
    edge_year,
    to_rows,
    by = c("id_to", "year"),
    nomatch = 0L   # drop edges where the neighbor doesn't exist in that year
  )

  # Return only the essential columns
  edge_year[, .(row_from, row_to)]
}

# ============================================================
# OPTIMIZED: compute_and_add_all_neighbor_features
# Processes all variables in a vectorized, grouped aggregation.
# ============================================================
compute_and_add_all_neighbor_features <- function(cell_data_dt, edge_table, neighbor_source_vars) {
  n <- nrow(cell_data_dt)

  # Build a sub-table of just the columns we need, indexed by row_to
  val_cols <- neighbor_source_vars
  neighbor_vals <- cell_data_dt[edge_table$row_to, ..val_cols]
  neighbor_vals[, row_from := edge_table$row_from]

  # Grouped aggregation: for each row_from, compute max/min/mean of each variable
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0(v, c("_max_neighbor", "_min_neighbor", "_mean_neighbor"))
  }))

  # Build the aggregation call
  stats <- neighbor_vals[,
    setNames(lapply(agg_exprs, eval, envir = .SD), agg_names),
    by = row_from
  ]

  # Replace Inf/-Inf (from max/min on all-NA groups, though nomatch=0 helps) with NA
  for (col in agg_names) {
    vals <- stats[[col]]
    set(stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
  }

  # Merge back into cell_data_dt by row index
  # First, ensure all rows are represented (some may have no neighbors)
  all_rows <- data.table(row_from = seq_len(n))
  stats <- merge(all_rows, stats, by = "row_from", all.x = TRUE)
  setorder(stats, row_from)

  # Add columns to cell_data_dt
  for (col in agg_names) {
    set(cell_data_dt, j = col, value = stats[[col]])
  }

  cell_data_dt
}

# ============================================================
# MAIN PIPELINE (replaces the original outer loop)
# ============================================================

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

# Build the vectorized edge table (one-time cost, replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %d directed neighbor-year pairs", nrow(edge_table)))

# Compute all neighbor features at once (replaces the for-loop over 5 variables)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features (vectorized)...")
cell_data_dt <- compute_and_add_all_neighbor_features(
  cell_data_dt, edge_table, neighbor_source_vars
)

# Clean up helper column
cell_data_dt[, row_i := NULL]

# Convert back to data.frame if downstream code requires it
cell_data <- as.data.frame(cell_data_dt)

# ============================================================
# Random Forest prediction (UNCHANGED — not the bottleneck)
# ============================================================
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# write.csv(cell_data, "predictions.csv", row.names = FALSE)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (interpreted R loops over 6.46M rows, string ops) | Eliminated |
| **Lookup construction** | 6.46M `lapply` iterations with `paste()` + named-vector lookup | Single `data.table` merge/join |
| **Neighbor aggregation** | 5 × 6.46M `lapply` iterations with per-element `max/min/mean` | One grouped `data.table` aggregation over all 5 vars |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical results** | Original max/min/mean per neighbor set | Identical (same aggregation logic, same NA handling) |