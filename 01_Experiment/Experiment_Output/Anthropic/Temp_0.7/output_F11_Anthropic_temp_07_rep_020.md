 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

**Why:**

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, each doing string construction and multiple named-vector lookups, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over **~6.46 million rows** with an `lapply` + subsetting + `max/min/mean` computation, then binding ~6.46M 3-element vectors with `do.call(rbind, ...)` — which is notoriously slow for large lists.

3. The total work is: ~6.46M iterations for the lookup build + 5 × ~6.46M iterations for neighbor stats = **~38.8 million R-level loop iterations** with per-iteration string operations, allocations, and named-vector lookups.

4. By contrast, Random Forest prediction on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M × 110 features. This is implemented in optimized C/Fortran within the `randomForest` or `ranger` package and typically completes in seconds to minutes, not hours.

**Conclusion:** The bottleneck is the row-level `lapply` loops with string-key lookups and the repeated `do.call(rbind, ...)` assembly. This is a classic R anti-pattern of element-wise iteration where vectorized or data.table-based operations should be used.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` with a vectorized `data.table` join.** Instead of building a list of neighbor row-indices per row (6.46M lists), construct an edge-list `data.table` mapping each `(id, year)` to its neighbor rows, then compute grouped statistics with `data.table`'s optimized `by=` grouping.

2. **Replace `compute_neighbor_stats()` with a single grouped `data.table` aggregation per variable.** Join the edge-list to the variable values and compute `max`, `min`, `mean` in one vectorized pass.

3. **Eliminate `do.call(rbind, ...)` on millions of small vectors.**

4. **Leave the Random Forest model and prediction step completely untouched.**

Expected speedup: from 86+ hours to roughly **minutes** (the bottleneck moves from O(N) R-level iterations to vectorized C-level data.table operations).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge list (replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────

build_neighbor_edge_list <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data must have columns: id, year (and be a data.table or coercible)
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: an nb object (list of integer neighbor index vectors)

  # --- Build directed edge list at the cell-ID level ---
  # Each element i of rook_neighbors_unique contains integer indices into id_order
  # representing neighbors of id_order[i].

  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Map indices back to actual cell IDs
  edges_id <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )

  # --- Cross with years to get row-level edges ---
  # Get unique years from the data
  dt <- as.data.table(cell_data)
  years <- sort(unique(dt$year))

  # Create a row-key table: (id, year) -> row_index in dt
  dt[, .row_idx := .I]
  key_table <- dt[, .(id, year, .row_idx)]

  # Expand edges across all years
  # This creates the full (focal_id, year, neighbor_id) table
  edge_year <- CJ_dt_edges(edges_id, years)

  # Join to get focal row index
  setnames(key_table, c("id", "year", ".row_idx"), c("focal_id", "year", "focal_row"))
  edge_year <- merge(edge_year, key_table, by = c("focal_id", "year"), all.x = FALSE)

  # Join to get neighbor row index
  setnames(key_table, c("focal_id", "year", "focal_row"), c("neighbor_id", "year", "neighbor_row"))
  edge_year <- merge(edge_year, key_table, by = c("neighbor_id", "year"), all.x = FALSE)

  # Restore names
  setnames(key_table, c("neighbor_id", "year", "neighbor_row"), c("id", "year", ".row_idx"))

  # Clean and return
  dt[, .row_idx := NULL]

  return(edge_year[, .(focal_row, neighbor_row)])
}

# Helper: cross-join edges with years efficiently
CJ_dt_edges <- function(edges_id, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge × every year
  result <- edges_id[, .(year = years), by = .(focal_id, neighbor_id)]
  return(result)
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats)
# ──────────────────────────────────────────────────────────────────────

compute_neighbor_stats_fast <- function(cell_data_dt, edge_list, var_name) {
  # edge_list has columns: focal_row, neighbor_row
  # Attach the neighbor values
  vals <- cell_data_dt[[var_name]]

  work <- edge_list[, .(focal_row, nval = vals[neighbor_row])]
  # Remove NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation
  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Build full-length result (NA for rows with no valid neighbors)
  n <- nrow(cell_data_dt)
  out <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  out[stats$focal_row, `:=`(
    nb_max  = stats$nb_max,
    nb_min  = stats$nb_min,
    nb_mean = stats$nb_mean
  )]

  # Name columns with variable prefix (matching original output convention)
  suffix <- c("_max", "_min", "_mean")
  setnames(out, paste0(var_name, "_neighbor", suffix))

  return(out)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Full optimized pipeline (drop-in replacement for outer loop)
# ──────────────────────────────────────────────────────────────────────

run_neighbor_feature_engineering <- function(cell_data, id_order, rook_neighbors_unique) {

  dt <- as.data.table(cell_data)

  message("Building vectorized edge list...")
  edge_list <- build_neighbor_edge_list(dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s rows", format(nrow(edge_list), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats_dt <- compute_neighbor_stats_fast(dt, edge_list, var_name)
    # Bind new columns to the main table
    dt <- cbind(dt, stats_dt)
  }

  message("Neighbor feature engineering complete.")
  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# Usage (replaces the original outer loop):
# ──────────────────────────────────────────────────────────────────────
# cell_data <- run_neighbor_feature_engineering(cell_data, id_order, rook_neighbors_unique)
#
# # Random Forest prediction is UNCHANGED — it was never the bottleneck:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup build | 6.46M `lapply` iterations with string ops | Vectorized `data.table` merge |
| Neighbor stats | 5 × 6.46M `lapply` + `do.call(rbind, ...)` | 5 × single grouped `data.table` aggregation |
| Estimated time | 86+ hours | Minutes |
| RF model | Untouched | Untouched |
| Numerical output | Preserved (same max/min/mean) | Preserved |

The true bottleneck is the **row-level R loop with string-key lookups repeated over ~6.46 million rows**, not Random Forest inference. The fix is to vectorize the entire neighbor feature computation using `data.table` joins and grouped aggregation.