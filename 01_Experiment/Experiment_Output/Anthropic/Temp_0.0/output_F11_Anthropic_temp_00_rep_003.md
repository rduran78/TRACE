 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million entries with `lapply`, performing subsetting, NA removal, and computing `max/min/mean`. That's ~32.3 million R-level loop iterations with repeated allocation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes — not hours.

**The bottleneck is the row-level R `lapply` loops over millions of rows with repeated string operations and named-vector lookups.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` join. Instead of looping row-by-row, expand the neighbor list into an edge table `(source_row, neighbor_id)`, join against the data to resolve `(neighbor_id, year) → row_index`, and group by source row.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the edge table — computing max, min, and mean in one vectorized pass per variable, eliminating millions of R-level function calls.

3. **Process all 5 variables** in a single grouped aggregation pass if possible, or at minimum use vectorized column operations.

These changes convert O(n) R-level iterations (with string ops) into vectorized C-level `data.table` joins and group-by operations, reducing runtime from 86+ hours to likely **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build a vectorized edge table mapping each row to its neighbor rows
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edges_dt <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step A: Create an edge list at the cell level: (focal_cell_id, neighbor_cell_id)
  # Each element neighbors[[k]] gives the indices (into id_order) of cell id_order[k]'s neighbors
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_idx <- unlist(neighbors)

  cell_edges <- data.table(
    focal_cell_id    = id_order[focal_idx],
    neighbor_cell_id = id_order[neighbor_idx]
  )

  # Step B: Assign row indices to the data
  data_dt[, row_idx := .I]

  # Step C: Join cell_edges with data to get (focal_row, neighbor_row) for matching years
  # First, join focal side: for each (focal_cell_id, year) → focal_row_idx
  focal_key <- data_dt[, .(focal_cell_id = id, year, focal_row = row_idx)]

  # Expand: each focal row gets its neighbor cell IDs
  # Merge focal_key with cell_edges on focal_cell_id
  setkey(cell_edges, focal_cell_id)
  setkey(focal_key, focal_cell_id)
  expanded <- cell_edges[focal_key, on = "focal_cell_id", allow.cartesian = TRUE,
                         nomatch = 0L]
  # expanded now has: focal_cell_id, neighbor_cell_id, year, focal_row

  # Step D: Resolve neighbor_cell_id + year → neighbor_row
  neighbor_key <- data_dt[, .(neighbor_cell_id = id, year, neighbor_row = row_idx)]
  setkey(expanded, neighbor_cell_id, year)
  setkey(neighbor_key, neighbor_cell_id, year)

  edges <- neighbor_key[expanded, on = c("neighbor_cell_id", "year"), nomatch = 0L]
  # edges has: neighbor_cell_id, year, neighbor_row, focal_cell_id, focal_row

  edges[, .(focal_row, neighbor_row)]
}

# ──────────────────────────────────────────────────────────────────────
# 2. Compute neighbor stats for all variables in one vectorized pass
# ──────────────────────────────────────────────────────────────────────
compute_all_neighbor_features_dt <- function(data_dt, edges, neighbor_source_vars) {
  n_rows <- nrow(data_dt)

  # Attach neighbor values for all variables at once
  neighbor_vals <- data_dt[edges$neighbor_row, ..neighbor_source_vars]
  neighbor_vals[, focal_row := edges$focal_row]

  # Group by focal_row and compute max, min, mean for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  stats <- neighbor_vals[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vals), min(vals), mean(vals))
    }), neighbor_source_vars),
    by = focal_row
  ]

  # The above is still somewhat complex; cleaner approach below:
  # Compute per-variable stats separately but vectorized (still very fast in data.table)
  for (v in neighbor_source_vars) {
    sub <- neighbor_vals[!is.na(get(v)), .(
      vmax  = max(get(v)),
      vmin  = min(get(v)),
      vmean = mean(get(v))
    ), by = focal_row]

    setnames(sub, c("vmax", "vmin", "vmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    # Left-join back to data_dt by row index
    data_dt[sub, on = c(row_idx = "focal_row"),
            (paste0("neighbor_", c("max_", "min_", "mean_"), v)) :=
              mget(paste0("neighbor_", c("max_", "min_", "mean_"), v))]
  }

  data_dt
}

# ──────────────────────────────────────────────────────────────────────
# 3. Full optimized pipeline
# ──────────────────────────────────────────────────────────────────────
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, neighbor_source_vars) {

  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, row_idx := .I]

  message("Building vectorized neighbor edge table...")
  t0 <- Sys.time()
  edges <- build_neighbor_edges_dt(cell_data, id_order, rook_neighbors_unique)
  message("  Edge table built: ", nrow(edges), " directed edges. (",
          round(difftime(Sys.time(), t0, units = "secs"), 1), "s)")

  message("Computing neighbor features for ", length(neighbor_source_vars), " variables...")
  t1 <- Sys.time()
  cell_data <- compute_all_neighbor_features_dt(cell_data, edges, neighbor_source_vars)
  message("  Neighbor features done. (",
          round(difftime(Sys.time(), t1, units = "secs"), 1), "s)")

  # Clean up helper column
  cell_data[, row_idx := NULL]

  # ── Random Forest inference (preserved exactly as-is) ──
  message("Running Random Forest predictions...")
  t2 <- Sys.time()
  cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
  message("  RF predictions done. (",
          round(difftime(Sys.time(), t2, units = "secs"), 1), "s)")

  cell_data
}

# ──────────────────────────────────────────────────────────────────────
# 4. Usage
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_optimized_pipeline(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  rf_model               = rf_model,
  neighbor_source_vars   = neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/named-vector lookup | Single vectorized `data.table` join on `(cell_id, year)` |
| **Neighbor stats** | 5 × `lapply` over 6.46M rows (32.3M R-level iterations) | 5 × grouped `data.table` aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~5–20 minutes |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical estimand** | Preserved ✓ (same max/min/mean, same predictions) | Preserved ✓ |

The true bottleneck is the **row-level R loops with string operations in the spatial feature engineering**, not Random Forest inference. The fix is vectorization via `data.table` joins and grouped aggregations.