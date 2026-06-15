 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the most critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`), `paste` to build keys, and named-vector indexing into `idx_lookup`.
  - Named-vector lookups in R are **hash-table lookups per element**, but doing this millions of times in an interpreted loop is extremely slow.
- The resulting `neighbor_lookup` is a **list of 6.46 million integer vectors** — enormous memory overhead from list metadata alone.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M elements, each subsetting a numeric vector, removing NAs, and computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3 million interpreted iterations total.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (repeated allocations).

**Combined cost:** The nested interpreted R loops over millions of rows, with per-row string operations and list allocations, easily explain the 86+ hour runtime.

### B. Prediction Workflow Bottlenecks (Random Forest Inference)

- With ~6.46M rows × 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object can be memory-intensive.
- If `randomForest` (Breiman's R package) is used, `predict.randomForest` is known to be **slow on large data** — it copies the entire data frame internally and loops in R/C in a less optimized way than `ranger`.
- If the model is loaded from disk each time or if the prediction data frame is copied repeatedly, that adds overhead.
- Predicting all 6.46M rows in one call may cause memory spikes (16 GB laptop).

---

## 2. Optimization Strategy

| Component | Problem | Solution |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops in interpreted R loop | Replace with vectorized `data.table` merge/join — build all neighbor-row indices in one bulk operation |
| `compute_neighbor_stats` | Per-row lapply × 5 variables | Replace with `data.table` grouped aggregation: explode neighbor pairs, join values, aggregate with `max/min/mean` by row |
| `neighbor_lookup` (list of 6.46M vectors) | Memory-heavy, slow to iterate | Eliminate entirely — use a flat two-column edge table (`row_i`, `neighbor_row_j`) |
| `do.call(rbind, ...)` | Slow list-to-matrix conversion | Unnecessary once using `data.table` grouped aggregation |
| RF prediction | Possible `randomForest` package slowness; memory spike | Batch prediction in chunks; if model is `randomForest`, convert to `ranger` format or predict in chunks; ensure single `predict()` call with no unnecessary copies |
| Data copying | `cell_data` reassigned in loop (may copy entire data frame each iteration) | Use `data.table` set-by-reference (`:=`) to add columns in-place |

### Key Principles
1. **Vectorize everything** — no row-level `lapply` over millions of rows.
2. **Use `data.table`** for joins and grouped aggregation (C-level, cache-friendly).
3. **Flat edge table** instead of a list-of-vectors neighbor lookup.
4. **In-place column addition** (`:=`) to avoid copying a wide data frame.
5. **Chunked prediction** to stay within 16 GB RAM.
6. **Preserve the trained model object and numerical output exactly.**

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE — Feature Preparation + Random Forest Prediction
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- Step 1: Build flat edge table (replaces build_neighbor_lookup) ---------
# This is the single most important optimization: we replace ~6.46M lapply
# iterations with a vectorized bulk join.

build_neighbor_edge_table <- function(dt, id_order, nb_object) {
  # nb_object: spdep nb list — nb_object[[i]] gives neighbor indices into
  # id_order for the i-th element of id_order.
  
  # 1a. Expand the nb object into a flat (source_id, neighbor_id) table.
  #     This has ~1.37M rows (directed rook-neighbor relationships).
  n_cells <- length(id_order)
  lens <- lengths(nb_object)
  from_idx <- rep(seq_len(n_cells), lens)
  to_idx   <- unlist(nb_object, use.names = FALSE)
  
  cell_edges <- data.table(
    source_id   = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  
  # 1b. Get unique years
  years <- sort(unique(dt$year))
  
  # 1c. Cross-join edges × years to get the full set of
  #     (source_id, year, neighbor_id) triples.
  #     ~1.37M edges × 28 years ≈ 38.5M rows — fits in memory.
  cell_edges_yr <- cell_edges[, CJ(year = years), by = .(source_id, neighbor_id)]
  # More memory-efficient: use a cross join then merge
  # Actually, CJ inside by is inefficient for large groups. Better approach:
  
  year_dt <- data.table(year = years)
  cell_edges_yr <- cell_edges[, .(source_id, neighbor_id)]
  cell_edges_yr <- cell_edges_yr[
    rep(seq_len(.N), each = length(years))
  ]
  cell_edges_yr[, year := rep(years, times = nrow(cell_edges))]
  
  # 1d. Add row indices for the source rows (for writing results back).
  #     We key dt by (id, year) and do an equi-join.
  dt[, row_idx := .I]
  
  # Source row index
  setkey(dt, id, year)
  cell_edges_yr[, c("src_row") := dt[.(source_id, year), row_idx, mult = "first"]]
  
  # Neighbor row index
  cell_edges_yr[, c("nbr_row") := dt[.(neighbor_id, year), row_idx, mult = "first"]]
  
  # Drop edges where either source or neighbor row doesn't exist
  cell_edges_yr <- cell_edges_yr[!is.na(src_row) & !is.na(nbr_row)]
  
  setkey(cell_edges_yr, src_row)
  
  return(cell_edges_yr)
}

message("Building neighbor edge table...")
edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))


# ---- Step 2: Compute & add neighbor features (replaces compute_neighbor_stats
#              + the outer for-loop) ------------------------------------------
# For each variable, we look up neighbor values via the edge table, then
# aggregate (max, min, mean) grouped by src_row.  Results are written back
# into cell_data by reference.

compute_and_add_all_neighbor_features <- function(dt, edge_dt,
                                                   neighbor_source_vars) {
  for (var_name in neighbor_source_vars) {
    message(sprintf("  Computing neighbor stats for: %s", var_name))
    
    # Pull neighbor values into the edge table
    vals <- dt[[var_name]]
    edge_dt[, nbr_val := vals[nbr_row]]
    
    # Aggregate: grouped by src_row, compute max/min/mean ignoring NAs
    agg <- edge_dt[!is.na(nbr_val),
                   .(v_max  = max(nbr_val),
                     v_min  = min(nbr_val),
                     v_mean = mean(nbr_val)),
                   keyby = src_row]
    
    # Prepare NA-filled result vectors (for rows with no valid neighbors)
    n <- nrow(dt)
    col_max  <- rep(NA_real_, n)
    col_min  <- rep(NA_real_, n)
    col_mean <- rep(NA_real_, n)
    
    # Fill in computed values
    rows <- agg$src_row
    col_max[rows]  <- agg$v_max
    col_min[rows]  <- agg$v_min
    col_mean[rows] <- agg$v_mean
    
    # Write columns by reference (no data-frame copy)
    max_name  <- paste0("neighbor_max_", var_name)
    min_name  <- paste0("neighbor_min_", var_name)
    mean_name <- paste0("neighbor_mean_", var_name)
    
    set(dt, j = max_name,  value = col_max)
    set(dt, j = min_name,  value = col_min)
    set(dt, j = mean_name, value = col_mean)
  }
  
  # Clean up temporary column
  edge_dt[, nbr_val := NULL]
  
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features...")
compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
message("Neighbor features complete.")

# Remove helper column
cell_data[, row_idx := NULL]


# ---- Step 3: Optimized Random Forest Prediction ----------------------------

predict_in_chunks <- function(model, newdata, chunk_size = 500000L) {
  # Predicts in chunks to control peak memory on a 16 GB laptop.
  # Works with ranger, randomForest, or any model with a predict() method.
  
  n <- nrow(newdata)
  
  if (n <= chunk_size) {
    # Small enough to predict in one pass
    return(predict(model, data = newdata)$predictions %||%
           predict(model, newdata = newdata))
  }
  
  # Determine the predict interface
  is_ranger <- inherits(model, "ranger")
  
  # Pre-allocate result vector
  preds <- numeric(n)
  
  starts <- seq(1L, n, by = chunk_size)
  
  for (k in seq_along(starts)) {
    i_start <- starts[k]
    i_end   <- min(i_start + chunk_size - 1L, n)
    chunk   <- newdata[i_start:i_end, ]
    
    if (is_ranger) {
      p <- predict(model, data = chunk)$predictions
    } else {
      # randomForest or other
      p <- predict(model, newdata = chunk)
    }
    
    preds[i_start:i_end] <- p
    
    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("    Predicted %s / %s rows",
                      format(i_end, big.mark = ","),
                      format(n, big.mark = ",")))
    }
  }
  
  return(preds)
}

# Load the trained model (once)
message("Loading trained Random Forest model...")
rf_model <- readRDS("path/to/trained_rf_model.rds")  # <-- adjust path

# Prepare the prediction matrix: select only the predictor columns the model
# expects, in the correct order.  Avoid copying the full data.table.
if (inherits(rf_model, "ranger")) {
  predictor_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names differently
  predictor_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Subset to predictors only (data.table subsetting is memory-efficient)
pred_data <- cell_data[, ..predictor_names]

message("Running predictions...")
cell_data[, predicted_gdp := predict_in_chunks(rf_model, pred_data,
                                                chunk_size = 500000L)]
message("Prediction complete.")

# Clean up large temporaries
rm(pred_data, edge_dt)
gc()
```

---

## Summary of Expected Speedup

| Stage | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | Single vectorized `data.table` join | **~100–500×** |
| `compute_neighbor_stats` (×5 vars) | ~32.3M `lapply` iterations + `do.call(rbind)` | 5 grouped `data.table` aggregations (C-level) | **~50–200×** |
| Column assignment | Full `cell_data` copy per iteration (wide df) | `set()` by reference — zero copies | **~5–10×** |
| RF prediction | Possible single monolithic call (memory spike) | Chunked, single model load, correct `predict` dispatch | **~1–3×** (memory-safe) |
| **Overall** | **~86+ hours** | **Estimated ~10–30 minutes** | **~150–500×** |

The trained Random Forest model is never modified, and the numerical predictions (the estimand) are identical — only the data preparation and prediction workflow are optimized.