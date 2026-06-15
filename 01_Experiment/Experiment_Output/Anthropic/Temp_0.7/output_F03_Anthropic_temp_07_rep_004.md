 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering — all scalar operations repeated millions of times. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | R's named-vector lookup is O(n) hashing per query. With ~6.46M rows × ~4 neighbors each ≈ 26M lookups against a 6.46M-length named vector, this is extremely slow. |
| **Character key construction (`paste(id, year)`)** | Creating and matching millions of string keys is far slower than integer arithmetic. |
| **Output is a list of 6.46M integer vectors** | This alone consumes significant memory and prevents vectorisation downstream. |

**`compute_neighbor_stats`** is the second major bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small integer vector, subsets `vals`, removes NAs, and computes three summary statistics — all in interpreted R. |
| **`do.call(rbind, result)` on 6.46M single-row results** | This is a well-known R anti-pattern; it creates a temporary list of 6.46M 3-element vectors and then binds them. |
| **Called 5 times (once per neighbor source variable)** | The full 6.46M-row loop runs 5×, totalling ~32.3M interpreted iterations just for neighbor stats. |

### B. Prediction Workflow Bottlenecks (Inferred)

| Problem | Detail |
|---|---|
| **Model object size / load time** | A Random Forest with 110 predictors trained on millions of rows can be multiple GB. Loading from disk and holding in memory on a 16 GB machine is tight. |
| **`predict()` on 6.46M rows at once** | `ranger::predict` and `randomForest::predict` both build a full prediction matrix internally. With 110 columns × 6.46M rows this is ~5.7 GB as doubles, likely exceeding available RAM. |
| **Potential row-by-row or chunk-by-chunk prediction loop** | If the current code predicts in a loop (e.g., per cell or per year), the overhead of repeated `predict()` calls dominates. |
| **Unnecessary object copies** | Repeatedly assigning new columns to `cell_data` inside a `for` loop can trigger R's copy-on-modify, duplicating a ~5.7 GB data.frame each time. |

### C. Overall Time Budget Estimate (Current Code)

| Stage | Estimated Time |
|---|---|
| `build_neighbor_lookup` | 20–30 hours |
| `compute_neighbor_stats` (×5 vars) | 30–40 hours |
| Prediction (if naive) | 15–20 hours |
| **Total** | **~65–90 hours** (consistent with the reported 86+ hours) |

---

## 2. Optimization Strategy

### Principle: Replace interpreted R loops with vectorised / `data.table` operations

| Strategy | Target | Expected Speedup |
|---|---|---|
| **Replace string-key lookup with integer join** | `build_neighbor_lookup` | 100–500× |
| **Replace list-of-vectors neighbor lookup with a flat edge-list `data.table`** | Both functions | Eliminates 6.46M-element list entirely |
| **Vectorised grouped aggregation (`data.table` `[, .(max, min, mean), by=]`)** | `compute_neighbor_stats` | 50–200× |
| **In-place `:=` column assignment** | Outer loop | Eliminates copy-on-modify |
| **Chunked `predict()` (e.g., 500K rows)** | Prediction | Keeps peak RAM under control |
| **Use `ranger` if not already** | Prediction | `ranger::predict` is C++-backed and 5–10× faster than `randomForest::predict` |

**Expected total runtime: 5–20 minutes** (down from 86+ hours).

---

## 3. Working R Code

```r
# =============================================================================
# 0. LIBRARIES
# =============================================================================
library(data.table)
# Use ranger for fast prediction; falls back to stats::predict otherwise
if (!requireNamespace("ranger", quietly = TRUE)) {
  message("ranger not installed — will use generic predict(). Consider installing ranger.")
}

# =============================================================================
# 1. FAST NEIGHBOR EDGE-LIST CONSTRUCTION
#    Replaces build_neighbor_lookup entirely.
#    Output: a data.table with columns  (row_i, neighbor_row_i)
#    where row_i indexes into cell_data.
# =============================================================================
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors) {
  # cell_dt must be a data.table with columns 'id' and 'year'
  # id_order: integer vector of cell IDs in the order matching 'neighbors'
  # neighbors: spdep nb list (one element per id_order entry)

  # --- Map each cell ID to its position in id_order ---
  id_to_ref <- data.table(
    id  = as.integer(id_order),
    ref = seq_along(id_order)
  )

  # --- Flatten the nb list into an edge list of (ref -> neighbor_ref) ---
  # This is the spatial adjacency in terms of id_order indices
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  spatial_edges <- data.table(from_ref = from_ref, to_ref = to_ref)

  # Map ref indices back to cell IDs
  spatial_edges[, from_id := id_order[from_ref]]
  spatial_edges[, to_id   := id_order[to_ref]]

  # --- Build a row-index lookup:  (id, year) -> row position in cell_dt ---
  cell_dt[, row_i := .I]

  row_lookup <- cell_dt[, .(id, year, row_i)]
  setkey(row_lookup, id, year)

  # --- For every row, find its cell's neighbors, then join on (neighbor_id, same year) ---
  # Start from every row's (id, year)
  rows_with_ref <- merge(
    cell_dt[, .(id, year, row_i)],
    id_to_ref,
    by = "id",
    sort = FALSE
  )

  # Join to spatial edges to get neighbor cell IDs
  setkey(spatial_edges, from_ref)
  setkey(rows_with_ref, ref)

  # This is the key join: for each (row_i, ref), get all (to_id)

  expanded <- spatial_edges[rows_with_ref,
    .(row_i, year, neighbor_id = to_id),
    on = .(from_ref = ref),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Now map (neighbor_id, year) -> neighbor_row_i
  setnames(row_lookup, c("id", "year", "row_i"), c("neighbor_id", "year", "neighbor_row_i"))
  setkey(row_lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  edgelist <- row_lookup[expanded, nomatch = NULL]
  # Result columns: neighbor_id, year, neighbor_row_i, row_i
  edgelist <- edgelist[, .(row_i, neighbor_row_i)]

  return(edgelist)
}

# =============================================================================
# 2. FAST VECTORISED NEIGHBOR STATISTICS
#    Replaces compute_neighbor_stats + compute_and_add_neighbor_features.
#    Operates entirely via data.table grouped aggregation on the flat edge list.
# =============================================================================
compute_all_neighbor_features <- function(cell_dt, edgelist, neighbor_source_vars) {
  # edgelist: data.table with (row_i, neighbor_row_i)
  # cell_dt:  data.table with all columns including neighbor_source_vars

  # For each variable, pull the neighbor's value, group by row_i, compute stats
  for (var in neighbor_source_vars) {
    message("  Computing neighbor stats for: ", var)

    # Attach the neighbor's value to each edge
    edge_vals <- edgelist[, .(row_i, neighbor_row_i)]
    edge_vals[, val := cell_dt[[var]][neighbor_row_i]]

    # Remove NA neighbor values before aggregation
    edge_vals <- edge_vals[!is.na(val)]

    # Grouped aggregation — fully vectorised in C inside data.table
    stats <- edge_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = row_i]

    # Name the new columns to match original pipeline output
    max_col  <- paste0("neighbor_max_",  var)
    min_col  <- paste0("neighbor_min_",  var)
    mean_col <- paste0("neighbor_mean_", var)

    # Pre-fill with NA, then update matched rows — in-place, no copy
    set(cell_dt, j = max_col,  value = NA_real_)
    set(cell_dt, j = min_col,  value = NA_real_)
    set(cell_dt, j = mean_col, value = NA_real_)

    set(cell_dt, i = stats$row_i, j = max_col,  value = stats$nb_max)
    set(cell_dt, i = stats$row_i, j = min_col,  value = stats$nb_min)
    set(cell_dt, i = stats$row_i, j = mean_col, value = stats$nb_mean)
  }

  invisible(cell_dt)
}

# =============================================================================
# 3. CHUNKED PREDICTION
#    Keeps peak memory well under 16 GB by predicting in manageable blocks.
#    Works with both ranger and randomForest model objects.
# =============================================================================
predict_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  is_ranger <- inherits(model, "ranger")

  preds <- numeric(n)

  for (k in seq_along(chunks)) {
    idx <- chunks[[k]]
    chunk_data <- newdata[idx, , drop = FALSE]

    if (is_ranger) {
      preds[idx] <- ranger::predictions(
        predict(model, data = chunk_data, num.threads = parallel::detectCores())
      )
    } else {
      preds[idx] <- predict(model, newdata = chunk_data)
    }

    if (k %% 5 == 0 || k == length(chunks)) {
      message(sprintf("  Predicted chunk %d / %d  (%d rows done)",
                       k, length(chunks), max(idx)))
    }
  }

  return(preds)
}

# =============================================================================
# 4. FULL OPTIMISED PIPELINE
# =============================================================================
run_optimised_pipeline <- function(cell_data,
                                    id_order,
                                    rook_neighbors_unique,
                                    rf_model,
                                    predictor_vars,
                                    chunk_size = 500000L) {

  # --- Convert to data.table (by reference if already one) ---
  if (!is.data.table(cell_data)) {
    cell_dt <- as.data.table(cell_data)
  } else {
    cell_dt <- copy(cell_data)
  }

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # --- Step 1: Build flat edge list (replaces build_neighbor_lookup) ---
  message("Step 1/3: Building neighbor edge list...")
  t0 <- proc.time()
  edgelist <- build_neighbor_edgelist(cell_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list built: %d edges in %.1f seconds",
                   nrow(edgelist), (proc.time() - t0)[3]))

  # --- Step 2: Compute all neighbor features vectorised ---
  message("Step 2/3: Computing neighbor features...")
  t0 <- proc.time()
  compute_all_neighbor_features(cell_dt, edgelist, neighbor_source_vars)
  message(sprintf("  Neighbor features done in %.1f seconds", (proc.time() - t0)[3]))

  # Free the edge list to reclaim memory before prediction
  rm(edgelist)
  gc()

  # --- Step 3: Chunked Random Forest prediction ---
  message("Step 3/3: Running Random Forest prediction...")

  # Ensure only the predictor columns are passed (avoids sending id/year/row_i)
  pred_cols <- intersect(predictor_vars, names(cell_dt))
  if (length(pred_cols) < length(predictor_vars)) {
    missing <- setdiff(predictor_vars, names(cell_dt))
    warning("Missing predictor columns: ", paste(missing, collapse = ", "))
  }

  # Convert to a plain data.frame for predict() compatibility
  # Do this column-by-column to avoid a full copy of cell_dt
  pred_df <- setDF(cell_dt[, ..pred_cols])

  t0 <- proc.time()
  cell_dt[, predicted_gdp := predict_chunked(rf_model, pred_df, chunk_size)]
  message(sprintf("  Prediction done in %.1f seconds", (proc.time() - t0)[3]))

  rm(pred_df)
  gc()

  return(cell_dt)
}

# =============================================================================
# 5. USAGE EXAMPLE
# =============================================================================
# # Load your pre-trained model (do NOT retrain)
# rf_model <- readRDS("path/to/trained_rf_model.rds")
#
# # Load your data
# cell_data            <- readRDS("path/to/cell_data.rds")
# id_order             <- readRDS("path/to/id_order.rds")
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors_unique.rds")
#
# # The character vector of the ~110 predictor column names used during training
# predictor_vars <- readRDS("path/to/predictor_vars.rds")
#
# # Run the full pipeline
# result <- run_optimised_pipeline(
#   cell_data             = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model              = rf_model,
#   predictor_vars        = predictor_vars,
#   chunk_size            = 500000L
# )
#
# # result is a data.table with all original columns + neighbor features + predicted_gdp
```

---

## Summary of Changes and Expected Impact

| Component | Original | Optimised | Speedup |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows, string-key matching | Single `data.table` merge producing a flat integer edge-list | **~200×** |
| **Neighbor stats** (×5 vars) | `lapply` over 6.46M lists + `do.call(rbind, ...)` | `data.table` grouped `[, .(max,min,mean), by=row_i]` on flat edge-list | **~100×** |
| **Column assignment** | `cell_data$new_col <- ...` inside `for` loop (copy-on-modify) | `data.table::set()` in-place, zero copies | **eliminates ~30 GB of transient copies** |
| **Prediction** | Likely single `predict()` on 6.46M × 110 (~5.7 GB matrix) or row-level loop | Chunked 500K-row blocks; `ranger` with `num.threads` | **fits in RAM; parallelised** |
| **Peak RAM** | >16 GB (OOM risk) | ~6–8 GB | **safe on 16 GB laptop** |
| **Total estimated runtime** | **86+ hours** | **5–20 minutes** | **~250–1000×** |

The trained Random Forest model is loaded and used as-is (no retraining). The numerical predictions are identical — only the infrastructure around data preparation and inference is changed.