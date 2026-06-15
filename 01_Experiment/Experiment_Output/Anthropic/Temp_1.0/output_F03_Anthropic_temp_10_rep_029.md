 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** is the most critical bottleneck in the feature-preparation phase:

| Issue | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, string pasting, and named-vector lookups — all O(n) or worse per call. With ~6.46M iterations this is catastrophically slow. |
| **Repeated `paste()` key construction** | `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates throwaway character vectors millions of times. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | Named vector lookup in R is an O(n) hash-miss-prone operation when the vector has ~6.46M names. This is done inside the inner loop. |
| **`as.character()` coercion inside loop** | Millions of redundant type conversions. |

**`compute_neighbor_stats`** has a secondary bottleneck:

| Issue | Detail |
|---|---|
| **Per-row `lapply` with subsetting and `c()` allocation** | 6.46M small vector allocations, each followed by `max/min/mean`. |
| **`do.call(rbind, result)` on a 6.46M-element list** | This is a known R anti-pattern; it copies the entire structure repeatedly. |

**Net effect:** The nested character-key approach turns what should be a vectorized integer-index join into ~6.46M × (string ops + hash lookups). This alone can account for most of the 86+ hour estimate.

### B. Random Forest Inference Bottleneck

| Issue | Detail |
|---|---|
| **Single `predict()` call on 6.46M rows × 110 features** | Even a moderately-sized RF (500 trees) must traverse every tree for every row. On a 16 GB laptop this can take hours and may cause memory pressure. |
| **Object size** | A 6.46M × 110 `data.frame` is ~5.7 GB in double precision alone. The RF model object, the prediction workspace, and the feature frame can together exceed 16 GB, triggering swap. |
| **Potential repeated model loading** | If the model is deserialized from disk inside a loop, overhead compounds. |
| **No batching** | One monolithic `predict()` call gives no opportunity to manage memory or parallelise. |

---

## 2. Optimization Strategy

### Feature Preparation — from O(n·k) string ops to O(n) integer joins

1. **Replace character-key lookups with a `data.table` equi-join.** Build the neighbor lookup as a two-column `data.table` (`id`, `neighbor_id`), then join on `(neighbor_id, year)` to get row indices — all vectorized, all integer-keyed.
2. **Compute neighbor stats with `data.table` grouped aggregation** (`[, .(max, min, mean), by = row_idx]`) — eliminates the 6.46M-element `lapply` and the `do.call(rbind, …)`.
3. **Pre-compute the full edge list once** and reuse it for every variable.

### Random Forest Inference — batched, memory-safe prediction

1. **Load the model once** at the top of the script.
2. **Predict in batches** (e.g., 500 K rows) to keep peak memory well under 16 GB.
3. **Convert the prediction input to a `matrix`** (not `data.frame`) — `ranger` and `randomForest` are both faster on matrices.
4. **Optionally parallelise** batches with `future.apply` if cores are available.

### Expected Speedup

| Phase | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~30–50 h | < 2 min |
| `compute_neighbor_stats` (×5 vars) | ~20–30 h | < 5 min |
| RF prediction (6.46M rows) | ~2–6 h | ~15–40 min |
| **Total** | **86+ h** | **~20–45 min** |

---

## 3. Working R Code

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)

# ============================================================
# 1. One-time setup: load model, convert data
# ============================================================

# --- Load the trained Random Forest model ONCE ---------------
# (Adjust the path / object name to match your pipeline.)
rf_model <- readRDS("trained_rf_model.rds")

# --- Convert cell_data to data.table in place ----------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Make sure id and year are integer (avoids any implicit coercion)
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Add a row-index column (needed for later join-back)
cell_data[, .row_idx := .I]


# ============================================================
# 2. Build the neighbor edge-list ONCE  (vectorised)
# ============================================================
build_neighbor_edgelist_dt <- function(cell_data, id_order, nb_object) {
  # id_order : integer vector mapping reference-index -> cell id
  # nb_object: spdep nb list (1-based indices into id_order)
  
  # Expand the nb list into a two-column integer edge list
  n_neighbors <- lengths(nb_object)                       # integer vector
  from_ref    <- rep(seq_along(nb_object), n_neighbors)   # reference indices
  to_ref      <- unlist(nb_object, use.names = FALSE)     # neighbor ref indices
  
  edge_dt <- data.table(
    id          = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )
  
  # For every (id, year) row we need (neighbor_id, year).
  # Join cell_data's row index onto the "from" side, then the
  # neighbor's value onto the "to" side.
  
  # Map id -> all years it appears in, with row index
  id_year_map <- cell_data[, .(id, year, .row_idx)]
  setkey(id_year_map, id)
  
  # Attach year and row_idx of the focal cell
  # Result: for every (focal row, neighbor_id) pair we have
  #         focal_row_idx and the year we need from the neighbor.
  edge_expanded <- edge_dt[id_year_map, on = "id",
                           allow.cartesian = TRUE,
                           nomatch = NULL]
  # Columns now: id, neighbor_id, year, .row_idx  (focal)
  
  # Now attach the ROW INDEX of the neighbor in that same year
  neighbor_year_map <- cell_data[, .(neighbor_id = id, year,
                                     neighbor_row_idx = .row_idx)]
  setkey(neighbor_year_map, neighbor_id, year)
  setkey(edge_expanded,     neighbor_id, year)
  
  edge_final <- neighbor_year_map[edge_expanded,
                                  on = c("neighbor_id", "year"),
                                  nomatch = NA]
  # Columns: neighbor_id, year, neighbor_row_idx, id, .row_idx
  
  # Drop rows where the neighbor doesn't exist in that year
  edge_final <- edge_final[!is.na(neighbor_row_idx)]
  
  # Return only what we need
  edge_final[, .(focal_row = .row_idx, neighbor_row = neighbor_row_idx)]
}

cat("Building neighbour edge-list …\n")
edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)
setkey(edge_dt, focal_row)
cat("Edge-list rows:", nrow(edge_dt), "\n")


# ============================================================
# 3. Compute neighbour statistics — fully vectorised
# ============================================================
compute_and_add_neighbor_features_dt <- function(cell_data, var_name, edge_dt) {
  # Pull the variable values for the neighbor rows
  vals <- cell_data[[var_name]]
  edge_dt[, nval := vals[neighbor_row]]
  
  # Drop NAs before aggregation
  agg <- edge_dt[!is.na(nval),
                 .(
                   nbr_max  = max(nval),
                   nbr_min  = min(nval),
                   nbr_mean = mean(nval)
                 ),
                 by = focal_row]
  
  # Create result columns initialised to NA
  col_max  <- paste0("n_", var_name, "_max")
  col_min  <- paste0("n_", var_name, "_min")
  col_mean <- paste0("n_", var_name, "_mean")
  
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)
  
  # In-place update — no copy
  set(cell_data, i = agg$focal_row, j = col_max,  value = agg$nbr_max)
  set(cell_data, i = agg$focal_row, j = col_min,  value = agg$nbr_min)
  set(cell_data, i = agg$focal_row, j = col_mean, value = agg$nbr_mean)
  
  # Clean up temporary column
  edge_dt[, nval := NULL]
  
  invisible(cell_data)
}

# --- Run for all 5 neighbour source variables -----------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Neighbor stats:", var_name, "…\n")
  compute_and_add_neighbor_features_dt(cell_data, var_name, edge_dt)
}

cat("Feature preparation complete.\n")


# ============================================================
# 4. Batched Random Forest Prediction
# ============================================================
predict_rf_batched <- function(model, data, feature_cols,
                               batch_size = 500000L) {
  # data       : data.table with all feature columns
  # feature_cols: character vector of the ~110 predictor column names
  
  n <- nrow(data)
  preds <- numeric(n)
  
  starts <- seq(1L, n, by = batch_size)
  cat(sprintf("Predicting %s rows in %d batches …\n", 
              formatC(n, big.mark = ","), length(starts)))
  
  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)
    
    # Extract batch as a plain matrix — fastest for predict()
    batch_mat <- as.matrix(data[i1:i2, ..feature_cols])
    
    # Works for both randomForest::predict and ranger::predict
    # Adjust the call if your model class differs.
    if (inherits(model, "ranger")) {
      batch_pred <- predict(model, data = batch_mat)$predictions
    } else {
      # randomForest or similar
      batch_pred <- predict(model, newdata = batch_mat)
    }
    
    preds[i1:i2] <- batch_pred
    
    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  batch %d / %d  (rows %s – %s)\n",
                  k, length(starts),
                  formatC(i1, big.mark = ","),
                  formatC(i2, big.mark = ",")))
    }
    
    # Free batch memory immediately
    rm(batch_mat, batch_pred)
    if (k %% 10 == 0) gc(verbose = FALSE)
  }
  
  preds
}

# --- Define your feature columns (adjust to your pipeline) ----
# Example: all columns except id, year, the target, and .row_idx
target_col   <- "gdp"                          # adjust if different
exclude_cols <- c("id", "year", target_col, ".row_idx")
feature_cols <- setdiff(names(cell_data), exclude_cols)
cat("Number of predictor features:", length(feature_cols), "\n")

# --- Run prediction -------------------------------------------
cell_data[, predicted_gdp := predict_rf_batched(
  model        = rf_model,
  data         = cell_data,
  feature_cols = feature_cols,
  batch_size   = 500000L
)]

cat("Prediction complete.\n")

# --- Clean up helper column ------------------------------------
cell_data[, .row_idx := NULL]
```

---

## Summary of Changes and Why They Work

| Original | Optimised | Why |
|---|---|---|
| `lapply` over 6.46M rows building character keys | Single `data.table` equi-join on integer columns | O(n log n) merge vs O(n²) string work |
| `paste(id, year)` as lookup key | Integer key join `(neighbor_id, year)` | Eliminates ~13M `paste()` calls and named-vector hash lookups |
| `do.call(rbind, 6.46M-element list)` | `data.table` grouped aggregation (`by = focal_row`) | Vectorised C-level aggregation; zero intermediate list allocation |
| Implicit full-`data.frame` copy in `cell_data <- …` | `data.table::set()` for in-place column updates | No 5.7 GB copy per variable |
| Single monolithic `predict()` on 6.46M rows | Batched prediction (500K rows per batch) as matrix | Peak memory stays well under 16 GB; faster matrix traversal |
| Potentially repeated `readRDS()` of model | Model loaded once at script start | Eliminates repeated deserialization |

The trained Random Forest model is never modified or retrained, and all numerical computations (`max`, `min`, `mean`, and the RF predictions themselves) are preserved identically — only the execution strategy changes.