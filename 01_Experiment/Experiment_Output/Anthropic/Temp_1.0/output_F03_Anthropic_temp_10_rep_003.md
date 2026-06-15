 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

The function iterates row-by-row with `lapply`, performing per-row string pasting (`paste(id, year)`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. With 6.46 million rows this is extremely slow because:

- **String key construction per row**: `paste()` and character-based named-vector lookup are O(k) per row where k = number of neighbors.
- **Named vector lookup is O(1) amortized but has high constant factor** in R due to hashing overhead on character keys, repeated millions of times.
- **`lapply` over 6.46M elements** with non-trivial closures creates massive interpreter overhead.
- The entire result is a **list of 6.46M integer vectors** — high memory overhead from list structure.

### B. `compute_neighbor_stats` — Called 5 times, each over 6.46M rows

Each call does `lapply` over the full 6.46M-element neighbor lookup, extracting subsets of a numeric vector and computing `max`, `min`, `mean`. This is pure R loop overhead × 5 variables × 6.46M rows = ~32.3 million R-level function invocations, each with vector subsetting and NA handling.

The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself very slow (repeated `rbind` allocations).

### C. Random Forest Prediction

With ~6.46M rows and ~110 predictors, a single `predict()` call on a `ranger` or `randomForest` object will:
- Attempt to allocate a prediction matrix of 6.46M × 110 ≈ 710M entries (≈5.3 GB for double).
- If using `randomForest::predict.randomForest`, it copies the data into a matrix internally, potentially doubling memory.
- If the model is large (hundreds of trees), tree traversal over 6.46M rows is CPU-bound but manageable — the bottleneck is memory and data copying.

### D. Overall

| Stage | Estimated Time | Bottleneck |
|---|---|---|
| `build_neighbor_lookup` | 20–40 hours | Per-row string ops, named vector lookup |
| `compute_neighbor_stats` ×5 | 30–50 hours | R-level `lapply`, per-row subsetting |
| RF prediction | 2–8 hours | Memory pressure, object copying |
| **Total** | **~86+ hours** | |

---

## 2. OPTIMIZATION STRATEGY

### Strategy Summary

| Problem | Solution | Speedup Factor |
|---|---|---|
| Per-row string-key lookup | Replace with integer join via `data.table` | ~100–500× |
| `lapply` over 6.46M rows for neighbor stats | Vectorized `data.table` grouped aggregation | ~100–500× |
| `do.call(rbind, ...)` on millions of elements | Eliminated (aggregation returns `data.table`) | N/A |
| RF prediction memory | Batch prediction in chunks | Keeps within 16 GB |
| Neighbor lookup stored as list of 6.46M vectors | Replaced with flat edge-table (`data.table`) | Major memory savings |

### Core Idea

Instead of building a per-row list of neighbor indices and then looping over it, we:

1. **Build a flat edge table**: each row is `(row_idx, neighbor_row_idx)` — a `data.table` with ~tens of millions of rows.
2. **Join the variable values** onto the neighbor side in one vectorized operation.
3. **Group-by aggregate** (`max`, `min`, `mean`) by `row_idx` in one `data.table` call.
4. **Predict in batches** to avoid memory blowout.

Expected total runtime: **5–20 minutes** for feature preparation, **10–60 minutes** for prediction (depending on model type and size). Total: **under 2 hours**, likely under 30 minutes for the feature stage.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest — handled below)
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table if not already ----
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure there is a row index for stable reference
cell_data[, .row_idx := .I]

# ---- Step 1: Build flat neighbor edge table (vectorized) ----
# This replaces build_neighbor_lookup entirely.
# Inputs:
#   cell_data      — data.table with columns 'id' and 'year' (and all features)
#   id_order       — integer/character vector mapping position -> cell id
#   rook_neighbors_unique — spdep nb object (list of integer vectors of neighbor positions)

build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {
  message("Building neighbor edge table...")
  
  # --- Map each cell id to its position in id_order ---
  # id_order[pos] = cell_id, so neighbors[[pos]] gives neighbor positions
  n_ids <- length(id_order)
  
  # Build edge list at the cell-id level: (focal_id, neighbor_id)
  # neighbors[[i]] are positions in id_order for neighbors of id_order[i]
  focal_pos <- rep(seq_len(n_ids), lengths(neighbors))
  neighbor_pos <- unlist(neighbors, use.names = FALSE)
  
  # Convert positions to cell ids
  id_edges <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(focal_pos, neighbor_pos)
  
  # --- Build row-index lookup: (id, year) -> .row_idx ---
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # --- Get unique years ---
  years <- sort(unique(cell_data$year))
  
  # --- Cross join edges with years, then map to row indices ---
  # For each (focal_id, neighbor_id) pair, both must exist in the same year
  # Expand edges across years
  message("  Expanding edges across years...")
  
  # Use CJ-style expansion: repeat id_edges for each year
  edge_years <- CJ(edge_idx = seq_len(nrow(id_edges)), year = years)
  edge_years[, focal_id    := id_edges$focal_id[edge_idx]]
  edge_years[, neighbor_id := id_edges$neighbor_id[edge_idx]]
  edge_years[, edge_idx := NULL]
  
  # Map focal (id, year) -> row_idx
  message("  Joining focal row indices...")
  setkey(edge_years, focal_id, year)
  edge_years[row_lookup, focal_row := i..row_idx, on = .(focal_id = id, year = year)]
  
  # Map neighbor (id, year) -> row_idx
  message("  Joining neighbor row indices...")
  setkey(edge_years, neighbor_id, year)
  edge_years[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year = year)]
  
  # Drop rows where either side is missing
  edge_table <- edge_years[!is.na(focal_row) & !is.na(neighbor_row),
                           .(focal_row, neighbor_row)]
  
  rm(edge_years, row_lookup, id_edges)
  gc()
  
  setkey(edge_table, focal_row)
  message("  Edge table complete: ", formatC(nrow(edge_table), big.mark = ","), " edges.")
  return(edge_table)
}

# ---- Step 1 (alternative, memory-efficient): chunk by year ----
# If the CJ expansion above exceeds memory (1.37M edges × 28 years ≈ 38.5M rows,
# which is fine for 16 GB), use this version instead.

build_neighbor_edge_table_chunked <- function(cell_data, id_order, neighbors) {
  message("Building neighbor edge table (chunked by year)...")
  
  n_ids <- length(id_order)
  focal_pos <- rep(seq_len(n_ids), lengths(neighbors))
  neighbor_pos <- unlist(neighbors, use.names = FALSE)
  
  id_edges <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(focal_pos, neighbor_pos)
  
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id)
  
  years <- sort(unique(cell_data$year))
  
  edge_list <- lapply(years, function(yr) {
    rl_yr <- row_lookup[year == yr]
    setkey(rl_yr, id)
    
    et <- copy(id_edges)
    # focal
    et[rl_yr, focal_row := i..row_idx, on = .(focal_id = id)]
    # neighbor
    et[rl_yr, neighbor_row := i..row_idx, on = .(neighbor_id = id)]
    
    et[!is.na(focal_row) & !is.na(neighbor_row), .(focal_row, neighbor_row)]
  })
  
  edge_table <- rbindlist(edge_list)
  rm(edge_list, row_lookup, id_edges)
  gc()
  
  setkey(edge_table, focal_row)
  message("  Edge table complete: ", formatC(nrow(edge_table), big.mark = ","), " edges.")
  return(edge_table)
}


# ---- Step 2: Vectorized neighbor stats computation ----
# Replaces compute_neighbor_stats + compute_and_add_neighbor_features

compute_and_add_all_neighbor_features <- function(cell_data, edge_table, var_names) {
  message("Computing neighbor features for ", length(var_names), " variables...")
  n_rows <- nrow(cell_data)
  
  for (var_name in var_names) {
    message("  Processing: ", var_name)
    
    # Pull neighbor values via the edge table
    vals <- cell_data[[var_name]]
    
    # Build a temporary table: for each (focal_row), the neighbor value
    et <- edge_table[, .(focal_row, nval = vals[neighbor_row])]
    
    # Remove NAs in neighbor values
    et <- et[!is.na(nval)]
    
    # Aggregate by focal_row
    agg <- et[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    set(cell_data, j = max_col,  value = rep(NA_real_, n_rows))
    set(cell_data, j = min_col,  value = rep(NA_real_, n_rows))
    set(cell_data, j = mean_col, value = rep(NA_real_, n_rows))
    
    # Fill in aggregated values
    set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
    
    rm(et, agg)
  }
  
  gc()
  message("  Neighbor features complete.")
  invisible(cell_data)
}


# ---- Step 3: Batched Random Forest Prediction ----
# Works with both ranger and randomForest models.

predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  message("Starting batched RF prediction...")
  n <- nrow(newdata)
  n_batches <- ceiling(n / batch_size)
  predictions <- numeric(n)
  
  # Detect model type
  is_ranger <- inherits(model, "ranger")
  
  for (b in seq_len(n_batches)) {
    start_i <- (b - 1L) * batch_size + 1L
    end_i   <- min(b * batch_size, n)
    
    if (b %% 5 == 1 || b == n_batches) {
      message("  Batch ", b, "/", n_batches,
              " (rows ", formatC(start_i, big.mark = ","),
              "-", formatC(end_i, big.mark = ","), ")")
    }
    
    batch_data <- newdata[start_i:end_i, ]
    
    if (is_ranger) {
      pred <- predict(model, data = batch_data)$predictions
    } else {
      # randomForest
      pred <- predict(model, newdata = batch_data)
    }
    
    predictions[start_i:end_i] <- pred
    
    # Free batch memory
    rm(batch_data, pred)
    if (b %% 10 == 0) gc()
  }
  
  message("  Prediction complete.")
  return(predictions)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# ---- Build edge table (run once; can be cached to disk) ----
edge_table <- build_neighbor_edge_table_chunked(
  cell_data, id_order, rook_neighbors_unique
)

# Optional: save/load for reuse
# fst::write_fst(edge_table, "edge_table.fst")
# edge_table <- fst::read_fst("edge_table.fst", as.data.table = TRUE)

# ---- Compute all neighbor features at once ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# Edge table no longer needed
rm(edge_table)
gc()

# ---- Prepare prediction data ----
# Get the feature columns the model expects
if (inherits(rf_model, "ranger")) {
  model_features <- rf_model$forest$independent.variable.names
} else {
  # randomForest
  model_features <- attr(rf_model$terms, "term.labels")
  if (is.null(model_features)) {
    # If model was trained with x/y interface, use colnames of importance
    model_features <- rownames(rf_model$importance)
  }
}

# Subset to model features only (avoids copying unneeded columns)
pred_data <- cell_data[, ..model_features]

# ---- Run prediction ----
cell_data[, predicted_gdp := predict_rf_batched(rf_model, pred_data, batch_size = 500000L)]

rm(pred_data)
gc()

# ---- Done ----
message("Pipeline complete. Predictions stored in cell_data$predicted_gdp")
```

---

## 4. EXPECTED PERFORMANCE SUMMARY

| Stage | Original | Optimized | Why |
|---|---|---|---|
| **Neighbor lookup** | 20–40 hrs (per-row string ops) | **1–3 min** (vectorized `data.table` joins) | Integer joins replace character hashing; no per-row R calls |
| **Neighbor stats ×5** | 30–50 hrs (`lapply` + `rbind`) | **2–5 min** (grouped aggregation) | `data.table` grouped `max/min/mean` on ~38.5M-row edge table is trivial |
| **RF prediction** | 2–8 hrs (single call, memory thrashing) | **10–30 min** (batched, controlled memory) | 500K-row batches keep peak memory ~2–3 GB; no swap pressure |
| **Total** | **86+ hours** | **~15–40 minutes** | |

### Key Invariants Preserved

- **Trained RF model**: never retrained; used only via `predict()`.
- **Numerical estimand**: `neighbor_max`, `neighbor_min`, `neighbor_mean` computed identically (same `max`, `min`, `mean` of non-NA neighbor values per cell-year). Column names follow the pattern `neighbor_{stat}_{var}` matching the original `compute_and_add_neighbor_features` output.
- **Spatial structure**: rook neighbor relationships and year-matching logic are identical.