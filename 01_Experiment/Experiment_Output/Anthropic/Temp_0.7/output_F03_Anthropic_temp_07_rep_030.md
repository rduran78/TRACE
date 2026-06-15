 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` is extremely slow:**
- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`)
  - Named vector lookups (which are O(n) hash lookups but done millions of times with `paste`/string concatenation)
  - `paste(..., sep="_")` string construction per row
  - `is.na` filtering
- **Net effect:** Billions of small string allocations and named-vector lookups. This alone could take hours.

**`compute_neighbor_stats` is inefficient:**
- `lapply` over 6.46M elements, each subsetting a numeric vector, removing NAs, and computing three summary stats.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is a classic R anti-pattern — it creates millions of tiny row-vectors then binds them one-by-one.
- This is repeated **5 times** (once per neighbor source variable).

**Object copying in the outer loop:**
- `cell_data <- compute_and_add_neighbor_features(cell_data, ...)` likely copies the entire ~6.46M × 110+ column data.frame on each assignment, adding ~5 full copies.

### B. Prediction Workflow Bottlenecks (Random Forest Inference)

- Predicting 6.46M rows with ~110 features through a Random Forest (especially one with many trees) is inherently expensive.
- If `predict()` is called in a loop (e.g., per year or per chunk) with repeated model deserialization or unnecessary data copies, this multiplies cost.
- A single `predict.randomForest` call on 6.46M rows may spike memory (the model object + prediction matrix + internal tree traversal buffers).
- If the model was saved with `randomForest::randomForest`, the predict method is pure R tree traversal — much slower than C-backed alternatives.

### C. Memory Pressure

- 6.46M rows × 110 columns × 8 bytes ≈ **5.7 GB** for the numeric matrix alone.
- With data.frame overhead, copies, and the RF model in memory, 16 GB RAM is tight. Swapping to disk would catastrophically slow everything.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops, named vector lookup | Vectorized integer-key join via `data.table`; build lookup once as a flat integer mapping |
| `compute_neighbor_stats` | Per-row `lapply`, `do.call(rbind,...)` | Flatten neighbor pairs into a `data.table`, compute grouped `max/min/mean` in one vectorized pass |
| Outer loop copying | Repeated full data.frame copy | Use `data.table` set-by-reference (`:=`) — zero copies |
| RF prediction | Possibly looped, R-level tree traversal | Single batched `predict()` call; optionally port to `ranger` for C++-backed prediction if model format allows; chunk if memory-constrained |
| Memory | ~6 GB data + copies + model | `data.table` in-place ops; `gc()` between stages; chunked prediction if needed |

**Key invariants preserved:**
- The trained Random Forest model is never retrained.
- The numerical estimand (predicted GDP) is identical — we only change how features are assembled and how `predict()` is dispatched.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place if possible) ---------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place — no copy
}

# Ensure id and year are integer for fast joins
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row index for the original data
cell_data[, .row_idx := .I]

# ---- Step 1: Build neighbor edge list (vectorized, no per-row strings) ------
build_neighbor_edgelist_dt <- function(cell_data, id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)
  
  # Expand neighbor list into a flat edge list: (focal_id, neighbor_id)
  # Each element neighbors[[i]] is an integer vector of indices into id_order
  n <- length(neighbors)
  
  # Pre-compute lengths for pre-allocation
  lens <- lengths(neighbors)
  total_edges <- sum(lens)
  
  focal_idx <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  # Remove 0-entries (spdep uses 0 for "no neighbors")
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  edge_dt <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  
  return(edge_dt)
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)

# ---- Step 2: Vectorized neighbor stats computation --------------------------
compute_and_add_all_neighbor_features <- function(cell_data, edge_dt, 
                                                   neighbor_source_vars) {
  # Build a minimal keyed lookup: (id, year) -> row_idx + variable values
  # We join edges × years to get all (focal_row, neighbor_row) pairs,

  # then compute grouped stats.
  
  # Unique years
  years <- unique(cell_data$year)
  
  # Create a key table: id, year, row index, and all source variable values
  key_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  key_dt <- cell_data[, ..key_cols]
  
  # --- Cross join edges with years ---
  # For each edge (focal_id, neighbor_id), the relationship holds for ALL years.
  # So the full set of (focal_id, year, neighbor_id) is edge_dt × years.
  
  # But that could be huge: ~1.37M edges × 28 years = ~38.4M rows. Manageable.
  
  cat("  Expanding edges across years...\n")
  edge_year_dt <- CJ_dt_edges(edge_dt, years)
  
  # Join to get neighbor variable values
  cat("  Joining neighbor values...\n")
  setkey(key_dt, id, year)
  
  # Join: for each (neighbor_id, year), get the neighbor's variable values
  edge_year_dt[key_dt, 
               (neighbor_source_vars) := mget(paste0("i.", neighbor_source_vars)),
               on = .(neighbor_id = id, year = year)]
  
  # Also need focal row index for grouping
  focal_key <- cell_data[, .(id, year, .row_idx)]
  setkey(focal_key, id, year)
  edge_year_dt[focal_key, focal_row := i..row_idx, 
               on = .(focal_id = id, year = year)]
  
  # Remove edges where focal_row is NA (shouldn't happen but safety)
  edge_year_dt <- edge_year_dt[!is.na(focal_row)]
  
  # --- Compute grouped stats per (focal_row) per variable ---
  cat("  Computing grouped neighbor stats...\n")
  
  for (var_name in neighbor_source_vars) {
    cat("    Variable:", var_name, "\n")
    
    vn <- var_name
    col_max  <- paste0("nb_max_", var_name)
    col_min  <- paste0("nb_min_", var_name)
    col_mean <- paste0("nb_mean_", var_name)
    
    # Compute stats grouped by focal_row
    stats <- edge_year_dt[!is.na(get(vn)), 
                          .(nb_max  = max(get(vn)),
                            nb_min  = min(get(vn)),
                            nb_mean = mean(get(vn))),
                          by = focal_row]
    
    # Assign back to cell_data by reference using row indices
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
    
    set(cell_data, i = stats$focal_row, j = col_max,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = col_min,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = col_mean, value = stats$nb_mean)
  }
  
  invisible(cell_data)
}

# Helper: cross-join edges with years (memory-efficient)
CJ_dt_edges <- function(edge_dt, years) {
  # Repeat each edge for each year
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  
  data.table(
    focal_id    = rep(edge_dt$focal_id,    times = n_years),
    neighbor_id = rep(edge_dt$neighbor_id,  times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ---- Step 3: Run feature preparation ---------------------------------------
cat("Computing all neighbor features...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up temporary column
cell_data[, .row_idx := NULL]

# Force garbage collection before prediction
rm(edge_dt)
gc()

cat("Feature preparation complete.\n")

# ---- Step 4: Random Forest Prediction (optimized) --------------------------
cat("Starting Random Forest prediction...\n")

# Load the trained model once
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify the predictor columns the model expects
predictor_cols <- names(rf_model$forest$xlevels)  # for randomForest package
# If using ranger: predictor_cols <- rf_model$forest$independent.variable.names

# Prepare the prediction matrix — extract only needed columns, as a data.frame
# (predict.randomForest / predict.ranger expect a data.frame)
cat("  Preparing prediction matrix...\n")
pred_input <- as.data.frame(cell_data[, ..predictor_cols])

# --- Option A: Single-batch prediction (if memory allows) ---
# Estimated memory: 6.46M rows × n_trees × 8 bytes for intermediate.
# For a model with ≤500 trees this should fit in 16 GB with the data.

cat("  Running predict()...\n")
cell_data[, predicted_gdp := predict(rf_model, newdata = pred_input)]

cat("Prediction complete.\n")

# --- Option B: Chunked prediction (if Option A causes memory issues) ---
# Uncomment below and comment out Option A if you hit memory limits.

# chunk_size <- 500000L  # 500K rows per chunk
# n_rows <- nrow(pred_input)
# n_chunks <- ceiling(n_rows / chunk_size)
# predictions <- numeric(n_rows)
# 
# for (ch in seq_len(n_chunks)) {
#   idx_start <- (ch - 1L) * chunk_size + 1L
#   idx_end   <- min(ch * chunk_size, n_rows)
#   cat(sprintf("  Chunk %d/%d (rows %d-%d)\n", ch, n_chunks, idx_start, idx_end))
#   predictions[idx_start:idx_end] <- predict(rf_model, 
#                                              newdata = pred_input[idx_start:idx_end, , drop = FALSE])
#   if (ch %% 5 == 0) gc()  # periodic GC
# }
# cell_data[, predicted_gdp := predictions]
# rm(predictions)

rm(pred_input)
gc()

cat("Pipeline complete.\n")
```

---

## 4. EXPECTED SPEEDUP SUMMARY

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M `paste` + named lookups (~hours) | Vectorized `data.table` edge list (~seconds) | **~1000×** |
| `compute_neighbor_stats` (×5 vars) | ~6.46M × 5 `lapply` + `do.call(rbind)` (~hours) | Grouped `data.table` aggregation (~minutes) | **~100–500×** |
| Data.frame copying (5 rounds) | ~5 × 5.7 GB copies | Zero-copy `:=` / `set()` | **eliminates ~28 GB allocation** |
| RF prediction | Same `predict()` call | Same, but with clean memory headroom; chunked option available | **1×** (but no OOM risk) |
| **Total estimated wall time** | **86+ hours** | **~10–30 minutes** | **~200–500×** |

### Why the numerical estimand is preserved:
- The neighbor statistics (`max`, `min`, `mean`) are computed over the identical neighbor sets with identical arithmetic — only the iteration mechanism changes (vectorized grouped ops vs. per-row `lapply`).
- The Random Forest model object is loaded and used as-is with `predict()` — no retraining, no modification.
- The predicted values are numerically identical (to floating-point precision of `data.table`'s grouped `mean`, which matches base R's `mean`).