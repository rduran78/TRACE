 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it does:
  - Character coercion (`as.character`), named-vector lookups, `paste` for key construction, subsetting, and `NA` filtering.
  - Named-vector lookups in R are **O(n)** hash probes per call, repeated 6.46M times with multiple neighbors each time.
- Estimated: billions of character operations and hash lookups.

**`compute_neighbor_stats`** is the second bottleneck:

- Also `lapply` over 6.46M rows, extracting variable subsets, removing NAs, computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3M iterations total.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is extremely slow — it creates millions of intermediate objects before binding.

**Object copying in the outer loop:**

- `cell_data <- compute_and_add_neighbor_features(...)` likely triggers full-copy semantics of the data.frame (~6.46M × 110+ columns) on each assignment — **5 full copies** of a multi-GB frame.

### 1.2 Prediction-Workflow Bottlenecks

- **Model loading**: If `readRDS` is used for a large RF object (potentially 1–4 GB), deserialization alone can take minutes.
- **Single-call prediction on 6.46M rows**: `predict.randomForest` or `predict.ranger` iterates every tree for every row. If done naively in a single call, memory for the prediction matrix (~6.46M × 110 features as a dense matrix) can spike to many GB, potentially exceeding 16 GB with the model in memory.
- **Data type conversion**: `predict()` may internally coerce a `data.frame` to a matrix — another full copy.
- **Garbage collection pressure**: Repeated large allocations trigger frequent GC pauses.

### 1.3 Summary of Time Sinks (estimated share of 86+ hours)

| Component | Estimated Share |
|---|---|
| `build_neighbor_lookup` | ~25–35% |
| `compute_neighbor_stats` (×5) | ~30–40% |
| Data.frame copying in outer loop | ~10–15% |
| RF prediction (single pass) | ~10–20% |
| Model loading / misc | ~2–5% |

---

## 2. OPTIMIZATION STRATEGY

| Problem | Solution |
|---|---|
| Slow row-by-row `lapply` in `build_neighbor_lookup` | Replace with vectorized `data.table` join; build integer index vectors without per-row character ops |
| Slow `lapply` + `do.call(rbind,...)` in `compute_neighbor_stats` | Use `data.table` grouped aggregation on an edge-list, fully vectorized |
| Repeated full-copy of `cell_data` | Use `data.table` with `:=` (in-place column addition — zero copies) |
| `paste`-based key construction | Use two-column integer keying (`id`, `year`) via `data.table` |
| RF prediction memory spike | Predict in chunked batches (~500K rows) to stay within 16 GB |
| Model loading time | Load once, keep in memory; use `qs::qread` for faster deserialization if re-serialized |

**Expected speedup**: from 86+ hours to approximately **15–45 minutes** depending on disk I/O and RF model size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- Step 0: Convert to data.table (one-time, in-place) --------------------

setDT(cell_data)

# Ensure key columns are integer for fast joining
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column (used for edge-list construction)
cell_data[, .row_idx := .I]

# Key for fast binary-search joins
setkey(cell_data, id, year)


# ---- Step 1: Build edge list (replaces build_neighbor_lookup) ---------------
# This converts the spdep::nb object + id_order into a flat data.table of
# (source_row, neighbor_id) pairs, then joins to get neighbor rows.

build_edge_list_dt <- function(cell_data, id_order, neighbors) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)
  
  id_order <- as.integer(id_order)
  n_ids <- length(id_order)
  
  # --- Build cell-level neighbor table: from_id -> to_id --------------------
  # Pre-calculate total length for pre-allocation
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  from_idx <- rep.int(seq_len(n_ids), lens)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  cell_neighbors <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, lens)
  
  # --- Expand to cell-year level by joining with cell_data ------------------
  # Get the unique (id, year, .row_idx) from cell_data
  id_year <- cell_data[, .(id, year, .row_idx)]
  
  # Join: for every row in cell_data, find its neighbor cell IDs
  # Then for each neighbor cell ID + same year, find the neighbor's row index
  
  # First join: attach row index of source to each (from_id, year)
  setkey(id_year, id)
  
  # Expand cell_neighbors by all years that from_id appears in
  # Use a keyed join: cell_neighbors[id_year, on = .(from_id = id), ...] 
  setkey(cell_neighbors, from_id)
  
  edge_expanded <- cell_neighbors[id_year,
    .(source_row = .row_idx,   # row index of the source cell-year
      neighbor_id = x.to_id,   # cell ID of the neighbor
      year = i.year),
    on = .(from_id = id),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]
  
  # Second join: look up the neighbor's row index for the same year
  setkey(id_year, id, year)
  
  edge_expanded[, neighbor_row := id_year[.(neighbor_id, year), .row_idx, 
                                           on = .(id, year), 
                                           nomatch = NA]$V1]
  
  # Drop edges where the neighbor cell-year doesn't exist
  edge_expanded <- edge_expanded[!is.na(neighbor_row)]
  
  # We only need source_row and neighbor_row going forward
  edge_expanded[, c("neighbor_id", "year") := NULL]
  
  setkey(edge_expanded, source_row)
  
  return(edge_expanded)
}

cat("Building edge list...\n")
system.time({
  edge_list <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)
})
# edge_list has columns: source_row, neighbor_row


# ---- Step 2: Vectorized neighbor stats (replaces compute_neighbor_stats) ----

compute_and_add_all_neighbor_features <- function(cell_data, edge_list, 
                                                   var_names) {
  # Compute max, min, mean of each variable over neighbors in one pass per var,
  # fully vectorized via data.table grouped aggregation.
  
  n <- nrow(cell_data)
  
  for (var_name in var_names) {
    cat("  Processing neighbor features for:", var_name, "\n")
    
    # Extract the variable values at the neighbor rows
    # edge_list$neighbor_row indexes into cell_data
    vals <- cell_data[[var_name]][edge_list$neighbor_row]
    
    # Build a temporary DT for grouped aggregation
    tmp <- data.table(
      source_row = edge_list$source_row,
      val = vals
    )
    
    # Remove NA values before aggregation
    tmp <- tmp[!is.na(val)]
    
    # Grouped aggregation — single pass, vectorized C-level
    agg <- tmp[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = source_row]
    
    # Prepare full-length result columns (NA for cells with no valid neighbors)
    max_col  <- rep(NA_real_, n)
    min_col  <- rep(NA_real_, n)
    mean_col <- rep(NA_real_, n)
    
    max_col[agg$source_row]  <- agg$nb_max
    min_col[agg$source_row]  <- agg$nb_min
    mean_col[agg$source_row] <- agg$nb_mean
    
    # In-place column addition (no copy of cell_data)
    col_max  <- paste0("nb_max_", var_name)
    col_min  <- paste0("nb_min_", var_name)
    col_mean <- paste0("nb_mean_", var_name)
    
    set(cell_data, j = col_max,  value = max_col)
    set(cell_data, j = col_min,  value = min_col)
    set(cell_data, j = col_mean, value = mean_col)
    
    rm(tmp, agg, vals, max_col, min_col, mean_col)
  }
  
  invisible(NULL)  # cell_data modified in place
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
})

# Clean up helper column
cell_data[, .row_idx := NULL]

# Free edge list memory
rm(edge_list)
gc()


# ---- Step 3: Optimized Random Forest Prediction ----------------------------

# Load model once (use qs for speed if available)
if (requireNamespace("qs", quietly = TRUE) && file.exists("rf_model.qs")) {
  cat("Loading model with qs::qread...\n")
  rf_model <- qs::qread("rf_model.qs")
} else {
  cat("Loading model with readRDS...\n")
  rf_model <- readRDS("rf_model.rds")
}

# Identify predictor columns (exclude id, year, and the target variable)
# Adjust 'target_var' to whatever your GDP column is named
target_var <- "gdp"  # <-- adjust as needed
exclude_cols <- c("id", "year", target_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Chunked prediction to manage memory on 16 GB laptop
predict_chunked <- function(model, data, predictor_cols, chunk_size = 500000L) {
  n <- nrow(data)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat("Predicting in", n_chunks, "chunks of up to", chunk_size, "rows...\n")
  
  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    
    # Extract chunk as a plain data.frame (required by most RF predict methods)
    chunk_df <- as.data.frame(data[start_row:end_row, ..predictor_cols])
    
    predictions[start_row:end_row] <- predict(model, newdata = chunk_df)
    
    if (i %% 2 == 0 || i == n_chunks) {
      cat("  Chunk", i, "/", n_chunks, "done.\n")
      gc(verbose = FALSE)
    }
  }
  
  return(predictions)
}

cat("Running prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(
    rf_model, cell_data, predictor_cols, chunk_size = 500000L
  )]
})

cat("Done. Predictions stored in cell_data$predicted_gdp\n")


# ---- Optional: Re-serialize model in qs format for faster future loads ------
if (requireNamespace("qs", quietly = TRUE) && !file.exists("rf_model.qs")) {
  cat("Saving model in qs format for faster future loads...\n")
  qs::qsave(rf_model, "rf_model.qs", preset = "fast")
}
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Requirement | How It Is Met |
|---|---|
| **Trained RF model preserved** | The model object is loaded read-only; `predict()` is the only function called on it. No retraining occurs. |
| **Original numerical estimand preserved** | The same `max`, `min`, `mean` neighbor statistics are computed over identical neighbor sets using the same rook-neighbor definitions. Column names follow the same `nb_{stat}_{var}` pattern. The `predict()` call returns the same numerical output for the same input features. |
| **Same neighbor definitions** | The edge list is derived directly from the same `rook_neighbors_unique` nb object and `id_order` mapping — just restructured from a nested list to a flat join table. |

## 5. EXPECTED PERFORMANCE

| Stage | Before (est.) | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~20–30 hrs | ~1–3 min (edge-list construction via vectorized `data.table` joins) |
| `compute_neighbor_stats` ×5 | ~30–40 hrs | ~2–5 min (grouped aggregation in C) |
| Data.frame copying ×5 | ~10–12 hrs | ~0 s (`set()` — zero-copy in place) |
| RF prediction (6.46M rows) | ~5–10 hrs | ~5–15 min (chunked, same algorithm) |
| **Total** | **~86+ hrs** | **~15–30 min** |

The dominant gains come from eliminating per-row R-level iteration (replaced by vectorized `data.table` operations executed in C) and eliminating full-frame copies (replaced by in-place `:=` / `set()`).