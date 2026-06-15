 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs `paste()` string concatenation, character-keyed named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and `is.na` filtering. Named-vector lookup in R is O(n) hash probing per call, and doing this 6.46M × k-neighbors times is extremely expensive.
- String keys (`paste(id, year, sep="_")`) are created both at setup and inside every iteration, generating enormous temporary character vectors and GC pressure.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M rows, each extracting a subset of a numeric vector, removing NAs, and computing three summary statistics. The `do.call(rbind, result)` at the end binds 6.46M small vectors — this alone can take minutes.
- This function is called **5 times** (once per neighbor source variable), multiplying the cost.

**Combined cost estimate:** ~6.46M iterations × 5 variables × expensive per-iteration work ≈ tens of billions of R-level operations. This explains the 86+ hour estimate.

### 1.2 Prediction-Workflow Bottlenecks

- **Model loading:** If the serialized Random Forest is large (110 predictors, many trees), `readRDS()` can take significant time and memory.
- **Single-call vs. chunked prediction:** If `predict()` is called row-by-row or in small chunks instead of on the full matrix at once, overhead is enormous.
- **Object copying:** R's copy-on-modify semantics mean that repeatedly doing `cell_data$new_col <- ...` inside a loop triggers full data.frame copies (~6.46M × 110+ cols). Each copy of a 6.46M-row, 110-column data.frame is ~5–7 GB.
- **Memory pressure:** 16 GB RAM is tight. Repeated copies and large intermediate objects cause swapping to disk.

---

## 2. OPTIMIZATION STRATEGY

| Problem | Solution | Expected Speedup |
|---|---|---|
| String-keyed lookup in `build_neighbor_lookup` | Replace with integer-indexed `data.table` join; pre-compute a (cell_id, year) → row mapping as a keyed data.table | 50–100× |
| Per-row `lapply` in `compute_neighbor_stats` | Explode neighbor pairs into a long table, join values, then `data.table` grouped aggregation (`max`, `min`, `mean`) | 50–200× |
| `do.call(rbind, ...)` on millions of small vectors | Eliminated by vectorized `data.table` grouping | 10–50× |
| 5× repeated full-dataset passes | Melt all 5 variables, compute all neighbor stats in one grouped pass, then dcast back | 3–5× |
| Data.frame copy-on-modify (`cell_data$x <- ...`) | Use `data.table` with `:=` in-place assignment | 5–10× (eliminates multi-GB copies) |
| RF prediction overhead | Single `predict()` call on full matrix; use `ranger` re-read if model is `ranger` class for faster predict | 2–5× |
| Memory pressure | Drop intermediate objects; `gc()` at key points; use single-precision where possible | Avoids swap |

**Overall target: 86+ hours → minutes.**

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# =============================================================================
# Prerequisites: install.packages(c("data.table", "ranger")) if needed

library(data.table)

# ---- STEP 0: Load data and model -------------------------------------------

# Assume cell_data is already loaded as a data.frame or data.table
# Assume rook_neighbors_unique is a precomputed nb object (list of integer vecs)
# Assume id_order is the vector mapping nb-list positions to cell IDs
# Assume rf_model is the trained Random Forest (ranger or randomForest object)

# Convert to data.table in place (no copy if already data.table)
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- STEP 1: Build neighbor lookup (vectorized, integer-indexed) -----------

build_neighbor_lookup_dt <- function(data, id_order, neighbors) {
  # Map: nb-list index -> cell_id
  # neighbors[[k]] gives nb-list indices of neighbors of cell id_order[k]
  
  # 1a. Build cell_id -> nb-list-index map
  id_to_ref <- data.table(
    cell_id = as.integer(id_order),
    ref_idx = seq_along(id_order)
  )
  setkey(id_to_ref, cell_id)
  
  # 1b. Build row-index lookup: (cell_id, year) -> row number
  data[, .row_idx := .I]
  row_map <- data[, .(cell_id = id, year, .row_idx)]
  setkey(row_map, cell_id, year)
  
  # 1c. Explode neighbor list into a long edge table:
  #     For each nb-list index k, neighbors[[k]] gives neighbor nb-indices.
  #     We need: (focal_cell_id, neighbor_cell_id) pairs.
  
  cat("Building edge list from nb object...\n")
  
  # Efficiently build edge list
  n_cells <- length(neighbors)
  focal_ref  <- rep(seq_len(n_cells), times = lengths(neighbors))
  nbr_ref    <- unlist(neighbors, use.names = FALSE)
  
  # Remove zero-entries (spdep uses 0 for no-neighbor sentinel)
  valid <- nbr_ref > 0L
  focal_ref <- focal_ref[valid]
  nbr_ref   <- nbr_ref[valid]
  
  edges <- data.table(
    focal_cell_id = id_order[focal_ref],
    nbr_cell_id   = id_order[nbr_ref]
  )
  rm(focal_ref, nbr_ref, valid)
  
  cat(sprintf("Edge list: %s directed neighbor pairs\n", format(nrow(edges), big.mark = ",")))
  
  # 1d. For each row in data, we need to find its neighbors in the same year.
  #     Strategy: join data rows to edges on focal_cell_id, then join to row_map
  #     on (nbr_cell_id, year) to get neighbor row indices.
  
  # Focal rows: (focal_cell_id, year, focal_row_idx)
  focal <- data[, .(focal_cell_id = id, year, focal_row_idx = .row_idx)]
  
  cat("Joining focal rows to edge list...\n")
  
  # Join: focal × edges -> (focal_row_idx, nbr_cell_id, year)
  setkey(edges, focal_cell_id)
  setkey(focal, focal_cell_id)
  
  # This is the big join: ~6.46M focal rows × ~4 neighbors each ≈ ~26M rows
  joined <- edges[focal, on = "focal_cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # joined has columns: focal_cell_id, nbr_cell_id, year, focal_row_idx
  
  cat(sprintf("Joined table: %s rows\n", format(nrow(joined), big.mark = ",")))
  
  # Now find the row index of each neighbor in the same year
  setkey(joined, nbr_cell_id, year)
  setkey(row_map, cell_id, year)
  
  joined[row_map, nbr_row_idx := i..row_idx, on = .(nbr_cell_id = cell_id, year)]
  
  # Drop rows where neighbor wasn't found (boundary / missing year)
  joined <- joined[!is.na(nbr_row_idx)]
  
  cat(sprintf("Final neighbor-pair table: %s rows\n", format(nrow(joined), big.mark = ",")))
  
  # Clean up temporary column
  data[, .row_idx := NULL]
  
  # Return the long neighbor-pair table
  # Columns needed downstream: focal_row_idx, nbr_row_idx
  joined[, .(focal_row_idx, nbr_row_idx)]
}

cat("=== Building neighbor pair table ===\n")
t0 <- proc.time()
nbr_pairs <- build_neighbor_lookup_dt(cell_data, id_order, rook_neighbors_unique)
setkey(nbr_pairs, focal_row_idx)
cat(sprintf("Neighbor lookup built in %.1f seconds\n", (proc.time() - t0)[3]))
gc()


# ---- STEP 2: Compute all neighbor stats at once (vectorized) ---------------

compute_all_neighbor_features <- function(data, nbr_pairs, var_names) {
  # For each var_name, attach neighbor values, then group-aggregate.
  
  n <- nrow(data)
  
  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
    t1 <- proc.time()
    
    # Get the values vector
    vals <- data[[var_name]]
    
    # Attach neighbor values to the pair table
    nbr_pairs[, nbr_val := vals[nbr_row_idx]]
    
    # Remove NAs in neighbor values before aggregation
    valid_pairs <- nbr_pairs[!is.na(nbr_val)]
    
    # Grouped aggregation: max, min, mean per focal row
    stats <- valid_pairs[, .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ), by = focal_row_idx]
    
    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    set(data, j = max_col,  value = rep(NA_real_, n))
    set(data, j = min_col,  value = rep(NA_real_, n))
    set(data, j = mean_col, value = rep(NA_real_, n))
    
    # Fill in computed values by row index (in-place)
    set(data, i = stats$focal_row_idx, j = max_col,  value = stats$nb_max)
    set(data, i = stats$focal_row_idx, j = min_col,  value = stats$nb_min)
    set(data, i = stats$focal_row_idx, j = mean_col, value = stats$nb_mean)
    
    cat(sprintf("    Done in %.1f seconds\n", (proc.time() - t1)[3]))
  }
  
  # Clean up temp column
  nbr_pairs[, nbr_val := NULL]
  
  invisible(data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("=== Computing neighbor features ===\n")
t0 <- proc.time()
compute_all_neighbor_features(cell_data, nbr_pairs, neighbor_source_vars)
cat(sprintf("All neighbor features computed in %.1f seconds\n", (proc.time() - t0)[3]))

rm(nbr_pairs)
gc()


# ---- STEP 3: Random Forest Prediction (optimized) --------------------------

cat("=== Running Random Forest prediction ===\n")

# Load model once
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify predictor columns (exclude target, id, year, etc.)
# Adjust these exclusions to match your actual target and ID columns:
exclude_cols <- c("gdp", "id", "year")  
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Ensure predictor_cols matches what the model expects
# If using ranger:
if (inherits(rf_model, "ranger")) {
  model_vars <- rf_model$forest$independent.variable.names
  # Verify all model vars are present
  missing_vars <- setdiff(model_vars, names(cell_data))
  if (length(missing_vars) > 0) {
    warning("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
  }
  predictor_cols <- intersect(model_vars, names(cell_data))
}

# Build prediction matrix ONCE (avoid repeated subsetting)
cat("Preparing prediction matrix...\n")
t0 <- proc.time()

# For ranger: pass data.frame/data.table directly (it handles it efficiently)
# For randomForest: may need a matrix
if (inherits(rf_model, "ranger")) {
  # ranger::predict accepts data.table directly
  pred_data <- cell_data[, ..predictor_cols]
  
  cat("Running ranger predict (single call, all rows)...\n")
  pred_result <- predict(rf_model, data = pred_data, num.threads = parallel::detectCores())
  cell_data[, gdp_pred := pred_result$predictions]
  
} else if (inherits(rf_model, "randomForest")) {
  # randomForest::predict — single call on full dataset
  pred_data <- cell_data[, ..predictor_cols]
  
  cat("Running randomForest predict (single call, all rows)...\n")
  # For very large data, predict in chunks to manage memory, but single call is fastest
  # if memory allows (~6.46M × 110 doubles ≈ 5.3 GB)
  
  # If memory is tight, chunk it:
  chunk_size <- 500000L
  n_rows <- nrow(pred_data)
  preds <- numeric(n_rows)
  
  n_chunks <- ceiling(n_rows / chunk_size)
  for (ch in seq_len(n_chunks)) {
    idx_start <- (ch - 1L) * chunk_size + 1L
    idx_end   <- min(ch * chunk_size, n_rows)
    cat(sprintf("  Chunk %d/%d (rows %d-%d)\n", ch, n_chunks, idx_start, idx_end))
    preds[idx_start:idx_end] <- predict(
      rf_model, 
      newdata = as.data.frame(pred_data[idx_start:idx_end])
    )
  }
  cell_data[, gdp_pred := preds]
  rm(preds)
  
} else {
  # Generic fallback
  cat("Running generic predict...\n")
  cell_data[, gdp_pred := predict(rf_model, newdata = cell_data[, ..predictor_cols])]
}

rm(pred_data)
gc()

cat(sprintf("Prediction complete in %.1f seconds\n", (proc.time() - t0)[3]))

# ---- STEP 4: Verify output -------------------------------------------------
cat(sprintf(
  "Result: %s rows, gdp_pred range [%.4f, %.4f], NAs: %d\n",
  format(nrow(cell_data), big.mark = ","),
  min(cell_data$gdp_pred, na.rm = TRUE),
  max(cell_data$gdp_pred, na.rm = TRUE),
  sum(is.na(cell_data$gdp_pred))
))
```

---

## 4. SUMMARY OF KEY CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Why It's Faster |
|---|---|---|---|
| **Data structure** | `data.frame` | `data.table` | In-place `:=` / `set()` — no multi-GB copies |
| **Neighbor lookup** | 6.46M `lapply` iterations with `paste`/named-vector lookup | Single vectorized `data.table` equi-join producing a ~26M-row long pair table | Eliminates per-row string ops; hash join is O(n) total |
| **Neighbor stats** | 6.46M `lapply` iterations × 5 vars, `do.call(rbind, ...)` | `data.table` grouped `[, .(max, min, mean), by=]` on the long pair table, 5 passes | Vectorized C-level grouping; no R-level loop |
| **Feature assignment** | `cell_data$col <- ...` (triggers copy) | `set(data, i, j, value)` (true in-place) | Zero-copy column writes |
| **RF prediction** | Potentially row-by-row or small-chunk | Single `predict()` call (or large chunks for `randomForest`) with max threads | Amortizes tree-traversal overhead; parallelism |
| **Memory** | ~Multiple 5–7 GB copies coexisting | One `data.table` + one ~26M-row pair table (~400 MB) | Stays well within 16 GB |

**Expected total runtime:** The feature-preparation phase should complete in **2–10 minutes** (down from 80+ hours). The prediction phase depends on the RF model size but should take **5–30 minutes** for a `ranger` model with multi-threading, or **30–90 minutes** for a `randomForest` model. **Total: under 2 hours** on a 16 GB laptop, with no change to the trained model or the numerical estimand.