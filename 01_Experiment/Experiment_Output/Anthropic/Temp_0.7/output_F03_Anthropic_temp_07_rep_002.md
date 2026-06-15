 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks (in estimated severity order)

**A. `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows with per-row string operations**

- Each of the ~6.46 million rows does: `as.character()`, `paste()`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but the overhead of creating ~6.46M character keys and performing millions of individual hash lookups is enormous.
- The function returns a list of 6.46M integer vectors — a large, fragmented memory structure.

**B. `compute_neighbor_stats` — repeated `lapply` over 6.46M rows, called 5 times**

- For each of the 5 neighbor source variables, another `lapply` iterates over 6.46M elements, subsetting a numeric vector and computing `max/min/mean`. That's ~32.3M R-level function calls total.
- The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is itself a well-known R anti-pattern (slow row-binding).

**C. Repeated `data.frame` column assignment in the outer loop**

- `cell_data <- compute_and_add_neighbor_features(...)` likely copies the entire data.frame (or at least triggers copy-on-modify) 5 times. With ~110+ columns and 6.46M rows, each copy is multiple GB.

**D. Random Forest prediction on 6.46M rows × 110 features**

- If `predict()` is called in a loop (e.g., per year or per chunk) rather than in a single vectorized call, overhead multiplies.
- Loading the model repeatedly (if done inside a loop) is wasteful.
- `ranger` is much faster than `randomForest` for prediction; if the model was trained with `randomForest`, prediction alone on 6.46M rows can take hours.

### Memory Pressure

On a 16 GB laptop, 6.46M rows × 110 numeric columns ≈ 5.4 GB (double precision). Two copies already exceed RAM, causing swapping. The neighbor lookup list (6.46M elements, each a small integer vector) adds another ~1–2 GB of fragmented memory.

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Replace per-row `paste`/hash with a `data.table` equi-join on `(id, year)` → integer row indices. Build lookup as a flat CSR (compressed sparse row) structure instead of a list. | 50–200× |
| `compute_neighbor_stats` | Vectorized grouped aggregation using `data.table` on the flattened neighbor-edge table. Eliminate `lapply` entirely. | 50–100× |
| Data.frame copying | Use `data.table` with `:=` (in-place column assignment). Zero copies. | 5–10× |
| Random Forest predict | Single `predict()` call on the full matrix. Use `ranger` if possible; if model is `randomForest`, convert or predict in one batch. Set `num.threads`. | 2–10× |
| Memory | Use `data.table` throughout; drop intermediate objects; `gc()` at key points. | Enables 16 GB feasibility |

**Overall target: from 86+ hours → under 30 minutes.**

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest), spdep (for nb object)
# Preserves: trained RF model (loaded once), original numerical estimand
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
 setDT(cell_data)
}

# Ensure id and year are the types we expect
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- Step 1: Build flat neighbor edge table (replaces build_neighbor_lookup) 
# id_order: integer vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_neighbor_edges <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  # Build source_id and target_id vectors directly
  source_idx <- rep.int(seq_along(neighbors), lengths(neighbors))
  target_idx <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    source_id = id_order[source_idx],
    target_id = id_order[target_idx]
  )
}

cat("Building neighbor edge table...\n")
edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- Step 2: Build row-index lookup via data.table keyed join ---------------
# We need to map (target_id, year) -> row index in cell_data so we can fetch
# neighbor variable values.

# Add a row index column to cell_data
cell_data[, .row_idx := .I]

# Create a keyed lookup: (id, year) -> row_idx
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# ---- Step 3: Vectorized neighbor stats (replaces compute_neighbor_stats) ----

compute_and_add_all_neighbor_features <- function(cell_data, edge_dt,
                                                   row_lookup,
                                                   neighbor_source_vars) {
  # Get unique years
  years <- unique(cell_data$year)
  
  # Expand edges across all years: each edge exists in every year
  # This creates the full (source_id, year, target_id) table
  cat("Expanding edges across years...\n")
  
  # More memory-efficient: join edges with row_lookup to get target row indices

  # For each (source_id, year), we need all target_id neighbors and their values
  
  # Step A: Create (source_id, target_id, year) by cross-joining edges with years
  # To avoid a massive cross-join (edges × years), we instead:
  # 1. For each source row in cell_data, find its neighbors via edge_dt
  # 2. Look up the neighbor's row in the same year
  
  # Build: source_row -> (source_id, year) from cell_data
  # Then join to edge_dt on source_id to get target_ids
  # Then join to row_lookup on (target_id, year) to get target rows
  
  # source info: (source_row_idx, source_id, year)
  source_info <- cell_data[, .(.row_idx, id, year)]
  setnames(source_info, c("source_row", "source_id", "year"))
  setkey(source_info, source_id)
  
  # Join source_info with edge_dt on source_id
  setkey(edge_dt, source_id)
  cat("Joining source rows to neighbor edges...\n")
  expanded <- edge_dt[source_info, on = "source_id",
                      .(source_row = i.source_row,
                        target_id  = x.target_id,
                        year       = i.year),
                      allow.cartesian = TRUE,
                      nomatch = NULL]
  
  cat(sprintf("  Expanded neighbor table: %s rows\n",
              format(nrow(expanded), big.mark = ",")))
  
  # Join to row_lookup to get target row indices
  setkey(expanded, target_id, year)
  setkey(row_lookup, id, year)
  expanded[row_lookup, target_row := i..row_idx,
           on = .(target_id = id, year = year)]
  
  # Drop rows where target_row is NA (neighbor cell-year doesn't exist)
  expanded <- expanded[!is.na(target_row)]
  
  cat("Computing neighbor stats for all variables...\n")
  
  # For each variable, extract target values, group by source_row, compute stats
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    
    # Extract the variable values for target rows
    vals <- cell_data[[var_name]]
    expanded[, v := vals[target_row]]
    
    # Group by source_row, compute max/min/mean (excluding NAs)
    stats <- expanded[!is.na(v),
                      .(nmax  = max(v),
                        nmin  = min(v),
                        nmean = mean(v)),
                      by = source_row]
    
    # Assign to cell_data by reference (in-place, no copy)
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)
    
    # Initialize with NA
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
    
    # Fill in computed values
    set(cell_data, i = stats$source_row, j = max_col,  value = stats$nmax)
    set(cell_data, i = stats$source_row, j = min_col,  value = stats$nmin)
    set(cell_data, i = stats$source_row, j = mean_col, value = stats$nmean)
  }
  
  # Clean up the temporary column
  expanded[, v := NULL]
  
  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data, edge_dt, row_lookup, neighbor_source_vars
)

# Free intermediate objects
rm(edge_dt, row_lookup)
gc()

# Remove helper column
cell_data[, .row_idx := NULL]

cat("Neighbor features complete.\n")

# ---- Step 4: Optimized Random Forest Prediction ----------------------------

cat("Preparing prediction matrix...\n")

# Load model once (adjust path as needed)
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify predictor columns (exclude id, year, and response variable)
# Adjust 'response_var' to your actual response variable name
response_var <- "gdp"  # adjust as needed
meta_cols    <- c("id", "year", response_var)
predictor_cols <- setdiff(names(cell_data), meta_cols)

# Ensure column order matches training data
# If rf_model is a ranger object:
if (inherits(rf_model, "ranger")) {
  # ranger stores feature names in rf_model$forest$independent.variable.names
  model_features <- rf_model$forest$independent.variable.names
  
  # Check all features are present
  missing <- setdiff(model_features, names(cell_data))
  if (length(missing) > 0) {
    stop(paste("Missing features:", paste(missing, collapse = ", ")))
  }
  
  # Predict in one vectorized call using all available threads
  cat("Running ranger prediction...\n")
  pred <- predict(
    rf_model,
    data = cell_data[, ..model_features],
    num.threads = parallel::detectCores(logical = FALSE),
    verbose = TRUE
  )
  
  cell_data[, predicted_gdp := pred$predictions]
  
} else if (inherits(rf_model, "randomForest")) {
  # randomForest package: slower, single-threaded
  # Option A: predict in one call (works but slow)
  # Option B: chunk to manage memory, still one predict() per chunk
  
  model_features <- rownames(rf_model$importance)
  missing <- setdiff(model_features, names(cell_data))
  if (length(missing) > 0) {
    stop(paste("Missing features:", paste(missing, collapse = ", ")))
  }
  
  cat("Running randomForest prediction (chunked for memory)...\n")
  
  n_rows    <- nrow(cell_data)
  chunk_size <- 500000L
  n_chunks  <- ceiling(n_rows / chunk_size)
  preds     <- numeric(n_rows)
  
  for (ch in seq_len(n_chunks)) {
    start_i <- (ch - 1L) * chunk_size + 1L
    end_i   <- min(ch * chunk_size, n_rows)
    cat(sprintf("  Chunk %d/%d (rows %d-%d)\n", ch, n_chunks, start_i, end_i))
    
    chunk_data <- as.data.frame(cell_data[start_i:end_i, ..model_features])
    preds[start_i:end_i] <- predict(rf_model, newdata = chunk_data)
    rm(chunk_data)
  }
  
  cell_data[, predicted_gdp := preds]
  rm(preds)
  
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

gc()
cat("Prediction complete.\n")

# ---- Step 5: Output ---------------------------------------------------------
# cell_data now contains 'predicted_gdp' alongside all original columns.
# The original numerical estimand is preserved (no transformations applied).

cat(sprintf("Output: %s rows, %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))
cat(sprintf("Predicted GDP range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Why It's Faster |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows; `paste()`+named-vector hash per row | Single `data.table` keyed join; flat edge table | Vectorized C-level join vs. 6.46M R function calls |
| **Neighbor stats** | 5 × `lapply` over 6.46M-element list; `do.call(rbind, ...)` | Single expanded edge table; `data.table` grouped `max/min/mean` with GForce | Vectorized grouped aggregation; no row-binding |
| **Column assignment** | `cell_data <- ...` (copies entire DF 5×) | `set()` / `:=` in-place by reference | Zero-copy; saves ~25 GB of allocation+GC |
| **Prediction** | Likely looped or suboptimal | Single `predict()` call; `num.threads` for `ranger`; chunked for `randomForest` | Minimizes R overhead; parallelizes tree traversal |
| **Memory** | ~2–3 full copies of 5+ GB data.frame in flight | One `data.table` + one temporary edge table; explicit `rm()`/`gc()` | Stays within 16 GB |

**Expected runtime: 10–30 minutes** (depending on model class and disk I/O), down from 86+ hours — roughly a **200× improvement**.