 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows with per-row string operations**

- `paste()` and named-vector lookups (`idx_lookup[neighbor_keys]`) are called **per row** inside `lapply`. For 6.46M rows this creates millions of temporary character vectors and performs hash-table lookups on a 6.46M-entry named vector. This alone can take hours.
- `as.character()` and `paste(…, sep="_")` are called redundantly inside the loop instead of being vectorised once.

**B. `compute_neighbor_stats` — repeated `lapply` over 6.46M rows, called 5 times**

- Each call extracts neighbour values, removes NAs, and computes `max/min/mean` in pure R. With ~1.37M neighbour edges and 6.46M rows, this is ~32.3M list operations across the 5 variables.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself very slow (repeated `rbind` allocation).

**C. Object copying in the outer loop**

- `cell_data <- compute_and_add_neighbor_features(cell_data, …)` likely copies the entire data.frame (6.46M × 110+ columns) on every column addition — 5 variables × 3 stats = 15 column additions, each triggering a copy.

**D. Random Forest prediction (downstream)**

- If `predict()` is called row-by-row or in small batches, overhead dominates. A single `predict(model, newdata)` call on the full 6.46M-row data.frame is the correct approach, but memory may spike (110 columns × 6.46M rows × 8 bytes ≈ 5.7 GB for numeric matrix alone, plus the forest's internal copies).
- Loading the model from disk with `readRDS` for a large RF object can itself take minutes and consume several GB.

**Estimated time breakdown (rough):**
| Step | Est. Time |
|---|---|
| `build_neighbor_lookup` | 20–40 hrs |
| `compute_neighbor_stats` (×5) | 20–30 hrs |
| Column binding / copying | 5–10 hrs |
| RF `predict()` (if done naively) | 2–6 hrs |
| **Total** | **~50–86+ hrs** |

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| Neighbor lookup construction | Replace per-row `lapply`+`paste` with a **vectorised `data.table` merge/join** that maps (cell_id, year) → row index, then expand the `nb` object into an edge-list and join once. | **100–500×** |
| Neighbor stats computation | Build a **sparse adjacency edge-list** with `data.table`, join the variable values, and compute `max/min/mean` with a **grouped aggregation** — no R-level loop at all. | **50–200×** |
| Column additions / copying | Use **`data.table` set-by-reference** (`:=`) to add columns in-place — zero copies. | **10–50×** |
| RF prediction | Single bulk `predict()` call; if memory-constrained, chunk into ~500K-row batches. Pre-convert to matrix if the RF package supports it. | **2–5×** vs naive |

**Target runtime: 5–20 minutes** for feature preparation, **10–60 minutes** for RF prediction, depending on forest size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# 0. LIBRARIES
# =============================================================================
library(data.table)
library(randomForest) # or library(ranger) — adjust predict() call accordingly

# =============================================================================
# 1. CONVERT TO data.table (IN-PLACE, NO COPY)
# =============================================================================
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Add a row-index column (used for fast joins)
cell_data[, .row_idx := .I]

# =============================================================================
# 2. BUILD NEIGHBOR EDGE-LIST (VECTORISED, REPLACES build_neighbor_lookup)
# =============================================================================
build_neighbor_edgelist <- function(id_order, nb_obj) {
  # nb_obj is an spdep::nb list: nb_obj[[i]] gives integer indices into

  # id_order that are neighbours of id_order[i].
  # We expand this into a two-column data.table: (focal_id, neighbor_id).

  n <- length(nb_obj)
  lens <- lengths(nb_obj)                       # number of neighbours per cell
  total_edges <- sum(lens)

  focal_idx    <- rep.int(seq_len(n), lens)     # index into id_order
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# =============================================================================
# 3. COMPUTE & ATTACH NEIGHBOR FEATURES (VECTORISED, IN-PLACE)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(dt, edge_dt, var_name) {
  # --- a. Build a lookup: (id, year) -> value & row_idx ---
  # We only need the variable column, id, and year.
  lookup <- dt[, .(id, year, .var_val = get(var_name))]

  # --- b. Expand edges across all years ---
  # Instead of crossing edges × years (huge), we join edges to the focal rows

  # to get the year, then join again to get the neighbour's value.

  # Step 1: focal side — get (focal_id, year) pairs with the focal row index
  #         We join edge_dt to dt on focal_id == id to get all (focal_id, year) combos.
  #         But this would be 1.37M edges × 28 years = 38.4M rows — manageable.


  # More efficient: join edge_dt to the unique years per focal_id.
  # Since every cell has all 28 years, we can cross-join edges with years.

  years_vec <- sort(unique(dt$year))

  # Cross join: each edge × each year  (~1.37M × 28 ≈ 38.4M rows)
  edge_year <- CJ_dt_year(edge_dt, years_vec)

  # Step 2: attach the neighbour's variable value
  setkey(lookup, id, year)
  setkey(edge_year, neighbor_id, year)
  edge_year[lookup, neighbor_val := i..var_val, on = .(neighbor_id = id, year)]

  # Step 3: group by (focal_id, year) and compute stats
  setkey(edge_year, focal_id, year)
  stats <- edge_year[!is.na(neighbor_val),
                     .(nb_max  = max(neighbor_val),
                       nb_min  = min(neighbor_val),
                       nb_mean = mean(neighbor_val)),
                     by = .(focal_id, year)]

  # Step 4: merge back into dt by reference
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Join stats to dt on (id, year)
  dt[stats, (c(max_col, min_col, mean_col)) :=
       .(i.nb_max, i.nb_min, i.nb_mean),
     on = .(id = focal_id, year = year)]

  invisible(dt)
}

# Helper: cross-join edge_dt with a vector of years
CJ_dt_year <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx     <- rep(seq_len(n_edges), each = n_years)
  result  <- edge_dt[idx]
  result[, year := rep(years_vec, times = n_edges)]
  result
}

# --- If not every cell appears in every year, use a smarter expansion: ---
# (Uncomment the block below and comment out CJ_dt_year above)
#
# CJ_dt_year <- function(edge_dt, years_vec) {
#   # Use a merge with a small year table
#   yr_dt <- data.table(year = years_vec)
#   merge(edge_dt, yr_dt, by = character(0), allow.cartesian = TRUE)
# }

# =============================================================================
# 3b. MEMORY-OPTIMISED VARIANT (if 38.4M-row cross is too large)
# =============================================================================
# If RAM is tight, process one year at a time inside the function.
# This trades a small amount of speed for much lower peak memory.

compute_and_add_neighbor_features_lowmem <- function(dt, edge_dt, var_name) {
  years_vec <- sort(unique(dt$year))

  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Pre-allocate result columns with NA
  dt[, (c(max_col, min_col, mean_col)) := NA_real_]

  setkey(dt, id, year)

  for (yr in years_vec) {
    # Subset to this year
    dt_yr <- dt[year == yr, .(id, .var_val = get(var_name))]
    setkey(dt_yr, id)

    # Attach neighbour values
    edge_yr <- copy(edge_dt)
    edge_yr[dt_yr, neighbor_val := i..var_val, on = .(neighbor_id = id)]

    # Aggregate
    stats_yr <- edge_yr[!is.na(neighbor_val),
                        .(nb_max  = max(neighbor_val),
                          nb_min  = min(neighbor_val),
                          nb_mean = mean(neighbor_val)),
                        by = .(focal_id)]
    stats_yr[, year := yr]

    # Write back by reference
    dt[stats_yr,
       (c(max_col, min_col, mean_col)) := .(i.nb_max, i.nb_min, i.nb_mean),
       on = .(id = focal_id, year)]
  }

  invisible(dt)
}

# =============================================================================
# 4. RUN FEATURE PREPARATION
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbour features...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  # Choose the standard version; switch to _lowmem if RAM is tight
  tryCatch(
    compute_and_add_neighbor_features_fast(cell_data, edge_dt, var_name),
    error = function(e) {
      cat(sprintf("    Fast version failed (%s), falling back to low-mem...\n",
                  conditionMessage(e)))
      compute_and_add_neighbor_features_lowmem(cell_data, edge_dt, var_name)
    }
  )
  gc()   # free intermediate memory
}

cat(sprintf("Feature prep done in %.1f minutes.\n",
            (proc.time() - t0)[["elapsed"]] / 60))

# Clean up helper column
cell_data[, .row_idx := NULL]

# =============================================================================
# 5. RANDOM FOREST PREDICTION (BULK, MEMORY-AWARE)
# =============================================================================
cat("Loading trained Random Forest model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify the predictor columns the model expects
# (works for both randomForest and ranger objects)
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all required columns are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# --- Chunked prediction to control memory ---
predict_chunked <- function(model, dt, pred_vars, chunk_size = 500000L) {
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)

  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    idx_start <- (i - 1L) * chunk_size + 1L
    idx_end   <- min(i * chunk_size, n)
    chunk_dt  <- dt[idx_start:idx_end, ..pred_vars]

    if (inherits(model, "ranger")) {
      preds[idx_start:idx_end] <- predict(model, data = chunk_dt)$predictions
    } else {
      # randomForest::predict
      preds[idx_start:idx_end] <- predict(model, newdata = chunk_dt)
    }

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %s–%s)\n",
                  i, n_chunks,
                  format(idx_start, big.mark = ","),
                  format(idx_end, big.mark = ",")))
    }
  }
  preds
}

cat("Running predictions...\n")
t1 <- proc.time()

cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, pred_vars,
                                              chunk_size = 500000L)]

cat(sprintf("Prediction done in %.1f minutes.\n",
            (proc.time() - t1)[["elapsed"]] / 60))

# =============================================================================
# 6. SUMMARY
# =============================================================================
cat("\n=== Pipeline Summary ===\n")
cat(sprintf("Rows:             %s\n", format(nrow(cell_data), big.mark = ",")))
cat(sprintf("Predictor cols:   %d\n", length(pred_vars)))
cat(sprintf("Neighbour vars:   %d (× 3 stats = %d new columns)\n",
            length(neighbor_source_vars), length(neighbor_source_vars) * 3))
cat(sprintf("Prediction range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))
cat(sprintf("Total elapsed:    %.1f minutes\n",
            (proc.time() - t0)[["elapsed"]] / 60))
```

---

## 4. WHY THIS WORKS — SUMMARY TABLE

| Original Problem | Root Cause | Fix Applied | Complexity Change |
|---|---|---|---|
| `build_neighbor_lookup`: 6.46M `paste` + hash lookups in `lapply` | Per-row string ops in R loop | Vectorised edge-list built once from `nb` object; `data.table` keyed joins replace hash lookups | O(n·k) R-loop → O(E) vectorised |
| `compute_neighbor_stats`: 6.46M `lapply` × 5 vars | R-level loop with per-element `max/min/mean` | `data.table` grouped aggregation (`by=`) on edge-year table | O(n·k) R-loop → O(E) C-level groupby |
| `do.call(rbind, …)` on 6.46M-element list | Repeated memory allocation | Eliminated entirely; stats returned as `data.table` | O(n²) alloc → O(n) |
| `cell_data <- …` column additions | Full data.frame copy on each `:=` | `data.table` `:=` modifies in-place | 15 copies → 0 copies |
| RF `predict()` on 6.46M rows | Potential single-call OOM or row-by-row overhead | Chunked 500K-row batches; single `predict()` per chunk | Balanced memory/speed |

**Expected total runtime: ~10–30 minutes** on a 16 GB laptop (down from 86+ hours), with the trained Random Forest model and original numerical estimand fully preserved.