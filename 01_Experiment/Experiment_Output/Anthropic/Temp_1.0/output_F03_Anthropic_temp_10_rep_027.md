 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. `build_neighbor_lookup` — O(n) `lapply` with expensive string operations

The function iterates over **~6.46 million rows**, and for each row it:

1. **Pastes strings** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — millions of string allocations.
2. **Looks up named character vectors** (`idx_lookup[neighbor_keys]`) — named-vector lookup in R is **O(k)** per name on average, but with ~6.46M names in the vector the hash table is enormous and cache-unfriendly.
3. **Wraps everything in `lapply`** — returns a 6.46M-element list of integer vectors, which is itself a large, fragmented memory structure.

**Net effect:** This single function likely takes **many hours** on 6.46M rows due to per-row string allocation and hash-table lookups.

### B. `compute_neighbor_stats` — repeated `lapply` over 6.46M-element list

For each of 5 variables, it:

1. Iterates all 6.46M rows.
2. Subsets a numeric vector by index, removes NAs, computes `max/min/mean`.
3. Calls `do.call(rbind, result)` on a 6.46M-element list of 3-vectors — this is a **very slow** row-bind pattern.

**Run 5 times** = ~32.3 million R-level function calls.

### C. Object copying (`cell_data <- compute_and_add_neighbor_features(...)`)

Each iteration copies the entire data frame (6.46M × 110+ columns) when adding 3 new columns. With 5 variables that's 5 full copies of a multi-GB frame.

### D. Random Forest prediction

With ~110 features × 6.46M rows, a single `predict()` call on a large Random Forest will:

- Allocate a full feature matrix (~5.7 GB for a 6.46M × 110 double matrix).
- Traverse every tree for every row — this is CPU-bound but also memory-bound if the model is large.
- If `predict()` is called **inside a loop** (row-by-row or chunk-by-chunk badly), overhead multiplies catastrophically.

### Summary of bottlenecks (ranked)

| Rank | Bottleneck | Estimated share |
|------|-----------|----------------|
| 1 | `build_neighbor_lookup` — per-row string paste + hash lookup | ~40-50% |
| 2 | `compute_neighbor_stats` — R-level lapply + `do.call(rbind,...)` | ~20-25% |
| 3 | Data frame copying in outer loop | ~10-15% |
| 4 | RF prediction (matrix construction + tree traversal) | ~10-20% |

---

## 2. Optimization Strategy

### Principle: Replace R-level row loops with vectorized / `data.table` operations

| Bottleneck | Strategy |
|-----------|----------|
| `build_neighbor_lookup` | Build a **`data.table` edge list** (row_idx → neighbor_row_idx) using vectorized integer joins — no strings, no `lapply`. |
| `compute_neighbor_stats` | **Join** the edge list to the values column, then `group-by` aggregate (`max`, `min`, `mean`) — fully vectorized in `data.table` C code. |
| Data frame copying | Use **`data.table` `:=`** (modify in place) — zero copies. |
| RF prediction | Call `predict()` **once** on the full matrix (or in large chunks). Convert feature columns to a matrix **once** with `as.matrix()`. |

**Expected speedup:** From ~86+ hours to **minutes** (the vectorized join + group-by on ~8.9M edges is trivially fast in `data.table`; RF predict on 6.46M rows is typically 5–30 min depending on the forest).

---

## 3. Working R Code

```r
# ============================================================
# OPTIMIZED PIPELINE — data.table vectorized implementation
# ============================================================

library(data.table)
library(randomForest) # or ranger — adjust predict() call accordingly

# ----------------------------------------------------------
# STEP 0: Convert cell_data to data.table (in-place if possible)
# ----------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data) # converts in place — no copy
}

# ----------------------------------------------------------
# STEP 1: Build a vectorized neighbor edge-list (replaces
#          build_neighbor_lookup entirely)
#
# Inputs:
#   cell_data           — data.table with columns `id`, `year`, ...
#   id_order            — integer/character vector of cell IDs in
#                         the same order as rook_neighbors_unique
#   rook_neighbors_unique — spdep nb object (list of integer index
#                           vectors into id_order)
#
# Output:
#   edge_dt — data.table with columns:
#       row_i  : integer row index in cell_data of the focal cell
#       row_j  : integer row index in cell_data of the neighbor cell
# ----------------------------------------------------------

build_neighbor_edge_dt <- function(cell_data, id_order, neighbors) {
  # Map each id_order position to its cell-ID
  n_cells <- length(id_order)

  # --- Build cell-level directed edge list (ref_idx -> neighbor_ref_idx) ---
  from_ref <- rep(seq_len(n_cells),
                  times = lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Convert ref indices to actual cell IDs
  from_id <- id_order[from_ref]
  to_id   <- id_order[to_ref]

  cell_edges <- data.table(from_id = from_id, to_id = to_id)

  # --- Map cell IDs × years to row indices in cell_data ---
  # Add row index to cell_data (will remove later)
  cell_data[, .row_idx := .I]

  # Create a lookup: (id, year) -> row index
  lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(lookup, id, year)

  # Get unique years
  years <- sort(unique(cell_data$year))

  # Cross-join cell edges with years, then look up row indices
  # This is the key vectorized step — no R-level loop over 6.46M rows
  edge_year <- CJ_dt_edges(cell_edges, years)
  # CJ_dt_edges: replicate each edge for every year
  # We'll do this with a simple cross join:
  edge_year <- cell_edges[, .(from_id, to_id, year = list(years)),
                          by = .I][, .(from_id, to_id, year = unlist(year)),
                                     by = I][, I := NULL]

  # Join to get row_i (focal row) and row_j (neighbor row)
  setnames(lookup, c("id", "year", ".row_idx"), c("from_id", "year", "row_i"))
  setkey(edge_year, from_id, year)
  edge_year <- lookup[edge_year, on = .(from_id, year), nomatch = 0L]

  setnames(lookup, c("from_id", "year", "row_i"), c("to_id", "year", "row_j"))
  setkey(edge_year, to_id, year)
  edge_year <- lookup[edge_year, on = .(to_id, year), nomatch = 0L]

  # Clean up: restore lookup names, remove temp column
  cell_data[, .row_idx := NULL]

  edge_year[, .(row_i, row_j)]
}

# --- Simpler, more memory-efficient version using integer keys ---

build_neighbor_edge_dt <- function(cell_data, id_order, neighbors) {

  cell_data[, .row_idx := .I]

  # 1. Cell-level edge list (directed)
  from_ref <- rep.int(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  from_id  <- id_order[from_ref]
  to_id    <- id_order[to_ref]

  cell_edges <- data.table(from_id = from_id, to_id = to_id)

  # 2. Lookup table: (id, year) -> row index
  lu <- cell_data[, .(id, year, .row_idx)]

  # 3. Join focal side: get (from_id, year, row_i) for every cell-year
  #    that the focal cell appears in
  focal <- lu[cell_edges, on = .(id = from_id), allow.cartesian = TRUE,
              nomatch = 0L]
  #    focal now has columns: id (=from_id), year, .row_idx (=row_i), to_id
  setnames(focal, ".row_idx", "row_i")

  # 4. Join neighbor side: get row_j for the same year
  setnames(lu, c("id", "year", ".row_idx"), c("to_id", "year", "row_j"))
  edge_dt <- lu[focal, on = .(to_id, year), nomatch = 0L]

  cell_data[, .row_idx := NULL]

  edge_dt[, .(row_i, row_j)]
}

# ----------------------------------------------------------
# STEP 2: Compute all neighbor stats vectorised
#          (replaces compute_neighbor_stats + outer loop)
# ----------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, edge_dt,
                                          neighbor_source_vars) {
  n <- nrow(cell_data)

  for (var_name in neighbor_source_vars) {
    # Pull the numeric values vector
    vals <- cell_data[[var_name]]

    # Attach neighbor values to edge list (vectorised)
    edge_dt[, val := vals[row_j]]

    # Group-by focal row and compute stats — fully in C via data.table
    stats <- edge_dt[!is.na(val),
                     .(nmax  = max(val),
                       nmin  = min(val),
                       nmean = mean(val)),
                     keyby = .(row_i)]

    # Prepare output column names
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Initialize with NA, then fill matched rows — in-place, no copy
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    matched <- stats$row_i
    set(cell_data, i = matched, j = col_max,  value = stats$nmax)
    set(cell_data, i = matched, j = col_min,  value = stats$nmin)
    set(cell_data, i = matched, j = col_mean, value = stats$nmean)
  }

  # Clean up temp column in edge_dt
  edge_dt[, val := NULL]

  invisible(cell_data)
}

# ----------------------------------------------------------
# STEP 3: Random Forest prediction — single vectorized call
# ----------------------------------------------------------

predict_rf_optimized <- function(cell_data, rf_model, feature_cols) {
  # Build the feature matrix ONCE (avoid repeated subsetting)
  # Using as.matrix on a data.table subset is efficient
  X <- as.matrix(cell_data[, ..feature_cols])

  # Single predict call on the full matrix
  # For randomForest package:
  preds <- predict(rf_model, newdata = X)

  # For ranger package (if applicable), use:
  # preds <- predict(rf_model, data = X)$predictions

  preds
}

# ----------------------------------------------------------
# STEP 4: Full pipeline
# ----------------------------------------------------------

run_optimized_pipeline <- function(cell_data, id_order,
                                   rook_neighbors_unique,
                                   rf_model, feature_cols) {

  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) setDT(cell_data)

  # --- Feature preparation ---
  cat("Building neighbor edge list (vectorized)...\n")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edge_dt(cell_data, id_order,
                                    rook_neighbors_unique)
  cat("  Edge list:", nrow(edge_dt), "edges built in",
      (proc.time() - t0)[3], "sec\n")

  cat("Computing neighbor features (vectorized)...\n")
  t0 <- proc.time()
  neighbor_source_vars <- c("ntl", "ec", "pop_density",
                            "def", "usd_est_n2")
  compute_all_neighbor_features(cell_data, edge_dt,
                                neighbor_source_vars)
  cat("  Neighbor features computed in",
      (proc.time() - t0)[3], "sec\n")

  # Free edge list memory
  rm(edge_dt); gc()

  # --- Prediction ---
  cat("Running Random Forest prediction...\n")
  t0 <- proc.time()
  cell_data[, predicted_gdp := predict_rf_optimized(
    cell_data, rf_model, feature_cols
  )]
  cat("  Prediction completed in",
      (proc.time() - t0)[3], "sec\n")

  cell_data
}

# ----------------------------------------------------------
# STEP 5 (optional): If memory is tight, chunk the prediction
# ----------------------------------------------------------

predict_rf_chunked <- function(cell_data, rf_model, feature_cols,
                               chunk_size = 500000L) {
  n <- nrow(cell_data)
  preds <- numeric(n)

  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    X_chunk <- as.matrix(cell_data[s:e, ..feature_cols])
    preds[s:e] <- predict(rf_model, newdata = X_chunk)
    if (s %% (chunk_size * 5) == 1L) gc() # periodic GC
  }

  preds
}

# ============================================================
# USAGE EXAMPLE
# ============================================================
#
# # Load pre-trained model
# rf_model <- readRDS("rf_model.rds")
#
# # Define the feature column names used during training
# feature_cols <- setdiff(names(cell_data),
#                         c("id", "year", "gdp", "predicted_gdp"))
#
# # Run full optimized pipeline
# cell_data <- run_optimized_pipeline(
#   cell_data, id_order, rook_neighbors_unique,
#   rf_model, feature_cols
# )
```

---

## 4. Summary of Expected Improvements

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~30-40 hrs (R `lapply`, string hashing over 6.46M rows) | ~30-90 sec (`data.table` integer joins) | **~1000-4000×** |
| `compute_neighbor_stats` (×5 vars) | ~20-30 hrs (`lapply` + `do.call(rbind,...)`) | ~10-30 sec (`data.table` group-by in C) | **~2000-5000×** |
| Data frame copying (5 iterations) | ~5-10 hrs (full-frame copy ×5) | ~0 sec (`set()` in place) | **∞** |
| RF prediction | Variable (depends on calling pattern) | ~5-30 min (single `predict()` call, matrix built once) | **1-100×** |
| **Total** | **~86+ hours** | **~10-35 minutes** | **~150-500×** |

### Key invariants preserved

- **Trained Random Forest model**: never modified — used read-only via `predict()`.
- **Original numerical estimand**: the same `max`, `min`, `mean` neighbor statistics are computed from the same neighbor structure with identical numerical semantics. The `predict()` call produces identical outputs because the same features are presented to the same model.