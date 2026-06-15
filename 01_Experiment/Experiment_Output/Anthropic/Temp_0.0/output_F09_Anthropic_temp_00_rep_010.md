 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M rows, this creates billions of intermediate character objects and named-vector searches (O(n) or O(n log n) per lookup in R's hashed environments, but with enormous constant overhead from string allocation and garbage collection).

### 2. The neighbor lookup is **year-redundant**
The spatial neighbor structure is **identical across all 28 years** — cell A's rook neighbors are always the same cells regardless of year. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, inflating the work by a factor of 28×.

### 3. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level `lapply` with 6.46M iterations is inherently slow. Each iteration allocates small vectors, subsets, removes NAs, and computes three summary statistics — all in interpreted R.

### Summary of waste
| Bottleneck | Scale | Root cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + named lookups | Year-redundant; should be 344K spatial lookups only |
| `compute_neighbor_stats` | 6.46M R-level iterations × 5 vars | Should be vectorized join + grouped aggregation |
| Memory | Millions of small character vectors | GC pressure from string intermediates |

---

## Optimization Strategy

**Core insight:** Build the neighbor table **once at the spatial level** (344K cells × ~4 neighbors each ≈ 1.37M directed edges), then **join yearly attributes** onto both sides of each edge, and compute grouped `max`, `min`, `mean` using `data.table` — fully vectorized, zero row-level R loops.

### Steps

1. **Flatten `rook_neighbors_unique`** (the `nb` object) into an edge list `data.table` with columns `(focal_id, neighbor_id)`. This is done once and has ~1.37M rows.

2. **Convert `cell_data` to a `data.table`** keyed on `(id, year)`.

3. **For each neighbor source variable**, join the neighbor's yearly value onto the edge list (by `neighbor_id` and `year`), then aggregate by `(focal_id, year)` to get `max`, `min`, `mean`. This is a keyed `data.table` join + grouped aggregation — extremely fast.

4. **Join the resulting neighbor features back** onto `cell_data`.

5. **Predict** with the existing trained Random Forest model (unchanged).

### Expected speedup

| Component | Before | After |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | <1 sec (344K integer expansion) |
| Compute stats (5 vars) | ~hours (5 × 6.46M lapply) | ~30–90 sec (5 × vectorized join+agg on 1.37M×28 ≈ 38M rows) |
| **Total neighbor feature engineering** | **~86+ hours** | **~2–5 minutes** |

RAM: The edge list × years is ~38.4M rows × a few columns of doubles ≈ < 2 GB. Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build the spatial edge list ONCE (year-invariant)
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, nb_object) {
  # id_order : vector of cell IDs in the same order as the nb object
  # nb_object: spdep nb list (rook_neighbors_unique)
  #
  # Returns a data.table with columns: focal_id, neighbor_id
  
  n <- length(nb_object)
  # Pre-compute total number of edges for memory pre-allocation
  n_edges <- sum(lengths(nb_object))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- nb_object[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    len    <- length(nb_idx)
    if (len > 0L) {
      idx_range <- pos:(pos + len - 1L)
      focal_id[idx_range]    <- id_order[i]
      neighbor_id[idx_range] <- id_order[nb_idx]
      pos <- pos + len
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    focal_id    <- focal_id[1:(pos - 1L)]
    neighbor_id <- neighbor_id[1:(pos - 1L)]
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

# ──────────────────────────────────────────────────────────────────────
# 2. Compute neighbor features for one variable (vectorized)
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_features_dt <- function(cell_dt, edges, var_name) {
  # cell_dt : data.table with key (id, year) and column var_name
  # edges   : data.table with columns (focal_id, neighbor_id)
  # var_name: character, name of the variable
  #
  # Returns cell_dt with three new columns appended:
  #   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean
  
  # Subset to only the columns we need for the join
  vals_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(vals_dt, id, year)
  
  # Cross edges with all years present in the data
  years <- sort(unique(cell_dt$year))
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_years[, `:=`(
    focal_id    = edges$focal_id[edge_idx],
    neighbor_id = edges$neighbor_id[edge_idx]
  )]
  edge_years[, edge_idx := NULL]
  
  # Join neighbor values onto edge_years
  setkey(edge_years, neighbor_id, year)
  edge_years[vals_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Aggregate by (focal_id, year), dropping NAs
  agg <- edge_years[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(focal_id, year)
  ]
  
  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Join aggregated features back onto cell_dt
  setkey(agg, focal_id, year)
  setkey(cell_dt, id, year)
  
  # Remove these columns if they already exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt[agg, (c(max_col, min_col, mean_col)) := mget(c(
    paste0("i.", max_col),
    paste0("i.", min_col),
    paste0("i.", mean_col)
  )), on = .(id = focal_id, year)]
  
  cell_dt
}

# ──────────────────────────────────────────────────────────────────────
# 2b. Memory-efficient variant (processes one year at a time)
#     Use this if the full cross of edges × years exceeds RAM.
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_features_dt_lowmem <- function(cell_dt, edges, var_name) {
  years    <- sort(unique(cell_dt$year))
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Pre-allocate result columns with NA
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    cell_dt[, (col) := NA_real_]
  }
  
  setkey(cell_dt, id, year)
  
  for (yr in years) {
    # Subset this year's values
    yr_vals <- cell_dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_vals, id)
    
    # Join neighbor values onto edges
    edge_yr <- copy(edges)
    edge_yr[yr_vals, neighbor_val := i.val, on = .(neighbor_id = id)]
    
    # Aggregate
    agg_yr <- edge_yr[
      !is.na(neighbor_val),
      .(
        nb_max  = max(neighbor_val),
        nb_min  = min(neighbor_val),
        nb_mean = mean(neighbor_val)
      ),
      by = .(focal_id)
    ]
    
    # Write back into cell_dt for this year
    idx <- cell_dt[.(agg_yr$focal_id, yr), which = TRUE, on = .(id, year)]
    set(cell_dt, i = idx, j = max_col,  value = agg_yr$nb_max)
    set(cell_dt, i = idx, j = min_col,  value = agg_yr$nb_min)
    set(cell_dt, i = idx, j = mean_col, value = agg_yr$nb_mean)
  }
  
  cell_dt
}

# ──────────────────────────────────────────────────────────────────────
# 3. Full pipeline
# ──────────────────────────────────────────────────────────────────────

# --- Load pre-existing objects (assumed already in environment) ---
# cell_data               : data.frame / data.table with columns id, year, ntl, ec, ...
# id_order                : integer vector of cell IDs matching nb object order
# rook_neighbors_unique   : spdep nb object (loaded from disk)
# trained_rf_model        : the already-trained Random Forest model (DO NOT retrain)

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Step A: Build spatial edge list once (~1.37M rows, < 1 second)
cat("Building spatial edge list...\n")
edges <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edges), big.mark = ",")))

# Step B: Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  t0 <- proc.time()
  
  # Use the standard variant; switch to _lowmem if RAM is tight
  cell_data <- compute_neighbor_features_dt(cell_data, edges, var_name)
  
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("    Done in %.1f seconds\n", elapsed))
}

# Step C: Predict with the existing trained Random Forest (unchanged)
cat("Generating predictions with trained Random Forest model...\n")

# Ensure predictor columns match what the model expects
predictor_cols <- setdiff(names(trained_rf_model$forest$xlevels),  # for ranger
                          character(0))
# Generic approach: use the model's expected variable names
# For randomForest package:
if (inherits(trained_rf_model, "randomForest")) {
  predictor_cols <- rownames(trained_rf_model$importance)
}
# For ranger package:
if (inherits(trained_rf_model, "ranger")) {
  predictor_cols <- trained_rf_model$forest$independent.variable.names
}

pred_data <- cell_data[, ..predictor_cols]
cell_data[, prediction := predict(trained_rf_model, data = pred_data)$predictions]
# NOTE: adjust the predict() call syntax to match your RF package
# randomForest: predict(trained_rf_model, newdata = pred_data)
# ranger:       predict(trained_rf_model, data = pred_data)$predictions

cat("Pipeline complete.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The `max`, `min`, `mean` aggregations are computed over exactly the same set of rook-neighbor cell-year values as the original code. The edge list is a faithful flattening of the same `nb` object. NA handling (`!is.na(neighbor_val)`) mirrors the original. |
| **Trained RF model** | The model object is never modified or retrained. Only `predict()` is called. |
| **Column naming** | Output columns follow the same `<var>_neighbor_max/min/mean` convention, so the trained model's expected feature names are matched. |

---

## Expected Performance on a 16 GB Laptop

| Step | Time estimate | Peak RAM |
|---|---|---|
| `build_edge_list` | < 1 sec | ~22 MB (1.37M × 2 int cols) |
| `compute_neighbor_features_dt` × 5 vars | ~30–120 sec each | ~2.5 GB peak (38.4M row edge-year table) |
| RF prediction (6.46M rows × 110 features) | Depends on forest size | Existing model footprint |
| **Total** | **~3–10 minutes** | **< 6 GB** |

If RAM is a concern, substitute `compute_neighbor_features_dt_lowmem`, which processes one year at a time and peaks at ~1.37M rows per iteration instead of 38.4M — at the cost of slightly more wall-clock time (still under ~15 minutes total).