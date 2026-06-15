You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F10_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query due to hashing overhead at scale, and `paste`/string operations on millions of keys are extremely slow. This single function likely accounts for 30â€“50% of total runtime.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max`/`min`/`mean`. This is called 5 times (once per variable), totaling ~32.3 million R-level loop iterations with per-element allocation overhead.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are a spatial relationship â€” cell *i*'s neighbors are the same in 1992 as in 2019. Yet the current code embeds year into the lookup keys, creating a massive yearÃ—cell structure instead of reusing a compact cell-only adjacency structure.

**Memory profile:** The 6.46M-element list of integer vectors, plus string keys, likely consumes 3â€“5 GB alone, leaving little headroom on a 16 GB laptop.

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure once over the 344,208 cells (not 6.46M cell-years). Rook contiguity is symmetric and year-invariant.

2. **Use a sparse matrix (CSR format via `Matrix::sparseMatrix`).** Store the ~1.37M directed edges as a 344,208 Ã— 344,208 sparse matrix. Neighbor aggregation then becomes sparse matrixâ€“dense matrix multiplication â€” a single vectorized BLAS-backed operation replacing millions of R-level loops.

3. **Reshape variables into cell Ã— year matrices (344,208 Ã— 28).** For each variable, `max`, `min`, and `mean` across neighbors can be computed column-by-column (year-by-year) using sparse matrix operations:
   - **Mean:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix.
   - **Sum:** `A %*% X` (binary adjacency).
   - **Max / Min:** Use the `{sparseMatrixStats}` package or a grouped operation on the CSC/CSR triplet representation. Alternatively, since max/min are not linear, iterate over cells using the CSR pointer arrays in compiled code â€” but a simpler R approach uses `data.table` grouped operations on the edge list.

4. **For max/min specifically**, sparse matrix multiplication doesn't directly apply. The most efficient pure-R approach: expand the edge list to (source, target) pairs, join target values, then group-by source to compute max/min/mean. With `data.table`, this is highly optimized and runs in seconds per variable-year.

5. **Estimated speedup:** From ~86 hours to **~5â€“15 minutes**.

## Optimized R Code

```r
library(data.table)
library(Matrix)
library(ranger)  # or randomForest â€” whichever holds the pre-trained model

# ============================================================
# 0. Assumptions about inputs already in memory / on disk:
#    - cell_data       : data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#    - id_order        : integer vector of cell IDs in the order matching rook_neighbors_unique
#    - rook_neighbors_unique : spdep nb object (list of integer index vectors), length = 344208
#    - rf_model        : pre-trained Random Forest model object
# ============================================================

# ============================================================
# 1. Build sparse adjacency edge list ONCE (year-invariant)
# ============================================================

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices (into id_order) of neighbors of cell i
  # We build a data.table of (source_idx, target_idx) in id_order space
  n <- length(nb_obj)
  src <- rep.int(seq_len(n), lengths(nb_obj))
  tgt <- unlist(nb_obj, use.names = FALSE)
  # Remove 0-entries (spdep uses 0 for "no neighbors")
  valid <- tgt > 0L
  data.table(src_idx = src[valid], tgt_idx = tgt[valid])
}

cat("Building edge list...\n")
edges <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  %d directed edges across %d cells\n", nrow(edges), length(id_order)))

# Map cell index (position in id_order) -> cell ID
idx_to_id <- id_order  # idx_to_id[i] gives the cell ID for index i

# ============================================================
# 2. Convert cell_data to data.table, keyed for fast joins
# ============================================================

dt <- as.data.table(cell_data)

# Create a cell-index column matching id_order positions
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
dt[, cell_idx := id_to_idx[as.character(id)]]

# Verify no NAs (all cell IDs should be in id_order)
stopifnot(!anyNA(dt$cell_idx))

# Key for fast joins
setkey(dt, cell_idx, year)

# ============================================================
# 3. Compute neighbor features: max, min, mean per variable
#    Strategy: for each variable, join edges with target values,
#    then aggregate by (source, year).
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract unique years
years <- sort(unique(dt$year))

compute_all_neighbor_features <- function(dt, edges, vars, years) {
  # Build a lookup table: (cell_idx, year) -> row index in dt
  dt[, row_id := .I]
  
  # For each year, we reuse the SAME edge list (topology is time-invariant)
  # Strategy: cross-join edges with years, then join variable values
  
  # Create edge-year table: each edge exists in every year
  # With ~1.37M edges Ã— 28 years = ~38.4M rows â€” fits in memory (~1-2 GB)
  
  cat("Expanding edges across years...\n")
  edge_year <- CJ(edge_row = seq_len(nrow(edges)), year = years)
  edge_year[, `:=`(
    src_idx = edges$src_idx[edge_row],
    tgt_idx = edges$tgt_idx[edge_row]
  )]
  edge_year[, edge_row := NULL]
  
  cat(sprintf("  Edge-year table: %d rows (%.1f M)\n", nrow(edge_year), nrow(edge_year)/1e6))
  
  # For memory efficiency, process one variable at a time
  for (var_name in vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)
    
    # Extract target values: (cell_idx, year) -> value
    val_dt <- dt[, .(cell_idx, year, val = get(var_name))]
    setkey(val_dt, cell_idx, year)
    
    # Join neighbor (target) values onto edge-year table
    setkey(edge_year, tgt_idx, year)
    edge_year[val_dt, neighbor_val := i.val, on = .(tgt_idx = cell_idx, year)]
    
    # Aggregate by (src_idx, year) â€” this is the neighborhood aggregation
    setkey(edge_year, src_idx, year)
    agg <- edge_year[!is.na(neighbor_val),
                     .(n_max  = max(neighbor_val),
                       n_min  = min(neighbor_val),
                       n_mean = mean(neighbor_val)),
                     by = .(src_idx, year)]
    
    setnames(agg, c("n_max", "n_min", "n_mean"), c(max_col, min_col, mean_col))
    
    # Join aggregated stats back to dt
    setkey(agg, src_idx, year)
    setkey(dt, cell_idx, year)
    
    dt[agg, (max_col)  := get(paste0("i.", max_col)),  on = .(cell_idx = src_idx, year)]
    dt[agg, (min_col)  := get(paste0("i.", min_col)),  on = .(cell_idx = src_idx, year)]
    dt[agg, (mean_col) := get(paste0("i.", mean_col)), on = .(cell_idx = src_idx, year)]
    
    # Clean up for next iteration
    edge_year[, neighbor_val := NULL]
    
    cat(sprintf("  Done: %s â€” added %s, %s, %s\n", var_name, max_col, min_col, mean_col))
  }
  
  dt[, row_id := NULL]
  return(dt)
}

dt <- compute_all_neighbor_features(dt, edges, neighbor_source_vars, years)

# ============================================================
# 4. Convert back and apply pre-trained Random Forest
# ============================================================

# If the RF model expects a data.frame (not data.table):
cell_data <- as.data.frame(dt)

# Drop helper columns not in the original predictor set
cell_data$cell_idx <- NULL

# Apply the pre-trained model (DO NOT retrain)
cat("Generating predictions from pre-trained Random Forest...\n")
predictions <- predict(rf_model, data = cell_data)  
# For ranger: predictions$predictions
# For randomForest: predictions (vector directly)

cat("Done.\n")
```

## Memory-Optimized Variant (If 38.4M-Row Edge-Year Table Is Too Large)

If the ~38.4M row `edge_year` table strains the 16 GB laptop, process year-by-year:

```r
compute_neighbor_features_by_year <- function(dt, edges, vars, years) {
  
  dt[, row_id := .I]
  
  # Pre-allocate result columns with NA
  for (var_name in vars) {
    dt[, paste0("n_max_",  var_name) := NA_real_]
    dt[, paste0("n_min_",  var_name) := NA_real_]
    dt[, paste0("n_mean_", var_name) := NA_real_]
  }
  
  for (yr in years) {
    cat(sprintf("Year %d...\n", yr))
    
    # Subset to this year
    dt_yr <- dt[year == yr, ]
    setkey(dt_yr, cell_idx)
    
    # Build value lookup for this year: cell_idx -> row in dt_yr
    val_lookup <- dt_yr[, .(cell_idx)]
    setkey(val_lookup, cell_idx)
    
    for (var_name in vars) {
      max_col  <- paste0("n_max_",  var_name)
      min_col  <- paste0("n_min_",  var_name)
      mean_col <- paste0("n_mean_", var_name)
      
      # Target values for this year
      tgt_vals <- dt_yr[, .(tgt_idx = cell_idx, val = get(var_name))]
      setkey(tgt_vals, tgt_idx)
      
      # Join target values onto edges
      edge_vals <- copy(edges)
      edge_vals[tgt_vals, neighbor_val := i.val, on = .(tgt_idx)]
      
      # Aggregate
      agg <- edge_vals[!is.na(neighbor_val),
                       .(n_max  = max(neighbor_val),
                         n_min  = min(neighbor_val),
                         n_mean = mean(neighbor_val)),
                       by = .(src_idx)]
      
      # Write back into dt for this year
      # Match on (cell_idx == src_idx) & (year == yr)
      setkey(agg, src_idx)
      idx_match <- dt[year == yr, which = TRUE]
      matched   <- dt[idx_match, .(cell_idx)]
      setkey(matched, cell_idx)
      
      # Efficient update via join
      dt[year == yr & cell_idx %in% agg$src_idx,
         (max_col) := agg[.(cell_idx), n_max, on = .(src_idx = cell_idx)]]
      dt[year == yr & cell_idx %in% agg$src_idx,
         (min_col) := agg[.(cell_idx), n_min, on = .(src_idx = cell_idx)]]
      dt[year == yr & cell_idx %in% agg$src_idx,
         (mean_col) := agg[.(cell_idx), n_mean, on = .(src_idx = cell_idx)]]
    }
  }
  
  dt[, row_id := NULL]
  return(dt)
}
```

However, the first (full expansion) approach is cleaner and faster if memory permits. Here is an even more streamlined year-loop that avoids repeated subsetting of `dt`:

```r
compute_neighbor_features_lean <- function(dt, edges, vars, years) {
  # Pre-allocate all output columns
  for (v in vars) {
    set(dt, j = paste0("n_max_",  v), value = NA_real_)
    set(dt, j = paste0("n_min_",  v), value = NA_real_)
    set(dt, j = paste0("n_mean_", v), value = NA_real_)
  }
  
  setkey(dt, year, cell_idx)
  
  for (yr in years) {
    cat(sprintf("  Year %d\n", yr))
    
    # Rows for this year
    yr_dt <- dt[.(yr)]  # keyed lookup, very fast
    
    # cell_idx -> values for all vars at once
    val_mat <- yr_dt[, c("cell_idx", vars), with = FALSE]
    setkey(val_mat, cell_idx)
    
    # Join all target values onto edges in one go
    edge_enriched <- val_mat[edges, on = .(cell_idx = tgt_idx), nomatch = NA, allow.cartesian = TRUE]
    # edge_enriched now has columns: cell_idx (=tgt), src_idx, and all vars
    
    # Aggregate by src_idx
    agg_exprs_max  <- lapply(vars, function(v) call("max",  as.name(v), na.rm = TRUE))
    agg_exprs_min  <- lapply(vars, function(v) call("min",  as.name(v), na.rm = TRUE))
    agg_exprs_mean <- lapply(vars, function(v) call("mean", as.name(v), na.rm = TRUE))
    
    all_exprs <- c(agg_exprs_max, agg_exprs_min, agg_exprs_mean)
    all_names <- c(paste0("n_max_", vars), paste0("n_min_", vars), paste0("n_mean_", vars))
    names(all_exprs) <- all_names
    
    agg <- edge_enriched[, lapply(all_exprs, eval, envir = .SD), by = .(src_idx)]
    
    # Handle -Inf/Inf from max/min on empty sets -> NA
    for (col_name in all_names) {
      set(agg, which(is.infinite(agg[[col_name]])), col_name, NA_real_)
    }
    
    # Update dt for this year
    setkey(agg, src_idx)
    yr_rows <- dt[.(yr), which = TRUE]
    yr_cell_idx <- dt$cell_idx[yr_rows]
    
    # Match order
    m <- match(yr_cell_idx, agg$src_idx)
    for (col_name in all_names) {
      set(dt, i = yr_rows, j = col_name, value = agg[[col_name]][m])
    }
  }
  
  return(dt)
}
```

## Final Recommended Solution (Cleanest & Fastest)

```r
library(data.table)
library(Matrix)

# -----------------------------------------------------------
# STEP 1: Build edge list from nb object (once, ~0.01 sec)
# -----------------------------------------------------------
n_cells <- length(rook_neighbors_unique)
src <- rep.int(seq_len(n_cells), lengths(rook_neighbors_unique))
tgt <- unlist(rook_neighbors_unique, use.names = FALSE)
keep <- tgt > 0L
edges <- data.table(src_idx = src[keep], tgt_idx = tgt[keep])
setkey(edges, tgt_idx)

# -----------------------------------------------------------
# STEP 2: Prepare data.table with cell index
# -----------------------------------------------------------
dt <- as.data.table(cell_data)
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
dt[, cell_idx := id_to_idx[as.character(id)]]
stopifnot(!anyNA(dt$cell_idx))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (v in neighbor_source_vars) {
  set(dt, j = paste0("n_max_",  v), value = NA_real_)
  set(dt, j = paste0("n_min_",  v), value = NA_real_)
  set(dt, j = paste0("n_mean_", v), value = NA_real_)
}

setkey(dt, year, cell_idx)
all_years <- sort(unique(dt$year))

# -----------------------------------------------------------
# STEP 3: Year-by-year neighborhood aggregation
# -----------------------------------------------------------
for (yr in all_years) {
  cat(sprintf("Year %d ... ", yr))
  t0 <- proc.time()
  
  yr_rows <- dt[.(yr), which = TRUE]
  yr_dt   <- dt[yr_rows, c("cell_idx", neighbor_source_vars), with = FALSE]
  setkey(yr_dt, cell_idx)
  
  # Join target-cell values onto edge list
  # Result: one row per (src_idx, tgt_idx) with target's variable values
  enriched <- yr_dt[edges, on = .(cell_idx = tgt_idx), nomatch = NA, allow.cartesian = TRUE]
  setnames(enriched, "cell_idx", "tgt_idx")
  # enriched has: tgt_idx, src_idx, ntl, ec, pop_density, def, usd_est_n2
  
  # Aggregate all 5 variables at once, grouped by src_idx
  agg <- enriched[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("n_max_",  v)]] <- NA_real_
        out[[paste0("n_min_",  v)]] <- NA_real_
        out[[paste0("n_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("n_max_",  v)]] <- max(vals)
        out[[paste0("n_min_",  v)]] <- min(vals)
        out[[paste0("n_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = .(src_idx), .SDcols = neighbor_source_vars]
  
  # Write results back into dt
  setkey(agg, src_idx)
  yr_cell_idx <- dt$cell_idx[yr_rows]
  m <- match(yr_cell_idx, agg$src_idx)
  
  out_cols <- c(paste0("n_max_",  neighbor_source_vars),
                paste0("n_min_",  neighbor_source_vars),
                paste0("n_mean_", neighbor_source_vars))
  
  for (col_name in out_cols) {
    set(dt, i = yr_rows, j = col_name, value = agg[[col_name]][m])
  }
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("%.1f sec\n", elapsed))
}

# -----------------------------------------------------------
# STEP 4: Apply pre-trained Random Forest (no retraining)
# -----------------------------------------------------------
dt[, cell_idx := NULL]
cell_data <- as.data.frame(dt)

# For ranger:
predictions <- predict(rf_model, data = cell_data)$predictions

# For randomForest:
# predictions <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | 6.46M string-key lookups | 1 `unlist` + 1 `data.table` (~0.1 sec) |
| **Neighbor aggregation** | 6.46M Ã— 5 R-level `lapply` iterations | 28 Ã— 1 `data.table` grouped join-aggregate |
| **Memory for topology** | ~3â€“5 GB (list of lists + string keys) | ~22 MB (2-column integer `data.table`) |
| **Total estimated time** | 86+ hours | **~5â€“15 minutes** |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` per (cell, year, variable) |
| **RF model** | Preserved | Preserved (predict only, no retraining) |

The key insight: **rook contiguity is a spatial property, not a spatiotemporal one.** By factoring the graph topology out of the time dimension, the adjacency structure shrinks from 6.46M entries to 344K entries, and `data.table`'s optimized grouped joins replace millions of R-level string operations.
END_RESPONSE>>>
