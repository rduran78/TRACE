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
run_id: Anthropic_F06_T07_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates 6.46 million R closures/list elements via `lapply`, and `compute_neighbor_stats` then iterates over that list 5 times (once per variable), each time doing per-row R-level loops.** This is ~32 million R-level iterations with repeated indexing, vector subsetting, and `is.na` checksâ€”all in interpreted R. The 86+ hour estimate is entirely explained by this.

Specific problems:

1. **`build_neighbor_lookup`**: For every one of 6.46M rows, it does string pasting, named-vector lookups (`id_to_ref`, `idx_lookup`), and `NA` filtering. Named vector lookup in R is O(n) hash probing per call, and doing it 6.46M Ã— (avg ~4 neighbors) times is devastating.

2. **`compute_neighbor_stats`**: An `lapply` over 6.46M elements calling `max`, `min`, `mean` individuallyâ€”each with NA handlingâ€”is slow. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

3. **The comment about raster focal/kernel operations** is a red flag hint: rook-neighbor aggregation on a regular grid *is* a focal operation with a cross-shaped (Von Neumann) kernel. But the data is a **panel** (cell Ã— year), the grid may have irregular boundaries/missing cells, and the neighbor structure is precomputed as an `spdep::nb` objectâ€”so a literal `terra::focal()` would require reshaping into a raster stack per year and careful handling of missing cells. It's a useful *analogy* but the best implementation uses **vectorized sparse-matrix multiplication**, which generalizes focal operations while preserving exact results for irregular grids.

## Optimization Strategy

**Replace the entire lookup + stats pipeline with a sparse neighbor matrix and vectorized column operations.**

1. **Build a sparse adjacency matrix `W`** (6.46M Ã— 6.46M) from the `nb` object and year-matching logicâ€”but crucially, build it once using vectorized operations (no per-row `lapply`).

2. **For each variable, compute neighbor max, min, mean** using the sparse matrix structure. Mean is trivial: `W %*% x / rowSums(W)`. Max and min require a grouped operation over the sparse entries, which can be done efficiently with `data.table` or with a custom approach using the sparse matrix's `i`, `j`, `x` slots.

3. This reduces 86 hours to **minutes** (sparse matrix construction ~1-2 min, each variable's stats ~30 sec).

**Key insight**: The neighbor relationships are *time-invariant* (rook neighbors don't change across years). So the 6.46M Ã— 6.46M sparse matrix has the same spatial pattern replicated across 28 year-blocks on the diagonal. We exploit this structure.

## Working R Code

```r
library(Matrix)
library(data.table)

# ============================================================
# STEP 1: Build the cell-year row index efficiently
# ============================================================
# cell_data must have columns: id, year
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer neighbor indices)

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  
  n_cells <- length(id_order)
  n_rows  <- nrow(cell_data)
  
  # --- Map each (id, year) to its row index in cell_data ---
  # Use data.table for speed
  dt <- data.table(
    id   = cell_data$id,
    year = cell_data$year,
    ridx = seq_len(n_rows)
  )
  setkey(dt, id, year)
  
  # --- Map cell id -> position in id_order ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Expand rook neighbor pairs (spatial, directed) ---
  # For each cell i, neighbors[[i]] gives positions in id_order
  # Build edge list: (from_pos, to_pos) in id_order space
  from_pos <- rep(seq_len(n_cells), lengths(rook_neighbors))
  to_pos   <- unlist(rook_neighbors)
  
  # Remove zero-length / empty neighbor entries (spdep uses 0 for no neighbors)
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  cat("Spatial edges:", length(from_pos), "\n")
  
  # Convert positions back to cell IDs
  from_id <- id_order[from_pos]
  to_id   <- id_order[to_pos]
  
  # --- Expand over all years ---
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  # Replicate edge list for each year
  edge_from_id <- rep(from_id, times = n_years)
  edge_to_id   <- rep(to_id,   times = n_years)
  edge_year    <- rep(years, each = length(from_id))
  
  cat("Total directed cell-year edges to resolve:", length(edge_from_id), "\n")
  
  # --- Look up row indices for (from_id, year) and (to_id, year) ---
  edges_dt <- data.table(
    from_id = edge_from_id,
    to_id   = edge_to_id,
    year    = edge_year
  )
  
  # Join to get "from" row index
  edges_dt[dt, on = .(from_id = id, year = year), from_ridx := i.ridx]
  # Join to get "to" row index
  edges_dt[dt, on = .(to_id = id, year = year), to_ridx := i.ridx]
  
  # Drop edges where either endpoint is missing (cell not in panel for that year)
  edges_dt <- edges_dt[!is.na(from_ridx) & !is.na(to_ridx)]
  
  cat("Valid cell-year edges:", nrow(edges_dt), "\n")
  
  # --- Build sparse matrix ---
  # W[i, j] = 1 means row j is a rook neighbor of row i (same year)
  W <- sparseMatrix(
    i    = edges_dt$from_ridx,
    j    = edges_dt$to_ridx,
    x    = 1,
    dims = c(n_rows, n_rows)
  )
  
  return(W)
}

# ============================================================
# STEP 2: Compute neighbor stats using sparse matrix
# ============================================================
compute_all_neighbor_stats_sparse <- function(cell_data, W, var_name) {
  
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- Handle NAs: set NA values to a sentinel so sparse ops work ---
  not_na <- !is.na(x)
  x_clean <- x
  x_clean[!not_na] <- 0  # will be masked out
  
  # Build a "valid neighbor" matrix: zero out columns where x is NA
  # W_valid[i,j] = W[i,j] * (x[j] is not NA)
  # Efficiently: multiply W by a diagonal matrix of not_na
  valid_diag <- Diagonal(x = as.numeric(not_na))
  W_valid <- W %*% valid_diag  # now W_valid[i,j] = 1 only if j is neighbor AND x[j] not NA
  
  # Count of valid neighbors per row
  n_valid <- as.numeric(W_valid %*% rep(1, n))
  
  # --- MEAN ---
  neighbor_sum  <- as.numeric(W_valid %*% x_clean)
  neighbor_mean <- ifelse(n_valid > 0, neighbor_sum / n_valid, NA_real_)
  
  # --- MAX and MIN ---
  # Extract the sparse triplet form of W_valid
  W_t <- as(W_valid, "TsparseMatrix")  # 0-indexed i, j
  row_idx <- W_t@i + 1L
  col_idx <- W_t@j + 1L
  
  # Get the actual x values for each edge
  edge_vals <- x_clean[col_idx]
  
  # Use data.table for grouped max/min (very fast)
  edge_dt <- data.table(row = row_idx, val = edge_vals)
  
  max_dt <- edge_dt[, .(nmax = max(val)), by = row]
  min_dt <- edge_dt[, .(nmin = min(val)), by = row]
  
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  neighbor_max[max_dt$row] <- max_dt$nmax
  neighbor_min[min_dt$row] <- min_dt$nmin
  
  # Where n_valid == 0, ensure NA
  no_valid <- n_valid == 0
  neighbor_max[no_valid] <- NA_real_
  neighbor_min[no_valid] <- NA_real_
  
  # Return named columns matching original convention
  prefix <- var_name
  result <- data.frame(
    nmax  = neighbor_max,
    nmin  = neighbor_min,
    nmean = neighbor_mean
  )
  names(result) <- paste0("n_", prefix, c("_max", "_min", "_mean"))
  
  return(result)
}

# ============================================================
# STEP 3: Main pipeline
# ============================================================
cat("Building sparse neighbor matrix...\n")
system.time({
  W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")
  stats_df <- compute_all_neighbor_stats_sparse(cell_data, W, var_name)
  cell_data <- cbind(cell_data, stats_df)
  cat("    Done.\n")
}

cat("All neighbor features computed. Ready for prediction.\n")

# ============================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# rf_model is the pre-trained model loaded from disk
# predictions <- predict(rf_model, newdata = cell_data)
```

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~40+ hours (6.46M `lapply` with string ops) | ~2-5 minutes (vectorized joins via `data.table`, one sparse matrix construction) |
| Compute stats (Ã—5 vars) | ~45+ hours (32M R-level iterations) | ~3-8 minutes (sparse matrix multiply + `data.table` grouped max/min) |
| **Total** | **~86+ hours** | **~5-15 minutes** |

## Why This Preserves Exact Results

- The sparse matrix `W` encodes **exactly** the same directed rook-neighbor relationships as the original `build_neighbor_lookup`, restricted to same-year pairs present in the data.
- `NA` handling is identical: neighbors with `NA` values are excluded before computing max, min, and mean.
- The `max`, `min`, and `mean` operations are mathematically identicalâ€”no approximation, no kernel smoothing, no raster resampling.
- The pre-trained Random Forest model is never modified; only the feature-engineering stage is accelerated.
END_RESPONSE>>>
