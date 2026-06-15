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
run_id: Anthropic_F10_T03_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`paste` + `idx_lookup[neighbor_keys]`). This is O(NÃ—K) with enormous constant factors from R's string hashing on millions of keys.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in an `lapply`**, calling `max`, `min`, `mean` individually per node. R-level loops over millions of elements are inherently slow.

3. **The neighbor lookup is year-coupled but the graph topology is year-invariant.** The rook adjacency is purely spatial (344,208 cells), yet the lookup is rebuilt across all 6.46M cell-year rows, redundantly replicating the same spatial structure 28 times.

**Key insight:** The adjacency graph is static across years. The topology needs to be built once over 344K cells, and then neighbor aggregation can be performed **per-year** using sparse matrixâ€“vector multiplication, which is O(nnz) and runs in optimized C/Fortran code via the `Matrix` package.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 Ã— 344,208, ~1.37M nonzeros). This is tiny in memory (~16 MB).

2. **For each year, extract the variable vector for all cells, then compute neighbor stats via sparse matrix operations:**
   - **Mean:** `A %*% x / A %*% 1` (sum of neighbor values / count of neighbors, handling NAs).
   - **Max / Min:** Use a single pass over the CSC/CSR structure of the sparse matrix â€” far faster than millions of R-level list lookups.

3. **Vectorize NA handling** using sparse matrix arithmetic: replace NAs with 0 for summation, track valid counts with a separate sparse multiply.

4. **Process year-by-year** to keep memory bounded (~344K vectors, trivially small).

**Expected speedup:** From 86+ hours to **~5â€“15 minutes** on the same laptop. The sparse matrix multiply is O(nnz) per variable-year (~1.37M operations), and we have 5 variables Ã— 28 years = 140 such operations for each of max/min/mean.

## Working R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 1: Build sparse adjacency matrix from spdep nb object (done ONCE)
# =============================================================================
build_sparse_adjacency <- function(nb_obj) {
  # nb_obj is a list of length N; nb_obj[[i]] gives integer indices of neighbors of cell i
  # (0L means no neighbors in spdep convention)
  n <- length(nb_obj)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # spdep uses 0L for "no neighbors" â€” remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Row i, Col j = 1 means j is a neighbor of i (i.e., we aggregate j's value for node i)
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

# =============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable across all years
#          using sparse matrix operations â€” numerically equivalent to original
# =============================================================================
compute_neighbor_features_sparse <- function(dt, var_name, A, id_to_row) {
  # Pre-allocate output columns
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  n_cells <- nrow(A)
  years   <- sort(unique(dt$year))
  
  # CSC structure for fast column-wise access (for max/min)
  # A is n x n: A[i,j]=1 means j is neighbor of i
  # For row-wise iteration (neighbors of i), we use the transpose in CSC = original in CSR
  At <- t(A)  # Now At in CSC format: column i of At = row i of A = neighbors of i
  p  <- At@p  # column pointers (0-indexed)
  j_idx <- At@i  # row indices (0-indexed) = neighbor cell indices for each node
  
  for (yr in years) {
    # Get row indices in dt for this year
    yr_rows <- which(dt$year == yr)
    
    # Map cell ids to spatial row index
    cell_ids <- dt$id[yr_rows]
    spatial_idx <- id_to_row[as.character(cell_ids)]
    
    # Build full-length vector aligned to spatial grid (NA for missing cells)
    x_full <- rep(NA_real_, n_cells)
    x_full[spatial_idx] <- dt[[var_name]][yr_rows]
    
    # --- MEAN via sparse matrix multiply ---
    # Handle NAs: replace with 0 for sum, track counts
    x_nona <- x_full
    x_nona[is.na(x_nona)] <- 0
    valid_mask <- as.double(!is.na(x_full))
    
    neighbor_sum   <- as.numeric(A %*% x_nona)
    neighbor_count <- as.numeric(A %*% valid_mask)
    
    n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN via CSC traversal (vectorized in C-level sparse structure) ---
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
    
    # Process each cell using the sparse structure
    # At column i contains the neighbor indices of cell i
    # We do this in vectorized chunks
    
    # Expand all neighbor pairs
    # For each cell i (0-indexed in p), neighbors are j_idx[(p[i]+1):p[i+1]]
    # We vectorize this:
    
    # Number of neighbors per cell
    n_neighbors <- diff(p)  # length n_cells
    
    # Cell index for each entry in j_idx
    cell_rep <- rep(seq_len(n_cells), times = n_neighbors)
    neighbor_spatial <- j_idx + 1L  # convert to 1-indexed
    
    # Get neighbor values
    neighbor_vals <- x_full[neighbor_spatial]
    
    # Use data.table for fast grouped max/min/mean
    if (length(cell_rep) > 0) {
      agg_dt <- data.table(cell = cell_rep, val = neighbor_vals)
      agg_dt <- agg_dt[!is.na(val)]
      
      if (nrow(agg_dt) > 0) {
        stats <- agg_dt[, .(vmax = max(val), vmin = min(val)), by = cell]
        n_max[stats$cell] <- stats$vmax
        n_min[stats$cell] <- stats$vmin
      }
    }
    
    # Write back to dt for this year's rows
    set(dt, i = yr_rows, j = col_max,  value = n_max[spatial_idx])
    set(dt, i = yr_rows, j = col_min,  value = n_min[spatial_idx])
    set(dt, i = yr_rows, j = col_mean, value = n_mean[spatial_idx])
  }
  
  dt
}

# =============================================================================
# STEP 3: Main pipeline
# =============================================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  
  # Convert to data.table for performance (non-destructive if already data.table)
  dt <- as.data.table(cell_data)
  
  # --- Build spatial index mapping: cell id -> row in adjacency matrix ---
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Build sparse adjacency matrix ONCE ---
  message("Building sparse adjacency matrix...")
  A <- build_sparse_adjacency(rook_neighbors_unique)
  message(sprintf("  Adjacency matrix: %d x %d, %d nonzeros", 
                  nrow(A), ncol(A), nnzero(A)))
  
  # --- Compute neighbor features for each source variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    dt <- compute_neighbor_features_sparse(dt, var_name, A, id_to_row)
  }
  
  # --- Predict using the pre-trained Random Forest (no retraining) ---
  message("Generating predictions with pre-trained Random Forest...")
  predictions <- predict(rf_model, newdata = dt)
  dt[, prediction := predictions]
  
  message("Done.")
  dt
}

# =============================================================================
# Usage (assuming objects are already loaded):
# =============================================================================
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized |
|-----------|----------|-----------|
| **Max** | `max(vals[idx])` after removing NAs | `data.table` grouped `max` over the same neighbor values, same NA exclusion |
| **Min** | `min(vals[idx])` after removing NAs | `data.table` grouped `min` over the same neighbor values, same NA exclusion |
| **Mean** | `mean(vals[idx])` after removing NAs | `sum(valid_neighbor_vals) / count(valid_neighbor_vals)` via sparse multiply â€” algebraically identical |
| **No neighbors / all-NA neighbors** | Returns `c(NA, NA, NA)` | Returns `NA` for all three â€” identical |

The sparse matrix `A` encodes exactly the same directed rook-neighbor relationships as `rook_neighbors_unique`. The per-year slicing ensures each cell-year sees only same-year neighbor values, exactly as the original `paste(id, year)` key matching does.

## Performance Estimate

| Component | Original | Optimized |
|-----------|----------|-----------|
| Neighbor lookup construction | ~hours (6.46M string keys) | ~2 seconds (sparse matrix from integer lists) |
| Per-variable aggregation | ~17 hours (6.46M `lapply` iterations) | ~30 seconds (28 sparse multiplies + grouped `data.table` ops) |
| Total (5 variables) | **86+ hours** | **~5â€“10 minutes** |
| Peak RAM | Potentially >16 GB (list of 6.46M vectors + string keys) | ~2â€“3 GB (sparse matrix is ~16 MB; data.table in-place) |
END_RESPONSE>>>
