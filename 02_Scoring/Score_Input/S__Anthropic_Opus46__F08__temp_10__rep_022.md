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
run_id: Anthropic_F08_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) mappings.** It creates a list of 6.46 million entries, where each entry maps a cell-year row to its neighbor cell-year rows. But the neighbor *relationship* between cells never changes â€” only the *year* changes. This means the function is redundantly recomputing the same cellâ†’neighbor mapping 28 times (once per year), and doing expensive string-based key lookups (`paste` + named vector indexing) across 6.46 million rows.

2. **`compute_neighbor_stats` iterates over 6.46 million list entries.** Each call to `lapply` over the full neighbor lookup is O(n_rows), repeated for each of the 5 variables â€” totaling ~32.3 million list-element evaluations.

3. **String key construction and lookup is O(n) per call.** `paste(id, year)` and named-vector indexing are extremely slow at scale (6.46M entries).

**The key insight:** Since the neighbor graph is static across years, we should:
- Build the neighbor topology **once** at the cell level (344K cells, not 6.46M rows).
- Compute neighbor statistics **per year** using fast vectorized/matrix operations on the static topology.

## Optimization Strategy

1. **Separate static structure from dynamic data.** Build a cell-level neighbor index once (344K cells), not a row-level index (6.46M rows).

2. **Use a sparse adjacency matrix.** Convert the `nb` object to a sparse row-normalized (or raw) adjacency matrix using `spdep::nb2listw` â†’ `as(listw, "CsMatrix")` or construct it directly. Sparse matrixâ€“vector multiplication computes neighbor sums in milliseconds.

3. **Compute neighbor stats via sparse matrix operations per year.** For each year and each variable:
   - Extract the variable vector for that year (344K values).
   - Use the sparse adjacency matrix to compute neighbor sums, counts, max, and min.
   - Neighbor **mean** = sparse_matrix %*% values / neighbor_count.
   - Neighbor **max** and **min** require one pass through the neighbor list (but only 344K cells, not 6.46M).

4. **Vectorize across years.** Loop over 28 years (trivial), not 6.46M rows.

**Expected speedup:** From ~86 hours to **minutes** (roughly 2,000â€“5,000Ã—).

## Working R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 1: Build static cell-level neighbor structures (done ONCE)
# =============================================================================

build_static_neighbor_structures <- function(id_order, neighbors) {
  # id_order: vector of 344,208 cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  
  n_cells <- length(id_order)
  
  # --- Sparse adjacency matrix (for fast sum and mean) ---
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
      from <- c(from, rep(i, length(nb_i)))
      to   <- c(to, nb_i)
    }
  }
  
  adj_matrix <- sparseMatrix(
    i = from, j = to, x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Neighbor count per cell (static)
  neighbor_count <- as.integer(rowSums(adj_matrix))  # length n_cells
  
  # --- Neighbor list as integer vectors (for max/min) ---
  # Clean the nb list: ensure each element is an integer vector of valid indices
  neighbor_list <- lapply(seq_len(n_cells), function(i) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 1 && nb_i[1] == 0L) return(integer(0))
    as.integer(nb_i)
  })
  
  list(
    id_order       = id_order,
    adj_matrix     = adj_matrix,
    neighbor_count = neighbor_count,
    neighbor_list  = neighbor_list,
    n_cells        = n_cells
  )
}

# =============================================================================
# STEP 2: Compute neighbor max & min using the static neighbor list
#          (vectorized in C++ style via vapply, but only 344K cells per year)
# =============================================================================

compute_neighbor_max_min <- function(vals, neighbor_list) {
  # vals: numeric vector of length n_cells (one year's data for one variable)
  # neighbor_list: list of integer vectors (static)
  # Returns: list(max = numeric(n_cells), min = numeric(n_cells))
  
  n <- length(vals)
  out <- vapply(neighbor_list, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_))
    c(max(nv), min(nv))
  }, numeric(2))
  
  # out is 2 x n matrix
  list(max = out[1L, ], min = out[2L, ])
}

# =============================================================================
# STEP 3: Compute all neighbor stats for one variable, all years
# =============================================================================

compute_neighbor_features_fast <- function(dt, var_name, static_nb) {
  # dt: data.table with columns: id, year, <var_name>
  # static_nb: output of build_static_neighbor_structures
  # Returns: dt with three new columns appended
  
  adj        <- static_nb$adj_matrix
  nb_count   <- static_nb$neighbor_count
  nb_list    <- static_nb$neighbor_list
  id_order   <- static_nb$id_order
  n_cells    <- static_nb$n_cells
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Create a mapping from cell ID to position in id_order (static)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Process each year independently
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    cat(sprintf("  %s | year %d\n", var_name, yr))
    
    # Get row indices in dt for this year
    row_idx <- which(dt$year == yr)
    
    # Build cell-level value vector aligned to id_order
    # (some cells may be missing in a given year; handle with NA)
    cell_vals <- rep(NA_real_, n_cells)
    pos <- id_to_pos[as.character(dt$id[row_idx])]
    cell_vals[pos] <- dt[[var_name]][row_idx]
    
    # --- Neighbor MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for matrix multiply, track non-NA counts
    vals_zero   <- cell_vals
    vals_nonNA  <- as.numeric(!is.na(cell_vals))
    vals_zero[is.na(vals_zero)] <- 0
    
    neighbor_sum     <- as.numeric(adj %*% vals_zero)
    neighbor_nonNA   <- as.numeric(adj %*% vals_nonNA)
    
    neighbor_mean <- ifelse(neighbor_nonNA > 0,
                            neighbor_sum / neighbor_nonNA,
                            NA_real_)
    
    # --- Neighbor MAX and MIN via list traversal (344K cells) ---
    maxmin <- compute_neighbor_max_min(cell_vals, nb_list)
    
    # --- Write results back to dt rows for this year ---
    result_pos <- pos  # position in cell-level vectors
    set(dt, i = row_idx, j = col_max,  value = maxmin$max[result_pos])
    set(dt, i = row_idx, j = col_min,  value = maxmin$min[result_pos])
    set(dt, i = row_idx, j = col_mean, value = neighbor_mean[result_pos])
  }
  
  dt
}

# =============================================================================
# STEP 4: Main pipeline â€” drop-in replacement
# =============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building static neighbor structures (one-time)...\n")
  static_nb <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
  cat(sprintf("  Cells: %d | Neighbor edges: %d\n",
              static_nb$n_cells, sum(static_nb$neighbor_count)))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for: %s\n", var_name))
    cell_data <- compute_neighbor_features_fast(cell_data, var_name, static_nb)
  }
  
  cat("Done. Neighbor features added.\n")
  cell_data
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop):
# =============================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup scope** | 6.46M row-level entries | 344K cell-level (static, built once) |
| **Key lookup mechanism** | String paste + named vector | Integer positional indexing |
| **Mean computation** | R `lapply` over 6.46M elements | Sparse matrix multiply (344K Ã— 344K, ~1.37M nonzeros) per year |
| **Max/Min computation** | R `lapply` over 6.46M elements | `vapply` over 344K cells per year |
| **Total iterations** | 5 vars Ã— 6.46M = 32.3M | 5 vars Ã— 28 years Ã— 344K = 48.2M (but vectorized, no string ops) |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** |
| **Numerical results** | Original estimand | **Preserved identically** (same max, min, mean logic) |
| **RF model** | Pre-trained | **Untouched â€” no retraining** |
| **Memory** | 6.46M-element list of integer vectors | Sparse matrix (~16 MB) + 344K-element list | 

The sparse matrix multiplication for **mean** handles NA values correctly by computing neighbor sums and neighbor non-NA counts separately, then dividing. The **max** and **min** are computed via direct neighbor-list traversal at the cell level (344K iterations per year), which is 28Ã— fewer iterations than the original per-row approach and avoids all string operations.
END_RESPONSE>>>
