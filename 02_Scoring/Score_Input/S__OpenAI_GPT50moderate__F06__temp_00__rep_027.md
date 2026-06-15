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
run_id: OpenAI_F06_T00_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F06_OpenAI_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly performing vector lookups and allocations.  
- Neighbor stats are computed row-by-row instead of in a vectorized or batched manner.  
- No parallelization or efficient data structures (e.g., matrix indexing) are used.  
- Memory overhead from millions of small lists and repeated `paste` operations.  

**Optimization Strategy**  
- Precompute neighbor indices once (already done) and store as integer vectors.  
- Reshape data into a matrix by `id` Ã— `year` for each variable, enabling fast matrix operations.  
- Compute neighbor stats using vectorized operations or `apply` over columns (years) and rows (cells).  
- Use `data.table` for efficient joins and updates.  
- Optionally parallelize across variables or years.  
- Avoid recomputing strings; use integer indexing.  

**Working R Code** (vectorized approach using `data.table` and matrices):  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

ids   <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))
n_id  <- length(ids)
n_yr  <- length(years)

# Build wide matrices for each variable
make_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_id, ncol = n_yr,
              dimnames = list(ids, years))
  idx <- match(cell_data$id, ids)
  jdx <- match(cell_data$year, years)
  m[cbind(idx, jdx)] <- cell_data[[var]]
  m
}

vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
mat_list <- lapply(vars, make_matrix)
names(mat_list) <- vars

# neighbor list as integer indices (1-based)
neighbor_list <- rook_neighbors_unique  # list of integer vectors per id index

# Function to compute neighbor stats for one variable
compute_stats_matrix <- function(mat) {
  max_mat <- min_mat <- mean_mat <- matrix(NA_real_, nrow = n_id, ncol = n_yr)
  for (i in seq_len(n_id)) {
    nb <- neighbor_list[[i]]
    if (length(nb) == 0) next
    nb_vals <- mat[nb, , drop = FALSE]
    max_mat[i, ]  <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
    min_mat[i, ]  <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    mean_mat[i, ] <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
  }
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# Compute for all variables
stats_list <- lapply(mat_list, compute_stats_matrix)

# Melt back to long and merge
result_dt <- CJ(id = ids, year = years)
for (v in vars) {
  s <- stats_list[[v]]
  for (stat in c("max", "min", "mean")) {
    colname <- paste0(v, "_nb_", stat)
    result_dt[[colname]] <- as.vector(s[[stat]])
  }
}

# Merge with original data
cell_data <- merge(cell_data, result_dt, by = c("id", "year"), all.x = TRUE)
```

**Why this is faster:**  
- Converts row-wise operations into matrix operations.  
- Eliminates repeated string concatenation and lookups.  
- Reduces overhead from millions of small list operations.  
- Complexity drops from O(n_rows Ã— avg_neighbors) in R loops to efficient matrix slicing.  

**Expected performance:**  
- From 86+ hours to a few hours or less on a standard laptop.  
- Further speedup possible with `parallel::mclapply` on `compute_stats_matrix` or using `Rcpp`.  

**Preserves:**  
- Original estimand (max, min, mean of rook neighbors per cell-year).  
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
