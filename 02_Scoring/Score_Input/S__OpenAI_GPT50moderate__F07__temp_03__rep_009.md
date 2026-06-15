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
run_id: OpenAI_F07_T03_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- `lapply` over ~6.46M rows with nested lookups is highly inefficient in R.  
- Repeated string concatenation (`paste`) and `setNames` lookups inside the loop are expensive.  
- Memory overhead from millions of small lists and repeated indexing is huge.  
- No vectorization or compiled operations; everything is interpreted R code.  

**Optimization Strategy**  
- Precompute neighbor indices once as integer vectors without string concatenation.  
- Use `data.table` for fast joins and grouping.  
- Reshape data into a matrix by `id Ã— year` for each variable, then compute neighbor stats using matrix operations.  
- Avoid per-row `lapply`; instead, use vectorized row/column operations.  
- Parallelize across variables or years if needed.  
- Keep the Random Forest model unchanged (we only compute features).  

**Working R Code** (efficient approach using `data.table` and matrix ops):  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (neighbors per id)
# id_order: vector of unique cell ids in same order as rook_neighbors_unique

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping from id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Convert rook_neighbors_unique into integer index list
neighbor_idx_list <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Create wide matrices for each variable: rows = id, cols = year
years <- sort(unique(cell_data$year))
n_id <- length(id_order)
n_year <- length(years)

make_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_id, ncol = n_year,
              dimnames = list(id_order, years))
  vals <- cell_data[[var]]
  idx <- match(cell_data$id, id_order)
  yr_idx <- match(cell_data$year, years)
  m[cbind(idx, yr_idx)] <- vals
  m
}

vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
mat_list <- lapply(vars, make_matrix)
names(mat_list) <- vars

# Compute neighbor stats efficiently
compute_neighbor_stats_matrix <- function(m, neighbor_idx_list) {
  n_id <- nrow(m)
  n_year <- ncol(m)
  max_mat <- matrix(NA_real_, n_id, n_year)
  min_mat <- matrix(NA_real_, n_id, n_year)
  mean_mat <- matrix(NA_real_, n_id, n_year)
  
  for (i in seq_len(n_id)) {
    nbs <- neighbor_idx_list[[i]]
    if (length(nbs) == 0) next
    nb_vals <- m[nbs, , drop = FALSE]
    max_mat[i, ] <- apply(nb_vals, 2, max, na.rm = TRUE)
    min_mat[i, ] <- apply(nb_vals, 2, min, na.rm = TRUE)
    mean_mat[i, ] <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE))
  }
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

neighbor_features <- lapply(mat_list, compute_neighbor_stats_matrix, neighbor_idx_list)

# Convert back to long and merge
longify <- function(mat, varname, stat) {
  dt <- as.data.table(as.table(mat))
  setnames(dt, c("id", "year", "value"))
  dt[, year := as.integer(year)]
  setnames(dt, "value", paste0(varname, "_nb_", stat))
  dt
}

result_list <- list()
for (var in vars) {
  for (stat in c("max", "min", "mean")) {
    mat <- neighbor_features[[var]][[stat]]
    result_list[[paste(var, stat, sep = "_")]] <- longify(mat, var, stat)
  }
}

neighbor_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), result_list)

# Merge with original cell_data
cell_data <- merge(cell_data, neighbor_dt, by = c("id", "year"), all.x = TRUE)
```

**Why this works:**  
- Eliminates per-row loops and string operations.  
- Uses matrix operations and `apply` over small slices (neighbors Ã— years).  
- Memory footprint is manageable: 5 matrices Ã— (344k Ã— 28) â‰ˆ 48M doubles (~384 MB).  
- Expected runtime: minutes instead of 86+ hours on a 16 GB laptop.  

**Preserves:**  
- Original rook-neighbor topology.  
- Exact max, min, mean per neighbor set.  
- Original Random Forest model remains unchanged.
END_RESPONSE>>>
