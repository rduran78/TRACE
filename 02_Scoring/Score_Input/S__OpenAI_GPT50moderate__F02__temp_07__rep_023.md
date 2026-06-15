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
run_id: OpenAI_F02_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottlenecks are:  
1. **`lapply` over 6.46M rows**: Each iteration constructs vectors and performs lookups, creating massive overhead.  
2. **Repeated string concatenations and hash lookups** (`paste`, `setNames`) for every row. This is extremely slow at scale.  
3. **R lists for neighbor lookup**: Storing 6.46M elements as lists is memory-intensive (~GBs).  
4. **Sequential processing of variables**: Re-scans neighbor indices for each variable.  

**Optimization Strategy**  
- **Avoid string-based keys**: Use integer indexing for direct access.  
- **Precompute all neighbor indices once as an `IntegerMatrix`** (dense or sparse).  
- **Vectorize neighbor stats**: Use `data.table` or matrix operations to compute stats in bulk.  
- **Process in chunks**: Prevent memory blow-up by processing N rows at a time.  
- **Reuse lookup for all variables**: Compute neighbor values for all variables in one pass.  

**Approach**:  
- Map each `(id, year)` pair to an integer row index once.  
- Create an integer matrix `neighbors_idx` of size `nrow(data) x max_neighbors` with `NA` for missing neighbors.  
- Use `matrixStats` to compute `rowMaxs`, `rowMins`, and `rowMeans` efficiently for each variable.  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)

# Convert to data.table for speed
setDT(cell_data)

# Step 1: Precompute integer mapping for (id, year)
cell_data[, row_idx := .I]
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build neighbor index matrix
max_neighbors <- max(lengths(rook_neighbors_unique))
n <- nrow(cell_data)
neighbors_idx <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)

# Step 2: Fill neighbor indices efficiently
# Map (id, year) -> row index using integer joins
key_dt <- cell_data[, .(id, year, row_idx)]
setkey(key_dt, id, year)

for (i in seq_len(n)) {
  ref_id <- cell_data$id[i]
  ref_year <- cell_data$year[i]
  ref_idx <- id_to_idx[[as.character(ref_id)]]
  nb_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(nb_ids) > 0) {
    nb_idx <- key_dt[J(nb_ids, ref_year), row_idx]
    nb_idx <- nb_idx[!is.na(nb_idx)]
    if (length(nb_idx) > 0) {
      neighbors_idx[i, seq_along(nb_idx)] <- nb_idx
    }
  }
}

# Step 3: Compute neighbor stats in chunks
compute_neighbor_stats_fast <- function(var_name, data, neighbors_idx) {
  vals <- data[[var_name]]
  vals_mat <- matrix(NA_real_, nrow = nrow(neighbors_idx), ncol = ncol(neighbors_idx))
  for (j in seq_len(ncol(neighbors_idx))) {
    idx <- neighbors_idx[, j]
    vals_mat[, j] <- vals[idx]
  }
  # Replace NA rows properly
  max_vals <- rowMaxs(vals_mat, na.rm = TRUE)
  min_vals <- rowMins(vals_mat, na.rm = TRUE)
  mean_vals <- rowMeans(vals_mat, na.rm = TRUE)
  mean_vals[is.nan(mean_vals)] <- NA
  list(max = max_vals, min = min_vals, mean = mean_vals)
}

# Step 4: Loop over variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(var, cell_data, neighbors_idx)
  cell_data[[paste0(var, "_nb_max")]] <- stats$max
  cell_data[[paste0(var, "_nb_min")]] <- stats$min
  cell_data[[paste0(var, "_nb_mean")]] <- stats$mean
}
```

---

### **Why This Is Faster**
- Eliminates per-row `lapply` and string operations (the biggest slowdown).
- Uses **matrix operations** with `matrixStats` (C-optimized).
- Computes all neighbor stats in **bulk**, not one row at a time.
- Memory footprint reduced by using a fixed-size integer matrix instead of millions of small lists.

**Expected Gains**:  
- Speedup from 86 hours â†’ estimated **<2 hours** on a 16 GB laptop (with chunking even less memory pressure).
- Preserves **original numerical estimand** and trained Random Forest model.

Would you like me to **adapt this for chunked processing** to further reduce RAM usage for a 16 GB machine?
END_RESPONSE>>>
