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
run_id: OpenAI_F10_T03_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is inefficient for large datasets.  
- Neighbor lookups are recomputed for each row and each variable, causing redundant work.  
- No vectorization or efficient data structures (e.g., matrices) are used.  
- Memory overhead from millions of small lists and repeated concatenations.  

**Optimization Strategy**  
- Build the neighbor index **once** as integer vectors referencing row positions.  
- Store data in a matrix for fast column access.  
- Compute all neighbor statistics in a **vectorized** manner using `rowsum` or matrix operations.  
- Avoid repeated `lapply`; instead, process variables in bulk.  
- Use `data.table` for efficient joins and updates.  
- Keep the Random Forest model unchanged and preserve numerical equivalence.  

---

### **Optimized R Implementation**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# Convert to data.table and sort
setDT(cell_data)
setkey(cell_data, id, year)

# Build neighbor lookup ONCE
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
neighbor_list <- lapply(rook_neighbors_unique, function(nbs) id_order[nbs])

# Flatten neighbor relationships for all years
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Map (id, year) -> row index
row_index <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))

# Build edge list for all years
edges <- rbindlist(lapply(seq_along(id_order), function(i) {
  src_id <- id_order[i]
  nbs <- neighbor_list[[i]]
  if (length(nbs) == 0) return(NULL)
  src_rep <- rep(src_id, length(nbs))
  data.table(src = src_rep, dst = nbs)
}))

# Expand edges for all years
edges_expanded <- edges[, .(src = rep(src, n_years),
                             dst = rep(dst, n_years),
                             year = rep(years, each = .N))]

# Map to row indices
edges_expanded[, src_row := row_index[paste(src, year, sep = "_")]]
edges_expanded[, dst_row := row_index[paste(dst, year, sep = "_")]]
edges_expanded <- edges_expanded[!is.na(src_row) & !is.na(dst_row)]

# Variables to process
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute neighbor stats efficiently
for (var in neighbor_vars) {
  vals <- cell_data[[var]]
  # Aggregate neighbor values by src_row
  agg <- edges_expanded[, .(
    max_val = max(vals[dst_row], na.rm = TRUE),
    min_val = min(vals[dst_row], na.rm = TRUE),
    mean_val = mean(vals[dst_row], na.rm = TRUE)
  ), by = src_row]
  
  # Merge back
  setnames(agg, c("src_row", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  cell_data[agg$src_row, c(paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")) := 
              agg[, -1]]
}

# cell_data now has neighbor features added
# Apply pre-trained Random Forest model without retraining
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This Is Fast**
- **Single pass edge expansion**: Build neighbor relationships once, reuse for all variables.  
- **Vectorized aggregation**: Use `data.table` grouping instead of millions of `lapply` calls.  
- **Memory efficiency**: Avoid nested lists and repeated concatenations.  
- **Preserves numerical equivalence**: Same max, min, mean per node-year as original.  

This approach should reduce runtime from **86+ hours to a few hours (or less)** on a 16 GB laptop.
END_RESPONSE>>>
