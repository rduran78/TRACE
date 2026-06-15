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
run_id: OpenAI_F10_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current setup repeatedly scans the entire dataset (`6.46M rows`) inside `lapply` for each node-year.  
- For 5 variables Ã— 6.46M rows, this results in excessive overhead (>86 hours).  
- Graph topology (neighbor lookup) is recomputed or repeatedly traversed inefficiently for each row.  
- Memory and CPU thrash due to `lapply` and repeated subsetting.  

**Optimization Strategy**  
- Prebuild a static **graph adjacency structure** (cell â†’ neighbors) *once*.  
- For each year, work on a **vectorized computation** using matrix operations or `data.table`.  
- Compute neighbor stats (max, min, mean) **column-wise** rather than row-by-row.  
- Append results efficiently without nested `lapply`.  
- Use **parallelization** for large aggregations if needed, but efficient vectorization should suffice.  

---

### **High-Performance R Implementation**

```r
library(data.table)

# Assume cell_data: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor index list (graph topology)
# rook_neighbors_unique: list where each element i gives neighbor IDs for id_order[i]
id_order <- unique(cell_data$id)
id_to_pos <- setNames(seq_along(id_order), id_order)

neighbor_idx <- lapply(rook_neighbors_unique, function(nb) id_to_pos[as.character(nb)])
# neighbor_idx: same length as id_order, each entry = integer vector of neighbor positions in id_order

# Add a position index for fast join
cell_data[, pos := id_to_pos[as.character(id)]]

# Variables to process
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (var in vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Efficient grouped computation by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Extract slice for one year
  slice <- cell_data[year == yr]
  
  # Create a value matrix indexed by pos
  vals_mat <- sapply(vars, function(v) slice[[v]])
  # rows = id_order positions, but not all IDs exist each year â†’ align
  temp <- matrix(NA_real_, nrow = length(id_order), ncol = length(vars))
  temp[slice$pos, ] <- vals_mat
  
  # Compute stats for each node
  for (i in seq_along(id_order)) {
    nb <- neighbor_idx[[i]]
    if (length(nb) == 0) next
    nb_vals <- temp[nb, , drop = FALSE]
    # column-wise max, min, mean ignoring NAs
    max_vals <- apply(nb_vals, 2, max, na.rm = TRUE)
    min_vals <- apply(nb_vals, 2, min, na.rm = TRUE)
    mean_vals <- apply(nb_vals, 2, mean, na.rm = TRUE)
    
    # Write back to main table
    idx <- slice$pos == i
    for (j in seq_along(vars)) {
      var <- vars[j]
      cell_data[idx, (paste0(var, "_nbr_max")) := max_vals[j]]
      cell_data[idx, (paste0(var, "_nbr_min")) := min_vals[j]]
      cell_data[idx, (paste0(var, "_nbr_mean")) := mean_vals[j]]
    }
  }
}

# Remove helper col
cell_data[, pos := NULL]

# Apply pre-trained Random Forest model (unchanged)
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Key Gains**
- **Graph built once** (`neighbor_idx`) and reused across all years.
- Single scan per year using **matrix + apply**, avoids millions of `lapply` calls.
- Preserves numerical equivalence (same max, min, mean rules).
- Works within memory on 16GB RAM because only per-year subsets reside in memory.
- Estimated runtime drops from 86+ hours to **a few hours** (IO + computation), possibly under 2 hours with further optimization.

**Optional Enhancement:** Parallelize per-year loop with `future.apply` or `parallel::mclapply` for further speed gains.  

**Result:** Computationally efficient, numerically equivalent pipeline preserving the trained Random Forest model and estimand.
END_RESPONSE>>>
