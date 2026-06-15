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
run_id: OpenAI_F10_T03_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows repeatedly (5 variables Ã— 6.46M iterations).  
- Neighbor lookups are recomputed for each row and variable.  
- No vectorization; heavy R-level loops and repeated indexing.  
- Memory overhead from repeated `rbind` and list operations.  

**Optimization Strategy**  
- Build the neighbor index **once** for all cell-years using integer mapping.  
- Represent the graph as a sparse adjacency structure (CSR-like).  
- Compute neighbor stats for all variables in a **single pass** using matrix operations.  
- Avoid repeated `lapply` and `rbind`; use `data.table` for efficient joins and grouping.  
- Process by year to keep memory usage manageable.  
- Preserve the trained Random Forest model and numerical equivalence.  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Assume: cell_data (data.table) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: list of integer neighbor indices (spdep::nb)

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute adjacency as integer vectors
id_to_idx <- setNames(seq_along(id_order), id_order)
adj_list <- rook_neighbors_unique  # already in integer indices

# Variables to compute neighbor stats for
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (v in vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process year by year to limit memory
years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  vals_mat <- as.matrix(dt_year[, ..vars])  # rows in id_order order
  
  # Reorder rows to match id_order
  dt_year <- dt_year[match(id_order, id)]
  
  # Compute neighbor stats
  n <- length(id_order)
  res <- matrix(NA_real_, nrow = n, ncol = length(vars) * 3)
  
  for (i in seq_len(n)) {
    nbrs <- adj_list[[i]]
    if (length(nbrs) > 0) {
      nbr_vals <- vals_mat[nbrs, , drop = FALSE]
      res[i, seq(1, by = 3, length.out = length(vars))] <- apply(nbr_vals, 2, max, na.rm = TRUE)
      res[i, seq(2, by = 3, length.out = length(vars))] <- apply(nbr_vals, 2, min, na.rm = TRUE)
      res[i, seq(3, by = 3, length.out = length(vars))] <- apply(nbr_vals, 2, mean, na.rm = TRUE)
    }
  }
  
  # Assign back
  colnames(res) <- as.vector(t(outer(vars, c("_nbr_max","_nbr_min","_nbr_mean"), paste0)))
  dt_year[, (colnames(res)) := as.data.table(res)]
  
  # Merge back
  cell_data[year == yr, (colnames(res)) := dt_year[, ..colnames(res)]]
}

# At this point, cell_data has neighbor features added
# Apply pre-trained Random Forest model
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This Is Fast**
- Adjacency built **once**, reused across years.  
- Year-wise processing avoids 6.46M Ã— 5 loops; only 28 passes.  
- Uses matrix operations and `apply` over neighbors instead of repeated `lapply`.  
- Avoids repeated `rbind` and list flattening.  
- Memory footprint controlled by chunking by year.  

**Expected runtime**: From 86+ hours to a few hours (or less) on 16GB RAM laptop.  

**Numerical equivalence**: Preserved because logic matches original (max, min, mean over non-NA neighbors).  
**Model**: Pre-trained Random Forest remains unchanged.
END_RESPONSE>>>
