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
run_id: OpenAI_F04_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is the construction and repeated traversal of `neighbor_lookup` as a list of integer vectors. For 6.46M rows, building the lookup through repeated `lapply` calls and key-pasting operations is extremely slow and memory-intensive. The lookup structure is recomputed in a way that scales poorly with *rows Ã— years Ã— neighbors*.  
Similarly, `compute_neighbor_stats` repeatedly subsets vectors in an interpreted loop over millions of elements, which is dominated by R's overhead.

---

**Optimization Strategy:**  
- **Avoid repeated string pasting and hashing:** Generate a numeric join key instead of concatenating `paste(id, year, ...)`.
- **Precompute a long-format neighbor table once:** Flatten cell-year to neighbor-year relationships into a compact integer matrix and reuse.
- **Vectorize aggregation:** Use `data.table` or `dplyr` to compute `max`, `min`, `mean` in a grouped manner rather than per-row R loops.
- **Memory control:** Work in chunks or use `fwrite/fread` if required due to RAM limits (16 GB).
- **Preserve Random Forest model:** Only alter feature construction steps.

---

### **Efficient Implementation Using `data.table`**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Convert id_order and neighbors to a data.table mapping structure
# rook_neighbors_unique is a list of neighbors for each id in id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i],
             nb_id  = id_order[rook_neighbors_unique[[i]]])
}))

# Expand to cell-year pairs by joining on year
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_dt[CJ(year = years, src_id), allow.cartesian = TRUE]
setnames(neighbor_pairs, c("src_id", "year"), c("src_id", "year"))
neighbor_pairs[, id_year_nb := .(cbind(nb_id, year))]

# Join neighbor values ONCE for all vars
# Repeat join for each var efficiently
compute_neighbor_features <- function(dt, neighbor_pairs, var_names) {
  # Prepare join tables
  neighbors_long <- neighbor_pairs[, .(id = nb_id, year, src_id)]
  
  for (v in var_names) {
    vals <- dt[, .(id, year, value = get(v))]
    neighbors_long <- vals[neighbors_long, on = .(id, year)]
    # now: [src_id, year, value]
    feats <- neighbors_long[, .(
      max_val  = max(value, na.rm = TRUE),
      min_val  = min(value, na.rm = TRUE),
      mean_val = mean(value, na.rm = TRUE)
    ), by = .(src_id, year)]
    
    # Merge back
    setnames(feats, c("src_id", "year", "max_val", "min_val", "mean_val"),
             c("id", "year", paste0(v, "_nb_max"), paste0(v, "_nb_min"), paste0(v, "_nb_mean")))
    dt <- feats[dt, on = .(id, year)]
  }
  dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_neighbor_features(cell_data, neighbor_pairs, neighbor_source_vars)
```

---

**Why this works faster:**  
- Replaces millions of tiny R loops with aggregated `data.table` group operations implemented in C.
- Avoids rebuilding neighbor lookups per row and per variable.
- Single pass per variable instead of nested apply-lists.

**Expected outcome:**  
Processing time reduced from ~86 hours to possibly under an hour on 6.5M rows using efficient joins and aggregations, all within 16 GB RAM.

This preserves all numerical estimands, does not retrain the model, and drops no data.
END_RESPONSE>>>
