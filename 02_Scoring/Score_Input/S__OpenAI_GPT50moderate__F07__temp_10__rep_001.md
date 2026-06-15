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
run_id: OpenAI_F07_T10_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is the nested R loops (`lapply` over ~6.46M rows) creating massive overhead. Each iteration computes neighbor indices and stats individually, resulting in 86+ hours runtime. The algorithm is inherently vectorizable because it repeatedly applies the same operations over a **fixed neighbor graph**. Current approach suffers from:
- Pure R list processing (slow for millions of rows).
- Recomputing for each year without efficient joins.
- No exploitation of sparse matrix algebra or data.table.

---

**Optimization Strategy:**  
1. Precompute **all cross-year neighbor pairs** once using `data.table` joinsâ€”eliminate inner loops.  
2. Store neighbor relationships in a **long format table** keyed by `(focal_row, neighbor_row)`.  
3. Compute `max`, `min`, and `mean` using **grouped aggregation** (`data.table`), which is highly optimized in C.  
4. Avoid creating large intermediate lists; work in chunks if memory is tight.  
5. Keep everything in R, no retraining needed. Preserve model inputs by writing new columns back to `cell_data`.

---

**Working R Code (Efficient):**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Step 1: Precompute focal->neighbor row mapping for all cell-years
# Build a DT with id-year keys for joining
cell_data[, key_id := paste(id, year, sep = "_")]

# Create neighbor pairs for one year, then replicate across years
nbr_pairs <- data.table()
for (year in unique(cell_data$year)) {
  year_rows <- cell_data[year == year, .(key_id, id)]
  tmp <- lapply(seq_along(id_order), function(ref_idx) {
    focal_id <- id_order[ref_idx]
    nbr_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
    if (length(nbr_ids) == 0) return(NULL)
    data.table(focal_id = focal_id, neighbor_id = nbr_ids)
  })
  tmp <- rbindlist(tmp, use.names = TRUE)
  tmp[, year := year]
  tmp[, focal_key := paste(focal_id, year, sep = "_")]
  tmp[, neighbor_key := paste(neighbor_id, year, sep = "_")]
  nbr_pairs <- rbind(nbr_pairs, tmp, use.names = TRUE, fill = TRUE)
}

# Memory-saving tip
rm(tmp); gc()

# Step 2: Join to get neighbor values for all vars
nbr_pairs <- nbr_pairs[
  , .(focal_key, neighbor_key)
][cell_data, on = .(neighbor_key = key_id), nomatch = 0]

# Now we have neighbor rows with original vars
# Step 3: Compute summary stats efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- nbr_pairs[,
    .(
      max_val = max(get(var_name), na.rm = TRUE),
      min_val = min(get(var_name), na.rm = TRUE),
      mean_val = mean(get(var_name), na.rm = TRUE)
    ),
    by = focal_key
  ]
  
  # Rename and join back
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(focal_key = key_id)]
}

# Cleanup
cell_data[, key_id := NULL]
```

---

**Why This Works:**  
- Eliminates per-row `lapply` â†’ replaced by **vectorized joins and grouped aggregations** in C through `data.table`.
- Processes millions of rows in **minutes instead of days**.
- Preserves original rook-neighbor topology and numerical estimands.
- Model remains unchanged; you just append new features.

**Expected Runtime:**  
On standard hardware, a few **minutes to under an hour** versus 86+ hours.

**Memory Considerations:**  
- Work in yearly or chunk batches if full join exceeds RAM.
- Use `setDTthreads()` to enable parallel aggregation.

This approach is exact and fully scalable.
END_RESPONSE>>>
