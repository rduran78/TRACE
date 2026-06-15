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
run_id: OpenAI_F06_T10_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is slow because for each of **6.46 million rows**, you iterate through a list and compute neighbor stats on the fly (`lapply` inside `compute_neighbor_stats`). This generates huge R loops and repeated indexing operations, which are very inefficient in R at this scale. Additionally, neighbor lookups are performed per row, even though neighbor relationships are static over time. Memory is also under pressure on a 16â€¯GB RAM laptop.

**Optimization Strategy**  
1. **Move from row-wise `lapply` to vectorized or data.table-based grouped operations**. Avoid millions of R function calls.  
2. **Precompute neighbor pairs as a long table** and join on years. Compute max, min, mean using `data.table` aggregations rather than millions of inner loops.  
3. **Process in chunks if memory-bound**, but minimize nested loops.  
4. Keep the Random Forest prediction step unchanged.

---

### **Optimized Approach**
- Expand neighbor relationships to a static pair list: `(source_id, neighbor_id)`.
- Use `data.table` to join this with panel data for each year and compute statistics in bulk.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# neighbor list -> long table of directed pairs
# rook_neighbors_unique: list where each element i has vector of neighbor indices
src <- rep(id_order, lengths(rook_neighbors_unique))
nbr <- unlist(rook_neighbors_unique, use.names = FALSE)
neighbor_pairs <- data.table(src_id = id_order[src], nbr_id = id_order[nbr])

# Repeat for all years via join: create expanded panel keys
# Create key for fast year matching
setkey(cell_data, id, year)

# Prepare result container
for (var_name in neighbor_source_vars) {
  
  # Extract only id, year, and var for neighbors
  vals_dt <- cell_data[, .(id, year, val = get(var_name))]
  
  # Join neighbors: replicate ids across years of cell_data (source_id + year)
  tmp <- merge(
    x = cell_data[, .(src_id = id, year)], 
    y = neighbor_pairs, 
    by = "src_id", 
    allow.cartesian = TRUE
  )
  
  # Join neighbor values by (nbr_id, year)
  tmp <- merge(
    tmp,
    vals_dt,
    by.x = c("nbr_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE
  )
  
  # Compute group-wise stats: by source_id-year
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(src_id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("src_id", "year"), c("id", "year"))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
  
  # Rename columns appropriately
  new_names <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  setnames(cell_data, c("max_val", "min_val", "mean_val"), new_names)
}

# At this point, cell_data has the new neighbor features.
# Run the pre-trained Random Forest
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

### **Why This Is Faster**
- Uses **bulk joins and group operations** in `data.table`, which are implemented in C.
- Eliminates millions of R function calls.
- Scales to millions of rows within reasonable time (hours â†’ minutes).
- Preserves original estimates and trained RF model.

**Expected runtime reduction:** 86+ hours â†’ ~30â€“60 minutes on a 16 GB machine (depending on join complexity and I/O).
END_RESPONSE>>>
