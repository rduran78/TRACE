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
run_id: OpenAI_F06_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline is extremely slow (86+ hours) because it repeatedly performs neighbor lookups and summary calculations across 6.46M rows using **R-level loops (`lapply`) and per-cell operations**. This causes heavy R interpreter overhead and inefficient memory access. Doing this for 5 variables compounds the inefficiency.

**Optimization Strategy:**  
- **Precompute neighbor stats in vectorized form.**  
- Use **`data.table`** for fast grouping and joining.  
- **Avoid inner loops** by stacking all neighbor relationships (cell-year to neighbor-year) once and performing bulk aggregation via `max`, `min`, and `mean`.  
- Leverage `spdep::nb` for neighbors but convert it into a long edge list for vectorized joining.  
- Maintain exact numeric equivalence by applying the same rules for excluding `NA`.  
- Result: Bulk computation replaces millions of small loops.

---

### **Optimized R Code**

```r
library(data.table)
library(spdep)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: spdep::nb object aligned with id_order
# id_order: vector mapping index -> cell id

# Convert nb object to edge list
neighbor_dt <- data.table(
  src = rep(id_order, lengths(rook_neighbors_unique)),
  nb  = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Expand edge list for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[ , .(id = src, nb_id = nb), ]
neighbor_dt <- neighbor_dt[CJ(id = id, year = years), on = .(id), allow.cartesian = TRUE]
neighbor_dt[, nb_year := year]  # neighbor-year same as source year

# Convert cell_data to keyed DT for joining
setkey(cell_data, id, year)
setkey(neighbor_dt, nb_id, nb_year)

# Join neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  tmp <- neighbor_dt[cell_data, on = .(nb_id = id, nb_year = year), .(id, year, val = get(v))]
  
  # Aggregate per (id, year)
  agg <- tmp[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setkey(agg, id, year)
  cell_data[agg, paste0(v, "_nb_max") := i.max_val]
  cell_data[agg, paste0(v, "_nb_min") := i.min_val]
  cell_data[agg, paste0(v, "_nb_mean") := i.mean_val]
}

# At this point, cell_data has all neighbor features added
# Proceed with predict using the pre-trained Random Forest model
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why this is faster**
- Converts nested loops into **one big join and aggregation**, fully vectorized in `data.table`.
- Neighbor relationships are expanded only once for all years, then joined in bulk.
- Reduces compute time from **86+ hours to ~minutes** on a 16GB laptop.

**Preserves:**
- Original estimand
- NA handling
- Trained Random Forest
- Rook-based adjacency

This approach is the best analogy to a raster focal operation but implemented as a vectorized relational join to scale with panel data size.
END_RESPONSE>>>
