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
run_id: OpenAI_F12_T03_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottlenecks is incorrect. While `rbind` on large lists is non-trivial, the deeper bottleneck lies in the **outer loop repeatedly scanning the entire dataset for each variable** and the **inefficient per-row neighbor lookup**. Specifically:  
- `build_neighbor_lookup()` creates a large list of integer vectors for ~6.46M rows, which is memory-heavy but computed only once.  
- `compute_neighbor_stats()` iterates over all rows for each variable (5 times), performing repeated indexing and filtering (`neighbor_vals <- vals[idx]; neighbor_vals[!is.na(...)]`).  
- This results in ~32M neighbor computations across 6.46M rows Ã— 5 variables, each involving R-level loops and allocations.  

The true bottleneck is **interpreted R loops and repeated NA filtering in `compute_neighbor_stats()`**, not `rbind`.  

---

### **Optimization Strategy**
- **Vectorize neighbor computations**: Precompute a sparse adjacency structure and use matrix operations or `data.table` joins instead of per-row loops.
- **Compute all neighbor stats in one pass**: Avoid looping over variables; reshape data to long format and aggregate.
- Use efficient packages (`data.table` or `Matrix`) for grouping and aggregation.
- Preserve the trained Random Forest model and estimands by ensuring identical numerical results.

---

### **Working R Code (Optimized)**
```r
library(data.table)

# Convert data to data.table for speed
dt <- as.data.table(cell_data)

# Precompute neighbor pairs (cell-year relationships)
# Flatten neighbor_lookup into long format
neighbor_pairs <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(
      src = i,
      nbr = neighbor_lookup[[i]]
    )
  })
)

# Map src and nbr to actual cell-year IDs
neighbor_pairs[, src_id := dt$id[src]]
neighbor_pairs[, year := dt$year[src]]
neighbor_pairs[, nbr_id := dt$id[nbr]]

# Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- melt(dt, id.vars = c("id", "year"), measure.vars = vars,
                variable.name = "var_name", value.name = "value")

# Join neighbor values
neighbor_pairs_long <- merge(
  neighbor_pairs,
  long_dt,
  by.x = c("nbr_id", "year"),
  by.y = c("id", "year"),
  allow.cartesian = TRUE
)

# Aggregate neighbor stats: max, min, mean
agg_stats <- neighbor_pairs_long[
  !is.na(value),
  .(nbr_max = max(value), nbr_min = min(value), nbr_mean = mean(value)),
  by = .(src_id, year, var_name)
]

# Reshape back to wide format and merge into original data
agg_wide <- dcast(agg_stats, src_id + year ~ var_name,
                  value.var = c("nbr_max", "nbr_min", "nbr_mean"))

# Merge with original data
setkey(dt, id, year)
setkey(agg_wide, src_id, year)
dt <- dt[agg_wide]

# Result: dt now contains neighbor features for all variables
```

---

### **Why This Works**
- Eliminates **5 full passes** over 6.46M rows.
- Uses `data.table` joins and aggregation, which are optimized in C.
- Preserves original estimands (max, min, mean for neighbors).
- Scales far better than interpreted R loops; expected runtime reduction from 86+ hours to a few hours or less on a 16 GB laptop.

**Summary:** Reject the colleagueâ€™s diagnosis. The bottleneck is repeated interpreted loops and NA filtering, not `rbind`. The correct optimization is to vectorize neighbor computations and aggregate in one pass using `data.table`.
END_RESPONSE>>>
