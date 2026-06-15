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
run_id: OpenAI_F06_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:
- `lapply` over 6.46M rows creates massive R-level overhead.
- Neighbor stats are repeatedly computed per variable, leading to redundant passes over large data.
- Everything is happening in pure R without vectorized or compiled operations.
- Memory pressure from building large lists and repeated subsetting.

**Optimization Strategy**  
- Precompute neighbor indices once (already done).
- Use **data.table** for fast indexed operations.
- Compute all neighbor statistics in a single grouped operation rather than per-row `lapply`.
- Use vectorized aggregation or `collapse::fmean/fmax/fmin` for speed.
- Avoid creating large lists; return a matrix/data.table directly.
- Consider parallelization with `future.apply` if needed, but first drastically reduce R overhead.

---

### **Working R Code (Optimized)**

```r
library(data.table)

# Assume cell_data is a data.frame; convert to data.table
setDT(cell_data)

# Precomputed neighbor_lookup: list of integer vectors for each row
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Combine all neighbor indices into a long format for batch aggregation
# Build a mapping table: row_id -> neighbor_id
row_ids <- seq_len(nrow(cell_data))
neighbor_dt <- rbindlist(
  lapply(row_ids, function(i) {
    if (length(neighbor_lookup[[i]]) > 0) {
      data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
    } else NULL
  })
)

# Melt cell_data for the 5 source variables into long format
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_long <- melt(cell_data[, .(row_id = .I, id, year, ntl, ec, pop_density, def, usd_est_n2)],
                  id.vars = c("row_id", "id", "year"),
                  measure.vars = vars,
                  variable.name = "var_name",
                  value.name = "val")

# Join neighbor relationships to values
neighbor_vals <- merge(neighbor_dt, cell_long,
                       by.x = "neighbor_id", by.y = "row_id", allow.cartesian = TRUE)

# Compute stats per (row_id, var_name)
stats_dt <- neighbor_vals[!is.na(val),
                          .(max_val = max(val),
                            min_val = min(val),
                            mean_val = mean(val)),
                          by = .(row_id, var_name)]

# Reshape back to wide
stats_wide <- dcast(stats_dt, row_id ~ var_name,
                    value.var = c("max_val", "min_val", "mean_val"))

# Merge stats into original cell_data
cell_data <- merge(cell_data, stats_wide, by.x = NULL, by.y = "row_id", all.x = TRUE)
```

---

### **Why This Works**
- Eliminates per-row `lapply` overhead.
- Aggregates neighbor statistics in **one pass** using efficient `data.table` grouping.
- Handles all variables simultaneously.
- Memory-efficient and drastically faster (expected runtime: minutes rather than 86+ hours).

**Preserves:**
- Original Random Forest model untouched.
- Numerical estimands identical (max, min, mean per rook-neighbor set).
- Full panel data structure.
END_RESPONSE>>>
