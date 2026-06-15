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
run_id: OpenAI_F02_T03_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `lapply` over 6.46M rows creates millions of small vectors and repeated string concatenations (`paste`) for every row.  
- `neighbor_lookup` is a list of length 6.46M, consuming huge memory.  
- `compute_neighbor_stats` repeatedly subsets large vectors in R loops, which is inefficient.  
- The process is single-threaded and not vectorized.  

**Optimization Strategy**  
1. **Avoid per-row string operations**: Use integer joins instead of `paste` keys.  
2. **Precompute neighbor relationships at the cell level** and join by year using `data.table`.  
3. **Use `data.table` for fast grouping and aggregation** rather than millions of `lapply` calls.  
4. **Compute all neighbor stats in one pass** using a long-format neighbor table.  
5. **Memory efficiency**: Avoid storing a 6.46M-length list; instead, store a two-column mapping of `(row_id, neighbor_row_id)`.  

**Optimized Approach**  
- Expand rook neighbors into a long table of `(cell_id, neighbor_id)`.  
- Join with panel data by year to get `(row_id, neighbor_row_id)`.  
- Melt variables and compute `max`, `min`, `mean` per `(row_id, var_name)`.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Add row_id for reference
cell_data[, row_id := .I]

# Expand neighbor pairs at cell level
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Join by year to create row-level neighbor mapping
# cell_data: id, year, row_id
neighbor_dt <- neighbor_dt[cell_data, on = .(cell_id = id), allow.cartesian = TRUE]
setnames(neighbor_dt, "row_id", "row_id_main")
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), allow.cartesian = TRUE]
setnames(neighbor_dt, "row_id", "row_id_neighbor")

# Keep only valid neighbor rows
neighbor_dt <- neighbor_dt[!is.na(row_id_neighbor), .(row_id_main, row_id_neighbor)]

# Reshape for neighbor variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor values
neighbor_vals <- melt(cell_data[, c("row_id", vars), with = FALSE],
                      id.vars = "row_id", variable.name = "var_name", value.name = "val")

# Join neighbor values
neighbor_dt <- neighbor_dt[neighbor_vals, on = .(row_id_neighbor = row_id), allow.cartesian = TRUE]

# Compute stats per main row and variable
agg_stats <- neighbor_dt[, .(
  neigh_max = max(val, na.rm = TRUE),
  neigh_min = min(val, na.rm = TRUE),
  neigh_mean = mean(val, na.rm = TRUE)
), by = .(row_id_main, var_name)]

# Reshape wide and merge back
agg_wide <- dcast(agg_stats, row_id_main ~ var_name, value.var = c("neigh_max", "neigh_min", "neigh_mean"))
setnames(agg_wide, "row_id_main", "row_id")

# Merge into original data
cell_data <- merge(cell_data, agg_wide, by = "row_id", all.x = TRUE)

# Drop helper columns
cell_data[, row_id := NULL]
```

---

**Why this is faster and feasible on 16 GB RAM**  
- Eliminates millions of small lists and string operations.  
- Uses `data.table` joins and aggregation, which are highly optimized in C.  
- Computes all variables in one pass instead of looping over variables.  
- Memory footprint is reduced because we store neighbor relationships as a long table (~1.3M Ã— 2 rows) instead of a 6.46M-length list.  

**Expected runtime**: From 86+ hours down to ~30â€“60 minutes on a standard laptop (depending on disk speed and CPU).  

This preserves the original estimand and does not require retraining the Random Forest model.
END_RESPONSE>>>
