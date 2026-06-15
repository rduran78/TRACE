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
run_id: OpenAI_F02_T10_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck comes from:  
- `lapply` loops over ~6.5M rows twice (once for neighbor index lookup, once for every variable), creating large intermediate lists.  
- Excessive string concatenation (`paste`) for keys, lookups, and repeated indexing on 6.5M observations.  
- `build_neighbor_lookup` materializes neighbor indices for each cell-year, multiplying the spatial structure by time unnecessarily.  
- Memory blowup: each row stores vector of indices â†’ huge list objects.  

**Optimization Strategy**  
- Leverage **vectorized joins** instead of per-row operations; precompute neighbor relationships as a long table and join with data.table/dplyr.  
- Expand neighbor relationships across years once, then compute aggregated stats by group.  
- Use `data.table` for efficient joins and aggregations.  
- Avoid repeated calculation for multiple vars; compute all neighbor stats together in one grouped operation.  
- Process in batches if needed, but a properly implemented data.table solution can handle it on a 16 GB laptop.  

---

### **Optimized Approach in R**
1. Convert to `data.table`.
2. Create a long table of `(cell_id, neighbor_id, year)` for all years.
3. Join with `cell_data` twice: once for source, once for neighbor variable values.
4. Compute `max`, `min`, `mean` by `(cell_id, year)` and variable.

---

#### **Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor relationships across years
# rook_neighbors_unique: list of integer vectors of neighbor ids matching id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand across years (~1.37M * 28 ~ 38M rows -> feasible on disk with data.table)
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Melt cell_data for easier join on multiple vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var_name", value.name = "value")

# Join neighbor values
setkey(long_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

joined <- neighbor_dt[long_data, on = .(neighbor_id = id, year), nomatch = 0]

# joined now has: id (source), neighbor_id, year, var_name, value
# Compute neighbor stats by (id, year, var_name)
neighbor_stats <- joined[
  !is.na(value),
  .(neighbor_max = max(value),
    neighbor_min = min(value),
    neighbor_mean = mean(value)),
  by = .(id, year, var_name)
]

# Reshape back to wide format
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var_name,
                              value.var = c("neighbor_max", "neighbor_min", "neighbor_mean"))

# Merge back into original cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why Itâ€™s Faster**
- Eliminates per-row loops and repeated string manipulation.
- Uses `data.table` grouping (C-optimized) over vectorized structures.
- Avoids storing giant lists in memory.
- Computes all 5 neighbor variables in **one pass**.

**Expected Performance**:  
On ~40M rows intermediate, `data.table` on 16 GB RAM is practical if processed in chunks, and runs in minutesâ€“1 hr instead of 86 hrs.  

**Preserves**:  
- Original model
- Identical numerical estimands
- Neighbor relationships
END_RESPONSE>>>
