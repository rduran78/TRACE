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
run_id: OpenAI_F02_T03_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `lapply` over 6.46M rows creates massive intermediate lists.  
- `build_neighbor_lookup` constructs a large list of integer vectors (millions of elements), consuming huge memory.  
- `compute_neighbor_stats` repeatedly subsets vectors and computes stats in R loops, which is inefficient for large datasets.  
- No vectorization or batching; everything is row-wise.  

**Optimization Strategy**  
1. **Avoid building a huge neighbor lookup list**: Instead, create a long-format table mapping each cell-year to its neighbors using `data.table`.  
2. **Leverage `data.table` joins and group operations** for computing neighbor statistics in a fully vectorized way.  
3. **Process variables in a single pass** rather than looping with `lapply`.  
4. **Memory efficiency**: Work in long format, compute aggregates, then join back to the main table.  
5. **Parallelization**: Optionally use `data.table` multithreading (built-in).  

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# Expand neighbor relationships to long format
# rook_neighbors_unique: list of integer vectors (neighbors) indexed by id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to create full panel for neighbors
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Merge neighbor values for all variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor_dt with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- cell_data[neighbor_dt, on = .(id = neighbor_id, year), nomatch = 0]
setnames(neighbor_dt, old = vars, new = paste0("nbr_", vars))

# Now compute stats grouped by (id, year)
agg_list <- lapply(vars, function(v) {
  list(
    max = as.name(paste0("max(nbr_", v, ", na.rm=TRUE)")),
    min = as.name(paste0("min(nbr_", v, ", na.rm=TRUE)")),
    mean = as.name(paste0("mean(nbr_", v, ", na.rm=TRUE)"))
  )
})
names(agg_list) <- vars

agg_expr <- unlist(lapply(names(agg_list), function(v) {
  list(
    substitute(max(nv, na.rm=TRUE), list(nv = as.name(paste0("nbr_", v)))),
    substitute(min(nv, na.rm=TRUE), list(nv = as.name(paste0("nbr_", v)))),
    substitute(mean(nv, na.rm=TRUE), list(nv = as.name(paste0("nbr_", v))))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))

neighbor_stats <- neighbor_dt[, eval(as.call(c(quote(list), agg_expr))), by = .(id, year)]
setnames(neighbor_stats, old = paste0("V", seq_along(agg_names)), new = agg_names)

# Merge back to cell_data
cell_data <- neighbor_stats[cell_data, on = .(id, year)]
```

---

**Why this works:**  
- Eliminates 6.46M-row `lapply` loops.  
- Uses `data.table` joins and aggregation, which are highly optimized in C.  
- Memory footprint is reduced because we avoid storing a massive list of neighbors; instead, use a long table and aggregate.  
- Preserves original estimand and trained model (only feature engineering changes).  

**Expected Performance:**  
- On 6.46M rows and ~1.37M neighbor links, this approach should run in **minutes to a few hours**, not 86+ hours, on a 16 GB laptop.  
- Built-in multithreading in `data.table` will speed up aggregation.  

**Optional further optimization:**  
- Process variables in chunks if memory is tight.  
- Save intermediate tables to disk with `fst` or `arrow` for large-scale workflows.
END_RESPONSE>>>
