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
run_id: OpenAI_F07_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because it performs `lapply`-based neighbor computations for ~6.46M rows and five variables, materializing large intermediate lists. Each neighbor query repeatedly builds subset vectors and runs `max`, `min`, and `mean` independently without leveraging vectorized or compiled operations. This causes huge overhead and memory pressure, especially since neighbor lookups repeat per year.

**Optimization Strategy**  
- Represent data as keyed `data.table` for fast join/merge.
- Flatten (cell_id, neighbor_id, year) triplets once, then `merge` to get neighbor values.
- Compute `max`, `min`, `mean` with `data.table` group aggregation (compiled C-level loops).
- Use integer indices and avoid repeated string concatenation inside loops.
- Compute all neighbor stats in one pass using `melt`/`dcast` or grouped aggregation rather than looping over rows.
- Preserve estimands by doing pure aggregation on the same neighbor sets, just more efficiently.
- Memory is manageable: ~6.46M rows, ~1.37M edges Ã— 28 years â‰ˆ 38.5M triplets â†’ still feasible with `data.table` on 16â€¯GB if processed in chunks.
- Keep the Random Forest model unchanged; generate enhanced feature columns via join.

---

### **Efficient R Implementation**

```r
library(data.table)

# Assume: cell_data (data.table), columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Inputs: id_order (vector), rook_neighbors_unique (list of integer vectors)
# Steps:

# 1. Prepare edge list from neighbors
build_edge_dt <- function(id_order, neighbors) {
  # Directed edges: for each ref cell, list its neighbors
  src <- rep(id_order, lengths(neighbors))
  dst <- unlist(neighbors, use.names = FALSE)
  data.table(ref_id = src, neigh_id = dst)
}

edges <- build_edge_dt(id_order, rook_neighbors_unique)

# 2. Expand for each year
years <- sort(unique(cell_data$year))
edges_expanded <- CJ(year = years, edge_id = seq_len(nrow(edges)))[, 
  c("ref_id", "neigh_id") := .(edges$ref_id[edge_id], edges$neigh_id[edge_id])
][, edge_id := NULL]

# 3. Join to get neighbor values
setkey(cell_data, id, year)
setkey(edges_expanded, neigh_id = neigh_id, year)

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Add neighbor values
edges_expanded <- cell_data[edges_expanded, on = .(id = neigh_id, year), 
                             .(year, ref_id, neigh_id, ntl, ec, pop_density, def, usd_est_n2)]

# 4. Compute stats grouped by ref_id and year for each var
agg_list <- lapply(neighbor_vars, function(v) {
  as.call(list(quote(.(max = max(x, na.rm=TRUE),
                       min = min(x, na.rm=TRUE),
                       mean = mean(x, na.rm=TRUE))),
               substitute(x, list(x = as.name(v)))))
})

# Build expression programmatically
expr <- parse(text = sprintf(
  "edges_expanded[, .(%s), by = .(ref_id, year)]", 
  paste(sprintf('%s_max = max(%s, na.rm=TRUE), %s_min = min(%s, na.rm=TRUE), %s_mean = mean(%s, na.rm=TRUE)', 
                neighbor_vars, neighbor_vars, neighbor_vars, neighbor_vars, neighbor_vars, neighbor_vars),
        collapse = ', ')
))

result_dt <- eval(expr)

# 5. Join back to cell_data
setkey(result_dt, ref_id, year)
cell_data <- result_dt[cell_data, on = .(ref_id = id, year)]

# Done: cell_data now has *_max, *_min, *_mean columns for each source var
```

---

**Why Itâ€™s Fast**  
- Eliminates per-row neighbor lookup.
- Uses `data.table` joins and vectorized aggregations in C.
- Scales linearly with number of edges (~38M rows after join), which is feasible with `fread`-class memory efficiency and chunking if necessary.

**Expected Runtime**  
From >86 hours to a few minutes to an hour, depending on I/O and hardware.

**Key Guarantees**  
- Original rook adjacency preserved.
- Same numerical estimands (max, min, mean of valid neighbors).
- No retraining: model remains unchanged.

You can further optimize by:
- Processing variables in chunks or using `fst`/`arrow` for disk-backed joins if RAM is tight.
- Parallelize by year using `future.apply` or `data.table`â€™s `by = .EACHI` across subsets.
END_RESPONSE>>>
