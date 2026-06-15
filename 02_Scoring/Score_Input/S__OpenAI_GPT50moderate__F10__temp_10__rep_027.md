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
run_id: OpenAI_F10_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is inefficient because it repeatedly computes neighbor indices and iterates (with R `lapply`) across 6.46 million rows and multiple variablesâ€”operations that are memory- and time-intensive in pure R. Bottlenecks:  
- Building neighbor lookup returns a list of vectors for 6.46M rows (huge memory/time).
- Computing stats via `lapply` for every var repeats a similar scan.
- No vectorization or efficient graph representation is leveraged.
- Each loop redundantly processes neighbors per year, rather than exploiting stable topology.

---

**Optimization Strategy**  
1. Represent the topology once as an edge list from rook neighbors: (cell_id, neighbor_id).  
2. Use integer mapping to map `(cell_id, year)` to row indices without producing massive lookup lists.  
3. Compute neighbor aggregates via fast joins and grouping instead of nested `lapply`. Use `data.table` for efficiency.  
4. Process all years in one pass by joining `data` to its neighbors through the edge list and same-year condition. This uses relational joins rather than building giant lists.
5. Compute max, min, mean using `data.table` `by=`.  
6. Append features back in vectorized fashion.

Memory fit: edge list ~1.37M pairs * 28 â‰ˆ 38.5M joined rows, which is large but feasible with `data.table` streaming in chunks.  
Preserves numerical equivalence.

---

**Efficient Implementation in R (`data.table` solution)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object

# 1. Build edge list once
edge_list <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique, use.names = FALSE)
)

# 2. Prepare repeated join keys: replicate edges for each year
years <- sort(unique(cell_data$year))
edge_list_years <- edge_list[, .(id = from, neighbor_id = to)][
  , .(id = rep(id, each = length(years)), 
      neighbor_id = rep(neighbor_id, each = length(years)), 
      year = rep(years, times = .N))
]

# 3. Join on neighbor_id/year to fetch neighbor attributes
setkey(cell_data, id, year)
setkey(edge_list_years, neighbor_id, year)

neighbor_joined <- cell_data[edge_list_years, on = .(id = neighbor_id, year)]

# neighbor_joined now has columns from edge_list_years (id, neighbor_id, year) and 
# neighbor attributes from cell_data (ntl, ec, etc.)
# Rename 'id' from edge_list_years (target cell) to target_id
setnames(neighbor_joined, "id", "target_id")

# 4. Compute aggregates by target_id + year
agg_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_exprs <- unlist(lapply(agg_vars, function(v) {
  list(
    substitute(max(x, na.rm=TRUE), list(x=as.name(v))),
    substitute(min(x, na.rm=TRUE), list(x=as.name(v))),
    substitute(mean(x, na.rm=TRUE), list(x=as.name(v)))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(agg_vars, function(v) c(paste0(v, "_nbr_max"),
                                                   paste0(v, "_nbr_min"),
                                                   paste0(v, "_nbr_mean"))))

neighbor_stats <- neighbor_joined[, c(agg_exprs), by = .(target_id, year)]
setnames(neighbor_stats, old = names(neighbor_stats)[-(1:2)], new = agg_names)

# 5. Merge back into cell_data
setkey(cell_data, id, year)
setkey(neighbor_stats, target_id, year)
cell_data <- neighbor_stats[cell_data, on = .(target_id = id, year)]

# Now cell_data has new columns with neighbor-based features
# Apply pre-trained Random Forest model:
pred <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why Fast?**
- Avoids creating a 6.46M-length list; uses grouped aggregation over an expanded edge-year table.
- Uses compiled `data.table` joins and aggregations instead of R loops.
- Topology is computed once and reused.
- Fully vectorized, numerically equivalent (same stats per node-year).

**Estimated Performance**: Depending on I/O, likely hours â†’ minutes on 16 GB RAM.  
**Preserves trained model & estimand**: Same node-year aggregates and attributes, model unchanged.
END_RESPONSE>>>
