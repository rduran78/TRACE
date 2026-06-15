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
run_id: OpenAI_F07_T03_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is extremely slow (86+ hours) because it repeatedly performs expensive list lookups and vector operations for ~6.46M rows across multiple variables. The main bottlenecks are:  
- `lapply` over millions of rows in `build_neighbor_lookup` and `compute_neighbor_stats`.  
- Repeated string concatenation (`paste`) and dictionary lookups for every row.  
- No vectorization; everything is row-wise.  
- Large intermediate lists consuming memory.  

**Optimization Strategy:**  
1. **Precompute neighbor indices once** as integer vectors without string operations.  
2. **Flatten the neighbor relationships** into an edge list and use `data.table` for fast joins and aggregations.  
3. **Compute max, min, mean in a single grouped aggregation** rather than row-wise loops.  
4. Avoid recomputing for each variable; process all variables in a long format or loop efficiently.  
5. Use `data.table` keys and grouping for speed and memory efficiency.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of ids in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Map id to index for fast lookup
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build edge list (directed)
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    from = id_order[i],
    to   = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand edges for all years
years <- unique(cell_data$year)
edges_expanded <- edges[, .(id = from, neighbor_id = to), by = years]
setnames(edges_expanded, "years", "year")

# Merge to get row indices
edges_expanded[, id := as.integer(id)]
edges_expanded[, neighbor_id := as.integer(neighbor_id)]

# Join with cell_data to get neighbor values
# Melt cell_data for the 5 variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var", value.name = "val")

# Join neighbor values
edges_long <- merge(edges_expanded, long_data, by.x = c("neighbor_id", "year"),
                    by.y = c("id", "year"), allow.cartesian = TRUE)

# Compute stats by (id, year, var)
neighbor_stats <- edges_long[, .(
  neigh_max = max(val, na.rm = TRUE),
  neigh_min = min(val, na.rm = TRUE),
  neigh_mean = mean(val, na.rm = TRUE)
), by = .(id, year, var)]

# Reshape back to wide format
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var,
                             value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- **No per-row loops**: All operations are vectorized via `data.table` grouping.
- **Single pass aggregation**: Compute max, min, mean in one grouped step.
- **Memory efficient**: Avoids building millions of small lists.
- **Scales well**: `data.table` handles 6.5M rows efficiently on 16 GB RAM.

**Expected runtime**: From 86+ hours down to a few minutes on a standard laptop.

**Preserves**:  
- Original rook-neighbor topology.  
- Exact numerical estimand (max, min, mean of neighbors).  
- Trained Random Forest model remains untouched.
END_RESPONSE>>>
