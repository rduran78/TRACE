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
run_id: OpenAI_F02_T03_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list (length = 6.46M rows), each element storing neighbor indices. This is highly inefficient in R due to list overhead and repeated string concatenation.  
- `compute_neighbor_stats` iterates over this huge list for each variable, performing repeated subsetting and aggregation, leading to ~O(N Ã— neighbors Ã— vars) operations in pure R loops.  
- Memory pressure is high because of redundant storage and repeated intermediate objects.  

**Optimization Strategy**  
1. **Avoid per-row lists**: Instead of building a giant list, use a long-format edge table (cell-year â†’ neighbor-year) and join operations.  
2. **Vectorize aggregation**: Compute neighbor stats using `data.table` group operations rather than `lapply`.  
3. **Leverage keys and joins**: `data.table` can handle 6.5M rows efficiently on 16 GB RAM if operations are vectorized.  
4. **Precompute static neighbor relationships**: Expand neighbors across years once, then join with variable columns.  
5. **Compute all neighbor stats in one pass**: Melt data and aggregate by `(cell_id, year, var_name)`.  

---

### **Optimized Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor edges (static across years)
# rook_neighbors_unique: list of integer vectors (spdep nb object)
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = i, nbr = rook_neighbors_unique[[i]])
}))

# Expand edges across all years
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, .(id = rep(src, length(years)), nbr_id = rep(nbr, length(years)), year = years), by = 1:nrow(edges)]
edges_expanded[, nrow := NULL]  # drop helper column

# Join neighbor values
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare lookup for neighbor values
setkey(cell_data, id, year)
setkey(edges_expanded, nbr_id, year)
edges_expanded <- cell_data[edges_expanded, on = .(id = nbr_id, year), nomatch = 0]
# Now edges_expanded has: id (src), nbr_id, year, and neighbor vars

# Melt neighbor values for aggregation
melted <- melt(edges_expanded, id.vars = c("id", "year"), measure.vars = neighbor_vars,
               variable.name = "var_name", value.name = "nbr_val", na.rm = TRUE)

# Compute neighbor stats: max, min, mean
agg_stats <- melted[, .(
  nbr_max = max(nbr_val, na.rm = TRUE),
  nbr_min = min(nbr_val, na.rm = TRUE),
  nbr_mean = mean(nbr_val, na.rm = TRUE)
), by = .(id, year, var_name)]

# Reshape wide to merge back
agg_wide <- dcast(agg_stats, id + year ~ var_name, value.var = c("nbr_max", "nbr_min", "nbr_mean"))

# Merge back into cell_data
cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- **No giant lists**: We store neighbor relationships in a compact long format.
- **Single pass aggregation**: All neighbor stats for all variables computed in one grouped operation.
- **Memory efficiency**: `data.table` uses references and avoids copying large objects repeatedly.
- **Speed**: Joins and group operations in `data.table` are highly optimized in C.

---

**Expected Performance**  
On 6.5M rows and ~1.37M neighbor edges, this approach should complete in **minutes to a few hours**, not 86+ hours, and fit in 16 GB RAM with careful memory management.

**Preserves**:
- Original estimand (numeric values unchanged).
- Trained Random Forest model (not retrained).
- All neighbor-based features computed correctly.
END_RESPONSE>>>
