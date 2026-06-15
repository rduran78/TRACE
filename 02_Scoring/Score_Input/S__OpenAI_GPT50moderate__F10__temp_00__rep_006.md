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
run_id: OpenAI_F10_T00_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind)`, which is inefficient in R for large data.  
- Neighbor lookup is rebuilt for every variable, and aggregation is done row-wise instead of vectorized.  
- No reuse of graph topology across years; repeated string concatenation and lookups are costly.  

**Optimization Strategy**  
- Build the neighbor graph once using integer indices (avoid string keys).  
- Use `data.table` for fast joins and grouped operations.  
- Compute all neighbor stats in a single pass per variable using vectorized aggregation.  
- Avoid repeated `lapply` over millions of rows; instead, flatten edges and aggregate with `by` groups.  
- Preserve numerical equivalence by using the same max, min, mean definitions.  

---

### **Optimized R Implementation**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: list of integer neighbor indices (spdep::nb)

# Convert to data.table
setDT(cell_data)

# Precompute graph edges (directed)
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand edges across years
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, .(id = rep(src, length(years)),
                             nbr = rep(nbr, length(years)),
                             year = rep(years, each = .N))]

# Merge neighbor values for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor attributes
setkey(cell_data, id, year)
setkey(edges_expanded, nbr, year)
edges_expanded <- cell_data[edges_expanded, on = .(id = nbr, year), 
                             .(id = i.id, year, val_ntl = ntl, val_ec = ec,
                               val_pop = pop_density, val_def = def, val_usd = usd_est_n2)]

# Compute stats for each variable in one grouped pass
agg_stats <- edges_expanded[, .(
  ntl_max = max(val_ntl, na.rm = TRUE),
  ntl_min = min(val_ntl, na.rm = TRUE),
  ntl_mean = mean(val_ntl, na.rm = TRUE),
  ec_max = max(val_ec, na.rm = TRUE),
  ec_min = min(val_ec, na.rm = TRUE),
  ec_mean = mean(val_ec, na.rm = TRUE),
  pop_max = max(val_pop, na.rm = TRUE),
  pop_min = min(val_pop, na.rm = TRUE),
  pop_mean = mean(val_pop, na.rm = TRUE),
  def_max = max(val_def, na.rm = TRUE),
  def_min = min(val_def, na.rm = TRUE),
  def_mean = mean(val_def, na.rm = TRUE),
  usd_max = max(val_usd, na.rm = TRUE),
  usd_min = min(val_usd, na.rm = TRUE),
  usd_mean = mean(val_usd, na.rm = TRUE)
), by = .(id, year)]

# Merge back to main data
cell_data <- agg_stats[cell_data, on = .(id, year)]

# Now cell_data has neighbor stats for all variables
# Apply pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This is Efficient**
- **Single graph build**: Edges computed once, reused for all years and variables.  
- **Vectorized aggregation**: `data.table` groups by `(id, year)` and computes all stats in one pass.  
- **Memory efficiency**: Avoids millions of small lists; uses fast joins and grouping.  
- **Preserves numerical equivalence**: Same max, min, mean logic, just vectorized.  

**Expected runtime**: From 86+ hours to a few minutes on a 16 GB laptop.
END_RESPONSE>>>
