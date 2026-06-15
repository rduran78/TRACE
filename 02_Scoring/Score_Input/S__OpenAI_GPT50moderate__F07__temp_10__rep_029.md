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
run_id: OpenAI_F07_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the per-row `lapply` lookup and repeated vector filtering inside `compute_neighbor_stats()` for ~6.46M rows. R loops and list operations scale poorly. The combination of `build_neighbor_lookup()` and neighbor calculations is extremely inefficient because it repeatedly constructs and filters vectors across millions of rows. Memory overhead is also high since `neighbor_lookup` is a huge nested list.

Direct neighborhood computation with joins or matrix representation is more efficient because:
- The neighbor graph is sparse but fixed (~1.37M edges).
- Vectorizing avoids expensive list iteration.
- Using integer keys and `data.table` joins avoids repeated lookups.
- We can precompute cross-year relationships by joining on `(id, year)` pairs.

---

### **Optimization Strategy**
1. **Precompute as an Edge List**: Instead of a nested list, build a long â€œedgesâ€ table mapping each cell-year to its neighbors for the same year.
2. **Use `data.table` for Joins**: Perform neighbor aggregation with `by` grouping after joining neighbor values.
3. **Compute all statistics in one pass per variable** instead of looping over rows.

---

### **Optimized R Code**  
Uses `data.table` to handle 6.46M rows efficiently within 16GB RAM.

```r
library(data.table)

# Assume cell_data is a data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Assume id_order and rook_neighbors_unique are available, rook_neighbors_unique is the neighbor structure

setDT(cell_data)  # Convert to data.table for speed

# Step 1: Build static neighbor edges across cells
edges <- data.table(from_id = rep(id_order, lengths(rook_neighbors_unique)),
                    to_id   = unlist(rook_neighbors_unique, use.names = FALSE))

# Step 2: Expand edges across years for same-year relationships
years <- unique(cell_data$year)
edges_expanded <- edges[ , .(id = rep(from_id, each = length(years)),
                              neigh_id = rep(to_id, each = length(years)),
                              year = rep(years, times = .N))]

# Step 3: Join neighbor values by (neigh_id, year)
setkey(cell_data, id, year)
setkey(edges_expanded, neigh_id, year)
edges_expanded[cell_data, val_ntl := i.ntl, on = .(neigh_id = id, year)]
edges_expanded[cell_data, val_ec := i.ec, on = .(neigh_id = id, year)]
edges_expanded[cell_data, val_pop_density := i.pop_density, on = .(neigh_id = id, year)]
edges_expanded[cell_data, val_def := i.def, on = .(neigh_id = id, year)]
edges_expanded[cell_data, val_usd := i.usd_est_n2, on = .(neigh_id = id, year)]

# Step 4: Collapse stats to original id-year level
stats <- edges_expanded[ , .(
  ntl_max = max(val_ntl, na.rm = TRUE),
  ntl_min = min(val_ntl, na.rm = TRUE),
  ntl_mean = mean(val_ntl, na.rm = TRUE),
  ec_max = max(val_ec, na.rm = TRUE),
  ec_min = min(val_ec, na.rm = TRUE),
  ec_mean = mean(val_ec, na.rm = TRUE),
  pop_max = max(val_pop_density, na.rm = TRUE),
  pop_min = min(val_pop_density, na.rm = TRUE),
  pop_mean = mean(val_pop_density, na.rm = TRUE),
  def_max = max(val_def, na.rm = TRUE),
  def_min = min(val_def, na.rm = TRUE),
  def_mean = mean(val_def, na.rm = TRUE),
  usd_max = max(val_usd, na.rm = TRUE),
  usd_min = min(val_usd, na.rm = TRUE),
  usd_mean = mean(val_usd, na.rm = TRUE)
), by = .(id, year)]

# Step 5: Merge back to cell_data
cell_data <- stats[cell_data, on = .(id, year)]
```

---

**Why is this faster?**
- Completely eliminates per-row `lapply`.
- Uses a sparse edge representation (~1.37M * years â‰ˆ 38M rows), which is manageable in `data.table`.
- Aggregates statistics in a vectorized manner (`by` groups).
- Handles all variables in one join step.

---

**Expected Performance Improvement**  
From 86+ hours â†’ ~30â€“60 minutes depending on disk I/O and RAM (with 16 GB this should fit).  
If memory is tight, process in chunks per subset of years (e.g., 1992â€“2000, 2001â€“2010, etc.) since edges are static.

**Preserves**:
- Original rook topology
- Panel structure
- Numerical estimand
- Random Forest model (unchanged)

This approach is safe and fully aligned with the requirement.
END_RESPONSE>>>
