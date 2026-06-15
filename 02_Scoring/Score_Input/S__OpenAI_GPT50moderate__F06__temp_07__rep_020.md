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
run_id: OpenAI_F06_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` loops in `compute_neighbor_stats`, which iterate across ~6.46M rows and repeatedly subset vectors. This is highly inefficient in R for such large datasets. Each call allocates vectors and performs redundant lookups. The complexity grows with rows Ã— neighbors Ã— variables, leading to the 86+ hour estimate.  

**Optimization Strategy**  
- **Avoid per-row loops**: Vectorize operations or use highly optimized aggregation packages (`data.table` or `dplyr`).
- **Pre-flatten neighbor relationships**: Use a long-form "edge list" (cell-year â†’ neighbor-year) so stats can be computed via grouped aggregations.
- **Chunking or parallelization**: Use `data.table` for fast joins and aggregations in memory.
- **Preserve original estimands**: Compute max, min, mean exactly over non-NA neighbors for the same year, no approximations.
- **Do not retrain model**: Only augment `cell_data` with new features.

---

### **Optimized Implementation (data.table)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Flatten neighbor relationships into an edge list
# id_order aligns with rook_neighbors_unique
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  nb_id = unlist(rook_neighbors_unique)
)

# Repeat for all years (cartesian join on year)
years <- unique(cell_data$year)
edges <- edges[, .(id = rep(id, length(years)),
                   nb_id = rep(nb_id, length(years)),
                   year = rep(years, each = .N))]

# Create keys for fast join
setkey(cell_data, id, year)
setkey(edges, nb_id, year)

# Join neighbor values for all neighbor variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for efficient processing
long_data <- melt(cell_data[, c("id", "year", neighbor_source_vars), with = FALSE],
                  id.vars = c("id", "year"),
                  variable.name = "var",
                  value.name = "value")

setkey(long_data, id, year)

# Join edges with neighbor values
edges_long <- long_data[edges, on = .(id = nb_id, year), allow.cartesian = TRUE]

# Compute stats grouped by focal id-year and variable
stats <- edges_long[, .(
  nb_max = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
  nb_min = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
  nb_mean = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
), by = .(focal_id = id, year, var)]

# Reshape wide to join back
stats_wide <- dcast(stats, focal_id + year ~ var,
                    value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge back to original cell_data
cell_data <- merge(cell_data, stats_wide, by.x = c("id", "year"), by.y = c("focal_id", "year"), all.x = TRUE)
```

---

### **Why This Is Faster**
- Single-pass joins and grouped aggregations via `data.table` are optimized in C.
- Eliminates millions of nested loops.
- Memory use remains manageable by processing in long form.
- Preserves the exact computation (max, min, mean of non-NA rook neighbors per year).

**Expected runtime**: Hours reduced from 86+ to ~15â€“30 minutes on a 16 GB laptop (dependent on I/O and disk speed).  

**Next steps**: After this, directly feed `cell_data` with new features into the pre-trained Random Forest model for prediction.
END_RESPONSE>>>
