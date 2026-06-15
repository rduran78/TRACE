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
run_id: OpenAI_F10_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `build_neighbor_lookup` produces a per-row neighbor index list for all 6.46M rows, repeating work for 28 years.  
- `compute_neighbor_stats` iterates over rows with `lapply`, repeatedly subsetting, which is highly inefficient.  
- The pipeline recomputes neighbor features per variable, causing redundant traversals.  
- No vectorization; memory overhead from large lists and repeated indexing.  

**Optimization Strategy**  
- Represent the panel as a graph with nodes = cell-year pairs and edges = rook neighbors across same year.  
- Precompute topology: for each cell, store its neighbors (rook adjacency). Reuse across years by broadcasting via year grouping.  
- Avoid per-row `lapply`: use `data.table` or matrix aggregation for vectorized computations.  
- Process all years in blocks using joins instead of looping through 6.46M rows individually.  
- Compute neighbor statistics for multiple variables in one pass if possible.  
- Preserve numerical equivalence: same max, min, mean for neighbor attributes per node-year.  
- Memory: keep adjacency as integer vectors and use `data.table` joins for speed.  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor adjacency once for all cells
# rook_neighbors_unique: list of integer neighbor indices corresponding to id_order
adj_list <- rook_neighbors_unique

# Create a mapping table for (id, year) -> row index
cell_data[, key := .I]
id_year_map <- cell_data[, .(key, id, year)]

# Expand adjacency across years
# For each cell-year, find neighbor keys via join
neighbor_edges <- rbindlist(
  lapply(seq_along(adj_list), function(i) {
    if (length(adj_list[[i]]) == 0) return(NULL)
    # Current cell id
    src_id <- id_order[i]
    # Neighbor ids
    nbr_ids <- id_order[adj_list[[i]]]
    data.table(src_id = src_id, nbr_id = nbr_ids)
  })
)

# Join to years: cross with all years in cell_data
years <- unique(cell_data$year)
neighbor_edges <- neighbor_edges[, .(year = years), by = .(src_id, nbr_id)]

# Map to keys for fast lookup
neighbor_edges <- merge(neighbor_edges, id_year_map[, .(src_id = id, year, src_key = key)],
                        by = c("src_id", "year"))
neighbor_edges <- merge(neighbor_edges, id_year_map[, .(nbr_id = id, year, nbr_key = key)],
                        by = c("nbr_id", "year"))

# Compute neighbor stats in one pass for all variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for easier aggregation
long_data <- melt(cell_data, id.vars = "key", measure.vars = vars, variable.name = "var", value.name = "val")

# Join neighbor values
neighbor_long <- merge(neighbor_edges[, .(src_key, nbr_key)], long_data, by.x = "nbr_key", by.y = "key")

# Aggregate stats: max, min, mean per src_key and var
agg_stats <- neighbor_long[, .(
  nbr_max = max(val, na.rm = TRUE),
  nbr_min = min(val, na.rm = TRUE),
  nbr_mean = mean(val, na.rm = TRUE)
), by = .(src_key, var)]

# Reshape back to wide and merge into cell_data
agg_wide <- dcast(agg_stats, src_key ~ var, value.var = c("nbr_max", "nbr_min", "nbr_mean"))
setnames(agg_wide, "src_key", "key")
cell_data <- merge(cell_data, agg_wide, by = "key", all.x = TRUE)

# Drop helper columns
cell_data[, key := NULL]

# Apply pre-trained Random Forest model (already trained)
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

### **Key Gains**
- **Single adjacency computation** reused for all years.
- **Vectorized aggregation** using `data.table` instead of per-row loops.
- Processes all variables in one pass.
- Preserves original numerical estimand.
- Expected runtime: **minutes to a few hours**, not 86+ hours, given efficient joins and aggregation.

This design minimizes memory overhead and leverages fast joins and group operations in `data.table`.
END_RESPONSE>>>
