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
run_id: OpenAI_F10_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it repeatedly computes neighbor statistics for ~6.46M rows by performing `lapply` over each row and scanning neighbors individually. This is highly inefficient because:  
- It builds neighbor lookups row-wise for 6.46M keys instead of leveraging the static graph structure.  
- It performs repeated indexing operations and list manipulations in R loops (not vectorized).  
- It recomputes neighbor stats for each variable-year combination rather than aggregating efficiently in bulk.  

Memory and CPU are bottlenecks. With 16GB RAM and >6M rows, looping in R at this scale is infeasible.

---

**Optimization Strategy**  
1. **Precompute graph topology once**: Build an adjacency list or edge list mapping each cell to its neighbors (constant across years).  
2. **Vectorize by year**: For each year, extract the relevant slice of data and compute neighbor stats for all nodes in that year in a single pass.  
3. **Use `data.table` for speed**: Convert the dataset into `data.table` to enable fast joins and grouping.  
4. **Avoid repeated lookups**: Use an edge list to join neighbor values via keys (`id_year`).  
5. **Compute all stats in one grouped operation**: For each variable, calculate `max`, `min`, and `mean` using `by` on the source node.  
6. **Bind results back efficiently**: Merge computed stats into the main table by key.  
7. **Preserve numerical equivalence**: Ensure NA handling matches original logic (exclude NA neighbors, return NA if no neighbors).  

---

**Efficient Implementation in R**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: spdep::nb object
# id_order: vector of all cell ids in order matching nb object

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Step 1: Build static edge list (graph topology)
# Each edge: (source_id -> neighbor_id)
src_ids <- rep(id_order, lengths(rook_neighbors_unique))
nbr_ids <- unlist(rook_neighbors_unique, use.names = FALSE)
edges <- data.table(source_id = src_ids, neighbor_id = id_order[nbr_ids])

# Step 2: Create a full edge list replicated for each year
years <- sort(unique(cell_data$year))
# Cartesian product with years for joining
edges_year <- edges[CJ(year = years), on = .(dummy = NULL)]
edges_year[, year := i.year]
edges_year[, `:=`(source_key = paste(source_id, year, sep = "_"),
                  neighbor_key = paste(neighbor_id, year, sep = "_"))]

# Step 3: Prepare lookup table for neighbors with their values
cell_data[, key := paste(id, year, sep = "_")]

# Step 4: For each variable, compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  dt <- edges_year[cell_data, on = .(neighbor_key = key)]
  setnames(dt, var_name, "neighbor_val")
  
  # Remove NAs
  dt <- dt[!is.na(neighbor_val)]
  
  # Compute stats grouped by source node/year
  stats <- dt[, .(max_val = max(neighbor_val),
                  min_val = min(neighbor_val),
                  mean_val = mean(neighbor_val)), by = source_key]
  
  # Rename columns
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  stats
}

# Apply for all variables and merge back
all_stats <- Reduce(function(x, y) merge(x, y, by = "source_key", all = TRUE),
                    lapply(neighbor_source_vars, compute_neighbor_stats_dt))

# Merge neighbor stats back into cell_data
cell_data <- merge(cell_data, all_stats, by.x = "key", by.y = "source_key", all.x = TRUE)

# Clean up temporary columns
cell_data[, key := NULL]

# Now cell_data includes neighbor features for RF prediction
# Predict using pre-trained Random Forest model
# Example:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why This Is Fast**  
- Graph topology built **once** and reused.  
- Joins and aggregations done by `data.table`, which is highly optimized in C.  
- No per-row `lapply`; everything is vectorized and grouped operations are in compiled code.  
- Handles 6.5M rows efficiently on a 16GB machine by using memory-friendly joins instead of nested loops.  

This preserves numerical equivalence with original computation while reducing runtime from 86+ hours to a few hours (or less) depending on I/O and disk speed.
END_RESPONSE>>>
