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
run_id: OpenAI_F06_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` approach over *6.46 million rows*. For each row, the lookup and summary statistics recompute repeatedly in pure R lists, causing massive overhead and memory pressure. With roughly 6.5M Ã— 5 variables Ã— 3 stats, the current loop is highly inefficient.  

**Optimization Strategy**  
- Flatten the neighbor relationships into an edge list and compute `max`, `min`, and `mean` using **vectorized `data.table` joins and group operations** instead of iterating per row.  
- Leverage `data.table` for fast aggregation: duplicate targetâ€™s year on joins, join each cell-year with its neighbors, compute grouped stats in one pass.  
- Process one variable at a time but without row-wise loops.  
- Preserve the Random Forest model by producing identical new features.  

---

### **Working R Code**

```r
library(data.table)

# Assuming: cell_data (data.frame) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# Build edge list of (id, neighbor_id) from rook_neighbors_unique and id_order
make_edge_dt <- function(id_order, nb) {
  # nb is a list from spdep::nb
  i_vec <- rep(seq_along(nb), lengths(nb))
  j_vec <- unlist(nb, use.names = FALSE)
  data.table(id = id_order[i_vec], neighbor_id = id_order[j_vec])
}

edges_dt <- make_edge_dt(id_order, rook_neighbors_unique)

# Replicate edges across all years (Cartesian join for 28 years)
years <- sort(unique(cell_data$year))
edges_dt <- edges_dt[, .(id = rep(id, each = length(years)),
                         neighbor_id = rep(neighbor_id, each = length(years)),
                         year = rep(years, times = .N))]
setkey(edges_dt, neighbor_id, year)

# Set keys on cell_data
setkey(cell_data, id, year)

# Function to compute and add neighbor features efficiently
compute_and_add_neighbor_features <- function(dt, var_name, edges_dt) {
  # Join neighbors' values
  neighbor_vals <- dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)
  joined <- edges_dt[neighbor_vals, on = .(neighbor_id, year)]

  # Now joined has: id, neighbor_id, year, val
  # Compute stats by (id, year)
  stats <- joined[, .(
    max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]

  # Merge back into main dt
  setnames(stats, c("id", "year", paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))))
  dt[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, edges_dt)
}

# cell_data now contains all neighbor-based features with original estimand preserved
```

---

**Why This Works Faster**  
- Eliminates 6.46M row-wise R loops using `lapply`.
- Uses `data.table` joins and grouped operations, scaling closer to O(N) per variable.
- Memory-efficient vs. constructing 6M lists.
- Maintains correctness by replicating year alignment exactly.

**Expected Impact**  
Reduction from **86+ hours** to **minutes-to-low-hours**, as `data.table` can handle tens of millions of rows efficiently on 16â€¯GB RAM.

**Model**  
No retraining neededâ€”predictor matrix now includes optimized neighbor stats as before.
END_RESPONSE>>>
