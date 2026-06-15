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
run_id: OpenAI_F09_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow because `build_neighbor_lookup()` recomputes neighbor indices for every cell-year combination (6.46M rows), and `compute_neighbor_stats()` repeatedly loops through full vectors for each year-variable combination. This results in heavy list allocations, many `paste()` string operations, and repeated subset operations, leading to exponential overhead.

**Optimization Strategy**  
- **Precompute adjacency once**: Use cell-level adjacency table from `rook_neighbors_unique` and reuse it across years.
- **Vectorize yearly computation**: Process one year at a time and join neighbor stats without repeatedly constructing strings.
- **Pre-index neighbors by numeric positions**: Avoid costly string concatenations; operate on numeric indices.
- **Compute stats in `data.table`** for efficient joins and group operations.
- **Reuse in-memory objects**, keep 16 GB RAM in mind.

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.
setkey(cell_data, id, year)

# Precompute adjacency once (neighbors as integer positions)
id_order <- sort(unique(cell_data$id))
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Collapse into edge list (cell_id -> neighbor_id)
edge_list <- data.table(
  id = rep(id_order, lengths(neighbor_lookup)),
  neighbor = unlist(lapply(neighbor_lookup, function(x) id_order[x]))
)

# Function to compute neighbor stats for a given variable
compute_neighbor_stats_fast <- function(dt, var_name) {
  # Melt relevant columns by year
  val_dt <- dt[, .(id, year, val = get(var_name))]

  # Join edge_list twice: origin and neighbor for same year
  joined <- edge_list[val_dt, on = .(neighbor = id)][, neighbor_val := i.val]
  joined <- joined[val_dt, on = .(id, year)][, origin_val := i.val]

  # Compute stats by id, year
  res <- joined[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]

  # Replace Inf/-Inf with NA (when all neighbor_val are NA)
  res[!is.finite(max_val), c("max_val", "min_val", "mean_val") := .(NA, NA, NA)]

  # Rename columns
  setnames(res, old = c("max_val", "min_val", "mean_val"),
                new = paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  res
}

# Compute and merge for each neighbor source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  feature_dt <- compute_neighbor_stats_fast(cell_data, var_name)
  cell_data <- merge(cell_data, feature_dt, by = c("id", "year"), all.x = TRUE)
}

# Predict using pre-trained Random Forest (model preserved)
cell_data$rf_pred <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Faster**
- **Adjacency built once** and kept as numeric IDsâ€”no string concatenation.
- Uses efficient **joins in `data.table`** rather than `lapply` loops over 6.46M rows.
- **No copy-heavy operations inside loops; process in vectorized form**.
- Expected to reduce runtime from **86+ hours to a few hours** on a 16 GB laptop, depending on I/O and join complexity.

This preserves the original estimand and the trained Random Forest model, while drastically improving computational efficiency.
END_RESPONSE>>>
