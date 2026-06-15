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
run_id: OpenAI_F08_T07_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current design computes neighbor statistics for **each cell-year row** using `lapply` over 6.46M rows.  
- Repeated lookups on a large list (`neighbor_lookup`) and repeated filtering cause massive overhead.  
- Neighbor relationships are **static** across years, but the code recomputes neighbor stats row by row.  
- This results in ~6.46M Ã— 5 variables Ã— 3 stats operations, which is very slow and memory-inefficient.  

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** (already done by `build_neighbor_lookup`).  
2. **Exploit panel structure**:  
   - For each year, take the subset of data for that year.  
   - Compute neighbor stats **vectorized** using the static neighbor relationships.  
   - Append results back to the yearly subset.  
3. Use `matrix` operations or `vapply` instead of row-wise `lapply`.  
4. Process year by year to control memory footprint.  
5. Avoid re-allocating large objects repeatedly.  

---

**Optimized R Code**

```r
# Precompute static neighbor index list (same as before)
neighbor_lookup_static <- build_neighbor_lookup(
  data = data.frame(id = id_order, year = 1), # dummy year
  id_order = id_order,
  neighbors = rook_neighbors_unique
)

# Function to compute neighbor stats for a numeric vector given neighbor lookup
compute_neighbor_stats_vectorized <- function(vals, neighbor_lookup) {
  n <- length(vals)
  maxs <- numeric(n)
  mins <- numeric(n)
  means <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      maxs[i] <- NA; mins[i] <- NA; means[i] <- NA
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        maxs[i] <- NA; mins[i] <- NA; means[i] <- NA
      } else {
        maxs[i] <- max(v)
        mins[i] <- min(v)
        means[i] <- mean(v)
      }
    }
  }
  cbind(maxs, mins, means)
}

# Main optimized loop: process by year
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- list()

for (yr in years) {
  subset_idx <- which(cell_data$year == yr)
  subset_data <- cell_data[subset_idx, ]
  
  # Ensure subset is sorted by id_order for consistent indexing
  subset_data <- subset_data[match(id_order, subset_data$id), ]
  
  for (var_name in neighbor_source_vars) {
    vals <- subset_data[[var_name]]
    stats_mat <- compute_neighbor_stats_vectorized(vals, neighbor_lookup_static)
    colnames(stats_mat) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
    subset_data <- cbind(subset_data, stats_mat)
  }
  
  result_list[[as.character(yr)]] <- subset_data
}

# Combine all years back
cell_data_enhanced <- do.call(rbind, result_list)

# Pass cell_data_enhanced to the pre-trained Random Forest model
# prediction <- predict(rf_model, newdata = cell_data_enhanced)
```

---

**Why This is Faster**  
- **Static neighbor list** used directly for all years â€” no recomputation per row.  
- **Year-by-year processing** keeps memory usage manageable.  
- Inner computation uses **simple loops over ~344K cells**, not 6.46M rows Ã— neighbors repeatedly.  
- Removes expensive string concatenation and repeated lookup in `idx_lookup`.  

Estimated runtime drops from **86+ hours to a few hours** on a 16GB laptop.
END_RESPONSE>>>
