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
run_id: OpenAI_F08_T07_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because `compute_neighbor_stats()` recomputes neighbor-based max, min, and mean for each variable across all 6.46M rows using `lapply`. This results in *O(N Ã— neighbors Ã— vars)* repeated lookups and calculations, where `N â‰ˆ 6.46M`. The neighbor structure is static across years, but the algorithm rebuilds neighbor relationships per row-year repeatedly. Memory access is also inefficient due to repeated indexing in large lists.  

**Optimization Strategy**  
1. **Leverage Static Neighbors**: Precompute a neighbor index map **only once** at the cell level, not per cell-year, since neighbors do not change across years.
2. **Vectorize by Year**: For each year, compute neighbor stats in bulk using matrix operations or `data.table` grouping rather than row-wise loops.
3. **Avoid Nested Loops**: Replace `lapply(row_ids, ...)` with vectorized joins or grouped aggregation.
4. **Efficient Storage**: Use `data.table` keyed by `id` and `year` for fast joins and aggregation.
5. **Parallelization (optional)**: If needed, parallelize by year using `future.apply` or `parallel::mclapply`.

This approach reduces complexity from repeated row-year lookups to a yearly grouped operation over ~28 subsets of ~344k rows each, which is feasible on a 16â€¯GB laptop.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute static neighbor map: list of integer neighbor IDs per cell
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

setkey(neighbor_dt, id)

# Function to compute neighbor stats for one variable and one year
compute_neighbor_stats_year <- function(dt_year, var_name) {
  # Join dt_year to neighbors (left join on id)
  joined <- neighbor_dt[dt_year, on = .(id), allow.cartesian = TRUE]
  
  # joined has columns: id, neighbor_id, year, var_name
  setnames(joined, var_name, "target_val")
  
  # Compute neighbor stats grouped by id
  stats <- joined[!is.na(target_val),
                  .(max_val = max(target_val),
                    min_val = min(target_val),
                    mean_val = mean(target_val)),
                  by = .(id)]
  
  # Merge back to dt_year
  setnames(stats, c("max_val","min_val","mean_val"),
           paste0(var_name, c("_nb_max","_nb_min","_nb_mean")))
  
  dt_year[stats, on = .(id)]
}

# Main computation: loop over years and variables
years <- sort(unique(cell_data$year))
vars  <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Split by year for memory efficiency
result_list <- vector("list", length(years))
names(result_list) <- years

for (yr in years) {
  dt_year <- cell_data[year == yr]
  for (var_name in vars) {
    dt_year <- compute_neighbor_stats_year(dt_year, var_name)
  }
  result_list[[as.character(yr)]] <- dt_year
}

# Combine all years back
cell_data <- rbindlist(result_list)
```

---

### **Why This is Faster**
- **Static neighbor map** avoids rebuilding per-row neighbor relationships.
- **Year-level batching** reduces memory overhead and improves cache locality.
- **data.table joins and grouped aggregation** are highly optimized in C.
- Complexity drops from `O(N Ã— neighbors Ã— vars)` row-wise operations to ~28 Ã— 5 grouped aggregations.

**Expected Runtime**: Hours reduced from 86+ to roughly 1â€“2 hours on a 16â€¯GB laptop, depending on I/O and disk speed.

**Preserves**:
- Original numerical estimands (max, min, mean).
- Pretrained Random Forest usage (only feature computation changed).

This design meets all constraints with a massive performance gain.
END_RESPONSE>>>
