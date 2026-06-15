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
run_id: OpenAI_F08_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over 6.46 million rows and repeatedly subsets vectors within an R `lapply`. For each row, the `neighbor_lookup` is list-based and recomputes basic stats by scanning neighborsâ€”resulting in millions of small operations. This design scales poorly:  
- 6.46M rows Ã— 5 variables Ã— 3 stats = ~97M operations in pure R loops.  
- The approach does not exploit the fact that the neighbor structure is static across years; it repeats row-by-row for each year.  
- Memory and CPU thrash on a 16GB laptop.

---

**Optimization Strategy**  
1. **Precompute static neighbor graph (already done) and map to IDs**, but for multi-year panels, transform the problem:
   - Instead of iterating per row, compute **neighbor stats per variable-year block** using vectorized operations.
   - For each year, subset data, and calculate neighbor max, min, mean for all cells using the neighbor adjacency list, then merge back.
2. Use **matrix operations with `rowsum` and `vapply` or `parallel::mclapply`**, not row-by-row R loops.
3. Ensure **no Random Forest retraining**â€”just regenerate neighbor features and feed them into the existing model.
4. Memory-friendly: process **year by year** (28 slices) instead of all 6.46M rows at once.

---

**Optimized Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique is an nb object (spdep)
# Precompute neighbor list as integer vectors
nb <- rook_neighbors_unique
neighbor_index <- nb  # list of integer vectors
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Convert cell_data to data.table for efficiency
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute structure once
base_ids <- sort(unique(cell_data$id)) # IDs are consistent across years
id_pos <- setNames(seq_along(base_ids), base_ids)

# Build a matrix or index for fast lookup
neighbor_map <- lapply(neighbor_index, function(nbs) base_ids[nbs])

# Function to compute stats for one variable in one year
compute_neighbor_year_stats <- function(dt_year, var_name) {
  vals <- setNames(dt_year[[var_name]], dt_year$id)
  res <- vapply(neighbor_map, function(nbs) {
    neighbor_vals <- vals[nbs]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  t(res)
}

# Apply for all years and variables
result_list <- list()
years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_neighbor_year_stats(dt_year, var_name)
    var_prefix <- paste0("neighbor_", var_name, "_")
    # stats_mat rows match base_ids order
    df_stats <- data.table(
      id = base_ids,
      year = yr,
      setNames(as.data.frame(stats_mat),
               paste0(var_prefix, c("max", "min", "mean")))
    )
    result_list[[length(result_list) + 1]] <- df_stats
  }
}

# Combine all computed stats
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")),
                             result_list)

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

**Why This is Fast and Memory-Efficient**  
- Computes stats **once per neighbor group per year**, not per row repeatedly.
- Vectorized `vapply` inside each year (~344K rows) instead of 6.46M row loops.
- Processes one year at a time: ~344K rows Ã— 5 vars Ã— 28 years, fully manageable on 16GB RAM.
- Reuses static neighbor structure for every year without rebuilding.

---

**Outcome**  
- No change to the trained Random Forest.
- Preserves original estimand (neighbor max, min, mean = same formula).
- Estimated runtime drops from 86+ hours to a few hours (or less with parallelization).  
- Pipeline ready for prediction using the trained model.
END_RESPONSE>>>
