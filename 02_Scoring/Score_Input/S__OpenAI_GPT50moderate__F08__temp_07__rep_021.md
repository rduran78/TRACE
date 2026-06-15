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
run_id: OpenAI_F08_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because it recomputes neighbor statistics for all 6.46 million rows (`cell-year`) in an unoptimized manner. The neighbor relationships across cells are static, but the code repeatedly processes them per row. For 344,208 cells Ã— 28 years, this leads to massive redundant computation and memory overhead. The approach uses `lapply` over millions of rows and repeatedly filters vectors, which is inefficient.  

---

**Optimization Strategy**  
- **Exploit Static Neighbor Structure**: Build the neighbor index **once per cell**, not per cell-year row.  
- **Process by Year in Blocks**: For each year, compute neighbor stats for all cells using fast vectorized operations.  
- **Avoid Repeated Lookups**: Use integer indexing and precomputed neighbor lists instead of repeatedly calling string-based lookups.  
- **Memory Efficiency**: Work year-by-year instead of full panel to avoid massive intermediate objects.  
- **Preserve Model and Estimand**: Compute the exact same statistics (max, min, mean) and merge them back into `cell_data` without altering values.  

---

**Optimized Working R Code**  

```r
# Precompute: neighbor list by cell index (static, from rook_neighbors_unique)
neighbor_list <- rook_neighbors_unique  # list of integer vectors, length = number of cells
n_cells <- length(neighbor_list)

# Function to compute neighbor stats for one variable in one year
compute_year_stats <- function(vals, neighbor_list) {
  # vals: numeric vector of length n_cells for one year
  n <- length(vals)
  max_vec <- numeric(n)
  min_vec <- numeric(n)
  mean_vec <- numeric(n)
  
  for (i in seq_len(n)) {
    nb <- neighbor_list[[i]]
    if (length(nb) == 0) {
      max_vec[i] <- NA
      min_vec[i] <- NA
      mean_vec[i] <- NA
    } else {
      nb_vals <- vals[nb]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) {
        max_vec[i] <- NA
        min_vec[i] <- NA
        mean_vec[i] <- NA
      } else {
        max_vec[i] <- max(nb_vals)
        min_vec[i] <- min(nb_vals)
        mean_vec[i] <- mean(nb_vals)
      }
    }
  }
  cbind(max_vec, min_vec, mean_vec)
}

# Main optimized loop
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

years <- sort(unique(cell_data$year))
cell_ids <- sort(unique(cell_data$id))
stopifnot(length(cell_ids) == n_cells)

# Prepare result storage
feature_names <- unlist(lapply(neighbor_source_vars, function(v) paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))))

# Split data for efficiency
library(data.table)
dt <- as.data.table(cell_data)
setkey(dt, id, year)

for (var_name in neighbor_source_vars) {
  # Create new columns
  for (suffix in c("_nb_max", "_nb_min", "_nb_mean")) {
    dt[[paste0(var_name, suffix)]] <- NA_real_
  }
}

# Process year by year
for (yr in years) {
  sub <- dt[year == yr]
  # Ensure order matches neighbor_list order
  sub <- sub[match(cell_ids, id)]
  
  for (var_name in neighbor_source_vars) {
    vals <- sub[[var_name]]
    stats <- compute_year_stats(vals, neighbor_list)
    dt[year == yr, paste0(var_name, "_nb_max") := stats[, 1]]
    dt[year == yr, paste0(var_name, "_nb_min") := stats[, 2]]
    dt[year == yr, paste0(var_name, "_nb_mean") := stats[, 3]]
  }
}

cell_data <- as.data.frame(dt)
```

---

**Key Improvements**  
- **Static neighbor structure used once**: No repeated string concatenation or hash lookups.  
- **Year-block processing**: Reduces memory footprint and repetitive operations.  
- **Vectorized assignment**: Directly updates `data.table` columns for speed.  
- **Preserves all original values and estimands**.  

**Expected Performance**  
From 86+ hours â†’ likely under 1 hour on a 16GB machine because complexity drops from `O(N*T*k)` string-based lookups to efficient numeric indexing and year-wise loops.  

This approach **keeps the pre-trained Random Forest model intact** and **produces exactly the same numerical neighbor stats** as before.
END_RESPONSE>>>
