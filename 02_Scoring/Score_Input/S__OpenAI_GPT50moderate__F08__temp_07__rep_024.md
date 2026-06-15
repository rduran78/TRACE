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
run_id: OpenAI_F08_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor statistics (max, min, mean) for each variable across all 6.46M rows using `neighbor_lookup` that includes year-specific keys. This creates massive redundant computations because the neighbor structure is static, but the lookup is rebuilt for every row-year combination. This leads to high memory usage and extreme runtime (86+ hours).  

**Optimization Strategy**  
- Exploit the static neighbor structure: Build neighbor lookup **once per cell ID**, independent of year.  
- For each year, subset data and compute neighbor stats using precomputed cell-to-neighbor mapping.  
- Avoid repeated string concatenation; use integer-based indices for efficiency.  
- Use `data.table` for fast subsetting and assignment.  
- Compute stats in blocks (year-wise) and append to the main dataset to preserve the Random Forest pipeline.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for efficiency
cell_data <- as.data.table(cell_data)

# Precompute neighbor lookup: cell_id -> neighbor_ids
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]]  # neighbor cell IDs for this cell
  })
}

neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one year and one variable
compute_neighbor_stats_year <- function(dt_year, var_name, neighbor_lookup_static, id_to_row) {
  vals <- dt_year[[var_name]]
  n <- nrow(dt_year)
  res <- matrix(NA_real_, n, 3)  # columns: max, min, mean

  for (i in seq_len(n)) {
    neigh_ids <- neighbor_lookup_static[[ dt_year$id[i] ]]
    neigh_idx <- id_to_row[neigh_ids]
    neigh_idx <- neigh_idx[!is.na(neigh_idx)]
    if (length(neigh_idx) > 0) {
      nv <- vals[neigh_idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  res
}

# Map cell_id -> row index for convenience
# Within each year, create mapping for fast access
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (var_name in neighbor_source_vars) {
  cell_data[, paste0(var_name, "_nbr_max") := NA_real_]
  cell_data[, paste0(var_name, "_nbr_min") := NA_real_]
  cell_data[, paste0(var_name, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  id_to_row <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  for (var_name in neighbor_source_vars) {
    stats <- compute_neighbor_stats_year(dt_year, var_name, neighbor_lookup_static, id_to_row)
    cell_data[year == yr, paste0(var_name, "_nbr_max") := stats[, 1]]
    cell_data[year == yr, paste0(var_name, "_nbr_min") := stats[, 2]]
    cell_data[year == yr, paste0(var_name, "_nbr_mean") := stats[, 3]]
  }
}

# The Random Forest prediction step remains unchanged:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This Works and Improves Performance**
- **Static neighbor lookup** reduces repeated work (built once for 344K cells instead of 6.46M rows).  
- **Year-wise block processing** avoids constructing giant lookups per row-year combination.  
- **Matrix-based stats computation** reduces overhead.  
- **Memory-efficient assignment with data.table** handles large data on 16GB RAM without blowing up.  
Expected runtime improvement: **From 86+ hours to a few hours** on a standard laptop.  

This preserves the trained Random Forest model and the original estimand while drastically optimizing computation.
END_RESPONSE>>>
