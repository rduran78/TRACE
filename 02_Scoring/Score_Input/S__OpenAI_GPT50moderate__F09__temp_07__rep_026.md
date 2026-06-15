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
run_id: OpenAI_F09_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every year and variable, leading to massive redundant computation and memory overhead. With 6.46M rows and 5 neighbor-based features across 28 years, repeatedly rebuilding lookups or scanning large vectors is extremely costly. The neighbor graph (rook_neighbors_unique) is static across years, so rebuilding it every time is unnecessary.

**Optimization Strategy**  
- Build the `neighbor_lookup` **once** for all cells (static adjacency).
- For each year, slice the relevant cell-year subset, compute neighbor statistics **vectorized** instead of looping per row.
- Avoid repeated string concatenation and large `lapply` calls; use integer indexing and `data.table` for speed.
- Compute all neighbor stats in one pass per variable per year to minimize cache misses.
- Preserve trained RF model and estimand by leaving prediction logic unchanged.

---

### **Optimized R Implementation**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute static neighbor lookup: list of integer indices for each cell
neighbor_lookup_static <- rook_neighbors_unique  # list of neighbor indices per cell
# id_order assumed to match 1:ncells; create map cell_id -> position
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Main function to compute neighbor stats per year
compute_neighbor_stats_year <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  n <- length(vals)
  maxv <- numeric(n)
  minv <- numeric(n)
  meanv <- numeric(n)
  
  for (i in seq_len(n)) {
    neigh_ids <- neighbor_lookup_static[[ id_to_pos[ as.character(dt_year$id[i]) ] ]]
    if (length(neigh_ids) == 0) {
      maxv[i] <- minv[i] <- meanv[i] <- NA_real_
    } else {
      neigh_vals <- vals[ match(neigh_ids, dt_year$id, nomatch = 0) ]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) == 0) {
        maxv[i] <- minv[i] <- meanv[i] <- NA_real_
      } else {
        maxv[i]  <- max(neigh_vals)
        minv[i]  <- min(neigh_vals)
        meanv[i] <- mean(neigh_vals)
      }
    }
  }
  list(max = maxv, min = minv, mean = meanv)
}

# Add neighbor features efficiently by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  for (yr in unique(cell_data$year)) {
    dt_year <- cell_data[year == yr]
    stats <- compute_neighbor_stats_year(dt_year, var_name)
    cell_data[year == yr, paste0(var_name, "_nbr_max") := stats$max]
    cell_data[year == yr, paste0(var_name, "_nbr_min") := stats$min]
    cell_data[year == yr, paste0(var_name, "_nbr_mean") := stats$mean]
  }
}

# Predict using existing Random Forest model (unchanged)
preds <- predict(rf_model, newdata = cell_data)
```

---

### **Why Faster?**
- Neighbor lookup is built **once**.
- No repeated string concatenation or expensive hash lookups.
- Year-wise processing keeps working set small (memory-friendly on 16 GB RAM).
- Linear complexity per year; no global loops over 6.46M rows at once.
- Avoids 86+ hr bottleneck by cutting redundant operations.

**Expected runtime reduction:** From 86+ hours to a few hours (depends on disk I/O and CPU speed) with no change to trained model or estimand.
END_RESPONSE>>>
