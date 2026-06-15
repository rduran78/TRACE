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
run_id: OpenAI_F08_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline rebuilds and scans neighbor relationships for **each row** and **each variable** across ~6.46M rows, using R loops and `lapply`. With 28 years and 5 variables, this results in huge repeated computations and memory overhead. Although neighbor topology is static across time, the algorithm redundantly recomputes lookups that could be reused per year. The static neighbor structure should be separated from changing values to avoid repetition.

**Optimization Strategy**  
1. **Precompute static neighbor indices once per cell** (not per row).  
2. For each year, extract data slice and apply vectorized operations (aggregate by neighbors).  
3. Use **matrix operations** instead of nested `lapply`.  
4. Store results in preallocated arrays rather than repeatedly binding rows.  
5. Use an **adjacency list or sparse matrix** for neighbor relationships to do fast aggregation.  

This reduces complexity from `O(#rows Ã— neighbors Ã— variables)` to approximately `O(#neighbors Ã— years Ã— variables)` using efficient vectorization.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Parameters
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
id_order <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# 1. Precompute static neighbor lookup (indices for each cell ID)
neighbor_indices <- lapply(rook_neighbors_unique, function(nb) match(id_order[nb], id_order))

# 2. Initialize result columns for each variable and stat
for (var_name in neighbor_source_vars) {
  for (stat in c("max", "min", "mean")) {
    col_name <- paste0(var_name, "_nbr_", stat)
    cell_data[[col_name]] <- NA_real_
  }
}

# 3. Compute stats by year in blocks, using fast lookups
for (yr in years) {
  # Slice data for this year
  year_data <- cell_data[year == yr]
  
  # For each source variable
  for (var_name in neighbor_source_vars) {
    vals <- year_data[[var_name]]
    
    # Preallocate matrices for stats
    nbr_max <- numeric(n_cells)
    nbr_min <- numeric(n_cells)
    nbr_mean <- numeric(n_cells)
    
    # Compute neighbor stats efficiently
    for (i in seq_len(n_cells)) {
      nb_idx <- neighbor_indices[[i]]
      if (length(nb_idx) == 0) {
        nbr_max[i] <- NA
        nbr_min[i] <- NA
        nbr_mean[i] <- NA
      } else {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0) {
          nbr_max[i] <- NA
          nbr_min[i] <- NA
          nbr_mean[i] <- NA
        } else {
          nbr_max[i] <- max(nb_vals)
          nbr_min[i] <- min(nb_vals)
          nbr_mean[i] <- mean(nb_vals)
        }
      }
    }
    
    # Assign back to main table
    year_idx <- which(cell_data$year == yr)
    cell_data[year_idx, (paste0(var_name, "_nbr_max")) := nbr_max]
    cell_data[year_idx, (paste0(var_name, "_nbr_min")) := nbr_min]
    cell_data[year_idx, (paste0(var_name, "_nbr_mean")) := nbr_mean]
  }
}

# At this point, cell_data has neighbor features ready for RF prediction
```

---

### **Performance Gain**
- Old approach: 86+ hours (due to full-row `lapply` over 6.46M rows Ã— 5 vars).
- New approach: processes **per year Ã— per cell Ã— per variable**, reducing redundant indexing.
- Expected runtime: drops from days to a few hours on 16GB RAM machine, since:
  - Neighbor index computed once.
  - Loops only over 344k cells Ã— 5 vars Ã— 28 years (â‰ˆ 48 million iterations) with light operations.
  - Can further optimize using `future.apply` for parallel years.

---

**Outcome**  
Preserves:
- Original neighbor-based computation logic.
- Trained Random Forest model.
- Original estimand (same numerical output as before).  
Greatly improves compute efficiency while respecting static neighbor topology and dynamic yearly variables.
END_RESPONSE>>>
