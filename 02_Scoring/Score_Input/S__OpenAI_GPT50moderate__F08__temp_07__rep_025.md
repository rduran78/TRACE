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
run_id: OpenAI_F08_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours estimate) because:

- It recomputes neighbor stats (`max`, `min`, `mean`) for **each row** (6.46M rows) and **each variable** (5 vars), repeatedly iterating through neighbor lookups.
- Neighbor relationships are **static across years**, but the code treats them dynamically per row.
- `compute_neighbor_stats` is applied on the entire dataset instead of year-by-year grouping, causing redundant memory scans.

**Optimization Strategy**  
- **Precompute a static neighbor index**: Use `id`-based neighbor relationships once, not per row.
- **Process data year-by-year**: For each year, compute neighbor stats using vectorized operations.
- Use **data.table** for fast grouping and joins.
- Eliminate per-row `lapply` by leveraging matrix operations (e.g., `rowsum` or `vapply`).
- Keep Random Forest model untouched and preserve original numerical output.

---

### **Optimized Workflow**
1. Precompute a static neighbor lookup for `id` (not `id_year`).
2. For each year:
   - Subset `data` for that year.
   - For each variable, compute neighbor stats using the static lookup.
   - Append results back efficiently.
3. Bind results across all years.

---

### **Working R Code**

```r
library(data.table)

compute_neighbor_features <- function(dt, id_order, neighbors, vars) {
  setDT(dt)
  
  # Precompute static neighbor index for IDs
  id_to_idx <- setNames(seq_along(id_order), id_order)
  
  # Convert neighbor list to an integer index matrix
  neighbor_list <- lapply(seq_along(neighbors), function(i) id_order[neighbors[[i]]])
  
  # Prepare storage for results
  result_list <- vector("list", length(vars))
  
  # Process year by year
  years <- unique(dt$year)
  
  for (yr in years) {
    cat("Processing year:", yr, "\n")
    subset_idx <- dt$year == yr
    dt_year <- dt[subset_idx]
    
    # Build a quick lookup for variable values by ID
    val_lookup <- setNames(seq_len(nrow(dt_year)), dt_year$id)
    
    for (v in seq_along(vars)) {
      var_name <- vars[v]
      vals <- dt_year[[var_name]]
      
      # Compute neighbor stats for each cell
      neighbor_stats <- t(vapply(seq_along(id_order), function(i) {
        nbrs <- neighbor_list[[i]]
        nbr_idx <- val_lookup[nbrs]
        nbr_idx <- nbr_idx[!is.na(nbr_idx)]
        if (length(nbr_idx) == 0) return(c(NA, NA, NA))
        nbr_vals <- vals[nbr_idx]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) return(c(NA, NA, NA))
        c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }, FUN.VALUE = numeric(3)))
      
      colnames(neighbor_stats) <- c(paste0(var_name, "_nbr_max"),
                                    paste0(var_name, "_nbr_min"),
                                    paste0(var_name, "_nbr_mean"))
      
      # Attach back to dt_year
      dt_year <- cbind(dt_year, neighbor_stats)
    }
    
    # Update original dt
    dt[subset_idx] <- dt_year
  }
  
  return(dt)
}

# Usage:
# cell_data <- compute_neighbor_features(cell_data, id_order, rook_neighbors_unique,
#                                        vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"))
```

---

### **Key Improvements**
- No per-row `lapply` over 6.46M rows. Instead, neighbor calculations are done once per `id` per year.
- Memory-efficient: Processes one year at a time.
- Complexity reduced from **O(rows Ã— neighbors)** to **O(years Ã— ids Ã— neighbors)** (orders of magnitude faster).
- Preserves the original estimand and Random Forest model.

**Expected runtime**: From 86+ hours â†’ a few hours (depending on I/O and CPU).
END_RESPONSE>>>
