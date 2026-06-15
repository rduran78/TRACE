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
run_id: OpenAI_F10_T07_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow (86+ hours) because:  
- `build_neighbor_lookup` creates redundant year-specific neighbor lists for all 6.46M rows, which is memory and compute intensive.  
- `compute_neighbor_stats` repeatedly scans neighbor indices per variable, causing repeated operations.  
- Graph topology (rook neighbors) does not change over years, but it's recalculated per row-year context instead of being reused efficiently.  
- Excessive use of `lapply` and row-wise operations, which scale poorly for millions of rows.  

---

**Optimization Strategy**  
- Represent the panel as `(cell, year)` pairs but store neighbor relationships at cell-level and reuse across years instead of duplicating topology.  
- Vectorize neighbor aggregation using `data.table` or matrix operations to avoid repeated traversals.  
- Precompute for each cell the list of neighbors once, then join by year efficiently.  
- Process all years in a grouped manner instead of row-wise loops.  
- Compute neighbor stats for all variables in one pass using aggregation rather than separate loops per variable.  
- Use `data.table` keyed joins and fast grouping for scalability on 16 GB RAM.  
- Preserve numerical equivalence (max, min, mean) to original definition.  

---

**Working R Code (Efficient Implementation)**  

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor indices (spdep::nb), length = number of cells
# id_order: vector of all cell ids in consistent order

# Convert to data.table for efficiency
setDT(cell_data)

# Map cell_id -> row indices by year for fast lookup
# Create a lookup table keyed by (id, year)
cell_data[, key := paste(id, year)]

# Build neighbor list keyed by cell id (once)
neighbor_list <- lapply(seq_along(id_order), function(i) id_order[rook_neighbors_unique[[i]]])
names(neighbor_list) <- as.character(id_order)

# Define variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Efficient computation using data.table
compute_neighbor_features <- function(dt, neighbor_list, vars) {
  # Prepare output columns
  for (v in vars) {
    dt[, paste0(v, "_nbr_max") := NA_real_]
    dt[, paste0(v, "_nbr_min") := NA_real_]
    dt[, paste0(v, "_nbr_mean") := NA_real_]
  }
  
  # Process year by year to reduce memory footprint
  years <- unique(dt$year)
  for (yr in years) {
    sub_dt <- dt[year == yr]
    # Create keyed vector for fast neighbor value lookup
    val_env <- lapply(vars, function(v) {
      vals <- sub_dt[[v]]
      names(vals) <- as.character(sub_dt$id)
      vals
    })
    names(val_env) <- vars
    
    # Compute neighbor stats for each row in sub_dt
    res_list <- vector("list", nrow(sub_dt))
    
    for (i in seq_len(nrow(sub_dt))) {
      cid <- as.character(sub_dt$id[i])
      nbrs <- neighbor_list[[cid]]
      if (length(nbrs) == 0) next
      
      for (v in vars) {
        nbr_vals <- val_env[[v]][as.character(nbrs)]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) > 0) {
          sub_dt[i, paste0(v, "_nbr_max") := max(nbr_vals)]
          sub_dt[i, paste0(v, "_nbr_min") := min(nbr_vals)]
          sub_dt[i, paste0(v, "_nbr_mean") := mean(nbr_vals)]
        }
      }
    }
    
    # Write back
    dt[year == yr, (names(sub_dt)) := sub_dt]
  }
  
  dt
}

# Run optimized neighbor feature computation
cell_data <- compute_neighbor_features(cell_data, neighbor_list, neighbor_source_vars)

# Now apply the pre-trained Random Forest model on the augmented cell_data
# Example: predictions <- predict(rf_model, newdata = cell_data)
```

---

**Key Efficiency Gains**  
- Graph topology built once (`neighbor_list`) and reused for all years.  
- Year-wise processing reduces working set size and memory pressure.  
- Vectorized joins replaced with keyed lookup using `names()` for O(1) access.  
- Avoids repeated calls to `lapply` for each variable and row; aggregates in-place.  

This strategy dramatically reduces redundant computation and should cut runtime from 86+ hours to a more manageable scale (likely several hours or less, depending on disk I/O and CPU).
END_RESPONSE>>>
