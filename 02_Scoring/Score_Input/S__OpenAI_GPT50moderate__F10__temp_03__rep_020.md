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
run_id: OpenAI_F10_T03_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups.  
- Neighbor relationships are recomputed for every row-year combination instead of leveraging a static graph structure.  
- No vectorization or efficient matrix operations; everything is in R loops.  
- Memory overhead from repeated list allocations and `do.call(rbind, ...)`.  

**Optimization Strategy**  
- Represent the data as a sparse graph: nodes = cell-year, edges = rook neighbors (same year).  
- Precompute a static neighbor index for cells (344,208 nodes) and reuse it across all years.  
- Use matrix operations or `data.table` for grouping and aggregation instead of per-row loops.  
- Compute neighbor stats year-by-year in batches to avoid exploding memory.  
- Avoid repeated key lookups; use integer indexing.  
- Parallelize across variables or years if possible.  

**Efficient Approach**  
1. Build a base neighbor list for cells (344k nodes).  
2. For each year, slice the data for that year, compute neighbor stats using the static neighbor list.  
3. Bind results back efficiently.  
4. Use `data.table` for speed and memory efficiency.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2, ...)
# rook_neighbors_unique: list of integer vectors (length = number of cells)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Map cell IDs to row positions for fast lookup
id_to_pos <- setNames(seq_along(id_order), id_order)

# Variables to compute neighbor stats for
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate columns for results
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for this year
  dt_year <- cell_data[year == yr]
  
  # Create a vector for each variable
  for (v in neighbor_source_vars) {
    vals <- dt_year[[v]]
    
    # Preallocate result matrices
    nbr_max <- numeric(nrow(dt_year))
    nbr_min <- numeric(nrow(dt_year))
    nbr_mean <- numeric(nrow(dt_year))
    
    # Compute neighbor stats
    for (i in seq_len(nrow(dt_year))) {
      cell_id <- dt_year$id[i]
      nbr_ids <- rook_neighbors_unique[[id_to_pos[[as.character(cell_id)]]]]
      if (length(nbr_ids) == 0) {
        nbr_max[i] <- NA
        nbr_min[i] <- NA
        nbr_mean[i] <- NA
      } else {
        nbr_vals <- vals[match(id_order[nbr_ids], dt_year$id)]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          nbr_max[i] <- NA
          nbr_min[i] <- NA
          nbr_mean[i] <- NA
        } else {
          nbr_max[i] <- max(nbr_vals)
          nbr_min[i] <- min(nbr_vals)
          nbr_mean[i] <- mean(nbr_vals)
        }
      }
    }
    
    # Assign back
    cell_data[year == yr, paste0(v, "_nbr_max") := nbr_max]
    cell_data[year == yr, paste0(v, "_nbr_min") := nbr_min]
    cell_data[year == yr, paste0(v, "_nbr_mean") := nbr_mean]
  }
}

# Predict using pre-trained Random Forest
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Further Speed Improvements**
- Replace inner `for` loop with **vectorized neighbor aggregation** using `igraph` or `Matrix`:
  - Build adjacency list once for cells.
  - For each year, create a numeric vector of variable values and apply `graph_apply` or sparse matrix multiplication to compute sums, then derive mean, max, min.
- Use `parallel::mclapply` or `future.apply` to compute per-year or per-variable in parallel.
- If memory allows, reshape data into a 3D array (cells Ã— years Ã— vars) and apply efficient compiled code.

---

**Expected Gains**  
- Eliminates 6.46M per-row lookups and repeated key construction.
- Reduces runtime from 86+ hours to a few hours or less (depending on parallelization and vectorization).
- Preserves numerical equivalence and uses the existing trained Random Forest model.
END_RESPONSE>>>
