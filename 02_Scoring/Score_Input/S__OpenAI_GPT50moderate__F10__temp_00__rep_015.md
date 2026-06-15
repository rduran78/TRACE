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
run_id: OpenAI_F10_T00_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is inefficient in R.  
- Neighbor lookup is rebuilt for every variable, and aggregation is done row-by-row instead of vectorized.  
- No use of efficient data structures (e.g., `data.table`, matrix operations).  
- The graph topology is recomputed or iterated redundantly across years.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it for all variables and years.  
- Use **vectorized operations** with `data.table` or `matrixStats` instead of per-row `lapply`.  
- Represent the neighbor relationships as an adjacency list or sparse matrix and apply fast aggregation.  
- Process data in **year-wise chunks** to reduce memory overhead.  
- Avoid repeated `rbind` calls; preallocate result matrices.  
- Preserve numerical equivalence by using the same max, min, mean definitions.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data (data.table) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object
# neighbor_source_vars: c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 1. Build adjacency list once
build_adjacency <- function(id_order, rook_neighbors_unique) {
  n <- length(id_order)
  # Create sparse adjacency matrix (directed)
  i <- rep(seq_len(n), sapply(rook_neighbors_unique, length))
  j <- unlist(rook_neighbors_unique)
  adj <- sparseMatrix(i = i, j = j, x = 1, dims = c(n, n))
  adj
}

adj <- build_adjacency(id_order, rook_neighbors_unique)

# 2. Convert cell_data to data.table and index
setDT(cell_data)
setkey(cell_data, id, year)

# 3. Precompute mapping from id to row index per year
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# 4. Function to compute neighbor stats for one variable across all years
compute_neighbor_stats_fast <- function(var_name) {
  result_list <- vector("list", n_years)
  
  for (y in seq_along(years)) {
    yr <- years[y]
    # Extract values for this year in id_order
    vals <- cell_data[year == yr, ..var_name][[1]]
    # Ensure order matches id_order
    vals <- vals[match(id_order, cell_data[year == yr, id])]
    
    # Compute neighbor aggregates using sparse matrix multiplication
    # For mean: sum / count
    neighbor_sum <- as.numeric(adj %*% vals)
    neighbor_count <- rowSums(adj)
    
    # For max and min: need apply over adjacency
    # Efficient approach: iterate over adjacency list
    max_vals <- numeric(n_cells)
    min_vals <- numeric(n_cells)
    
    for (i in seq_len(n_cells)) {
      nbrs <- rook_neighbors_unique[[i]]
      if (length(nbrs) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
      } else {
        nbr_vals <- vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vals[i] <- NA
          min_vals[i] <- NA
        } else {
          max_vals[i] <- max(nbr_vals)
          min_vals[i] <- min(nbr_vals)
        }
      }
    }
    
    mean_vals <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA)
    
    result_list[[y]] <- data.table(
      id = id_order,
      year = yr,
      paste0(var_name, "_nbr_max") := max_vals,
      paste0(var_name, "_nbr_min") := min_vals,
      paste0(var_name, "_nbr_mean") := mean_vals
    )
  }
  
  rbindlist(result_list)
}

# 5. Compute for all variables and merge
neighbor_features <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
neighbor_features_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), neighbor_features)

# 6. Merge back to cell_data
cell_data <- merge(cell_data, neighbor_features_dt, by = c("id", "year"), all.x = TRUE)

# 7. Apply pre-trained Random Forest model (already loaded as rf_model)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This is Efficient**
- **Adjacency built once** and reused.
- **Sparse matrix multiplication** for sums and means (fast).
- **Year-wise processing** avoids huge memory blow-up.
- **Vectorized merging** instead of row-by-row `lapply`.
- Preserves **numerical equivalence** with original logic.

This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop.
END_RESPONSE>>>
