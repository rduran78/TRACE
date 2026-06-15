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
run_id: OpenAI_F10_T00_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over 6.46M rows.  
- Neighbor lookups are recomputed for every variable, causing redundant work.  
- The approach is not vectorized and does not leverage efficient data structures.  
- Memory overhead from repeated list operations and `do.call(rbind, ...)` is high.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it across all variables and years.  
- Represent the panel as a matrix where rows = cell-year, columns = variables.  
- Use **vectorized aggregation** with `rowsum` or `data.table` for speed.  
- Precompute a sparse adjacency matrix (cell Ã— cell) and apply it year by year.  
- Compute max, min, mean in a single pass per variable-year using fast grouping.  
- Avoid loops over 6.46M rows; instead, loop over 28 years and 5 variables (140 iterations).  
- Use `Matrix` for sparse adjacency and `data.table` for efficient joins.  

---

### **Efficient Implementation in R**

```r
library(data.table)
library(Matrix)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: spdep::nb object
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Build adjacency matrix (sparse)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Map cell IDs to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Add index column for fast join
cell_data[, idx := id_to_idx[as.character(id)]]

# Variables to process
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result columns
for (v in neighbor_vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  sub <- cell_data[year == yr]
  idx <- sub$idx
  
  for (v in neighbor_vars) {
    vals <- sub[[v]]
    
    # Compute neighbor stats using adjacency
    # Multiply adjacency by vals to get sums and counts
    # For max/min, iterate neighbors efficiently
    # Extract neighbor indices for each cell
    nbr_idx_list <- adj_list
    
    max_vec <- numeric(length(idx))
    min_vec <- numeric(length(idx))
    mean_vec <- numeric(length(idx))
    
    for (i in seq_along(idx)) {
      nbrs <- nbr_idx_list[[idx[i]]]
      if (length(nbrs) == 0) {
        max_vec[i] <- NA
        min_vec[i] <- NA
        mean_vec[i] <- NA
      } else {
        nbr_vals <- vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vec[i] <- NA
          min_vec[i] <- NA
          mean_vec[i] <- NA
        } else {
          max_vec[i] <- max(nbr_vals)
          min_vec[i] <- min(nbr_vals)
          mean_vec[i] <- mean(nbr_vals)
        }
      }
    }
    
    # Assign back
    cell_data[year == yr, paste0(v, "_nbr_max") := max_vec]
    cell_data[year == yr, paste0(v, "_nbr_min") := min_vec]
    cell_data[year == yr, paste0(v, "_nbr_mean") := mean_vec]
  }
}

# At this point, cell_data has neighbor features added
# Apply pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This is Efficient**
- **Adjacency built once** and reused across all years and variables.  
- **Year-wise processing** reduces memory footprint (only ~344k rows at a time).  
- **Direct index-based lookup** avoids repeated string concatenation and hashing.  
- **Single pass per variable-year** instead of per row-variable combination.  

**Expected runtime**:  
- 28 years Ã— 5 variables = 140 iterations over ~344k rows each.  
- On a 16 GB laptop, this should complete in **a few hours** instead of 86+.  

**Numerical equivalence**:  
- Logic for max, min, mean matches original implementation exactly.  
- No retraining of Random Forest; predictions remain consistent.
END_RESPONSE>>>
