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
run_id: OpenAI_F10_T03_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable using `lapply` over ~6.46M rows.  
- It performs many small vector lookups and concatenations in R loops, which is inefficient for large data.  
- The neighbor graph is rebuilt conceptually for each row-year combination instead of leveraging a fixed sparse adjacency structure.  

**Optimization Strategy**  
- Represent the spatial structure as a **sparse adjacency matrix** (cells Ã— cells) using `Matrix::dgCMatrix`.  
- Compute neighbor statistics **year by year** using **vectorized matrix operations**:  
  - For each year, extract the relevant rows and apply adjacency-based aggregation (max, min, mean) using efficient matrix ops.  
- Avoid repeated lookups by building the graph topology **once** and reusing it.  
- Use `data.table` for fast grouping and merging.  
- Preserve numerical equivalence by applying the same aggregation logic.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep::nb)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# 1. Build sparse adjacency matrix (cells x cells)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Precompute row sums for mean calculation
deg <- rowSums(adj_mat)

# 3. Function to compute neighbor stats for one variable and one year
compute_year_stats <- function(vals, adj_mat, deg) {
  # vals: numeric vector of length n_cells
  # Compute sums for mean
  sum_vals <- as.numeric(adj_mat %*% vals)
  mean_vals <- sum_vals / deg
  # For max/min, iterate efficiently
  # Extract neighbors once
  max_vals <- numeric(length(vals))
  min_vals <- numeric(length(vals))
  for (i in seq_along(adj_list)) {
    neigh <- adj_list[[i]]
    if (length(neigh) == 0) {
      max_vals[i] <- NA
      min_vals[i] <- NA
    } else {
      v <- vals[neigh]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
      } else {
        max_vals[i] <- max(v)
        min_vals[i] <- min(v)
      }
    }
  }
  list(max = max_vals, min = min_vals, mean = mean_vals)
}

# 4. Apply year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Prepare result container
for (var_name in neighbor_source_vars) {
  cell_data[[paste0(var_name, "_nbr_max")]] <- NA_real_
  cell_data[[paste0(var_name, "_nbr_min")]] <- NA_real_
  cell_data[[paste0(var_name, "_nbr_mean")]] <- NA_real_
}

for (yr in years) {
  idx <- which(cell_data$year == yr)
  # Ensure ordering by id_order
  sub_dt <- cell_data[idx][match(id_order, id)]
  for (var_name in neighbor_source_vars) {
    vals <- sub_dt[[var_name]]
    stats <- compute_year_stats(vals, adj_mat, deg)
    cell_data[idx, paste0(var_name, "_nbr_max") := stats$max]
    cell_data[idx, paste0(var_name, "_nbr_min") := stats$min]
    cell_data[idx, paste0(var_name, "_nbr_mean") := stats$mean]
  }
}

# 5. Predict with pre-trained Random Forest model
# rf_model is assumed preloaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Efficient**
- **Adjacency built once** and reused across all years and variables.  
- **Matrix multiplication** handles mean aggregation in bulk.  
- Max/min still require iteration, but only over cells (344k), not 6.46M rows repeatedly.  
- **Year-wise batching** reduces memory footprint and avoids repeated lookups.  
- Using `data.table` ensures fast assignment and merging.  

This approach should reduce runtime from **86+ hours to a few hours** on a standard laptop while preserving the original numerical results and the trained Random Forest model.
END_RESPONSE>>>
