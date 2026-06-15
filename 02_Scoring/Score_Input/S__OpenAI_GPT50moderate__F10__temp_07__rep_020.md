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
run_id: OpenAI_F10_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics row-by-row in nested `lapply` calls.  
- The neighbor lookup is built for every row-year combination instead of reusing a sparse structure.  
- There is no vectorization; repeated indexing and list traversals create huge overhead with 6.46M rows.  

**Optimization Strategy:**  
- Represent the entire panel as a **sparse graph** using `Matrix` or `igraph`.  
- Build the adjacency matrix once using `rook_neighbors_unique` and reuse it across years.  
- Compute neighbor max, min, and mean using **matrix operations** per year (block processing) rather than row-wise loops.  
- Avoid creating per-row lists; instead, aggregate with fast vectorized operations.  
- Use `data.table` for memory-efficient joins and grouping.  
- Preserve numerical equivalence by matching original NA-handling logic.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2)
# id_order: vector of cell IDs in graph order
# rook_neighbors_unique: spdep::nb object
# pre-trained RF model: rf_model

# 1. Convert to data.table and ensure ordering
setDT(cell_data)
setkey(cell_data, id, year)

# 2. Build adjacency matrix once (binary rook adjacency)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 3. Prepare fast lookup for row indices
id_to_row <- setNames(seq_along(id_order), id_order)
cell_data[, cell_idx := id_to_row[as.character(id)]]

# 4. Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 5. Allocate output columns
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nb_max") := NA_real_]
  cell_data[, paste0(v, "_nb_min") := NA_real_]
  cell_data[, paste0(v, "_nb_mean") := NA_real_]
}

# 6. Compute neighbor stats year by year using matrix ops
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  # Subset for this year
  dt_year <- cell_data[year == yr]
  idx <- dt_year$cell_idx
  
  for (v in neighbor_source_vars) {
    vals <- dt_year[[v]]
    
    # Build numeric vector aligned to id_order
    vec <- rep(NA_real_, n_cells)
    vec[idx] <- vals
    
    # Multiply adjacency by vec to get sums (for mean) and neighbor values
    # For max and min, iterate neighbors efficiently
    # Extract neighbors only once
    nb_indices <- adj_list
    
    nb_max <- nb_min <- nb_sum <- numeric(length(idx))
    nb_count <- integer(length(idx))
    
    for (i in seq_along(idx)) {
      nbs <- nb_indices[[idx[i]]]
      if (length(nbs) > 0) {
        nb_vals <- vec[nbs]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          nb_max[i] <- max(nb_vals)
          nb_min[i] <- min(nb_vals)
          nb_sum[i] <- sum(nb_vals)
          nb_count[i] <- length(nb_vals)
        } else {
          nb_max[i] <- nb_min[i] <- nb_sum[i] <- NA_real_
        }
      } else {
        nb_max[i] <- nb_min[i] <- nb_sum[i] <- NA_real_
      }
    }
    
    # Assign back
    cell_data[year == yr, paste0(v, "_nb_max") := nb_max]
    cell_data[year == yr, paste0(v, "_nb_min") := nb_min]
    cell_data[year == yr, paste0(v, "_nb_mean") := ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)]
  }
}

# 7. Predict with pre-trained RF model (unchanged)
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Key Improvements**
- Adjacency built **once** and reused.
- Avoid per-row `lapply`; instead, loop over years and variables with **direct indexing**.
- Handles 6.46M rows efficiently on 16 GB RAM.
- Preserves NA handling and numerical equivalence.
- Complexity reduced from O(N * neighbors * vars) with heavy R overhead to streamlined loops.

**Estimated runtime:** From 86+ hours â†’ a few hours (depending on disk I/O and CPU). Further speedups possible using `Rcpp` for neighbor loops or parallelization (`future.apply`).
END_RESPONSE>>>
