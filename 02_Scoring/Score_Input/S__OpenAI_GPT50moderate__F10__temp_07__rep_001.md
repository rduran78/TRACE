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
run_id: OpenAI_F10_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
1. **Repeated R loops and list processing**: For ~6.46M rows, `lapply` iterates for each row and each variable, causing massive interpreter overhead.  
2. **Neighbor lookup recomputed for each function call**: Though `neighbor_lookup` is precomputed, the aggregation for 5 variables still re-traverses millions of lists.  
3. **Inefficient memory usage**: Each `compute_neighbor_stats` call allocates intermediate lists and matrices repeatedly.  
4. **No vectorization or graph-based optimization**: The rook neighbor structure is fixed and reused across years, but computation is not leveraging this.  

---

### **Optimization Strategy**
- Represent the full panel as a **graph adjacency list or sparse matrix** using `Matrix` or `igraph`.  
- Precompute a **mapping vector** from (cell, year) â†’ row index once.  
- Use **vectorized matrix ops** instead of `lapply` for neighbor aggregation:
  - Build a sparse adjacency matrix `A` for cells (n_cells x n_cells).
  - For each year, extract the row block and compute `A %*% values` for sums, and use fast group operations for min/max.
- Process **one year at a time** to control memory (28 chunks).
- Use **data.table** for fast joins and column updates.
- Compute all five variables in one pass per year.
- Append results and write back to the main dataset.
- Keep results numerically identical (max, min, mean ignoring `NA`).

---

### **Efficient R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2), rook_neighbors_unique, id_order
setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Build sparse adjacency matrix once ----
nb_list <- rook_neighbors_unique
rows <- rep(seq_along(nb_list), lengths(nb_list))
cols <- unlist(nb_list, use.names = FALSE)
A <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# ---- Helper to compute neighbor stats for one year ----
compute_stats_year <- function(dt_year) {
  idx <- match(dt_year$id, id_order)  # map rows to adjacency
  res_list <- vector("list", length(neighbor_vars))
  
  for (v in neighbor_vars) {
    vals <- dt_year[[v]]
    vals[is.na(vals)] <- NA_real_  # keep NA
    # Compute mean: sum / count
    sum_neighbors <- as.numeric(A %*% vals)
    count_neighbors <- as.numeric(A %*% (!is.na(vals)))
    
    mean_neighbors <- sum_neighbors / ifelse(count_neighbors == 0, NA, count_neighbors)
    
    # For min/max: use apply over adjacency indices
    # Efficient approach: iterate rows of A
    max_vals <- numeric(n_cells)
    min_vals <- numeric(n_cells)
    
    for (i in seq_len(n_cells)) {
      nbr_idx <- nb_list[[i]]
      if (length(nbr_idx) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
      } else {
        nbr_vals <- vals[nbr_idx]
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
    
    res_list[[v]] <- data.table(
      paste0(v, "_nbr_max") = max_vals[idx],
      paste0(v, "_nbr_min") = min_vals[idx],
      paste0(v, "_nbr_mean") = mean_neighbors[idx]
    )
  }
  
  cbind(dt_year, do.call(cbind, res_list))
}

# ---- Process by year to manage memory ----
result_list <- vector("list", length(years))
for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  result_list[[i]] <- compute_stats_year(dt_year)
}

final_data <- rbindlist(result_list)

# final_data now has neighbor-derived features. Use the pre-trained RF model:
preds <- predict(pretrained_rf_model, newdata = final_data)
```

---

### **Key Improvements**
- **Graph built once** (`A` and `nb_list` reused across years).
- **Chunked processing** (per year) prevents memory overload.
- **Sparse matrix multiplication** accelerates mean calculation.
- **Pre-indexing** avoids repeated lookups.
- Preserves **numerical equivalence** with original stats.

---

**Expected Runtime**:  
This approach moves from 86+ hours to **~1â€“2 hours** on a 16 GB laptop due to vectorization, adjacency reuse, and chunked processing.
END_RESPONSE>>>
