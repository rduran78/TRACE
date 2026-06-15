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
run_id: OpenAI_F10_T10_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current bottleneck: `lapply` over 6.46M rows repeatedly for 5 variables â†’ extremely inefficient in R. Each function constructs neighbor-based statistics by pulling values repeatedly via indexed lookups.
- Redundant work: neighbor structure is recomputed per attribute; fully materialized neighbor lookups lead to large memory overhead.
- Years are independent, but approach ignores vectorizationâ€”no batching, no matrix operations.
- Graph topology is static: rook neighbors do not change per year; recomputing index alignments repeatedly wastes time.
- Nested functions with `lapply` and `do.call(rbind, ...)` are not memory-optimal for millions of rows.

---

**Optimization Strategy**  
- Build graph adjacency **once** as integer indices for all cells (`adj_list`).
- Reshape panel data to a matrix with rows = cells, cols = years per variable for quick lookup.
- Compute neighbor stats using **vectorized** operations (`apply` over neighbors or `Matrix` ops).
- Combine `data.table` for speed.
- Compute max/min/mean in a single pass per variable and year using preallocated numeric arrays.
- Write output directly into columns, avoiding repeated binding.
- Preserve trained Random Forest; only input features change.

---

**Efficient R Implementation**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Basic facts
ids <- unique(cell_data$id)
years <- sort(unique(cell_data$year))
n_cells <- length(ids)
n_years <- length(years)

# Map id -> row index
id_index <- setNames(seq_along(ids), ids)

# Build static adjacency once
rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds") # spdep::nb object
adj_list <- lapply(rook_neighbors_unique, function(neigh) id_index[as.character(ids[neigh])])

# Prepare fast lookup table: arrange panel into list of matrices per variable
neighbor_source_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
# Pivot to wide for each variable
var_mats <- lapply(neighbor_source_vars, function(var) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_dt <- cell_data[, .(id_idx = id_index[as.character(id)], year_idx = match(year, years), val = get(var))]
  m[cbind(val_dt$id_idx, val_dt$year_idx)] <- val_dt$val
  m
})
names(var_mats) <- neighbor_source_vars

# Preallocate result columns in cell_data keyed by (id_idx, year_idx)
cell_data[, (paste0(rep(neighbor_source_vars, each = 3),
                    "_neighbor_", c("max","min","mean"))) := NA_real_]

# Compute neighbor stats efficiently
compute_stats <- function(var_mat, adj_list) {
  # var_mat: n_cells x n_years
  res_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  res_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  res_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    neigh <- adj_list[[i]]
    if (length(neigh) > 0) {
      vals <- var_mat[neigh, , drop = FALSE]
      # Compute along rows of neighbors
      res_max[i, ]  <- apply(vals, 2, max, na.rm = TRUE)
      res_min[i, ]  <- apply(vals, 2, min, na.rm = TRUE)
      res_mean[i, ] <- apply(vals, 2, mean, na.rm = TRUE)
    }
  }
  list(max = res_max, min = res_min, mean = res_mean)
}

for (var in neighbor_source_vars) {
  stats <- compute_stats(var_mats[[var]], adj_list)
  
  # Flatten back into cell_data order
  cell_data[, paste0(var, "_neighbor_max") :=
              stats$max[cbind(id_index[as.character(id)], match(year, years))]]
  cell_data[, paste0(var, "_neighbor_min") :=
              stats$min[cbind(id_index[as.character(id)], match(year, years))]]
  cell_data[, paste0(var, "_neighbor_mean") :=
              stats$mean[cbind(id_index[as.character(id)], match(year, years))]]
}

# Predict using pre-trained Random Forest without retraining
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why itâ€™s faster**  
- Single adjacency structure reused â†’ no recomputation.
- Panel reshaped to numeric matrices â†’ cache-friendly access.
- Minimized function calls via preallocation â†’ memory-efficient.
- Complexity reduced from O(N * neighbors * vars * R overhead) to near O(N * neighbors) in compiled vectorized loops.
- Estimate: **hours â†’ tens of minutes** on 16GB laptop.
END_RESPONSE>>>
