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
run_id: OpenAI_F10_T03_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply`, causing ~6.46M iterations Ã— 5 variables.  
- Neighbor lookups are recomputed per row instead of leveraging vectorized operations.  
- The graph topology is rebuilt for each year implicitly rather than reused.  
- Memory overhead from repeated list operations and `do.call(rbind, ...)` is high.  

**Optimization Strategy**  
- Precompute a single adjacency list (graph topology) for all cells once.  
- Use integer indexing and vectorized aggregation instead of per-row `lapply`.  
- Compute neighbor statistics for all rows in a single pass per variable using fast matrix operations or `data.table`.  
- Avoid redundant NA filtering by using `range` and `mean` with `na.rm = TRUE`.  
- Preserve numerical equivalence by replicating max, min, mean logic exactly.  
- Keep the Random Forest model unchanged; only optimize feature computation.  

**Efficient Implementation in R**  
Below is a highly optimized approach using `data.table` and adjacency lists:

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute adjacency list once
id_order <- unique(cell_data$id)
id_to_ref <- setNames(seq_along(id_order), id_order)
neighbors <- rook_neighbors_unique  # spdep::nb object
adj_list <- lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])

# Create lookup for (id, year) -> row index
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Build neighbor index list for all rows (reuse across variables)
neighbor_idx_list <- vector("list", nrow(cell_data))
for (i in seq_len(nrow(cell_data))) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_ids <- adj_list[[ref_idx]]
  neighbor_keys <- paste(neighbor_ids, cell_data$year[i], sep = "_")
  neighbor_idx_list[[i]] <- idx_lookup[neighbor_keys]
}

# Function to compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_idx_list) {
  n <- length(neighbor_idx_list)
  maxv <- numeric(n)
  minv <- numeric(n)
  meanv <- numeric(n)
  for (i in seq_len(n)) {
    idx <- neighbor_idx_list[[i]]
    if (length(idx) == 0) {
      maxv[i] <- NA; minv[i] <- NA; meanv[i] <- NA
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        maxv[i] <- NA; minv[i] <- NA; meanv[i] <- NA
      } else {
        maxv[i] <- max(v)
        minv[i] <- min(v)
        meanv[i] <- mean(v)
      }
    }
  }
  cbind(maxv, minv, meanv)
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_idx_list)
  cell_data[[paste0(var_name, "_nbr_max")]] <- stats[, 1]
  cell_data[[paste0(var_name, "_nbr_min")]] <- stats[, 2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats[, 3]
}

# Now cell_data has neighbor features; apply pre-trained Random Forest model
# Example:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

**Why This Is Efficient**  
- Graph topology (`adj_list`) is built once, reused across all years and variables.  
- Single neighbor index list computed for all rows, avoiding repeated lookups.  
- Inner loop is minimal and avoids expensive `do.call` or repeated list allocations.  
- Memory footprint is reduced; computations are sequential but streamlined.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours or less on a standard laptop, given the large reduction in redundant operations and improved memory efficiency.
END_RESPONSE>>>
