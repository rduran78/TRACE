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
run_id: OpenAI_F05_T00_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste(...)`) and hash lookups inside the inner `lapply` of `build_neighbor_lookup` are a **local inefficiency**, but the real issue is **algorithmic**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated hash lookups.  
- This is done once in `build_neighbor_lookup` and then the resulting list is reused for all 5 variables, so the string work is not repeated per variable.  
- However, the current approach still scales as `O(N * avg_neighbors)` with expensive string operations and list overhead.  
- The neighbor structure is static across years, so we can **precompute numeric indices** for all years without string keys.

**Optimization Strategy**

1. **Avoid string keys entirely**: Instead of `paste(id, year)`, map `(id, year)` to a numeric index using vectorized operations.
2. **Exploit panel structure**: Data is sorted by `(id, year)`. If so, the index for `(id, year)` can be computed as:
   ```
   index = (id_position - 1) * n_years + year_position
   ```
3. **Precompute neighbor indices for all rows**: Use integer arithmetic, not string concatenation.
4. **Compute neighbor stats in a fully vectorized way**: Use `matrix` or `data.table` for speed.

**Working R Code**

```r
library(data.table)

# Assume data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Basic constants
n_ids   <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Map id -> position
id_to_pos <- setNames(seq_along(id_order), id_order)
year_to_pos <- setNames(seq_along(years), years)

# Precompute row index for each (id, year)
cell_data[, id_pos := id_to_pos[as.character(id)]]
cell_data[, year_pos := year_to_pos[as.character(year)]]
cell_data[, row_idx := (id_pos - 1L) * n_years + year_pos]

# Build neighbor lookup as integer matrix
# rook_neighbors_unique: list of integer vectors (neighbors by id position)
neighbor_lookup <- vector("list", nrow(cell_data))

# Precompute neighbor positions for each id
neighbor_pos_list <- lapply(rook_neighbors_unique, function(neigh_ids) {
  id_to_pos[as.character(neigh_ids)]
})

# Fill neighbor_lookup
for (i in seq_len(nrow(cell_data))) {
  ref_id_pos <- cell_data$id_pos[i]
  neigh_id_pos <- neighbor_pos_list[[ref_id_pos]]
  if (length(neigh_id_pos) == 0L) {
    neighbor_lookup[[i]] <- integer(0)
  } else {
    # Compute row indices for same year
    y_pos <- cell_data$year_pos[i]
    neighbor_lookup[[i]] <- (neigh_id_pos - 1L) * n_years + y_pos
  }
}

# Compute neighbor stats efficiently
compute_neighbor_stats <- function(vals, neighbor_lookup) {
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0L) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0L) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data[[var_name]], neighbor_lookup)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

**Why This Is Faster**
- Eliminates string concatenation and hash lookups.
- Uses integer arithmetic for index mapping.
- Reuses `neighbor_lookup` for all variables.
- Memory footprint is manageable: `neighbor_lookup` is a list of integer vectors, same as before but built without expensive string ops.

**Expected Impact**
- From 86+ hours to a few hours (or less) on a 16 GB laptop.
- Complexity remains `O(N * avg_neighbors)` but with minimal overhead.

**Preserves**
- Original estimand.
- Trained Random Forest model.
- Panel structure and neighbor relationships.
END_RESPONSE>>>
