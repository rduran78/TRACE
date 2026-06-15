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
run_id: OpenAI_F05_T00_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and repeated hash lookups inside the `lapply` loop are only part of the inefficiency. The deeper issue is that the algorithm repeatedly performs expensive string-based key generation and hash lookups for every row (â‰ˆ6.46M times), even though the neighbor relationships and year structure are highly regular. This is an **O(N Ã— avg_neighbors)** operation with heavy overhead from string manipulation and list indexing.

Since the neighbor structure is static across years, the current approach redundantly recomputes neighbor indices for each cell-year row. This is a **broader algorithmic inefficiency**: the neighbor mapping should be computed once per cell and then extended across years without string-based keys.

---

### **Optimization Strategy**
1. **Precompute numeric indices** instead of string keys:
   - Map each `(id, year)` to a row index using a fast integer-based approach.
   - Avoid `paste()` and `setNames()` in the inner loop.
2. **Exploit panel regularity**:
   - Each cell has the same neighbors every year.
   - Build a base neighbor index list for cells, then replicate across years.
3. **Vectorize neighbor stats computation**:
   - Use matrix operations or `data.table` joins instead of millions of small `lapply` calls.

---

### **Proposed Algorithm**
- Create a matrix `id_year_index` mapping `(id, year)` â†’ row index.
- For each cell, store its neighbor IDs once.
- For each year, compute neighbor row indices by adding an offset.
- Compute neighbor stats using vectorized operations.

---

### **Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Basic facts
n_ids   <- length(unique(cell_data$id))
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Precompute row index matrix: rows = id, cols = year
id_to_idx <- match(cell_data$id, unique(cell_data$id))
year_to_idx <- match(cell_data$year, years)
row_index_matrix <- matrix(seq_len(nrow(cell_data)), nrow = n_ids, ncol = n_years)
# Fill row indices by order
row_index_matrix[cbind(id_to_idx, year_to_idx)] <- seq_len(nrow(cell_data))

# Precompute neighbor list for IDs (not cell-years)
# rook_neighbors_unique: list of integer vectors (neighbors per id)
neighbor_list <- rook_neighbors_unique

# Build neighbor index list for all cell-years
neighbor_lookup <- vector("list", length = nrow(cell_data))

for (id_idx in seq_len(n_ids)) {
  neighbors <- neighbor_list[[id_idx]]
  if (length(neighbors) == 0) next
  for (yr_idx in seq_len(n_years)) {
    row_id <- row_index_matrix[id_idx, yr_idx]
    neighbor_rows <- row_index_matrix[neighbors, yr_idx]
    neighbor_lookup[[row_id]] <- neighbor_rows
  }
}

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  colnames(result) <- paste0(var_name, c("_max", "_min", "_mean"))
  result
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Why This Is Faster**
- Eliminates repeated string concatenation and hash lookups.
- Uses integer indexing and precomputed matrices.
- Reduces complexity from repeated key generation to simple numeric lookups.
- Memory footprint is manageable: `neighbor_lookup` is a list of integer vectors, same as before, but built efficiently.

---

**Expected Performance Gain**  
This approach should reduce runtime from **86+ hours to a few hours or less**, depending on I/O and CPU speed, because:
- String operations and hash lookups are eliminated.
- Neighbor mapping is computed once and reused.
- Still uses base R loops for stats, but can be further optimized with `data.table` or `Rcpp` if needed.

**Preserves**:
- Original estimand.
- Trained Random Forest model.
- Panel structure and neighbor definitions.
END_RESPONSE>>>
