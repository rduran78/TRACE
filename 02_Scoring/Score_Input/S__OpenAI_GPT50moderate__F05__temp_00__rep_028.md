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
run_id: OpenAI_F05_T00_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and hash lookups inside the `lapply` loop are only a *local inefficiency symptom of a larger algorithmic issue*: the current design performs **6.46 million Ã— average neighbor count** repeated lookups in R lists and character vectors. This is extremely costly because:

- String concatenation and name-based indexing are repeated for every row.
- The neighbor structure is static across years, but the code recomputes neighbor keys for every cell-year.
- The algorithm is effectively doing a nested loop over all rows and their neighbors, which is O(N Ã— avg_neighbors) with heavy overhead.

Given 6.46M rows and ~4 neighbors per cell, this results in tens of millions of expensive operations. The bottleneck is not just string operations but the entire repeated lookup pattern.

---

### **Optimization Strategy**
1. **Exploit panel structure**: The neighbor relationships are constant across years. We can:
   - Precompute a numeric neighbor index matrix for the base cells (344,208 rows).
   - Replicate this structure across years without recomputing string keys.
2. **Use integer indexing instead of string keys**:
   - Map `(id, year)` to a row index once.
   - Build a numeric matrix of neighbor indices for all rows.
3. **Vectorize neighbor stats computation**:
   - Avoid `lapply` over 6.46M rows.
   - Use matrix operations or `data.table` joins.

---

### **Proposed Reformulation**
- Precompute a **neighbor index matrix** for all cell-years using integer arithmetic.
- Compute neighbor stats in a fully vectorized way.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, and all variables
setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
n_years <- length(unique(cell_data$year))
years <- sort(unique(cell_data$year))

# Precompute neighbor indices for base cells
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
max_neighbors <- max(lengths(rook_neighbors_unique))
neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_neighbors)
for (i in seq_along(rook_neighbors_unique)) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) > 0) {
    neighbor_mat[i, seq_along(nb)] <- nb
  }
}

# Map id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Add a numeric index for each row in cell_data
cell_data[, idx := id_to_idx[as.character(id)]]

# Compute global row index for each cell-year
# Row index = (year_index - 1) * n_cells + idx
year_to_offset <- setNames(seq_along(years) - 1, years)
cell_data[, global_idx := year_to_offset[as.character(year)] * n_cells + idx]

# Build neighbor lookup for all rows as integer matrix
# For each row, compute neighbor global indices
neighbor_lookup <- matrix(NA_integer_, nrow = nrow(cell_data), ncol = max_neighbors)
for (y in seq_along(years)) {
  year_offset <- (y - 1) * n_cells
  rows_this_year <- ((y - 1) * n_cells + 1):(y * n_cells)
  neighbor_lookup[rows_this_year, ] <- neighbor_mat + year_offset
}

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  # vals: numeric vector of length nrow(cell_data)
  neighbor_vals <- vals[neighbor_lookup]
  # Mask invalid neighbors
  neighbor_vals[is.na(neighbor_lookup)] <- NA_real_
  
  # Compute row-wise max, min, mean ignoring NAs
  max_vals <- apply(neighbor_vals, 1, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE))
  min_vals <- apply(neighbor_vals, 1, function(x) if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE))
  mean_vals <- apply(neighbor_vals, 1, function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE))
  
  cbind(max_vals, min_vals, mean_vals)
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- **No repeated string concatenation**: All lookups are integer-based.
- **No per-row lapply**: We use matrix operations and `apply` (or could use `rowMaxs` from `matrixStats` for even more speed).
- **Precomputation**: Neighbor structure is computed once and reused.
- **Memory**: `neighbor_lookup` is large but feasible (~6.46M Ã— max_neighbors integers). With 4 neighbors, thatâ€™s ~26M integers (~200 MB).

---

**Expected Speedup**: From 86+ hours to a few minutes on a 16 GB laptop, as the algorithm is now O(N Ã— max_neighbors) with vectorized operations and no string overhead.

**Preserves**:
- Original estimand.
- Trained Random Forest model.
- All neighbor-based features.

**Further optimization**:
- Use `matrixStats::rowMaxs`, `rowMins`, and `rowMeans` for faster aggregation.
- If memory is tight, process in yearly chunks.
END_RESPONSE>>>
