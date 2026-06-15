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
run_id: OpenAI_F05_T00_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and hash lookups inside the `lapply` loop are only a *local inefficiency symptom of a larger algorithmic issue*: the current design performs **6.46 million Ã— average neighbor count** repeated string operations and hash lookups. This is extremely costly because:

- For each row, you rebuild `neighbor_keys` and perform a named vector lookup.
- The neighbor structure is static across years; only the year dimension changes.
- The entire neighbor lookup is recomputed for every row, even though the neighbor relationships are fixed and predictable.

Thus, the inefficiency is **algorithmic**: the code repeatedly does expensive string-based mapping instead of precomputing integer indices once and reusing them.

---

### **Optimization Strategy**
1. **Precompute integer-based neighbor indices for all rows**:
   - Avoid string concatenation and hash lookups inside the main loop.
   - Use vectorized mapping from `(id, year)` â†’ row index via a matrix reshape or integer arithmetic.
2. **Exploit panel structure**:
   - Data is sorted by `id` and `year` (or can be).
   - If sorted, row index = `(id_position - 1) * n_years + year_position`.
   - Precompute a matrix `neighbor_lookup_matrix` of size `nrow(data)` Ã— `max_neighbors` with integer indices.
3. **Compute neighbor stats in a fully vectorized way**:
   - Use `matrixStats` or `apply` on precomputed neighbor index matrix.
   - Avoid `lapply` over millions of rows.

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume data is sorted by id, then year
setDT(cell_data)
n_ids   <- length(id_order)
n_years <- length(unique(cell_data$year))
stopifnot(nrow(cell_data) == n_ids * n_years)

# Map id to position
id_to_pos <- setNames(seq_along(id_order), id_order)

# Precompute neighbor positions (id-level)
max_neighbors <- max(lengths(rook_neighbors_unique))
neighbor_id_mat <- matrix(NA_integer_, nrow = n_ids, ncol = max_neighbors)
for (i in seq_along(rook_neighbors_unique)) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) > 0) {
    neighbor_id_mat[i, seq_along(nb)] <- nb
  }
}

# Expand to cell-year level: build neighbor index matrix
n_rows <- nrow(cell_data)
neighbor_idx_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_neighbors)

# Compute row index formula: row = (id_pos - 1) * n_years + year_pos
year_vals <- sort(unique(cell_data$year))
year_to_pos <- setNames(seq_along(year_vals), year_vals)

for (row in seq_len(n_rows)) {
  id_pos   <- id_to_pos[[as.character(cell_data$id[row])]]
  year_pos <- year_to_pos[[as.character(cell_data$year[row])]]
  nb_ids   <- neighbor_id_mat[id_pos, ]
  nb_ids   <- nb_ids[!is.na(nb_ids)]
  if (length(nb_ids) > 0) {
    neighbor_idx_mat[row, seq_along(nb_ids)] <- (nb_ids - 1) * n_years + year_pos
  }
}

# Function to compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_idx_mat) {
  # vals: numeric vector of length n_rows
  # neighbor_idx_mat: integer matrix (n_rows x max_neighbors)
  n_rows <- nrow(neighbor_idx_mat)
  result <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  for (i in seq_len(n_rows)) {
    idx <- neighbor_idx_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      nb_vals <- vals[idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) > 0) {
        result[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
      }
    }
  }
  result
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_idx_mat)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**
- Eliminates repeated string concatenation and hash lookups.
- Precomputes neighbor indices once.
- Uses integer arithmetic for direct indexing.
- Reduces complexity from **O(n_rows Ã— neighbors Ã— string_ops)** to **O(n_rows Ã— neighbors)** with simple integer lookups.

---

**Expected Performance Gain**  
On 6.46M rows and ~1.37M neighbor links, this approach should reduce runtime from **86+ hours to a few hours or less**, depending on I/O and memory bandwidth, because:
- All expensive operations are moved out of the inner loop.
- Memory layout is contiguous and cache-friendly.

**Preserves**:
- Original estimand.
- Trained Random Forest model.
- Numerical results (only faster computation).
END_RESPONSE>>>
