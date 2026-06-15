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
run_id: OpenAI_F05_T00_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:

- For each of ~6.46M rows, the code:
  - Builds `neighbor_keys` by pasting neighbor IDs with the current year.
  - Looks up indices in `idx_lookup` using string keys.
- This is repeated for every row, even though the neighbor structure is static across years.
- Complexity:  
  `O(n_rows Ã— avg_neighbors)` â‰ˆ `6.46M Ã— ~8` â†’ ~50M string operations.

This dominates runtime and memory. The inefficiency is not just local; itâ€™s a symptom of a **broader repeated lookup pattern** caused by mixing spatial and temporal dimensions via string keys.

---

### **Optimization Strategy**

1. **Precompute a numeric index map** instead of string keys:
   - Sort `data` by `(id, year)`.
   - Create a matrix `neighbor_idx` of size `n_rows Ã— max_neighbors` with integer indices.
2. **Exploit panel structure**:
   - For each year, neighbors are the same set of IDs, just shifted by year.
   - Build neighbor indices year by year using vectorized operations.
3. **Avoid repeated `paste` and hash lookups**:
   - Use integer mapping from `id` to row index for each year.
4. **Compute neighbor stats in a fully vectorized way**:
   - Use `matrixStats` or `apply` on precomputed neighbor index matrix.

This reduces complexity to roughly `O(n_rows Ã— avg_neighbors)` **once**, without string overhead, and makes subsequent feature computations trivial.

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume: data.table with columns id, year, and all variables
setDT(cell_data)
setkey(cell_data, id, year)

# Basic facts
years <- sort(unique(cell_data$year))
n_years <- length(years)
id_order <- sort(unique(cell_data$id))
n_ids <- length(id_order)

# Precompute: map id -> position
id_to_pos <- setNames(seq_along(id_order), id_order)

# Precompute neighbor positions (static across years)
max_neighbors <- max(lengths(rook_neighbors_unique))
neighbor_pos <- matrix(NA_integer_, nrow = n_ids, ncol = max_neighbors)
for (i in seq_along(rook_neighbors_unique)) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) > 0) {
    neighbor_pos[i, seq_along(nb)] <- id_to_pos[id_order[nb]]
  }
}

# Build neighbor index matrix for all rows
n_rows <- nrow(cell_data)
neighbor_idx <- matrix(NA_integer_, nrow = n_rows, ncol = max_neighbors)

# Fill neighbor_idx year by year
row_start <- 1
for (yr in years) {
  year_rows <- which(cell_data$year == yr)
  # year_rows are in id order because of setkey
  neighbor_idx[year_rows, ] <- (neighbor_pos - 1) * n_years + match(yr, years)
  # Explanation: row index = (id_pos - 1)*n_years + year_index
  row_start <- row_start + length(year_rows)
}

# Remove invalid indices (neighbors that don't exist)
neighbor_idx[neighbor_idx < 1 | neighbor_idx > n_rows] <- NA_integer_

# Compute neighbor stats for each variable
compute_neighbor_stats_fast <- function(vals, neighbor_idx) {
  # vals: numeric vector length n_rows
  neighbor_vals <- matrix(vals[neighbor_idx], nrow = nrow(neighbor_idx))
  # Compute row-wise max, min, mean ignoring NAs
  maxv <- rowMaxs(neighbor_vals, na.rm = TRUE)
  minv <- rowMins(neighbor_vals, na.rm = TRUE)
  meanv <- rowMeans(neighbor_vals, na.rm = TRUE)
  cbind(maxv, minv, meanv)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_idx)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**

- **No string concatenation**: All lookups are integer-based.
- **One-time neighbor index computation**: `neighbor_idx` is built once and reused.
- **Vectorized stats**: Uses `matrixStats` for fast row-wise operations.
- **Memory**:  
  - `neighbor_idx`: ~6.46M Ã— 8 neighbors Ã— 4 bytes â‰ˆ 200 MB (fits in 16 GB RAM).
- **Speed**:  
  - Precomputation: minutes.
  - Each variable: seconds.

---

**Result**:  
From **86+ hours** to **under an hour** on a standard laptop, preserving the trained Random Forest model and original estimand.
END_RESPONSE>>>
