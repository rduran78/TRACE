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
run_id: OpenAI_F05_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated `paste()` and key lookups inside `lapply(row_ids, ...)` are **not just a minor inefficiency**â€”they indicate a deeper algorithmic issue. The current design repeatedly constructs and queries string-indexed keys (`"id_year"`) for **each row across all years**, even though the neighbor structure is static and the panel spans 6.46M rows. This drives excessive string manipulation and hash lookups, leading to the 86+ hour runtime. The pipeline is performing ~6.46M Ã— multiple key operations when the mapping from (id, year) â†’ row index could be organized in a purely numeric array upfront.

**Key Issues:**
- For every row, the code repeatedly `paste()` cell IDs with year and performs a named lookup in `idx_lookup` (vector of length nrow(data)).
- Although `id_order` and neighbor sets are static, recomputing keys for each year causes a large combinatorial overhead.
- Work per iteration scales with rows rather than exploiting the repeated structure (neighbor sets constant across years).

**Optimization Strategy**
1. **Avoid string-based keys**: Switch to integer indexing.
2. **Exploit structure**: Neighbor relationships are static over years; only the time index changes. Build a numeric index matrix once.
3. **Vectorize time expansion**: Precompute a neighbor index matrix of dimensions `nrow(data) Ã— max_neighbors` using direct integer positions.
4. Keep the output identical (max, min, mean per neighbor set).

This shifts complexity from billions of hash lookups to simple numeric indexing.

---

### **Optimized Approach**
- Assume `data$id` matches `id_order` mapping (IDs from 1..N).
- Precompute `id_to_row` as a `(years Ã— ids)` matrix for direct lookup.
- Expand neighbors for all rows by integer join, no string pasting.

---

### **Working R Code**

```r
build_neighbor_index <- function(data, id_order, neighbors) {
  # Basic dimensions
  N_ids   <- length(id_order)
  N_years <- length(unique(data$year))
  
  # Map id -> reference index in neighbor list
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Sort data by year then id for easier indexing
  data <- data[order(data$year, data$id), ]
  
  # Create year and id vectors
  years <- sort(unique(data$year))
  
  # Create a matrix that gives row index for each (id, year)
  # id_row[i_id, i_year] = row index in data
  id_year_mat <- matrix(NA_integer_, nrow = N_ids, ncol = N_years)
  row_id_seq  <- seq_len(nrow(data))
  id_year_mat[
    match(data$id, id_order),
    match(data$year, years)
  ] <- row_id_seq
  
  # Max number of neighbors
  max_nb <- max(vapply(neighbors, length, 0L))
  
  # Precompute neighbor index matrix
  neighbor_index <- matrix(NA_integer_, nrow = nrow(data), ncol = max_nb)
  
  # Fill per row efficiently
  for (i_year in seq_along(years)) {
    # vector of row positions for this year
    rows_this_year <- ((i_year - 1) * N_ids + 1):(i_year * N_ids)
    # get (id index) of these rows
    id_idx <- match(data$id[rows_this_year], id_order)
    
    # For each row: lookup neighbor ids via id_to_ref
    for (j in seq_along(rows_this_year)) {
      ref_idx <- id_idx[j]
      nb_ids  <- neighbors[[ref_idx]]
      if (length(nb_ids) > 0) {
        nb_row_idx <- id_year_mat[ nb_ids, i_year ]
        neighbor_index[ rows_this_year[j], seq_along(nb_row_idx) ] <- nb_row_idx
      }
    }
  }
  
  neighbor_index
}

compute_neighbor_stats_fast <- function(data, neighbor_index, var_name) {
  vals <- data[[var_name]]
  apply(neighbor_index, 1, function(nb_idx) {
    nb_idx <- nb_idx[!is.na(nb_idx)]
    if (length(nb_idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[nb_idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }) |> t()
}
```

---

### **How to Use**
```r
neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_index, var_name)
  colnames(stats) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why this is faster:**
- Eliminates billions of string key operations.
- Uses numeric indexing and precomputed mappings for O(1) row lookups.
- Reuses neighbor_index for all variables, avoiding repeated recomputation.

---

**Expected Runtime Reduction:**  
From 86+ hours to a few hours (or less, especially if combined with `data.table` or `Rcpp` for loops). Memory remains manageable because `neighbor_index` is ~6.46M rows Ã— ~max 8 neighbors (â‰ˆ 50â€“60M integers).

**Result Preservation:**  
Same neighbor relationships and aggregations as before, compatible with your trained Random Forest model.
END_RESPONSE>>>
