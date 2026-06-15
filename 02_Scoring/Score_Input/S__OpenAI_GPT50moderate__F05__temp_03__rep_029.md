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
run_id: OpenAI_F05_T03_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and repeated hash lookups inside `lapply` are only part of the inefficiency. The deeper issue is **algorithmic**:  
- For each of ~6.46M rows, you build neighbor keys and perform lookups in a large named vector.  
- This results in ~6.46M Ã— average neighbor count (â‰ˆ4â€“8) string operations and hash lookups, repeated for every neighbor source variable.  
- The outer loop over 5 variables multiplies this cost.  

Thus, the inefficiency is **not just local**; itâ€™s a symptom of a broader pattern of repeated key generation and lookup. The core problem: the neighbor relationships are static across variables, but you recompute neighbor indices for every row and every variable.

---

**Optimization Strategy**

1. **Precompute neighbor indices once** for all cell-years, avoiding repeated string concatenation and hash lookups.
2. Store neighbor indices in an integer matrix or list aligned with `data` rows.
3. Reuse this structure for all variables, so `compute_neighbor_stats` only does numeric operations.
4. Use `data.table` or `matrix` operations for speed and memory efficiency.

---

### **Proposed Algorithm**

- Build a **fast join** between `(id, year)` and row index using integer keys instead of strings.
- Expand the neighbor structure across years in a **vectorized way**.
- Compute neighbor stats in a single pass per variable using precomputed indices.

---

### **Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, and predictor vars
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping from (id, year) -> row index
cell_data[, row_id := .I]

# Expand neighbor relationships across years
years <- sort(unique(cell_data$year))
n_years <- length(years)

# rook_neighbors_unique: list of neighbors per id in id_order
id_order <- as.integer(id_order)  # ensure numeric
id_to_neighbors <- rook_neighbors_unique

# Precompute neighbor indices for all rows
neighbor_lookup <- vector("list", nrow(cell_data))

for (yr in seq_along(years)) {
  y <- years[yr]
  rows_this_year <- which(cell_data$year == y)
  ids_this_year  <- cell_data$id[rows_this_year]

  # For each row in this year, map neighbors
  for (i in seq_along(rows_this_year)) {
    ref_id <- ids_this_year[i]
    ref_idx <- match(ref_id, id_order)
    neighbor_ids <- id_order[id_to_neighbors[[ref_idx]]]

    # Find row indices for these neighbors in the same year
    neighbor_rows <- cell_data[J(neighbor_ids, y), row_id, nomatch = 0L]
    neighbor_lookup[[rows_this_year[i]]] <- neighbor_rows
  }
}

# Function to compute neighbor stats using precomputed indices
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_lookup)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**

- **No repeated string concatenation**: integer joins via `data.table` keys.
- **Neighbor indices computed once**: reused for all variables.
- **Memory-efficient**: neighbor_lookup is a list of integer vectors.
- **Time complexity reduced**: from O(rows Ã— vars Ã— neighbors Ã— string_ops) to O(rows Ã— neighbors + rows Ã— vars).

---

**Expected Performance Gain**  
On 6.46M rows and ~1.37M neighbor links, this approach should reduce runtime from **86+ hours to a few hours or less**, depending on disk I/O and CPU.

**Preserves**:  
- Original numerical estimand.  
- Trained Random Forest model (no retraining).  

**Next Step**: Parallelize `compute_neighbor_stats_fast` with `parallel::mclapply` or `future.apply` if needed.
END_RESPONSE>>>
