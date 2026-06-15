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
run_id: OpenAI_F10_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- For every row (`~6.46M`), it builds neighbor indices on-the-fly (via `lapply`), which repeatedly reconstructs lookups.
- `compute_neighbor_stats()` allocates and iterates for each neighbor computation per variable, causing redundant passes over large vectors.
- R-level `lapply` and list operations dominate runtime; no vectorization or bulk indexing.
- The neighbor graph topology does not change across years, yet it's indirectly rebuilt repeatedly instead of being reused.

---

**Optimization Strategy**  
1. **Build neighbor lookup once** as integer indices for the entire dataset (cell-year rows).
2. Use **vectorized operations or compiled code** to compute statistics across neighbors in a single pass.
3. Avoid repeated string concatenations and dictionary lookups; precompute ID-year index mapping.
4. Store results in a preallocated numeric matrix and bind as columns efficiently.
5. Leverage **data.table** or **matrix indexing** for speed and memory efficiency.
6. Ensure **numerical equivalence** (same max/min/mean logic).

---

### **Optimized Implementation**

```r
library(data.table)

# Assume: cell_data (id, year, vars), id_order (cells), rook_neighbors_unique (list)
setDT(cell_data)
setkey(cell_data, id, year)

n_rows <- nrow(cell_data)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Precompute: map (id, year) -> row index
id_index <- match(cell_data$id, id_order)
year_index <- match(cell_data$year, years)

# Build neighbor-row-index lookup once
build_neighbor_row_indices <- function(n_ids, n_years, neighbors) {
  # offsets for each year block
  offsets <- ((0:(n_years-1)) * n_ids)
  lapply(seq_len(n_ids * n_years), function(i) {
    y_off <- ((i-1) %/% n_ids)
    base_id <- (i-1) %% n_ids + 1
    nb_ids <- neighbors[[base_id]]
    if (length(nb_ids) == 0) return(integer(0))
    nb_idx <- nb_ids + offsets[y_off + 1]
    nb_idx
  })
}

neighbor_row_indices <- build_neighbor_row_indices(length(id_order), n_years, rook_neighbors_unique)

# Compute stats efficiently for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(vals, neighbor_lookup) {
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  # column order: max, min, mean
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    nb_vals <- vals[idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  res
}

# Preallocate list for results
all_stats <- vector("list", length(neighbor_source_vars))
names(all_stats) <- neighbor_source_vars

for (var in neighbor_source_vars) {
  vals <- cell_data[[var]]
  all_stats[[var]] <- compute_all_neighbor_stats(vals, neighbor_row_indices)
}

# Bind results
for (var in neighbor_source_vars) {
  m <- all_stats[[var]]
  cell_data[[paste0(var, "_nb_max")]]  <- m[, 1]
  cell_data[[paste0(var, "_nb_min")]]  <- m[, 2]
  cell_data[[paste0(var, "_nb_mean")]] <- m[, 3]
}
```

---

### **Key Improvements**
- **Single graph construction** via `neighbor_row_indices` reused for all years and variables.
- **Avoid repeated string concatenation/lookup** â†’ replaced by integer offsets.
- **Efficient loops**: outer loop over rows unavoidable, but simplified and reuses precomputed indices.
- **Memory-efficient binding**: appends columns after full computation.

---

**Expected Result**  
- Computational time drops from **86+ hours â†’ under a few hours** on 16 GB RAM laptop.
- Preserves **original estimands** (max, min, mean).
- Random Forest model remains unchanged and can be applied immediately.  
**This is the most efficient base-R/data.table approach without requiring external graph libraries.**
END_RESPONSE>>>
