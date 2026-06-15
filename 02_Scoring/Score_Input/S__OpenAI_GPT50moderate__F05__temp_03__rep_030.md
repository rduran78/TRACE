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
run_id: OpenAI_F05_T03_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste`) and hash lookups inside the innermost `lapply` loop are only part of the problem. The deeper issue is that the algorithm repeatedly constructs neighbor indices for each row across all variables, even though the neighbor relationships are static across years. This results in:

- **O(N Ã— k)** string operations and lookups for N = 6.46M rows and k â‰ˆ 5 neighbors.
- Recomputing neighbor indices for every row and every variable, even though they are invariant once built.
- Memory churn from repeated vector allocations.

Thus, the inefficiency is **global**, not just local. The entire neighbor-lookup pattern should be precomputed once and reused.

---

### **Optimization Strategy**
1. **Precompute neighbor indices for all rows once** (no repeated string concatenation).
2. Store them as an integer list aligned with `data` rows.
3. Use this precomputed structure for all variables.
4. Avoid per-row `paste` and hash lookups by mapping `(id, year)` â†’ row index using vectorized joins or integer arithmetic.
5. Use `data.table` for speed and memory efficiency.

---

### **Reformulated Approach**
- Assign each `(id, year)` a **row index**.
- Compute neighbor row indices by joining on `(neighbor_id, year)` in a vectorized way.
- Store as a list of integer vectors.
- Compute neighbor stats in a vectorized or partially vectorized manner.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all cell IDs in spatial order
# rook_neighbors_unique: spdep::nb object with neighbor indices (1-based)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor pairs (id -> neighbor_id)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# Expand neighbor pairs across years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# Join to get row indices for (id, year) and (neighbor_id, year)
neighbor_pairs[, row_id := cell_data[.SD, on = .(id, year), which = TRUE]]
neighbor_pairs[, neighbor_row_id := cell_data[.SD, on = .(id = neighbor_id, year), which = TRUE]]

# Drop invalid matches
neighbor_pairs <- neighbor_pairs[!is.na(row_id) & !is.na(neighbor_row_id)]

# Build neighbor lookup as a list of integer vectors
neighbor_lookup <- split(neighbor_pairs$neighbor_row_id, neighbor_pairs$row_id)

# Ensure full length (some rows have no neighbors)
neighbor_lookup <- neighbor_lookup[as.character(seq_len(nrow(cell_data)))]
neighbor_lookup[sapply(neighbor_lookup, is.null)] <- list(integer(0))

# Compute neighbor stats for each variable
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
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

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**
- **Single join-based computation** replaces millions of string concatenations and hash lookups.
- `neighbor_lookup` is computed **once** and reused.
- Memory footprint is minimized by using integer indices instead of strings.
- Complexity drops from `O(N Ã— k Ã— vars)` string ops to `O(N Ã— k)` integer ops + `O(N Ã— vars)` numeric ops.

---

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU speed. The bottleneck shifts to numeric aggregation rather than string manipulation.

**Preserves:**  
- Original Random Forest model.
- Original estimand and feature definitions.
END_RESPONSE>>>
