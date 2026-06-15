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
run_id: OpenAI_F05_T03_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated creation of `neighbor_keys` and lookup via `idx_lookup` in `build_neighbor_lookup` is a **local inefficiency** but also a **symptom of a broader algorithmic issue**:  
- For each of ~6.46M rows, the code repeatedly pastes strings and performs hash lookups.  
- This is done once in `build_neighbor_lookup` (O(N Ã— avg_neighbors)), but the bigger cost is that the neighbor lookup is computed **once per row**, and then `compute_neighbor_stats` iterates again over the same structure for each of 5 variables.  
- So while the string-paste overhead is local, the main inefficiency is that the neighbor relationships are recomputed in a costly way and then repeatedly traversed for each variable.  

**Optimization Strategy**  
1. **Precompute numeric indices** for neighbors once, avoiding string concatenation entirely.  
2. **Vectorize neighbor stats computation** across variables instead of looping 5 times.  
3. Use **matrix operations** or `data.table` for fast aggregation.  
4. Avoid repeated `lapply` over millions of rows; instead, store neighbor indices in a compressed structure and compute all features in one pass.  

**Algorithmic Reformulation**  
- Build a neighbor index matrix: each row corresponds to a cell-year, columns are neighbor row indices (NA for missing).  
- Then compute max, min, mean for all variables in one vectorized pass.  

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data is a data.frame with columns id, year, and predictor vars
# id_order: vector of unique cell ids
# rook_neighbors_unique: spdep::nb object

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping from (id, year) -> row index
id_year_to_idx <- cell_data[, .I, by = .(id, year)]

# Build neighbor index matrix
build_neighbor_index <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(cell_data)
  row_ids <- seq_len(n)
  
  # Precompute neighbor indices without string operations
  neighbor_list <- vector("list", n)
  for (i in row_ids) {
    ref_idx <- id_to_ref[[as.character(cell_data$id[i])]]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    # Join to get row indices for same year
    year_val <- cell_data$year[i]
    idx <- id_year_to_idx[J(neighbor_cell_ids, year_val), I]
    neighbor_list[[i]] <- idx[!is.na(idx)]
  }
  neighbor_list
}

neighbor_lookup <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for multiple variables in one pass
compute_neighbor_stats_multi <- function(data, neighbor_lookup, vars) {
  m <- length(neighbor_lookup)
  res_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- data[[vars[v]]]
    out <- matrix(NA_real_, nrow = m, ncol = 3)
    for (i in seq_len(m)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          out[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    res_list[[v]] <- out
  }
  res_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_list <- compute_neighbor_stats_multi(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind results back to cell_data
for (v in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[v]
  cell_data[, paste0(var, "_nbr_max") := stats_list[[v]][, 1]]
  cell_data[, paste0(var, "_nbr_min") := stats_list[[v]][, 2]]
  cell_data[, paste0(var, "_nbr_mean") := stats_list[[v]][, 3]]
}
```

---

### **Why This Helps**
- Eliminates repeated string concatenation and hash lookups (big win).
- Computes neighbor stats in **one pass per variable**, not per variable Ã— per row Ã— per neighbor.
- Still uses lists for neighbor indices (memory efficient vs full dense matrix).
- On 6.46M rows, this should reduce runtime from 86+ hours to a few hours on a 16 GB laptop.

**Further Speedups**
- Use `Rcpp` or `data.table` row-wise joins for building neighbor indices.
- Parallelize the loop over variables or rows using `future.apply` or `parallel`.

**Preserves**:  
- Original estimand (same neighbor stats).
- Trained Random Forest model (unchanged).  
- Data structure and variable names remain consistent.
END_RESPONSE>>>
