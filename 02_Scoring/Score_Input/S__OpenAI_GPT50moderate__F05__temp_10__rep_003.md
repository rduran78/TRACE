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
run_id: OpenAI_F05_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated creation of `neighbor_keys` (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) and subsequent lookups in `idx_lookup` inside the inner `lapply` loop are only symptoms of a **larger algorithmic inefficiency**:

- You are building neighbor index lookups on-the-fly **for every row** (`6.46M` rows) by repeatedly doing:
  - String concatenation (`paste`) per neighbor relation.
  - Name-based lookup in `idx_lookup`, which creates unnecessary overhead at this scale.
- This effectively scales with  
  `O(#rows Ã— avg_neighbors)` â‰ˆ `6.46M Ã— 4-8` â†’ tens of millions of map lookups.
- Because the neighbor topology **does not change across years**, the entire repeated-string logic is avoidable: compute a block-based index once and reuse it across all years.

The bottleneck is therefore *algorithmic*, not just micro-inefficiency. Optimal plan:  
â†’ Precompute neighbor ID â†’ row-index relationships structured as integer arrays **without string keys**, leveraging the fact that data is sorted by `(id, year)` or can be ordered so.  

---

### **Optimization Strategy**
1. **Sort `data` by (id, year)** so that rows for each `id` form a contiguous block of `n_years` rows.
2. Compute:
   - `years <- sort(unique(data$year))`
   - `n_years <- length(years)`
   - `id_pos <- match(data$id, id_order)`
3. Build a **static neighbor index matrix** (`neighbor_lookup`) using integer indices instead of repeated string pastes:
   - For cell `i` at year `t` (`row_index r`), its neighbors are:
     ```
     neighbor_idx = neighbors[[id_pos[i]]]
     neighbor_rows = (neighbor_idx - 1) * n_years + year_index[t]
     ```
   - This is pure integer math: `row = (id_index - 1)*n_years + year_idx`.
4. Use these indices once for all variable computations.

Memory check:  
- 6.46M rows â†’ integer indices = ~ 25MB for a list of vectors (manageable with 16GB RAM).

---

### **Working Optimized R Code**

```r
# Assume: data has columns id, year, sorted by id then year
optimize_neighbor_lookup <- function(data, id_order, neighbors) {
  years <- sort(unique(data$year))
  n_years <- length(years)

  # Mapping id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # For fast year index lookup
  year_index <- match(data$year, years)

  # Confirm data is sorted by id, then year (if not, do order)
  o <- order(data$id, data$year)
  if (!isTRUE(all.equal(o, seq_len(nrow(data))))) {
    data <- data[o, ]
    year_index <- year_index[o]
  }

  # Build lookup: row i -> integer vector of neighbor row indices
  id_ref_vec <- id_to_ref[as.character(data$id)]
  n_rows <- nrow(data)

  neighbor_lookup <- vector("list", n_rows)

  # Precompute multiplier for block offset
  block_offset <- (seq_along(id_order) - 1L) * n_years

  for (r in seq_len(n_rows)) {
    ref_idx <- id_ref_vec[r]
    n_ids <- neighbors[[ref_idx]]
    if (length(n_ids) == 0L) {
      neighbor_lookup[[r]] <- integer(0L)
    } else {
      neighbor_rows <- block_offset[n_ids] + year_index[r]
      neighbor_lookup[[r]] <- neighbor_rows
    }
  }

  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0L) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0L) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Usage:
neighbor_lookup <- optimize_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Runtime Reduction**
- Removes repeated string concatenation and name-based lookup.
- Each neighbor-index computation now is pure integer arithmetic done once.
- For 6.46M rows and moderate neighbor density, this should reduce runtime from **86+ hours â†’ a few hours (or less)** on a standard 16â€¯GB machine.

**Core Principle:** Precompute numeric neighbor indices once; reuse across all target variables, cutting complexity drastically while preserving the trained model and original estimand.
END_RESPONSE>>>
