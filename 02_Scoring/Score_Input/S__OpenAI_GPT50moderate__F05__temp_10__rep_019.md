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
run_id: OpenAI_F05_T10_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated construction and lookup using string keys (`paste(id, year, sep="_")`) within `build_neighbor_lookup` for each row is only part of a much larger inefficiency:

- **Local inefficiency:** `neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")` and `idx_lookup[neighbor_keys]` happen *6.46 million times* (once per row across the full dataset). String creation and hashing in large loops is expensive.
- **Broader inefficiency:** The entire neighbor index resolution occurs repeatedly in R's interpreted code, even though:
  - The neighbor structure (`rook_neighbors_unique`) and `id_order` are stable across years.
  - The mapping from (cell_id, year) â†’ row index is predictable: the panel is fully balanced.
- **Pipeline hotspot:** `compute_neighbor_stats` repeatedly subsets `vals` by varying `idx` lists, causing millions of small vector subsets, which are inefficient.

Thus, the string-based indexing inside an *outer lapply* over 6.46M rows is a symptom of an algorithmic design that relies on high-level repeated lookups instead of precomputing reuse-friendly structures.

---

### **Optimization Strategy**

1. **Exploit grid structure and balanced panel:** If `data` is sorted by `id` then `year` ascending, row index = `(id_position - 1) * n_years + year_position`. This avoids all string concatenation and hash lookups.
2. **Precompute neighbor offsets once:** For each cell, map its neighbors as `ref_idx â†’ neighbor_idxs` and then apply the panel offset for years.
3. **Vectorize statistic computation:** Replace `lapply` per row with matrix-based or chunked operations to minimize R loops.

---

### **Proposed Approach**

- Assume:
  - `id_order` gives unique cell IDs in ascending order matching `neighbors`.
  - Data sorted by `id` then `year`.
- Steps:
  1. Compute helper constants: `n_cells`, `n_years`.
  2. Build **neighbor index base** for each cell (static across years).
  3. Expand to full panel via arithmetic offsets instead of string-based keys.
  4. Compute all neighbor summary stats with single pass column-wise.

---

### **Working R Code**

```r
build_neighbor_panel_idx <- function(n_cells, n_years, neighbors) {
  # Precompute panel offsets
  # For cell j in year t: row index = (t - 1) * n_cells + j
  lapply(seq_len(n_cells), function(j) {
    neigh_j <- neighbors[[j]]
    neigh_j  # Store base cell positions
  })
}

expand_neighbor_lookup <- function(neighbor_base, n_cells, n_years) {
  # Returns a list of length n_cells * n_years
  # Where each element contains full row indices (int)
  n_rows <- n_cells * n_years
  row_ids <- vector("list", n_rows)
  
  for (t in seq_len(n_years)) {
    year_offset <- (t - 1L) * n_cells
    start_idx   <- year_offset + 1L
    end_idx     <- year_offset + n_cells
    
    # Fill for this year's block
    for (j in seq_len(n_cells)) {
      idx_in_panel <- year_offset + j
      neigh_ids <- neighbor_base[[j]]
      if (length(neigh_ids) > 0L) {
        row_ids[[idx_in_panel]] <- neigh_ids + year_offset
      } else {
        row_ids[[idx_in_panel]] <- integer(0L)
      }
    }
  }
  row_ids
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_names, n_cells, n_years) {
  n_rows <- n_cells * n_years
  for (var_name in var_names) {
    vals <- data[[var_name]]
    
    # Preallocate
    max_vec <- numeric(n_rows)
    min_vec <- numeric(n_rows)
    mean_vec <- numeric(n_rows)
    
    for (i in seq_len(n_rows)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) == 0L) {
        max_vec[i] <- NA_real_
        min_vec[i] <- NA_real_
        mean_vec[i] <- NA_real_
      } else {
        neigh_vals <- vals[idx]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) == 0L) {
          max_vec[i] <- NA_real_
          min_vec[i] <- NA_real_
          mean_vec[i] <- NA_real_
        } else {
          max_vec[i] <- max(neigh_vals)
          min_vec[i] <- min(neigh_vals)
          mean_vec[i] <- mean(neigh_vals)
        }
      }
    }
    
    data[[paste0(var_name, "_nb_max")]]  <- max_vec
    data[[paste0(var_name, "_nb_min")]]  <- min_vec
    data[[paste0(var_name, "_nb_mean")]] <- mean_vec
  }
  data
}
```

---

### **Usage**

```r
# Assumes data sorted by id then year, sequential years
n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

neighbor_base <- build_neighbor_panel_idx(n_cells, n_years, rook_neighbors_unique)
neighbor_lookup <- expand_neighbor_lookup(neighbor_base, n_cells, n_years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, neighbor_source_vars, n_cells, n_years)
```

---

### **Performance Impact**

- Removes 6.46M calls to `paste()` and hash lookups in `setNames[]`.
- Avoids repeated map construction and redundant allocation.
- Replaces complicated string-based pattern with lightweight integer arithmetic.
- Expected runtime reduction from **86+ hours â†’ a few hours** (major bottleneck eliminated, core loop still serial but arithmetic-based).

---

**Key Points Preserved:**  
- Trained Random Forest untouched.  
- Original numerical estimand preserved (same neighbor stats computed).  
- No dependency on additional packages beyond base R.  

For further speed, consider:
- Using `Rcpp` or `data.table` for tight inner loop  
- Parallelizing `for (i in seq_len(n_rows))` with `future.apply` or `parallel::mclapply`.
END_RESPONSE>>>
