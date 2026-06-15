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
run_id: OpenAI_F05_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(...)`) and named-index lookups inside `lapply(row_ids, ...)` are a **symptom of a larger inefficiency**: for each of 6.46 million rows, the code builds `neighbor_keys` and repeatedly hashes through `idx_lookup`. This creates a quadratic blow-up in work because these lookups are performed for every row-year combination even though neighbor relationships are invariant across years. The logic essentially rebuilds and searches mapping for each row instead of vectorizing it or leveraging precomputed structures.

The inefficiency is not just localâ€”itâ€™s algorithmic:  
- You have **344k cells Ã— 28 years = 6.46M rows**.  
- Each rowâ€™s neighbors are drawn from **id_order** and **neighbors**, which do not change across years.  
- For every row, you compute string keys for `neighbor_cell_ids` joined with the year and then look up positions in `idx_lookup`.  
- This is repeated for every neighbor-linked statistic and every variable (5 variables).  

Thus, the algorithm repeatedly executes expensive string-based mapping that could be eliminated by precomputing a fully numeric index-based neighbor structure expanded across the 28 years.

---

**Optimization Strategy**  
Reformulate the algorithm so that:  
1. We precompute `neighbor_lookup` **once as integer indices**, using numeric positions instead of string concatenation.  
2. Use vectorized operations for computing statistics instead of repeated `lapply`.  

Approach:  
- Instead of building string keys like `"id_year"`, create a direct mapping from `(id, year_index)` â†’ row index.  
- Expand neighbor relationships across years with numeric arrays: for each row index `i`, store the corresponding neighbor row indices across the same year.  
- This avoids string concatenation, dictionary lookups, and repeated hashing.  
- Then, apply `vapply` or `Rcpp` to quickly compute neighbor stats.

---

### **Working R Code**

```r
# Precompute (id, year) -> row index map as matrix
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Assume data is sorted by id then year with equal years per id.
  ids   <- data$id
  years <- data$year
  n_ids <- length(id_order)
  
  # Create mapping from id to sequential block
  year_levels <- sort(unique(years))
  n_years     <- length(year_levels)
  
  # Precompute index matrix: row[i, y] = row in data where id=i and year=y
  idx_matrix <- matrix(NA_integer_, n_ids, n_years)
  row_ids    <- seq_len(nrow(data))
  
  # Fill index matrix
  id_pos <- match(ids, id_order)
  year_pos <- match(years, year_levels)
  idx_matrix[cbind(id_pos, year_pos)] <- row_ids
  
  # Build neighbor lookup: list of length nrow(data)
  neighbor_lookup <- vector("list", length(row_ids))
  
  for (rid in row_ids) {
    this_id <- id_pos[rid]
    this_year <- year_pos[rid]
    neigh_ids <- neighbors[[this_id]]
    if (length(neigh_ids) > 0) {
      # Directly pick rows from idx_matrix (same year)
      neigh_rows <- idx_matrix[neigh_ids, this_year]
      neighbor_lookup[[rid]] <- neigh_rows[!is.na(neigh_rows)]
    } else {
      neighbor_lookup[[rid]] <- integer(0)
    }
  }
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  # Vectorized apply
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    nvals <- vals[idx]
    nvals <- nvals[!is.na(nvals)]
    if (length(nvals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nvals), min(nvals), mean(nvals))
  }, numeric(3))
  t(result)
}

# --- MAIN USAGE ---
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste(var_name, c("max", "min", "mean"), sep = "_neighbor_")
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

**Why this is faster:**  
- Eliminates all string concatenation and hashing (`paste` + named lookup).  
- Precomputes a numeric mapping table (`idx_matrix`), so retrieval is direct O(1).  
- Avoids repeated construction of keys for each observation.  
- Uses `vapply` for speed and safety.  

Given 6.46M rows and ~1.37M neighbor edges, this vectorized approach will reduce runtime **from tens of hours to a few hours or less**, bounded mainly by memory bandwidth and R loop overhead (further optimizable via `data.table` or Rcpp).  

**Preserves:**  
- Random Forest model unchanged.  
- Same estimands and numeric outputs.
END_RESPONSE>>>
