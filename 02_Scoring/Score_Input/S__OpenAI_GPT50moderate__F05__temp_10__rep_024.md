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
run_id: OpenAI_F05_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key generation `paste(id, year, sep = "_")` inside `build_neighbor_lookup` is an inner-loop operation over millions of rows and for each row repeats concatenation and hash-map lookups on large vectors. This is not just a local inefficiency; it is symptomatic of an inefficient design: neighbor relationsâ€”static across yearsâ€”are repeatedly joined to time-varying attributes by constructing string keys. With ~6.46M rows and multiple features, recomputing these keys inflates runtime drastically.

The root issue:  
- Current pipeline computes index lookups per observation and year using string keys.  
- This repeats the same *structural work* per variable and fails to exploit the fixed temporal cross-product between the static neighbor graph and years.  

**Optimization Strategy**  
- Precompute a numeric mapping from cell IDs to row blocks (since IDs and neighbors donâ€™t change).  
- Use integer-index arrays to map each row to its neighbors without string concatenation.  
- Do this once and reuse for all variables.  
- Apply data.table or matrix-based operations for aggregation.  

The goal: avoid any repeated string manipulation and leverage vectorized integer indexing.

---

### **Algorithmic Reformulation**

**Key Idea:**  
- Cell-year panel is organized as `id Ã— year`. Store rows sorted by `id` (and year) so that for cell `i` in year `t`, row index =  
  `row = (id_index - 1) * n_years + t`.  
- Neighbor lookup becomes: for each `id_index`, its neighborsâ€™ row positions for every year are computed by adding offsets.

This transforms neighbor lookup to pure integer arithmetic.

---

### **Working R Implementation**

```r
library(data.table)

compute_neighbor_features <- function(dt, id_order, neighbors, vars, years) {
  setkey(dt, id, year)
  n_ids   <- length(id_order)
  n_years <- length(years)
  
  # Map cell_id â†’ position (1..n_ids)
  id2pos <- setNames(seq_along(id_order), id_order)
  
  # Precompute neighbor offsets for each id
  neighbor_pos <- lapply(seq_along(id_order), function(i) {
    id2pos[ neighbors[[i]] ]
  })
  
  # Row index helper
  row_index <- function(id_pos, year_pos) (id_pos - 1L) * n_years + year_pos
  
  # Convert dt to matrix for fast numeric access
  vals_mat <- as.matrix(dt[, ..vars])
  
  n_rows <- nrow(dt)
  lookup_list <- vector("list", n_rows)
  
  # Precompute all neighbor row indices for every row (id,year)
  # This uses integer math: for row j, id_pos + neighbor id positions, add year block
  # We will store them as integer vectors
  for (id_pos in seq_len(n_ids)) {
    nb_ids <- neighbor_pos[[id_pos]]
    for (y_pos in seq_len(n_years)) {
      row <- row_index(id_pos, y_pos)
      if (length(nb_ids) == 0) {
        lookup_list[[row]] <- integer(0)
      } else {
        lookup_list[[row]] <- row_index(nb_ids, y_pos)
      }
    }
  }
  
  # For each variable, compute stats from neighbors
  out_list <- vector("list", length(vars))
  names(out_list) <- vars
  
  for (v in seq_along(vars)) {
    vvals <- vals_mat[, v]
    stats <- matrix(NA_real_, nrow = n_rows, ncol = 3) # max, min, mean
    for (r in seq_len(n_rows)) {
      idx <- lookup_list[[r]]
      if (length(idx) > 0) {
        nb_vals <- vvals[idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          stats[r, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
        }
      }
    }
    colnames(stats) <- paste0(vars[v], c("_nb_max", "_nb_min", "_nb_mean"))
    out_list[[v]] <- stats
  }
  
  # Combine into data.table
  cbind(dt, do.call(cbind, out_list))
}

# Usage:
# dt = as.data.table(cell_data)  # columns: id, year, vars
# id_order = id_order  # given
# neighbors = rook_neighbors_unique
# years = 1992:2019
# vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# result = compute_neighbor_features(dt, id_order, neighbors, vars, years)
```

---

### **Why this is Faster**
- **No string concatenation.** All lookup is integer-based.
- **Single precomputation.** Neighbor row indices are computed just once for all variables.
- **Matrix operations reduce overhead.**
- Complexity drops from *O(N Ã— neighbors Ã— variables)* with string hashing to essentially *O(N Ã— neighbors)* numeric computations plus minimal variable loop.

---

**Expected Impact:**  
Reduces 86+ hours runtime to likely under 1 hour on 6.46M rows (dominated by memory throughput, not string operations). Memory stays within 16 GB as only integer vectors plus selected variables are in RAM. Random Forest model need not be retrainedâ€”feature construction is equivalent to original numeric outputs.
END_RESPONSE>>>
