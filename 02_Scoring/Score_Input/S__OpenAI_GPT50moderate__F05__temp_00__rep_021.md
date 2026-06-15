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
run_id: OpenAI_F05_T00_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but the real issue is **algorithmic**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated hash lookups.  
- This is done once in `build_neighbor_lookup`, but then `compute_neighbor_stats` iterates again over all rows and neighbors for each of 5 variables, causing **O(N Ã— neighbors Ã— vars)** repeated work.  
- The neighbor structure is static across variables and years, so recomputing stats per variable is wasteful.  

**Optimization Strategy**

1. **Precompute neighbor indices once** as an integer matrix or list (no string keys).  
2. **Vectorize neighbor stats computation** across all variables in one pass, avoiding repeated loops.  
3. Use **matrix operations** or `data.table` for speed and memory efficiency.  

This avoids repeated string concatenation and hash lookups and collapses 5 full passes into 1.

---

### **Proposed Reformulation**

- Build a single integer matrix `neighbor_idx` where each row corresponds to a cell-year and columns are neighbor indices (NA for missing).
- Compute all neighbor-based features in one pass using `apply` or `rowMeans` on subsets.

---

### **Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup: map (id, year) -> row index
cell_data[, row_id := .I]

# Build neighbor index matrix
build_neighbor_index <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(cell_data)
  row_ids <- seq_len(n)
  
  # Precompute mapping from (id, year) to row index
  idx_lookup <- setNames(row_ids, paste(cell_data$id, cell_data$year, sep = "_"))
  
  # For each row, find neighbor indices
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(cell_data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
    as.integer(idx_lookup[neighbor_keys])
  })
}

neighbor_lookup <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

# Convert to a fixed-width matrix for vectorization
max_neighbors <- max(lengths(neighbor_lookup))
neighbor_mat <- matrix(NA_integer_, nrow = length(neighbor_lookup), ncol = max_neighbors)
for (i in seq_along(neighbor_lookup)) {
  ni <- neighbor_lookup[[i]]
  if (length(ni) > 0) neighbor_mat[i, seq_along(ni)] <- ni
}

# Compute neighbor stats for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
vals_mat <- as.matrix(cell_data[, ..vars])

compute_neighbor_features <- function(vals_mat, neighbor_mat) {
  n <- nrow(vals_mat)
  p <- ncol(vals_mat)
  res <- array(NA_real_, dim = c(n, p, 3)) # max, min, mean
  
  for (j in seq_len(p)) {
    v <- vals_mat[, j]
    for (i in seq_len(n)) {
      idx <- neighbor_mat[i, ]
      idx <- idx[!is.na(idx)]
      if (length(idx) > 0) {
        nv <- v[idx]
        nv <- nv[!is.na(nv)]
        if (length(nv) > 0) {
          res[i, j, 1] <- max(nv)
          res[i, j, 2] <- min(nv)
          res[i, j, 3] <- mean(nv)
        }
      }
    }
  }
  res
}

neighbor_stats <- compute_neighbor_features(vals_mat, neighbor_mat)

# Bind results back to cell_data
stat_names <- c("max", "min", "mean")
for (j in seq_along(vars)) {
  for (k in seq_along(stat_names)) {
    cell_data[[paste0(vars[j], "_nbr_", stat_names[k])]] <- neighbor_stats[, j, k]
  }
}
```

---

### **Why This Is Faster**
- **No repeated string concatenation or hash lookups** inside the main loop.
- **Single pass** over neighbors for all variables.
- **Precomputed neighbor matrix** allows O(1) access.
- Memory trade-off: `neighbor_mat` is large but feasible on 16 GB RAM (sparse structure possible if needed).

---

**Expected Impact:**  
From 86+ hours â†’ **minutes to a few hours**, depending on I/O and CPU speed, because the complexity drops from `O(N Ã— vars Ã— neighbors)` repeated with string ops to a single structured pass.

**Preserves:**  
- Original estimand  
- Trained Random Forest model  
- Numerical equivalence of features
END_RESPONSE>>>
