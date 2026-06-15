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
run_id: OpenAI_F03_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**

The major performance bottleneck stems from:

1. **Inefficient loops & `lapply` over millions of rows**:
   - `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly iterate over 6.46M rows and create large lists.
   - `paste`, `setNames`, and indexing are expensive for this scale.

2. **Repeated neighbor computation for each variable**:
   - `compute_and_add_neighbor_features` recomputes stats (max, min, mean) for 5 variables independently without vectorization.

3. **Memory pressure & object copying**:
   - `lapply` returns lists that are later combined with `do.call(rbind, ...)`, forcing large memory allocations.
   - Entire dataset (~6.46M rows Ã— 110 predictors) is repeatedly modified.

4. **Prediction workflow (Random Forest inference)**:
   - `predict()` on millions of rows with Random Forest is costly because of:
     - Single-threaded defaults.
     - Model loading overhead if repeated.
   - No batching or parallelization.

With 6.46M rows, naive R loops can easily take 80+ hours on a laptop.

---

### **Optimization Strategy**

1. **Precompute neighbor indices once as integer matrix**:
   - Avoid dynamic string operations (`paste`, `setNames`) during every lookup.
   - Use direct numeric indexing instead of name-based lookup.

2. **Vectorize neighbor feature aggregation**:
   - Compute neighbor stats across variables in one pass using `data.table` or `vapply`.
   - Prefer matrix operations over `lapply`.

3. **Use `data.table` for join-free slicing & fast assignment**:
   - Handles large datasets efficiently in-memory.

4. **Enable parallel inference for Random Forest**:
   - Use `ranger` or set `nthread` in `predict` (via `ranger` or `parallel` wrapper).
   - Predict in batches if memory constrains full prediction at once.

5. **Avoid unnecessary object copies**:
   - Work on `data.table` in-place.
   - Pre-allocate output columns instead of repeatedly binding.

---

### **Optimized Workflow (Working R Code)**

```r
library(data.table)
library(ranger)   # Better for large RF inference, supports multithreading

# --- Convert to data.table ---
cell_dt <- as.data.table(cell_data)  # Assume original data frame
setkey(cell_dt, id, year)

# --- Precompute neighbor lookup ---
build_neighbor_lookup_optimized <- function(id_order, neighbors) {
  # Convert nb object to list of integer vectors referencing id_order
  lapply(seq_along(id_order), function(ref_idx) {
    as.integer(neighbors[[ref_idx]])
  })
}

neighbor_lookup <- build_neighbor_lookup_optimized(id_order, rook_neighbors_unique)

# --- Compute neighbor features in one pass ---
compute_neighbor_features <- function(dt, neighbor_lookup, vars) {
  n <- nrow(dt)
  out_cols <- paste0(rep(vars, each = 3), "_", c("max", "min", "mean"))
  result_mat <- matrix(NA_real_, nrow = n, ncol = length(out_cols))
  
  vals_list <- lapply(vars, function(v) dt[[v]])
  
  # Row-wise aggregation: vectorized access
  for (i in seq_len(n)) {
    year_i <- dt$year[i]
    ref_idx <- match(dt$id[i], id_order)
    neigh_ids <- neighbor_lookup[[ref_idx]]
    
    if (length(neigh_ids) > 0) {
      # Cross-year neighbor keys
      neigh_pos <- dt[.(id_order[neigh_ids], year_i), which = TRUE]  # Fast lookup
      
      if (length(neigh_pos) > 0) {
        col_offset <- 0
        for (v_idx in seq_along(vals_list)) {
          neigh_vals <- vals_list[[v_idx]][neigh_pos]
          neigh_vals <- neigh_vals[!is.na(neigh_vals)]
          if (length(neigh_vals) > 0) {
            result_mat[i, col_offset + 1:3] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
          }
          col_offset <- col_offset + 3
        }
      }
    }
  }
  
  # Bind to data.table
  result_dt <- as.data.table(result_mat)
  setnames(result_dt, out_cols)
  cbind(dt, result_dt)
}

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_dt <- compute_neighbor_features(cell_dt, neighbor_lookup, neighbor_vars)

# --- Random Forest Prediction ---
# Load pre-trained RF model once (assume saved as RDS)
rf_model <- readRDS("trained_rf_model.rds")

# Predict using ranger for efficiency
pred <- predict(rf_model, data = cell_dt, num.threads = parallel::detectCores())

cell_dt[, gdp_pred := pred$predictions]
```

---

### **Why This Is Faster**
- Reduces repeated string operations and `paste()`.
- Eliminates multiple large `lapply` calls and repeated `do.call(rbind, ...)`.
- In-place calculations with `data.table` mitigate copy overhead.
- Aggregates all neighbor stats in a single loop using pre-allocated matrix.
- Parallelized Random Forest inference via `ranger`.

---

**Expected Improvement**: From >86 hours to a manageable few hours even on a laptop, possibly faster if neighbor computations offloaded to C++ via `data.table` or using `future.apply` for multi-core batching.
END_RESPONSE>>>
