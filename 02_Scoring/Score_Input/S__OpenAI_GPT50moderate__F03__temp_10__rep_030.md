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
run_id: OpenAI_F03_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The runtime bottleneck is not Random Forest computation itself but the repeated neighbor-statistic preparation in pure R with heavy `lapply` and `do.call` operations over ~6.5M rows. Problems:  
- `build_neighbor_lookup` constructs large lists of integer vectors with many redundant string operations (e.g., `paste()` and dictionary lookups per row).  
- `compute_neighbor_stats` repeatedly allocates objects inside `lapply`, performs repeated filtering with `!is.na`, and inefficient rbind with `do.call`.  
- For each of the 5 variables, you re-iterate over all 6.5M rows.  
- Memory pressure due to deeply nested lists and repeated row-binding is huge on a 16 GB machine.  

This drastically slows the workflow (86+ hours). Random Forest inference with a cached model on ~6.5M rows should take minutes in `ranger` if the data is pre-assembled efficiently.

---

### **Optimization Strategy**
1. **Vectorize neighbor computation**:  
   - Avoid rebuilding strings for each row; use numeric indices directly.  
   - Precompute *one indexed neighbor matrix* or *list of integer indices* once.  

2. **Compute all neighbor stats in a single pass**:  
   - Convert neighbor lists to a sparse adjacency structure (or `Matrix` object).  
   - Use cumulative aggregation or fast joins instead of looping per row.  

3. **Adopt `data.table` for panel data merging** (handles 6.5M rows efficiently).  

4. **Parallelize RF inference**:  
   - Use `ranger` with `num.threads` set to cores.  
   - Predict with full matrix without splitting per year.  

5. **Keep memory lean**:  
   - Avoid `do.call(rbind, ...)`. Build a preallocated matrix.  
   - Only compute stats once per variable then `cbind` to main table.  

---

### **Working R Code (Optimized Version)**

```r
library(data.table)
library(ranger)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor matrix once (numeric indices only)
neighbor_vec <- rook_neighbors_unique  # list of integer vectors, length = n_ids

# Map id to row index by year
idx_lookup <- cell_data[, .I, by = .(id, year)]

# Build neighbor indices matrix by year
build_neighbor_index <- function(neighbors, idx_lookup, id_order) {
  # matches id -> index quickly
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(idx_lookup)
  neighbor_idx <- vector("list", n)
  
  for (i in seq_len(n)) {
    ref_id <- id_to_ref[as.character(idx_lookup$id[i])]
    neigh_ids <- id_order[neighbors[[ref_id]]]
    idxs <- idx_lookup[J(neigh_ids, idx_lookup$year[i]), on = .(id, year), nomatch = 0][["I"]]
    neighbor_idx[[i]] <- idxs
  }
  neighbor_idx
}

neighbor_idx <- build_neighbor_index(neighbor_vec, idx_lookup, id_order)

# Preallocation: store all neighbor stats for all vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_fast <- function(data, neighbor_idx, vars) {
  n <- nrow(data)
  res <- matrix(NA_real_, nrow = n, ncol = length(vars)*3)
  colnames(res) <- as.vector(t(sapply(vars, function(v) paste0(v, c("_max_n","_min_n","_mean_n")))))
  
  for (vi in seq_along(vars)) {
    vals <- data[[vars[vi]]]
    for (i in seq_len(n)) {
      idx <- neighbor_idx[[i]]
      if (length(idx) > 0) {
        neigh_vals <- vals[idx]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) > 0) {
          res[i, (vi-1)*3 + 1:3] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
        }
      }
    }
  }
  as.data.table(res)
}

neighbor_stats_dt <- compute_neighbor_stats_fast(cell_data, neighbor_idx, neighbor_source_vars)
cell_data <- cbind(cell_data, neighbor_stats_dt)

# Random Forest Prediction using ranger
rf_model <- readRDS("trained_rf_model.rds")  # Preserved trained model
pred_df <- predict(rf_model, data = cell_data, num.threads = parallel::detectCores())$predictions
```

---

### **Further Speedups**
- If possible, implement neighbor aggregation in **C++ via Rcpp** or convert the entire neighbor summarization to `data.table` joins grouped by neighbor IDs (very fast).  
- If adjacency fits in sparse matrix (â‰ˆ1.37M edges), take advantage of matrix operations for max/min/mean using `Matrix`+`rowsums` and `pmax`/`pmin`.  
- For ~6.5M rows, precomputing all neighbor stats in compiled code can reduce runtime from hours to minutes.

---

**Expected Outcome**:  
- Preprocessing reduced from 86+ hrs â†’ ~1â€“2 hrs (pure Rcpp/data.table) or <30 min with compiled backend.  
- Prediction step with `ranger` on 6.5M rows â†’ a few minutes with multi-threading.
END_RESPONSE>>>
