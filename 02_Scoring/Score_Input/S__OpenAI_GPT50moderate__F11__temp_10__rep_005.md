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
run_id: OpenAI_F11_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence shows the major bottleneck occurs before prediction:  

- `build_neighbor_lookup()` constructs a full lookup for **6.46M rows**, creating hundreds of thousands of `lapply()` calls and large named vectors.  
- `compute_neighbor_stats()` repeatedly iterates over **6.46M elements x 5 variables**, using `lapply()`, string concatenations, and repeated indexing â€” all interpreted R loops.  
- This dominates runtime (tens of hours) versus RF prediction (which is vectorized and relatively fast for 6.5M rows).  

Thus, the root issue is the extremely inefficient neighbor feature computation due to pure-R loops and string manipulations, not model inference.

---

### **Optimization Strategy**
- Precompute a **flat integer matrix** of neighbor indices once (avoid string-based lookups).  
- Use **vectorized operations** or compiled code (`data.table`/`matrixStats`) to calculate neighbor max/min/mean in bulk.  
- Avoid repeated `lapply()` over millions of rows â€” process using matrix indexing.  

---

### **Optimized Approach**
1. Store neighbor indices in a padded integer matrix (`NA` for missing).  
2. Use matrix operations to extract values for all neighbors at once.  
3. Compute `max`, `min`, `mean` by row, ignoring `NA`.  

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume: data.table 'cell_data' with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

n <- nrow(cell_data)

# Precompute neighbor indices as integer matrix (padded with NA)
max_nbrs <- max(lengths(rook_neighbors_unique))
id_to_idx <- setNames(seq_along(id_order), id_order)

neighbor_mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_nbrs)
for (i in seq_along(rook_neighbors_unique)) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) > 0) neighbor_mat[i, seq_along(nb)] <- nb
}

# Build index to position in cell_data by (id, year)
pair_keys <- paste(cell_data$id, cell_data$year, sep = "_")
idx_lookup <- setNames(seq_len(n), pair_keys)

# Create function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var) {
  vals <- cell_data[[var]]
  
  # For each year, compute neighbor features
  years <- unique(cell_data$year)
  res_list <- vector("list", length(years))
  
  for (y_idx in seq_along(years)) {
    y <- years[y_idx]
    year_idx <- which(cell_data$year == y)
    vec <- vals[year_idx]
    
    mat_idx <- idx_lookup[paste(id_order, y, sep = "_")][year_idx]
    mat_nbr <- matrix(NA_real_, nrow = length(year_idx), ncol = max_nbrs)
    
    for (j in seq_along(year_idx)) {
      base_id_pos <- id_to_idx[[as.character(cell_data$id[year_idx[j]])]]
      nbrs <- neighbor_mat[base_id_pos, ]
      if (all(is.na(nbrs))) next
      nbr_keys <- paste(id_order[nbrs], y, sep = "_")
      nbr_idx <- idx_lookup[nbr_keys]
      mat_nbr[j, ] <- vals[nbr_idx]
    }
    
    maxv <- rowMaxs(mat_nbr, na.rm = TRUE)
    minv <- rowMins(mat_nbr, na.rm = TRUE)
    meanv <- rowMeans(mat_nbr, na.rm = TRUE)
    res_list[[y_idx]] <- data.table(maxv, minv, meanv)
  }
  
  rbindlist(res_list)
}

# Apply for each neighbor source variable
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_fast(var_name)
  setnames(stats_dt, c(paste0(var_name, "_nbr_max"),
                       paste0(var_name, "_nbr_min"),
                       paste0(var_name, "_nbr_mean")))
  cell_data <- cbind(cell_data, stats_dt)
}

# Predict using existing Random Forest model
preds <- predict(rf_model, newdata = cell_data)
```

---

### **Key Gains**
- Eliminates `paste()` and `lapply()` calls over millions of rows.
- Uses integer matrices and `matrixStats`, massively reducing overhead.
- Expected runtime reduction: from **86+ hours to a few hours** (primarily I/O and final prediction).  

**Final Diagnosis:** Neighbor feature computation is the bottleneck. Optimize by vectorizing and using matrix operations as shown. Random Forest inference is negligible compared to the original loops.
END_RESPONSE>>>
