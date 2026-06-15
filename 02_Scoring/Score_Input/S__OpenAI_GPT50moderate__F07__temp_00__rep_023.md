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
run_id: OpenAI_F07_T00_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F07_OpenAI_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Neighbor lookups are recomputed for every row and every variable, causing redundant work.  
- The approach is not vectorized and does not leverage efficient data structures or compiled code.  
- Memory overhead is high due to repeated object creation.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as an integer matrix or list of integer vectors aligned with `data` rows.  
2. **Avoid repeated string concatenation** (`paste`) and hash lookups inside loops.  
3. **Vectorize aggregation**: Use `data.table` or `matrixStats` to compute max, min, mean in bulk.  
4. **Process by year**: Since neighbors are within the same year, split by year and compute in chunks to reduce memory pressure.  
5. **Parallelize**: Use `parallel::mclapply` or `future.apply` for multi-core execution.  
6. **Preserve model**: Do not retrain; only compute new features efficiently.  

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)
library(parallel)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: id -> neighbor ids
id_to_neighbors <- rook_neighbors_unique
names(id_to_neighbors) <- as.character(id_order)

# Split by year for memory efficiency
years <- sort(unique(cell_data$year))

# Function to compute neighbor stats for one year
compute_year_stats <- function(yr) {
  dt <- cell_data[year == yr]
  n <- nrow(dt)
  
  # Map row index for fast lookup
  idx_lookup <- setNames(seq_len(n), as.character(dt$id))
  
  # Precompute neighbor indices for this year
  neighbor_idx <- lapply(dt$id, function(cid) {
    neigh_ids <- id_to_neighbors[[as.character(cid)]]
    idx_lookup[as.character(neigh_ids)]
  })
  
  # Convert to list of integer vectors
  neighbor_idx <- lapply(neighbor_idx, function(x) as.integer(x[!is.na(x)]))
  
  # Compute stats for each variable
  res_list <- list()
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    
    # Preallocate result matrix
    mat <- matrix(NA_real_, nrow = n, ncol = 3)
    
    for (i in seq_len(n)) {
      idx <- neighbor_idx[[i]]
      if (length(idx) > 0) {
        neigh_vals <- vals[idx]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) > 0) {
          mat[i, ] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
        }
      }
    }
    
    colnames(mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    res_list[[var_name]] <- mat
  }
  
  cbind(dt, do.call(cbind, res_list))
}

# Parallel execution by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cl <- detectCores() - 1
result_list <- mclapply(years, compute_year_stats, mc.cores = cl)

# Combine all years
cell_data_enriched <- rbindlist(result_list)

# Preserve trained model: no retraining, just use cell_data_enriched for prediction
```

---

**Expected Performance Gains**  
- Eliminates repeated string operations and redundant lookups.  
- Processes by year to keep memory manageable.  
- Parallelization across years uses all CPU cores.  
- Still uses R loops inside each year, but only over ~12k rows/year instead of 6.46M at once.  
- Estimated runtime: **hours â†’ minutes** on a 16 GB laptop.  

**Numerical Estimand Preserved**:  
- Same rook-neighbor relationships.  
- Same max, min, mean definitions.  
- No change to trained Random Forest model.
END_RESPONSE>>>
