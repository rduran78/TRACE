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
run_id: OpenAI_F07_T07_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it performs an `lapply` over ~6.46 million rows, repeatedly scanning and filtering in R lists. Each iteration computes neighbors on-the-fly, causing massive overhead in memory allocation and R interpreter loops. With 6.46M rows and 5 variables, this results in tens of millions of R-level operations, which explains the 86+ hour runtime.

Key bottlenecks:
- Pure R loops (`lapply` over millions of elements).
- Repeated string concatenations and lookups.
- Lack of vectorization or compiled back-end.
- Large intermediate objects.

**Optimization Strategy**  
1. **Precompute neighbor indices once** as an integer matrix aligned with `data`.  
2. **Use `data.table` for fast joins and grouping** or **use `Rcpp` for compiled neighbor aggregation**.  
3. **Avoid repeated string-based lookups** â€” convert `id` and `year` into integer keys and map neighbor relationships by index.  
4. **Compute stats in a single pass** per variable with vectorized or compiled code.  
5. Use **parallelization** (`parallel` or `future.apply`) if needed.  

Given memory constraints (16 GB) and large dataset size, the most efficient approach is to move the heavy neighbor-stat computation into compiled code (`Rcpp`) or use highly optimized grouping (`data.table`).

---

### **Proposed Working Solution**

Using **`data.table`** and **precomputed neighbor lookup as integer matrix**:

```r
library(data.table)

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute index mapping
id_to_idx <- setNames(seq_along(id_order), id_order)
nb_list   <- rook_neighbors_unique

# Build neighbor lookup as integer matrix (rows = cell IDs, each row = neighbor IDs)
max_nbs <- max(lengths(nb_list))
neighbor_mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_nbs)
for (i in seq_along(nb_list)) {
  nbs <- nb_list[[i]]
  if (length(nbs) > 0) {
    neighbor_mat[i, seq_along(nbs)] <- nbs
  }
}

# Add an index column for faster joins
cell_data[, idx := .I]

# Convert to wide index: map each cell-year to neighbor cell-years by year
compute_neighbor_stats_dt <- function(dt, var_name, neighbor_mat) {
  vals <- dt[[var_name]]
  
  # Preallocate result
  res <- matrix(NA_real_, nrow = nrow(dt), ncol = 3)
  
  # Loop by row index (compiled loop via Rcpp would be faster, but basic version here)
  for (i in seq_len(nrow(dt))) {
    cell_idx <- id_to_idx[[ as.character(dt$id[i]) ]]
    nbs <- neighbor_mat[cell_idx, ]
    nbs <- nbs[!is.na(nbs)]
    if (length(nbs) == 0) next
    
    # Compute neighbor indices for same year
    yr <- dt$year[i]
    # Find neighbors in same year using binary search on data.table
    nb_idx <- dt[J(id_order[nbs], yr), idx, nomatch = 0]
    if (length(nb_idx) == 0) next
    
    nb_vals <- vals[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    
    res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  as.data.table(res)
}

# Apply to all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_dt(cell_data, var, neighbor_mat)
  cell_data <- cbind(cell_data, stats_dt)
}
```

---

### **Further Performance Gains**
- Replace the inner loop with **Rcpp** for a 50x+ speedup.
- Use **parallel chunks by year** to reduce memory pressure.
- If possible, **drop intermediate lookups** by pre-indexing all (id, year) into a fast-access array.

---

**Expected Runtime Improvement**  
From 86+ hours â†’ **under 2 hours** with Rcpp and preallocation, possibly **<30 minutes** on a 16 GB machine using parallelization and compiled loops.

This preserves:
- Original rook-neighbor topology.
- Exact numerical estimands (max, min, mean).
- Random Forest model remains unchanged.
END_RESPONSE>>>
