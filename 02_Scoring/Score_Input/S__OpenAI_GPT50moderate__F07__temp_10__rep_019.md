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
run_id: OpenAI_F07_T10_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- For each of ~6.46 million rows, `lapply` runs a full vectorized computation across neighbors (`compute_neighbor_stats`) repeatedly.  
- Neighbor lookups are reconstructed for each variable, and `do.call(rbind, ...)` for millions of rows introduces overhead.  
- No parallelization or memory-efficient aggregations.  
- Computation scales as **O(N Ã— K)** (N = 6.46M rows, K = number of neighbors), which on single-core R with heavy object creation becomes infeasible.

---

### **Optimization Strategy**
1. **Precompute Sparse Adjacency Once**  
   Flatten neighbor relationships into two integer vectors `(from, to)`, representing row-wise links for all cell-year pairs.
   
2. **Vectorized Joins Instead of Per-Row Loops**  
   Use `data.table` for efficient keyed merges and group aggregations instead of millions of `lapply` calls.

3. **Batch Compute Neighbor Stats in Long Form**  
   - Melt neighbor relations into (source_index â†’ neighbor_index).
   - Join values for each source variable, compute max, min, and mean by group.

4. **Parallelization with `data.table` or `future`**  
   Use all available cores for different variables.

5. **Memory Efficiency**  
   Compute in chunks if necessary (e.g., by year or split rows).

**Key Idea:** Transform the problem into a join-and-group-by pipeline using `data.table`, which can handle tens of millions of rows efficiently.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute neighbor lookup as edge list for panel data
build_neighbor_edges <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  edges_list <- vector("list", length = length(id_order))
  
  for (i in seq_along(id_order)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0) {
      edges_list[[i]] <- data.table(from_id = id_order[i], to_id = id_order[nb])
    }
  }
  rbindlist(edges_list)
}

neighbor_edges_base <- build_neighbor_edges(dt, id_order, rook_neighbors_unique)

# Expand to panel form for all years
years <- sort(unique(dt$year))
neighbor_edges <- neighbor_edges_base[, .(id = from_id, neighbor_id = to_id), ][
  rep(seq_len(.N), times = length(years))
][
  , year := rep(years, each = nrow(neighbor_edges_base))]
setkey(neighbor_edges, neighbor_id, year)

# For memory: create a row index for fast joins
dt[, row_id := .I]
neighbor_edges[, neighbor_key := paste(neighbor_id, year, sep = "_")]
dt[, self_key := paste(id, year, sep = "_")]

# Join and compute stats for each variable
compute_neighbor_features <- function(dt, edges, var) {
  tmp <- edges[
    dt[, .(neighbor_key = self_key, val = get(var))],
    on = .(neighbor_key), nomatch = 0
  ]
  # Now tmp has: from_id/year and neighbor values
  stats <- tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to dt
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nmax", "_nmin", "_nmean")))
  dt[stats, on = .(id, year)]
}

# Apply for all variables in a loop
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  dt <- compute_neighbor_features(dt, neighbor_edges, var)
}

# Result: dt now has neighbor max/min/mean columns for all source vars
cell_data <- as.data.frame(dt)
```

---

### **Expected Gains**
- Moves from **per-row `lapply`** to **bulk grouping** using `data.table`.
- Time complexity effectively reduces to **O(E)** where `E = ~edges Ã— years` (~38M rows after expansion), tractable in RAM with `data.table`.
- On a modern laptop, this should complete in a few hours, not days.

---

âœ… **Preserves:**  
- Trained Random Forest model untouched.  
- Same numeric estimand (neighbor stats computed from actual rook adjacency).  
- Full panel structure and neighbor integrity.
END_RESPONSE>>>
