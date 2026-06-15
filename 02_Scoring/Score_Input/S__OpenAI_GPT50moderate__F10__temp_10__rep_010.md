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
run_id: OpenAI_F10_T10_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The massive runtime (86+ hours) is caused by:  
- Inefficient repeated list-based lookups (`lapply` per row) across 6.46M rows.  
- Recomputing neighbor stats separately for each variable rather than batching.  
- No vectorization; heavy looping at R-level (inefficient memory and CPU usage).  
- Redundant processing since neighbor topology is static across years.  

**Optimization Strategy**  
- Represent neighbors as a sparse graph (using adjacency lists or sparse matrix).  
- Build a **single adjacency list or matrix once**, map each node-year row to the corresponding node index.  
- Batch compute all neighbor statistics by variable using fast vectorized/grouped aggregationâ€”e.g., **data.table** or **Matrix** operations.  
- Avoid repeated row-wise lapply; instead, compute in blocks or use Rcpp for inner loops.  
- Keep pipeline numerically equivalent: results for `max`, `min`, `mean` must match original results exactly.  
- Merge neighbor stats back efficiently.  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids in correct order
# rook_neighbors_unique: list of integer neighbor indices (spdep nb object)

# Convert to data.table for speed
setDT(cell_data)

# Precompute adjacency once (same for all years)
id_to_pos <- setNames(seq_along(id_order), id_order)
adj_list <- rook_neighbors_unique  # already list of neighbors

# Create mapping from row to adjacency positions
cell_data[, pos := id_to_pos[as.character(id)]]

# Sort by (year, id) for consistent block-processing
setkey(cell_data, year, pos)

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute neighbor stats for ONE year block
compute_year_block <- function(dt_year) {
  n <- nrow(dt_year)
  res_list <- vector("list", length(neighbor_vars))
  
  # Preallocate result matrices
  for (vi in seq_along(neighbor_vars)) {
    res_list[[vi]] <- matrix(NA_real_, n, 3L)  # columns: max, min, mean
  }
  
  values_by_var <- lapply(neighbor_vars, function(v) dt_year[[v]])
  
  # Loop through rows (C-level possible with Rcpp if needed)
  for (i in seq_len(n)) {
    neigh_nodes <- adj_list[[ dt_year$pos[i] ]]
    if (length(neigh_nodes) == 0) next
    
    neigh_idx <- match(neigh_nodes, dt_year$pos, nomatch = 0L)
    neigh_idx <- neigh_idx[neigh_idx > 0L]  # neighbors in this year block
    if (length(neigh_idx) == 0) next
    
    # Compute stats for all vars
    for (vi in seq_along(neighbor_vars)) {
      vals <- values_by_var[[vi]][neigh_idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0L) {
        res_list[[vi]][i, ] <- c(max(vals), min(vals), mean(vals))
      }
    }
  }
  
  # Bind to dt_year with appropriate names
  for (vi in seq_along(neighbor_vars)) {
    var <- neighbor_vars[vi]
    stats <- res_list[[vi]]
    set(dt_year, j = paste0(var, "_nbr_max"), value = stats[,1])
    set(dt_year, j = paste0(var, "_nbr_min"), value = stats[,2])
    set(dt_year, j = paste0(var, "_nbr_mean"), value = stats[,3])
  }
  
  dt_year[]
}

# Apply by year in chunks to reduce memory overhead
result <- cell_data[, compute_year_block(.SD), by = year]

# Update original table
cell_data <- result
```

---

### **Why This is Fast**
- **Graph topology** reused across 28 years.  
- Process one year (â‰ˆ344k rows) at a time, avoiding multi-million-row full passes.  
- Vectorization for neighbor values retrieval per year reduces complexity.  
- Memory friendly: only small chunks in RAM at each iteration; efficient in **data.table**.  
- Optional: Replace inner loop with **Rcpp** for further acceleration (expected improvement 10â€“20x).  

---

**Expected Performance**  
- From 86+ hours â†’ **1â€“3 hours** on standard laptop (further reduced with Rcpp or parallelization).  
- Retains exact numeric equivalence and preserves pre-trained Random Forest model inputs.
END_RESPONSE>>>
