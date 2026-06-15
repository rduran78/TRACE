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
run_id: OpenAI_F07_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because:  
- It computes neighbor statistics in R loops over ~6.46 million rows (`lapply` per row) repeatedly for 5 variables.  
- Each iteration re-allocates vectors and repeatedly performs character-based lookups.  
- Memory thrash: multiple large lists and string operations kill performance and RAM.  
- Complexity â‰ˆ O(N Ã— avg_neighbors), with large overhead from Râ€™s list and string handling.  

---

**Optimization Strategy**  
1. **Convert to numeric indices early** and avoid repeated string concatenation.  
2. **Unnest neighbor relationships once** into a long numeric table joined by `data.table`.  
3. **Compute max/min/mean in grouped operations** using `data.table` or `collapse` for vectorized speed (>100Ã— faster).  
4. **Precompute neighbor links across years**: replicate spatial adjacency for all years as a single integer mapping.  

The key idea: turn the neighbor relation into a **two-column table** of `(row_id, neighbor_row_id)` and join on values in one pass.  

---

**Working R Code (Fast, Memory-Efficient)**  

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor indices (length = number of unique cell IDs)
# id_order: original cell IDs in the same order as rook_neighbors_unique

compute_neighbor_features_fast <- function(cell_data, id_order, neighbors, vars) {
  setDT(cell_data)
  # Map cell id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  cell_data[, ref_idx := id_to_ref[as.character(id)]]
  
  # Build adjacency for all rows across time:
  # For each row, expand neighbors
  message("Building adjacency table...")
  cell_ids <- cell_data$ref_idx
  years    <- cell_data$year
  
  # Generate mapping row_id -> neighbor_row_id
  # Precompute cumulative lengths for efficient unlisting
  n_per_cell <- lengths(neighbors[cell_ids])
  total_links <- sum(n_per_cell)
  
  row_idx <- rep(seq_along(cell_ids), n_per_cell)
  neigh_id <- unlist(neighbors[cell_ids], use.names = FALSE)
  
  # Convert neighbor IDs to ref positions
  neigh_ref <- neigh_id
  # Now join by year: replicate for same year
  # Build neighbor_row_id by matching (cell, year)
  # Create key of id + year -> row_id
  cell_data[, key := paste(ref_idx, year, sep = "_")]
  idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)
  
  # Key for neighbors
  neigh_keys <- paste(neigh_ref[row_idx], years[row_idx], sep = "_")
  neigh_row_ids <- idx_lookup[neigh_keys]
  
  # Drop NA (years where no matching record)
  valid <- !is.na(neigh_row_ids)
  row_idx <- row_idx[valid]
  neigh_row_ids <- neigh_row_ids[valid]
  
  # Build edge list table
  edges <- data.table(src = row_idx, nbr = neigh_row_ids)
  
  # Compute features for all vars
  for (v in vars) {
    message("Computing neighbor stats for ", v, "...")
    nbr_vals <- cell_data[[v]][edges$nbr]
    # Add neighbor values next to src
    edges[, val := nbr_vals]
    
    # Aggregate: max, min, mean per src
    agg <- edges[!is.na(val), .(max = max(val), min = min(val), mean = mean(val)), by = src]
    
    # Join back to cell_data by src -> row
    setnames(agg, c("src", paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
    cell_data[agg$src, c(paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))) := agg[, -1]]
    
    edges[, val := NULL]  # Clean up for next var
  }
  
  cell_data[, ref_idx := NULL]
  cell_data[, key := NULL]
  cell_data
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

system.time({
  cell_data <- compute_neighbor_features_fast(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
})
```

---

**Why This is Faster**  
- Removes repeated string concatenation in the inner loop.  
- Uses **vectorized joins and aggregations via data.table** (C-level performance).  
- Builds the adjacency edge list **once**, then reuses it for all variables.  
- Memory footprint is linear, no deep nested lists per row.  

**Expected Gain**  
From 86+ hours â†’ likely under **1 hour** (often 50â€“200Ã— speedup for this scale on 16 GB RAM).  

**Preserves:**  
- Original rook-neighbor topology  
- Year alignment per observation  
- Original numerical estimand  
- No retraining of the Random Forest model  

---  
**Bottom Line:** Transform the problem into an **edge aggregation** using data.table and reuse structure for all variables.
END_RESPONSE>>>
