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
run_id: OpenAI_F10_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current implementation (`lapply`-based) computes neighbor max, min, mean for ~6.46M rows, repeatedly scanning vectors for each observation.  
- Lookup building is repeated and not vectorized; neighbor traversal happens in pure R loops, causing high overhead.  
- For each row, dynamic indexing and multiple `!is.na` checks dominate runtime.  
- Overall complexity: `O(N * avg_degree)` in R loops â‡’ **86+ hrs** on 16GB system.  
- Key inefficiency: non-vectorized aggregation, redundant list traversals, and memory fragmentation.  

---

**Optimization Strategy**  
1. **Convert to graph representation once** using a sparse adjacency structure (CSR-like).  
2. **Exploit sorted panel data**: rows for the same cell by year allow block addressing.  
3. Use **`data.table`** or **matrix-based aggregation** to compute neighbor stats for each year in bulk.  
4. Precompute `(id â†’ rows)` mapping per year, then use vectorized join or matrix indexing instead of per-row loops.  
5. Minimize R overhead: fully vectorized or partially compiled (`Rcpp`) implementation.  
6. **Reuse neighbor graph for all years**; only attribute vectors change by year.  

---

**Working R Code (Efficient Implementation)**  

```r
library(data.table)

# Assume: cell_data[id, year, ntl, ec, pop_density, def, usd_est_n2, ...]
# Build adjacency once
build_sparse_adj <- function(id_order, rook_neighbors) {
  n <- length(id_order)
  src <- rep(seq_len(n), lengths(rook_neighbors))
  dst <- unlist(rook_neighbors, use.names = FALSE)
  list(src = src, dst = dst)  # directed edges
}

# Precompute row index map: cell-year to row position
prepare_key_map <- function(cell_data) {
  setDT(cell_data)
  setkey(cell_data, id, year)
  cell_data
}

# Vectorized neighbor aggregation by year
compute_neighbor_stats_all <- function(cell_data, adj, years, vars) {
  setDT(cell_data)
  n_years <- length(years)
  
  for (v in vars) {
    max_col <- paste0(v, "_nbr_max")
    min_col <- paste0(v, "_nbr_min")
    mean_col <- paste0(v, "_nbr_mean")
    cell_data[, c(max_col, min_col, mean_col) := .(NA_real_, NA_real_, NA_real_)]
  }
  
  # Loop only over years (28 iterations)
  for (yr in years) {
    # Subset rows for this year
    year_rows <- which(cell_data$year == yr)
    vals_mat <- as.matrix(cell_data[year_rows, ..vars])
    
    # Map cell IDs to row positions in this year
    id_to_pos <- integer(max(cell_data$id))
    id_to_pos[cell_data$id[year_rows]] <- seq_along(year_rows)
    
    # For each edge, get source & target row indices for this year
    src_idx <- id_to_pos[adj$src]
    dst_idx <- id_to_pos[adj$dst]
    
    # Remove edges where neighbor absent in this year (rare if full panel)
    valid <- which(src_idx > 0 & dst_idx > 0)
    src_idx <- src_idx[valid]
    dst_idx <- dst_idx[valid]
    
    # Aggregate using data.table fast grouping
    dt_edges <- data.table(src = src_idx, dst = dst_idx)
    
    for (j in seq_along(vars)) {
      w <- vals_mat[dst_idx, j]
      # compute max, min, mean per src node
      stats <- dt_edges[, .(max_v = max(w, na.rm = TRUE),
                             min_v = min(w, na.rm = TRUE),
                             mean_v = mean(w, na.rm = TRUE)),
                        by = src]
      # assign back
      target_rows <- year_rows[stats$src]
      set(cell_data, i = target_rows, j = paste0(vars[j], "_nbr_max"), value = stats$max_v)
      set(cell_data, i = target_rows, j = paste0(vars[j], "_nbr_min"), value = stats$min_v)
      set(cell_data, i = target_rows, j = paste0(vars[j], "_nbr_mean"), value = stats$mean_v)
    }
  }
  cell_data
}

# Usage
adj <- build_sparse_adj(id_order, rook_neighbors_unique)
cell_data <- prepare_key_map(cell_data)
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

system.time({
  cell_data <- compute_neighbor_stats_all(cell_data, adj, years, neighbor_vars)
})
```

---

**Why This Works**  
- **Adjacency built once** â†’ reused for all variables and years.  
- **Year loop only 28 iterations**; inside, highly vectorized calculations leverage `data.table` grouping.  
- Complexity drops to `O(E * vars)` with minimal R overhead (`E â‰ˆ 1.37M edges`).  
- Fits in memory: operates on year blocks (~230k rows) instead of entire 6.46M rows at once.  
- Preserves exact logic: `NA` handling identical, same max, min, mean semantics.  

**Expected runtime:** from 86+ hrs â†’ **under 1 hour** (often minutes) on 16GB laptop.  
**Random Forest model remains unchanged.**  

This is the most computationally efficient approach in R for the given constraints.
END_RESPONSE>>>
