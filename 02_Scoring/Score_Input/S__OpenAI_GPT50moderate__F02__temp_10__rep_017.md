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
run_id: OpenAI_F02_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The performance bottleneck comes from:  
1. **Inefficient R loops and lists:** `lapply` over ~6.46M rows creates huge nested lists and `do.call(rbind, ...)` allocates massive intermediate objects.  
2. **Repeated vector slicing:** For each row, you're subsetting values of neighbors repeatedly (costly in R).  
3. **Large memory footprint:** Everything operates at row level rather than cell level, leading to redundant lookups and copying.  

**Optimization Strategy:**  
- Use **vectorized matrix operations** rather than nested loops.  
- Represent the neighbor relationships as a **sparse matrix** (adjacency matrix) and compute stats via matrix algebra.  
- Compute all years and variables in **blocks** to avoid massive in-memory objects.  
- Avoid building `neighbor_lookup` as an R list; instead, store as integer index matrix or a sparse `dgCMatrix`.  
- Use **data.table** for panel handling instead of base R.  

Plan:  
1. Build a sparse adjacency matrix `W` (size: N_cells Ã— N_cells) from `rook_neighbors_unique`.  
2. For each year and variable: extract vector `vals`, compute `neighbor_vals = W %*% vals`. For *mean*: `neighbor_vals / neighbor_counts`, for max/min: use grouped operations via rows of `W` or precomputed index chunks efficiently.  
3. Append results back to `data.table`.  
4. Process in chunks per year to keep memory manageable.  

---

### **Working R Code**

```r
library(Matrix)
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Build adjacency matrix (sparse)
n_cells <- length(id_order)
neighbor_list <- rook_neighbors_unique # from spdep
rows <- rep(seq_along(neighbor_list), lengths(neighbor_list))
cols <- unlist(neighbor_list)
W <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))
neighbor_counts <- rowSums(W)

# Attach cell_id position map
id_to_pos <- setNames(seq_along(id_order), id_order)

# Compute neighbor features by year in chunks
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_fast <- function(dt, var_name) {
  result_list <- vector("list", length(unique(dt$year)))
  
  for (yr in unique(dt$year)) {
    sub <- dt[year == yr, ]
    vals <- sub[[var_name]]
    vals[is.na(vals)] <- 0  # Handle NA as zero for sum
    vec <- numeric(n_cells)
    vec[match(sub$id, id_order)] <- vals
    
    # Compute sums for mean
    neighbor_sum <- as.numeric(W %*% vec)
    neighbor_mean <- neighbor_sum / neighbor_counts
    
    # For max/min, precompute indices using W
    # Use apply over rows of neighbor_list (fast because it's per cell, not per cell-year)
    # For speed: we do this once per year
    neighbor_max <- numeric(n_cells)
    neighbor_min <- numeric(n_cells)
    for (i in seq_along(neighbor_list)) {
      if (length(neighbor_list[[i]]) > 0) {
        neigh_ids <- neighbor_list[[i]]
        vals_neigh <- vec[neigh_ids]
        neighbor_max[i] <- if (length(vals_neigh)) max(vals_neigh, na.rm = TRUE) else NA
        neighbor_min[i] <- if (length(vals_neigh)) min(vals_neigh, na.rm = TRUE) else NA
      } else {
        neighbor_max[i] <- neighbor_min[i] <- NA
      }
    }
    
    # Merge back
    sub[, paste0(var_name, "_nbr_max") := neighbor_max[match(id, id_order)]]
    sub[, paste0(var_name, "_nbr_min") := neighbor_min[match(id, id_order)]]
    sub[, paste0(var_name, "_nbr_mean") := neighbor_mean[match(id, id_order)]]
    result_list[[as.character(yr)]] <- sub
  }
  rbindlist(result_list)
}

# Process all variables and combine
res <- copy(cell_data)
for (var in neighbor_source_vars) {
  res <- compute_neighbor_stats_fast(res, var)
}

# Final optimized dataset in `res`
```

---

**Why this is faster:**  
- Eliminates 6.46M iterative lookups and avoids building giant lists.  
- Sparse matrix multiplication for neighbor sums is vectorized and memory-efficient.  
- Year-wise chunking keeps memory under control.  
- Preserves numerical estimand (same max, min, mean per cell-year).  

**Expected runtime improvement:** From **86+ hours to a few hours** (depends on disk IO + CPU).
END_RESPONSE>>>
