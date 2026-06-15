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
run_id: OpenAI_F10_T03_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It repeatedly computes neighbor statistics for each variable using `lapply` over 6.46M rows, causing heavy R-level loops and memory overhead.  
- Neighbor lookup is recomputed for each row and variable rather than vectorized or aggregated.  
- No use of efficient matrix operations or sparse representations despite the graph being sparse (â‰ˆ1.37M edges vs. 6.46M nodes).  

**Optimization Strategy**  
- Represent the neighbor relationships as a sparse adjacency matrix (rows = cell-year rows, columns = same).  
- Build this adjacency structure once and reuse it for all variables.  
- Use fast matrix operations (`Matrix` package) to compute max, min, and mean for neighbors.  
- Process data year by year to keep memory usage manageable (since adjacency across years is block-diagonal).  
- Append computed features back to `cell_data` without altering the Random Forest model or estimand.  

---

### **Efficient R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data (data.table) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: list of integer neighbor indices (spdep::nb object)
# neighbor_source_vars: c("ntl", "ec", "pop_density", "def", "usd_est_n2")

setDT(cell_data)
setkey(cell_data, id, year)

years <- sort(unique(cell_data$year))
n_ids <- length(id_order)
n_years <- length(years)

# Build adjacency for one year (block reused)
adj_list <- rook_neighbors_unique
row_idx <- rep(seq_along(adj_list), lengths(adj_list))
col_idx <- unlist(adj_list)
adj_one_year <- sparseMatrix(i = row_idx, j = col_idx, x = 1, dims = c(n_ids, n_ids))

compute_stats_for_year <- function(dt_year, adj, vars) {
  # dt_year: data.table for one year, sorted by id_order
  setorder(dt_year, id)
  res_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- dt_year[[vars[v]]]
    # Compute neighbor means
    sums <- as.numeric(adj %*% vals)
    counts <- rowSums(adj)
    means <- ifelse(counts > 0, sums / counts, NA_real_)
    
    # Compute neighbor max/min efficiently
    # For sparse graph, loop over adjacency rows
    maxs <- mins <- numeric(length(vals))
    maxs[] <- NA_real_
    mins[] <- NA_real_
    for (i in seq_len(n_ids)) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) > 0) {
        nbr_vals <- vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) > 0) {
          maxs[i] <- max(nbr_vals)
          mins[i] <- min(nbr_vals)
        }
      }
    }
    
    res_list[[v]] <- data.table(
      paste0(vars[v], "_nbr_max") = maxs,
      paste0(vars[v], "_nbr_min") = mins,
      paste0(vars[v], "_nbr_mean") = means
    )
  }
  
  cbind(dt_year, do.call(cbind, res_list))
}

# Process year by year
result_list <- vector("list", length(years))
for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  result_list[[i]] <- compute_stats_for_year(dt_year, adj_one_year, neighbor_source_vars)
}

cell_data_enriched <- rbindlist(result_list)
```

---

### **Key Improvements**
- **Adjacency built once** and reused for all years (block-diagonal structure handled by looping over years).
- **Matrix multiplication** for neighbor means (fast, vectorized).
- **Loop for max/min** only over neighbors per node (â‰ˆ1.37M edges, far less than 6.46M Ã— 5 variables).
- **Year-by-year processing** keeps memory footprint low (â‰ˆ344K rows per year).
- Preserves **numerical equivalence** with original pipeline.

**Expected runtime**: Hours reduced from 86+ to a few hours or less on a 16 GB laptop.
END_RESPONSE>>>
