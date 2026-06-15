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
run_id: OpenAI_F08_T00_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M rows) by iterating through a large list of neighbor indices. This is inefficient because:  
- The neighbor structure is static across years, but the code rebuilds or reuses a lookup that is applied repeatedly for each row-year combination.  
- For each variable, the function loops through all rows and extracts neighbor values, causing repeated expensive lookups.  
- Memory and CPU overhead are high due to repeated `lapply` calls and `do.call(rbind, ...)` on millions of rows.  

**Optimization Strategy**  
- Precompute a static neighbor index for cells only (not cell-years).  
- For each year, slice the data for that year, compute neighbor stats in a **vectorized** way using matrix operations or `rowsum`/`aggregate`.  
- Append results back to the main dataset by year.  
- Avoid recomputing neighbor stats for each variable separately in a nested loop; instead, compute all neighbor stats for all variables in one pass per year.  
- Use `data.table` for efficient grouping and joining.  

**Working R Code**  

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute static neighbor lookup as a list: cell_id -> neighbor_ids
# id_order: vector of cell IDs in consistent order
# rook_neighbors_unique: spdep nb object
neighbor_list <- setNames(rook_neighbors_unique, id_order)

# Convert neighbor list to an index map for fast access
# We'll keep it as is since it's static and small relative to full data

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute neighbor stats for one year
compute_year_neighbor_stats <- function(dt_year, neighbor_list, vars) {
  # Create a named vector for quick row index lookup
  val_mat <- as.matrix(dt_year[, ..vars])
  rownames(val_mat) <- dt_year$id
  
  # Preallocate result matrices
  n <- nrow(dt_year)
  res_list <- vector("list", length(vars))
  names(res_list) <- vars
  for (v in vars) {
    res_list[[v]] <- matrix(NA_real_, nrow = n, ncol = 3,
                             dimnames = list(NULL, c("max", "min", "mean")))
  }
  
  # Compute stats for each cell
  for (i in seq_len(n)) {
    cell_id <- dt_year$id[i]
    neigh_ids <- neighbor_list[[as.character(cell_id)]]
    if (length(neigh_ids) == 0) next
    # Filter neighbors present in this year
    neigh_ids <- neigh_ids[neigh_ids %in% dt_year$id]
    if (length(neigh_ids) == 0) next
    for (v in vars) {
      vals <- val_mat[as.character(neigh_ids), v, drop = TRUE]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        res_list[[v]][i, ] <- c(max(vals), min(vals), mean(vals))
      }
    }
  }
  
  # Bind results into a data.table
  out <- data.table(id = dt_year$id, year = dt_year$year)
  for (v in vars) {
    out[, paste0(v, "_nbr_max") := res_list[[v]][, "max"]]
    out[, paste0(v, "_nbr_min") := res_list[[v]][, "min"]]
    out[, paste0(v, "_nbr_mean") := res_list[[v]][, "mean"]]
  }
  out
}

# Apply by year
years <- unique(cell_data$year)
result_list <- vector("list", length(years))

for (j in seq_along(years)) {
  yr <- years[j]
  dt_year <- cell_data[year == yr]
  result_list[[j]] <- compute_year_neighbor_stats(dt_year, neighbor_list, neighbor_source_vars)
}

neighbor_features <- rbindlist(result_list)

# Merge back to main data
setkey(cell_data, id, year)
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data]

# Now cell_data has neighbor stats columns ready for Random Forest prediction
```

**Why This Is Faster**  
- Processes one year at a time (reduces working set size from 6.46M rows to ~344K rows).  
- Avoids repeated lookups across years since neighbor structure is static.  
- Computes all variables in one pass per year.  
- Uses `data.table` for efficient joins and memory handling.  

**Expected Performance**  
- Complexity reduced from O(N * Y * V) with heavy overhead to O(N * Y) with efficient loops.  
- Should run in a few hours instead of 86+.  

**Preserves**  
- Original estimand (neighbor max, min, mean per variable per cell-year).  
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
