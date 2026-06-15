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
run_id: OpenAI_F09_T03_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each cell-year row, which is extremely inefficient given 6.46M rows and 28 years. The `build_neighbor_lookup()` function is already a step toward optimization, but the main bottleneck is that neighbor stats are recomputed for every year without leveraging the static spatial adjacency. This leads to redundant lookups and memory overhead.  

**Optimization Strategy**  
- Build a **reusable neighbor adjacency table** keyed by `cell_id` only (not year), since spatial neighbors do not change over time.  
- For each year, **join yearly attributes** to this static adjacency table and compute neighbor statistics in a vectorized manner (using `data.table` or `dplyr`), rather than looping through 6.46M rows.  
- Avoid repeated `lapply` over millions of rows; instead, use grouped joins and aggregation.  
- Process one year at a time to keep memory within 16 GB.  

**Working R Code**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique is a list of integer vectors (spdep::nb object)
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique

# 1. Build static adjacency table (cell_id -> neighbor_id)
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]))
  data.table(id = from, neighbor_id = to)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)

# Convert cell_data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 2. Compute neighbor stats year by year
compute_neighbor_features <- function(cell_data, adj_table, vars) {
  years <- sort(unique(cell_data$year))
  results_list <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    message("Processing year: ", yr)
    
    # Subset data for this year
    year_data <- cell_data[year == yr, .(id, year, (vars))]
    setkey(year_data, id)
    
    # Join adjacency with year_data to get neighbor values
    joined <- adj_table[year_data, on = .(neighbor_id = id)]
    # joined has columns: id (from adj_table), neighbor_id, year, vars
    
    # Compute stats by original id
    agg <- joined[, lapply(.SD, function(x) {
      if (all(is.na(x))) return(c(NA_real_, NA_real_, NA_real_))
      c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))
    }), by = id, .SDcols = vars]
    
    # Flatten column names
    new_names <- unlist(lapply(vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
    setnames(agg, old = names(agg)[-1], new = new_names)
    
    # Add year
    agg[, year := yr]
    results_list[[i]] <- agg
  }
  
  # Combine all years
  rbindlist(results_list)
}

neighbor_features <- compute_neighbor_features(cell_data, adj_table, neighbor_source_vars)

# Merge back into cell_data
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data, on = .(id, year)]

# Now cell_data has neighbor stats for each variable
# Proceed with Random Forest prediction using the existing trained model
```

**Why This Is Faster**  
- Adjacency is computed once (1.37M relationships) instead of per row.  
- Yearly processing uses efficient joins and aggregation in `data.table`.  
- Avoids 6.46M `lapply` calls; instead, uses vectorized operations and grouping.  
- Memory is controlled by processing one year at a time.  

This approach should reduce runtime from 86+ hours to a few hours on a standard laptop.
END_RESPONSE>>>
