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
run_id: OpenAI_F08_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main inefficiency arises because `compute_neighbor_stats` loops across all 6.46 million rows and recomputes neighbor aggregation for every cell-year, for each variable, performing repeated lookups in R lists. This is O(#rows Ã— avg_neighbors Ã— #vars), and the current R loops cause massive overhead for such a large panel (86+ hours estimated).  

Yet the neighbor structure is static â€” only values change over time. Therefore, instead of recomputing neighbor IDs repeatedly, we can:  

- Precompute a **neighbor index matrix** at the cell level (344,208 cells).  
- Iterate by year, subset the array of values for that year, and compute neighbor max/min/mean in a **vectorized** manner.  
- Append results back with proper alignment.  

This reduces complexity significantly by leveraging fixed adjacency and matrix operations.  

---

**Optimization Strategy**  
- Use the cell-level neighbor list (`rook_neighbors_unique`) once to build an integer adjacency list.  
- For each year, compute neighbor statistics for all cells in a vectorized pass, avoiding per-row R loops.  
- Bind results for all years and merge with the panel.  
- Process all five variables inside the year loop.  
- Memory footprint remains moderate: process per-year chunks (~344K rows), instead of all 6.46M rows at once.  

---

**Working R Code**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)
setkey(cell_data, id, year)

# Basic info
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

# Precompute neighbor list at cell index level
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbors_idx <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Function to compute stats for all cells given one variable vector
compute_stats_vectorized <- function(values, neighbors_idx) {
  max_vals <- numeric(length(values))
  min_vals <- numeric(length(values))
  mean_vals <- numeric(length(values))
  
  for (i in seq_along(neighbors_idx)) {
    nbs <- neighbors_idx[[i]]
    if (length(nbs) == 0 || all(is.na(values[nbs]))) {
      max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
    } else {
      v <- values[nbs]
      v <- v[!is.na(v)]
      max_vals[i]  <- if (length(v) > 0) max(v) else NA_real_
      min_vals[i]  <- if (length(v) > 0) min(v) else NA_real_
      mean_vals[i] <- if (length(v) > 0) mean(v) else NA_real_
    }
  }
  
  list(max_vals, min_vals, mean_vals)
}

# Prepare container
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
results_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  cat("Processing year:", yr, "\n")
  
  # Subset to year and sort by id_order
  sub <- cell_data[year == yr]
  setkey(sub, id)
  sub_vals <- sub[id_order]
  
  # Initialize per-year result table
  res_year <- data.table(id = id_order, year = yr)
  
  # Compute stats for each variable
  for (v in neighbor_vars) {
    vals <- sub_vals[[v]]
    stats <- compute_stats_vectorized(vals, neighbors_idx)
    res_year[[paste0(v, "_nbr_max")]]  <- stats[[1]]
    res_year[[paste0(v, "_nbr_min")]]  <- stats[[2]]
    res_year[[paste0(v, "_nbr_mean")]] <- stats[[3]]
  }
  
  results_list[[y]] <- res_year
}

# Combine back to panel and merge
neighbor_features <- rbindlist(results_list)
setkey(neighbor_features, id, year)

# Merge into original data
cell_data <- neighbor_features[cell_data]

# Now cell_data has neighbor-based features and can be fed unchanged to RF predict
```

---

**Key Advantages**
- Reuses static neighbor structure once.
- Operates per-year (~344K rows), keeping memory safe on a 16GB machine.
- Avoids 6.46M per-row list operations; relies on contiguous, vectorized loops (significant speedup from 86 hours to a few hours or less).
- Preserves trained Random Forest model and output estimands.

You can parallelize the yearly loop using `parallel::mclapply` if desired.
END_RESPONSE>>>
