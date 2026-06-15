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
run_id: OpenAI_F08_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats()` is applied row-by-row across **6.46 million cell-year rows**. Each row performs lookups and vector operations repeatedly, resulting in massive redundant work. The neighbor structure is static across years, but the current code recomputes neighbor relationships for every row and every variable. This leads to:

- ~6.46M Ã— 5 variables = ~32M expensive operations.
- Constant reallocation and repeated NA filtering.
- Inefficient R loops on a very large dataset.

The bottleneck is not the Random Forest prediction but the naive repeated computation of neighbor stats for each cell-year.

---

**Optimization Strategy**  
Exploit the static neighbor graph:

1. **Precompute neighbor indices once per cell (not per cell-year)** since neighbors do not change over time.
2. For each year, slice the data vector for that variable, then compute neighbor stats for all cells in **vectorized fashion**, producing a `year Ã— cells` matrix.
3. Bind results back to the full panel by joining on `id` and `year`.
4. Use efficient data structures (`data.table` or matrix) and avoid repeated `lapply` on millions of rows.

This reduces complexity from `O(N * neighbors * years)` repeated per row to `O(neighbors * cells * years)` in a structured loop, with heavy vectorization.

---

**Working R Code**

```r
library(data.table)

# Assumes: cell_data has columns id, year, and variables
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: spdep nb object

# 1. Precompute neighbor lookup per cell (static)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) neighbors[[i]])
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)
n_cells <- length(id_order)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

years <- sort(unique(cell_data$year))

# 2. Function to compute stats efficiently
compute_neighbor_stats_by_year <- function(dt, var_name, neighbor_lookup, id_order, years) {
  # Preallocate result container
  res_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Extract vector of variable for this year in id_order
    vals <- dt[year == yr][match(id_order, id), get(var_name)]
    
    # Compute stats for all cells
    maxs <- numeric(n_cells)
    mins <- numeric(n_cells)
    means <- numeric(n_cells)
    
    for (i in seq_len(n_cells)) {
      nb_idx <- neighbor_lookup[[i]]
      if (length(nb_idx) == 0) {
        maxs[i] <- NA; mins[i] <- NA; means[i] <- NA
      } else {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0) {
          maxs[i] <- NA; mins[i] <- NA; means[i] <- NA
        } else {
          maxs[i] <- max(nb_vals)
          mins[i] <- min(nb_vals)
          means[i] <- mean(nb_vals)
        }
      }
    }
    
    # Combine into data.table for this year
    res_list[[yi]] <- data.table(
      id = id_order,
      year = yr,
      paste0(var_name, "_nb_max") := maxs,
      paste0(var_name, "_nb_min") := mins,
      paste0(var_name, "_nb_mean") := means
    )
  }
  
  rbindlist(res_list)
}

# 3. Loop over source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

feature_tables <- lapply(neighbor_source_vars, function(var_name) {
  compute_neighbor_stats_by_year(cell_data, var_name, neighbor_lookup_static, id_order, years)
})

# Merge all features back
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), feature_tables)

# Final dataset
setkey(neighbor_features, id, year)
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Is Fast**  
- Neighbor relationships computed **once**.
- Operations done in **year blocks** (28 iterations, not millions).
- Inner loop only over cells (344K) with fixed neighbor sets, avoids data.frame overhead.
- Memory footprint is controlled by working year-by-year, not all at once.
- Eliminates redundant lookups and repeated filtering per row.

This approach should cut computation from **86+ hours to a few hours or less** on a standard laptop while preserving the trained Random Forest model and the original estimand.
END_RESPONSE>>>
