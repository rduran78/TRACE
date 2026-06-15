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
run_id: OpenAI_F09_T00_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F09_OpenAI_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow because for each of ~6.46M rows, `build_neighbor_lookup` repeatedly constructs neighbor indices by string concatenation and lookup. This is highly inefficient due to repeated hashing and list operations. Additionally, `compute_neighbor_stats` iterates over all rows and recomputes neighbor statistics for each variable separately, causing redundant passes over the same neighbor structure.

**Optimization Strategy**  
- Precompute a reusable adjacency table (cell-to-neighbor relationships) once at the cell level, not per row.
- For each year, join yearly attributes to this adjacency table and compute neighbor statistics using vectorized operations or `data.table` grouping.
- Avoid repeated string concatenation and list lookups.
- Use `data.table` for efficient joins and aggregations.
- Preserve the trained Random Forest model and original estimand.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute adjacency table at cell level
# rook_neighbors_unique: list of integer vectors (neighbors for each cell)
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(lapply(seq_along(adj_list), function(i) {
  if (length(adj_list[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
}))

# Ensure keys for fast joins
setkey(adj_dt, neighbor_id)
setkey(cell_data, id, year)

# Function to compute neighbor stats for all variables in one pass per year
compute_neighbor_features <- function(cell_data, adj_dt, vars) {
  result_list <- vector("list", length(vars))
  names(result_list) <- vars
  
  # Process year by year to reduce memory footprint
  years <- unique(cell_data$year)
  
  for (yr in years) {
    # Subset for this year
    year_data <- cell_data[year == yr, .(id, year, (vars)), with = FALSE]
    setnames(year_data, c("id", "year", vars))
    
    # Join adjacency with neighbor values
    joined <- adj_dt[year_data, on = .(neighbor_id = id)]
    
    # Compute stats for each variable
    for (v in vars) {
      stats <- joined[, .(
        max = max(get(v), na.rm = TRUE),
        min = min(get(v), na.rm = TRUE),
        mean = mean(get(v), na.rm = TRUE)
      ), by = cell_id]
      
      # Rename columns
      setnames(stats, c("cell_id", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
      
      # Merge back into cell_data
      cell_data[year == yr, (names(stats)[-1]) := stats[.SD, on = .(id = cell_id), mget(names(stats)[-1])]]
    }
  }
  
  return(cell_data)
}

# Apply optimized function
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_neighbor_features(cell_data, adj_dt, neighbor_source_vars)
```

---

### **Why This Is Faster**
- `adj_dt` is built once and reused.
- No per-row string concatenation or list indexing.
- Uses `data.table` joins and grouped aggregations, which are highly optimized in C.
- Processes data year by year to keep memory usage within 16 GB.
- Reduces redundant passes: all neighbor stats for a variable computed in one grouped operation.

**Expected Performance Gain:**  
From 86+ hours to a few hours (depending on disk I/O and CPU), as the bottleneck shifts from R loops to efficient C-level operations in `data.table`.
END_RESPONSE>>>
