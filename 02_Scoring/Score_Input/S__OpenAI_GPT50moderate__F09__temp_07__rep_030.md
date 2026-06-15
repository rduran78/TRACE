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
run_id: OpenAI_F09_T07_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor relationships for each cell-year row repeatedly, causing severe inefficiency. With 6.46M rows and ~1.37M neighbor relationships, looping through all rows for every variable leads to massive redundant work. The major bottleneck: `build_neighbor_lookup` and `compute_neighbor_stats` iterate over all 6.46M rows multiple times, and neighbor lookup is recalculated for every year-variable combination.

**Optimization Strategy**  
1. **Precompute adjacency once at the cell level**: Build a reusable neighbor table mapping each cell to its neighbors (using `rook_neighbors_unique`).  
2. **Use year-based joins instead of nested loops**: For each year, join cell attributes with neighbors via an expanded table, compute aggregate stats (`max`, `min`, `mean`) in a vectorized way.  
3. **Leverage `data.table` for speed and memory efficiency**: Avoid per-row `lapply`. Use `data.table` joins and group operations.  
4. **Do not retrain the Random Forest model**: Only recompute predictors efficiently, preserving the original estimand.  
5. Process in yearly chunks to avoid memory overload.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build adjacency table ONCE
# rook_neighbors_unique: list of integer vectors, same order as id_order
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(
  lapply(seq_along(adj_list), function(i) {
    if (length(adj_list[[i]]) == 0) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
  })
)

setkey(adj_dt, cell_id)

# Variables for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output storage
results_list <- vector("list", length(neighbor_source_vars))

# Process year by year to keep memory manageable
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for year
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(dt_year, id)
  
  # Join adjacency
  merged <- adj_dt[dt_year, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # merged has: cell_id, neighbor_id, year data from neighbor

  for (var_name in neighbor_source_vars) {
    # Compute neighbor stats for this var and year
    stats <- merged[, .(
      max = max(get(var_name), na.rm = TRUE),
      min = min(get(var_name), na.rm = TRUE),
      mean = mean(get(var_name), na.rm = TRUE)
    ), by = cell_id]
    
    # Handle NaN if all NA
    stats[is.infinite(max), max := NA]
    stats[is.infinite(min), min := NA]
    stats[is.nan(mean), mean := NA]
    
    # Rename columns
    setnames(stats, c("cell_id", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
    
    # Merge back to dt_year
    dt_year <- stats[dt_year, on = .(cell_id = id)]
  }
  
  # Append year results back to main data
  cell_data[year == yr, (setdiff(names(dt_year), c("id","year"))) := dt_year[, setdiff(names(dt_year), c("cell_id","year"))]]
}

# At this point, cell_data includes all neighbor-based predictors, ready for RF prediction
```

---

### **Why This is Faster**
- **Adjacency computed once**: No repeated construction of neighbor lookups.
- **Yearly chunking**: Limits memory footprint while processing 6.46M rows.
- **Vectorized group operations via `data.table`**: Avoids costly per-row `lapply`.
- **Single-pass per variable per year**: Reduces redundancy from 86+ hours to likely under 2â€“4 hours on a 16GB laptop.

**Preserves**:
- Original Random Forest model.
- Original numerical estimand.
- Same neighbor-based feature logic, just computed efficiently.
END_RESPONSE>>>
