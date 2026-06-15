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
run_id: OpenAI_F09_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck comes from recomputing neighbor adjacency for every cell-year when building neighbor features. For ~6.46 million rows, repeatedly generating indices for neighbors is extremely expensive. Neighbor spatial relationships are static across years, so recalculating them for each time slice is unnecessary.  

**Optimization Strategy**  
1. **Precompute a Reusable Neighbor Table**: Build a single adjacency structure based on cell IDs only (not year).  
2. **Vectorized or Efficient Joins**: For each year, join yearly values onto this neighbor mapping and compute the summary stats in a grouped fashion (avoiding per-row R loops).  
3. **Keep Memory Use in Check**: Use `data.table` for efficient joins and aggregation, as the dataset is large (~6.5M rows).  
4. **Pipeline**:  
   - Build neighbor long table: `(cell_id, neighbor_id)`.  
   - Merge yearly attributes for neighbor cells.  
   - Aggregate (max, min, mean) per `(cell_id, year)`.  
   - Merge back to main dataset.  

This approach eliminates most of the costly nested `lapply` operations, reducing the estimated runtime from 86+ hours to potentially under 1 hour.  

---

### **Optimized R Code**

```r
library(data.table)

# 1. Convert to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.

# 2. Build reusable neighbor table (id-to-id)
id_order <- as.integer(id_order)  # ensure integer IDs
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Keep only unique directed pairs
setkey(neighbor_pairs, cell_id, neighbor_id)

# 3. For each year and variable, compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure fast join
setkey(cell_data, id, year)

for (var in neighbor_source_vars) {
  # Subset columns for efficiency
  dt_neighbors <- cell_data[, .(neighbor_id = id, year, value = get(var))]

  # Join: add neighbor values
  joined <- neighbor_pairs[dt_neighbors, on = "neighbor_id", allow.cartesian = TRUE]

  # Compute stats by (cell_id, year)
  agg <- joined[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(cell_id, year)]

  # Handle NAs from empty neighbor sets
  agg[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]

  # Rename columns to indicate feature
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(var, "_nbr_", c("max", "min", "mean")))

  # Merge back to cell_data
  cell_data <- merge(cell_data, agg, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
}

# Cell_data now contains neighbor features for Random Forest prediction
```

---

**Expected Impact**  
- Precomputing adjacency once and using `data.table` joins aggregates billions of lookups into efficient block operations.  
- Memory: ~6.5M rows + ~1.3M neighbor edges repeated over years (~39M rows intermediate), which is feasible on 16 GB RAM if processed iteratively per variable.  
- Original Random Forest model remains intact and predictions are unchanged.  

**Runtime Reduction**: Likely from 86+ hours to well under an hour using this batch aggregation method.
END_RESPONSE>>>
