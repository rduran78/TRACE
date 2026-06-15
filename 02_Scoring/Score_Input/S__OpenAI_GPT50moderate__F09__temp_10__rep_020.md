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
run_id: OpenAI_F09_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current bottleneck arises from repeatedly computing neighbor relationships for all 6.46 million cell-year rows when building neighbor features. For each variable and each row, the code performs list lookups and aggregation, causing:

- **Redundant adjacency computations:** The neighbor lookup is recomputed indirectly for each year-variable combination.
- **Inefficient per-row aggregation in R:** `lapply` over millions of rows and `rbind` assembly is very slow.
- Memory stress from repeatedly managing large intermediate lists.

**Optimization Strategy:**  
- Build a **reusable adjacency table once** using `data.table` or similar.
- Expand to cell-year combinations *once*, then `join` yearly cell attributes for all neighbor variables.
- Use **vectorized grouping** operations instead of millions of list traversals.
- Keep the Random Forest model intact and output features in the same structure.

This approach reduces repeated computation and uses efficient joins and aggregation.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build base neighbor table (cell-cell adjacency)
adj_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Expand with years (Cartesian join)
years <- unique(cell_data$year)
adj_year_dt <- adj_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(adj_year_dt, "year", "year")
# Add both cell-year combos
adj_year_dt[, id_year := paste(id, year, sep = "_")]
adj_year_dt[, neighbor_id_year := paste(neighbor_id, year, sep = "_")]

# Map to row indices for fast joining
cell_data[, id_year := paste(id, year, sep = "_")]

# Bring neighbor attributes by join for all years
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt/link efficiently
neighbor_stats_list <- list()
for (var in neighbor_source_vars) {
  # Join var from cell_data onto neighbor-year pairs
  tmp <- merge(
    adj_year_dt,
    cell_data[, .(neighbor_id_year = id_year, val = get(var))],
    by = "neighbor_id_year",
    all.x = TRUE,
    sort = FALSE
  )
  
  # Compute stats by original id-year
  stats <- tmp[, .(
    max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = id_year]
  
  setnames(stats, c("id_year", paste0(var, "_nb_max"), paste0(var, "_nb_min"), paste0(var, "_nb_mean")))
  
  neighbor_stats_list[[var]] <- stats
}

# Combine all stats into one table
neighbor_features <- Reduce(function(x, y) merge(x, y, by = "id_year", all = TRUE), neighbor_stats_list)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = "id_year", all.x = TRUE, sort = FALSE)

# Drop helper columns
cell_data[, id_year := NULL]

# Now cell_data has all neighbor-based features for Random Forest prediction
```

---

### **Why This Is Faster**
- **Adjacency calculated once:** `adj_dt` uses the rook neighbor list only once.
- **Single join per neighbor variable across all years** instead of per-row operations.
- **Vectorized aggregation using `data.table`** massively reduces overhead.
- Memory friendly since merges and group-aggregations are optimized in C.

---

**Expected improvement:** This approach should bring runtime down from **86+ hours to a few hours or less**, even on a standard laptop, by eliminating `lapply` loops and leveraging columnar operations.
END_RESPONSE>>>
