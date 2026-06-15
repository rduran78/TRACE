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
run_id: OpenAI_F02_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach builds an explicit `neighbor_lookup` list of length â‰ˆ 6.46M (rows), which consumes significant memory and forces repeated `lapply` operations over millions of elements for each of 5 variables. The combination of high row count, repeated string operations (`paste`), and non-vectorized lookups produces severe time and memory pressure.  

Key pain points:
- Constructing `neighbor_lookup` by iterating over each row.
- Recomputing neighbor statistics with nested `lapply` for every variable.
- Heavy reliance on R lists and repeated indexing instead of vectorized or table joins.

**Optimization Strategy:**  
Use **data.table** for efficient group joins instead of building a giant lookup list:
- Reshape data once into a keyed `data.table`.
- Expand neighbor relationships into a long table `(cell_id, neighbor_id, year)` and join on `id+year`.
- Compute aggregate statistics (`max`, `min`, `mean`) grouped by `(cell_id, year)`.
- Merge back into main data table.
This avoids memory-heavy list operations and leverages fast grouped aggregations.

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)
setkey(cell_data, id, year)

# Expand rook-neighbor relationships into a data.table
# id_order gives ordering and rook_neighbors_unique is nb object
neighbor_pairs <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Ensure correct types
neighbor_pairs[, from_id := as.integer(from_id)]
neighbor_pairs[, to_id   := as.integer(to_id)]

# We will join by year: replicate years for all relationships
years <- unique(cell_data$year)
neighbor_dt <- neighbor_pairs[ , .(year = years), by = .(from_id, to_id)]

# Join neighbor values for all variables in one pass
# Prepare a result container
result_list <- list()

for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  # Join to get neighbor variable values
  tmp <- merge(neighbor_dt, cell_data[, .(to_id = id, year, val = get(var_name))],
               by = c("to_id", "year"), all.x = TRUE)
  
  # Aggregate statistics by (from_id, year)
  tmp_stats <- tmp[!is.na(val),
                   .(max_val = max(val),
                     min_val = min(val),
                     mean_val = mean(val)),
                   by = .(from_id, year)]
  
  # Rename columns and merge back
  setnames(tmp_stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  # Merge into main data.table
  cell_data <- merge(cell_data, tmp_stats, by.x = c("id", "year"), by.y = c("from_id", "year"), all.x = TRUE)
}

# cell_data now contains added neighbor feature columns
```

---

**Why this works:**
- Eliminates large per-row list object.
- Uses **vectorized joins** and **grouped aggregation**.
- Leverages efficient data.table handling with billions of operations scaled to RAM.
- Avoids training or altering the Random Forest model; only feature engineering changes.
- Memory footprint manageable (processing via chunks optional if RAM tight).

**Expected speedup:**  
From 86+ hours to a few hours (or less), depending on disk I/O and machine capacity, since aggregation is highly optimized versus repeated `lapply` over millions of rows.
END_RESPONSE>>>
