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
run_id: OpenAI_F07_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The performance bottleneck is the per-row `lapply` approach over ~6.46M rows, triggering numerous R-level loops and repeated vector allocations. It also recomputes lookups repeatedly and relies on `lapply` for both neighbor mapping and summary computation, which is slow at scale. Memory usage is reasonable, but execution time is prohibitively long because of Râ€™s overhead in iterating millions of times.

**Optimization Strategy**  
1. **Vectorize and Precompute**:  
   - Flatten neighbor relationships into a single mapping table of `(row_idx, neighbor_idx)` pairs for all years at once.
   - Maintain each feature column as a numeric vector and compute max, min, mean per `row_idx` using efficient group aggregation.
2. **Use `data.table` for Speed**:  
   - `data.table`'s grouping (`by`) is far faster for millions of rows than base R loops.
3. **Memory Efficiency**:  
   - Avoid nested lists; store relationships as a long data frame/table only once.
4. **Reuse Neighbor Index Mapping** across all variables:  
   - Compute joins once, then aggregate per variable to avoid repeated neighbor lookups.

---

### **Optimized Working Code**

```r
library(data.table)

# Assumes: cell_data (data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2)
#          id_order (vector of unique ids)
#          rook_neighbors_unique (list from spdep::nb)

setDT(cell_data)
setkey(cell_data, id, year)

# Step 1: Build long index map of row -> neighbor_row for all years
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
adj_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  cid <- id_order[i]
  nbrs <- id_order[rook_neighbors_unique[[i]]]
  if (length(nbrs)) {
    data.table(id = cid, neighbor_id = nbrs)
  } else NULL
}))

# Repeat for all years (Cartesian)
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(year = years), by = .(id, neighbor_id)]
# Map to row indices in cell_data
adj_dt[, row_idx := cell_data[J(id, year), which = TRUE]]
adj_dt[, neighbor_idx := cell_data[J(neighbor_id, year), which = TRUE]]
adj_dt <- adj_dt[!is.na(row_idx) & !is.na(neighbor_idx)]
# Remove columns not needed
adj_dt <- adj_dt[, .(row_idx, neighbor_idx)]

# Step 2: Function to compute neighbor stats for a given column
compute_neighbor_stats_dt <- function(var_name) {
  vals <- cell_data[[var_name]]
  tmp <- data.table(row_idx = adj_dt$row_idx,
                    val = vals[adj_dt$neighbor_idx])
  # Remove NAs before aggregation
  tmp <- tmp[!is.na(val)]
  agg <- tmp[, .(max = max(val),
                 min = min(val),
                 mean = mean(val)), by = row_idx]
  setnames(agg, c("row_idx", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  agg
}

# Step 3: Apply for all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge all aggregates back to cell_data
for (agg in agg_list) {
  cell_data <- merge(cell_data, agg, by.x = ".I", by.y = "row_idx", all.x = TRUE)
}

# Drop helper column .I if needed
```

---

**Why this works better**  
- We **flatten adjacency once** (~1.37M edges Ã— 28 years â‰ˆ 38M rows in `adj_dt`), still manageable on 16 GB RAM (â‰ˆ few GB).
- All aggregations occur through **highly optimized C-backed grouping in `data.table`** rather than millions of `lapply` calls.
- Sequence of steps eliminates redundant computations for each variable.
- Preserves original rook topology and numerical estimand since we only refactored the computational approach.

**Expected runtime** on 6.5M rows with `data.table` grouping: **minutes rather than 86+ hours** on a standard laptop. Memory footprint is higher but within 16 GB.
END_RESPONSE>>>
