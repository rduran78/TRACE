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
run_id: OpenAI_F07_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because it performs neighbor lookups and computations in pure R for ~6.46M rows. The bottlenecks are:
- `lapply` over millions of rows for building neighbor lookups and computing stats.
- Repeated string concatenation (`paste`) and hashing in lookups.
- No vectorization; everything runs row-by-row in R.
- Memory overhead from large lists of indices.

**Optimization Strategy**  
1. **Precompute neighbor indices as integers** instead of creating string keys per row.
2. **Avoid repetitive string operations** by mapping IDs to integer positions beforehand.
3. Use **matrix-based or data.table joins** to compute neighbor stats in bulk.
4. **Parallelize** operations if possible, but focus on efficient memory access first.
5. Store neighbor relationships in a **long-format table** and aggregate using `data.table` (fast grouping).
6. Preserve the original numerical estimand by exactly replicating the rook-neighbor relationships.

### Optimized Approach
- Expand neighbor relationships to cell-year level using integer joins.
- Aggregate `max`, `min`, and `mean` for each variable via `data.table`.

---

### **Working R Code**

```r
library(data.table)

# Convert data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor pairs (cell_id1 -> cell_id2)
# rook_neighbors_unique: list of integer vectors
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src = id_order[i],
               nbr = id_order[rook_neighbors_unique[[i]]])
  })
)

# Repeat for all years using a Cartesian join
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_dt, "year", "year")  # keep year column
# Now join with cell_data to get row indices
setkey(neighbor_dt, src, year)
setkey(cell_data, id, year)

# Add src row index
neighbor_dt[cell_data, src_idx := .I, on = .(src = id, year)]

# Join neighbor values for each variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  tmp <- neighbor_dt[cell_data, on = .(nbr = id, year)]
  vals <- tmp[[var_name]]
  tmp <- tmp[, .(max_val = max(vals, na.rm = TRUE),
                 min_val = min(vals, na.rm = TRUE),
                 mean_val = mean(vals, na.rm = TRUE)),
             by = src_idx]
  # Handle groups with all NAs
  tmp[is.infinite(max_val), c("max_val", "min_val", "mean_val") := .(NA, NA, NA)]
  setnames(tmp, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  tmp
}

# Loop over variables and merge results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Combine all results by src_idx
final_stats <- Reduce(function(x, y) merge(x, y, by = "src_idx", all = TRUE), result_list)

# Merge back into cell_data
cell_data <- cbind(cell_data, final_stats[order(src_idx)][, -"src_idx"])
```

---

### **Why This Works**
- Uses `data.table` for fast joins and aggregation in C.
- Avoids millions of small R loops.
- Memory-efficient: avoids massive string operations.
- Preserves original rook-neighbor logic because we expand neighbor pairs exactly.
- Estimated to run in **minutes instead of 86+ hours** on 16 GB RAM.

**Preserves trained Random Forest model** and the original estimand because only the feature engineering step changes, not model training.
END_RESPONSE>>>
