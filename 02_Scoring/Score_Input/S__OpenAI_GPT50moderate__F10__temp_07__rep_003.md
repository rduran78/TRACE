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
run_id: OpenAI_F10_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is prohibitively slow (86+ hours) because:  
- It repeatedly performs expensive lookups and small list operations for ~6.46M rows.  
- `compute_neighbor_stats` uses `lapply` and `do.call(rbind, ...)` per variable.  
- Neighbor relationships are reconstructed year-by-year instead of leveraging the repeated topology.  
- No vectorization; the process is I/O and memory heavy.  

**Optimization Strategy**  
- Build the graph topology **once** (cell â†’ neighbor indices).  
- Use integer indices to map all rows efficiently: compute neighbor stats in a **fully vectorized** or batched manner.  
- Leverage `data.table` for fast grouping and joins.  
- Avoid repeated list allocations: preallocate result matrices.  
- Process all variables in one pass if possible.  
- Preserve numerical equivalence and Random Forest model (no retraining).  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Assume: cell_data has columns id, year, and predictor vars
setDT(cell_data)
setkey(cell_data, id, year)

# Parameters
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Precompute graph topology: a named integer list mapping cell_id -> neighbor ids
# rook_neighbors_unique: list of integer neighbor indices in id_order
id_order <- sort(unique(cell_data$id))
id_to_pos <- setNames(seq_along(id_order), id_order)

# Create an index for quick row lookup
cell_data[, row_idx := .I]

# Build neighbor lookup ONCE at the cell level
neighbor_lookup <- lapply(rook_neighbors_unique, function(neigh) id_order[neigh])

# Expand to (id, year) mapping using join instead of per-row lapply
# Build a big DT with columns: row_idx, neighbor_row_idx
years <- sort(unique(cell_data$year))

# Generate neighbor pairs for all years efficiently
pairs_list <- vector("list", length(id_order))
for (i in seq_along(id_order)) {
  if (length(neighbor_lookup[[i]]) == 0) next
  src_id <- id_order[i]
  dst_ids <- neighbor_lookup[[i]]
  # Cartesian product of years with dst_ids
  pairs_list[[i]] <- CJ(year = years, src_id = src_id, dst_id = dst_ids)
}
pairs_dt <- rbindlist(pairs_list, use.names = TRUE)

# Map to row indices for fast aggregation
pairs_dt[
  cell_data, on = .(src_id = id, year), src_idx := i.row_idx
][
  cell_data, on = .(dst_id = id, year), dst_idx := i.row_idx
]

pairs_dt <- pairs_dt[!is.na(src_idx) & !is.na(dst_idx), .(src_idx, dst_idx)]

# Compute neighbor stats for all variables
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  # Join neighbor values
  pairs_dt[, val := vals[dst_idx]]
  # Aggregate by src_idx
  agg <- pairs_dt[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = src_idx]
  # Merge results back
  setnames(agg, c("src_idx", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[agg, on = .(row_idx = src_idx), `:=`( 
    (paste0(var_name, "_nbr_max")) = i.get(paste0(var_name, "_nbr_max")),
    (paste0(var_name, "_nbr_min")) = i.get(paste0(var_name, "_nbr_min")),
    (paste0(var_name, "_nbr_mean")) = i.get(paste0(var_name, "_nbr_mean"))
  )]
  pairs_dt[, val := NULL] # clear for next var
}

# Drop helper column
cell_data[, row_idx := NULL]

# Predict with pre-trained Random Forest
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This Is Fast**
- Graph topology is built **once** at the cell level and reused across all years.
- Neighbor aggregation uses `data.table` grouping instead of millions of small `lapply` calls.
- Memory overhead minimized by reusing `pairs_dt` for all variables.
- The approach scales linearly with rows and is efficient on a 16 GB machine.

**Expected Runtime Improvement**  
From 86+ hours to a few hours (or less), depending on disk I/O, because all major bottlenecks (per-row `lapply` and repeated neighbor rebuilds) are eliminated.
END_RESPONSE>>>
