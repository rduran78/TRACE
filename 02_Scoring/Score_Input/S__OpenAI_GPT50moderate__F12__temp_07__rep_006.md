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
run_id: OpenAI_F12_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottlenecks is **incorrect**. While `rbind` on millions of rows has some cost, the dominant bottleneck is the **per-row lapply over ~6.46 million rows inside `compute_neighbor_stats()`**, which recalculates neighbor statistics for every row-variable combination. This results in roughly:

```
6.46 million rows Ã— 5 variables â‰ˆ 32 million neighbor scans
```

Each scan iterates over neighbor indices and applies `max`, `min`, and `mean`, creating enormous overhead in pure R.

The true performance issue is the **R-level interpretive overhead of millions of function calls and repeated indexing**, not the final `rbind`.

---

### **Optimization Strategy**
- **Avoid per-row R loops**: Precompute neighbor statistics in a vectorized or compiled manner.
- Use **matrix-based computation or data.table** to aggregate neighbor statistics in bulk.
- Compute stats for **all variables in one pass** instead of looping in `for (var_name in ...)`.
- Use **preallocated structures** to avoid repeated allocation.

---

### **Optimized Approach**
1. Flatten the neighbor relationships into an edge list: `(source_row, neighbor_row)`.
2. Join values for all required variables.
3. Use `data.table` aggregation to compute `max`, `min`, `mean` per source_row and variable in compiled C.
4. Bind results back to the main data in one pass.

---

### **Working R Code**

```r
library(data.table)

# Convert input to data.table
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Build edge list once
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(dt)), paste(dt$id, dt$year, sep = "_"))

edges <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
  src_id <- id_order[ref_idx]
  nbr_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(nbr_ids) == 0) return(NULL)
  list(src_id = src_id, nbr_id = nbr_ids)
}))

# Expand edges for all years (cartesian join)
years <- unique(dt$year)
edges_dt <- CJ(year = years, src_id = edges$src_id)[
  , nbr_id := edges$nbr_id, by = .I]

# Map to row indices
edges_dt[, src_key := paste(src_id, year, sep = "_")]
edges_dt[, nbr_key := paste(nbr_id, year, sep = "_")]
edges_dt[, src_idx := idx_lookup[src_key]]
edges_dt[, nbr_idx := idx_lookup[nbr_key]]
edges_dt <- edges_dt[!is.na(nbr_idx)]

# Compute neighbor stats for all variables in one pass
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_vals <- melt(dt[, c("id","year", neighbor_vars), with = FALSE],
                  id.vars = c("id","year"),
                  variable.name = "var", value.name = "val")

# Link values with edges
edges_long <- edges_dt[, .(src_idx, nbr_idx)]
edges_long <- merge(edges_long, long_vals[, .(nbr_idx = .I, val, var)],
                    by = "nbr_idx", allow.cartesian = TRUE)

# Aggregate neighbor stats
agg_stats <- edges_long[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(src_idx, var)]

# Reshape wide and join back to dt
agg_wide <- dcast(agg_stats, src_idx ~ var, value.var = c("max_val","min_val","mean_val"))
dt <- cbind(dt, agg_wide[match(seq_len(nrow(dt)), agg_wide$src_idx), -1])

# Result: dt now has neighbor features efficiently computed
```

---

### **Why This Works**
- Eliminates **6.46M lapply calls** and repeated indexing.
- Uses **data.table**â€™s highly optimized C-level aggregation for `max`, `min`, `mean`.
- One pass for all variables instead of five separate passes.
- Memory-efficient because edges are stored as integers and operations are vectorized.

**Expected speedup:** From ~86 hours to under 1 hour on a standard laptop, depending on I/O and disk speed.
END_RESPONSE>>>
