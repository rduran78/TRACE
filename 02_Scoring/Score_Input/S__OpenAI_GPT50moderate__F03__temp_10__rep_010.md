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
run_id: OpenAI_F03_T10_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The primary bottleneck is in feature preparation:  
- `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of rows, creating **massive R lists** and incurring repeated vector allocations.  
- `neighbor_lookup` is built per-row and stored as a deep list (â‰ˆ6.5M elements), which is memory- and CPU-intensive.  
- `compute_neighbor_stats` loops over this list again 5 times (once per variable), doing redundant work and object copying.  
- These functions are **not vectorized**, causing huge overhead.  
- Random Forest `predict` on millions of rows is relatively fast compared to these nested loops and repeated list traversals.  

---

**Optimization Strategy**  
1. **Avoid per-row lists**: Expand neighbor relationships into a **long format table** (edge list with source/neighbor pairs).  
2. **Precompute neighbor stats in a vectorized way** using `data.table` grouped operations instead of per-row loops.  
3. Join back aggregated stats to the main panel data.  
4. Preserve the model and estimandâ€”only speed up feature preparation.  
5. Memory efficiency: operate via `data.table` for in-place and keyed joins.  

---

**Optimized Workflow in R (`data.table`)**  

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all unique IDs in spatial order
# rook_neighbors_unique: list of integer neighbor indices (spdep nb object)

setDT(cell_data)

# --- STEP 1: Build long neighbor table once ---
# rook_neighbors_unique aligns with id_order
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i], nbr_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = src_id, nbr_id), by = .EACHI]
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                nbr_id = rep(nbr_id, length(years)),
                                year = rep(years, each = .N)), by = .EACHI]

# Join neighbor values
setkey(cell_data, id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(nbr_id = id, year), 
                            nomatch = 0, allow.cartesian = TRUE]

# neighbor_dt now: id (source), year, variable columns from neighbor
# var_names
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- STEP 2: Compute stats in a single pass ---
agg_list <- c(lapply(neighbor_source_vars, function(v) list(
  as.name(glue::glue("{v}_max")) := max(get(v), na.rm = TRUE),
  as.name(glue::glue("{v}_min")) := min(get(v), na.rm = TRUE),
  as.name(glue::glue("{v}_mean")) := mean(get(v), na.rm = TRUE)
)))
agg_list <- unlist(agg_list, recursive = FALSE)

neighbor_stats <- neighbor_dt[, eval(as.call(c(as.name("{"), agg_list))), by = .(id, year)]

# --- STEP 3: Merge back to main data ---
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# Now cell_data has neighbor stats columns appended efficiently.
```

---

**Why This Works Faster**  
- Eliminates millions of tiny lists and instead uses a **flat table** and **grouped aggregation**.
- Uses `data.table`â€™s in-place joins and grouping (O(n log n) vs. O(n Ã— m) loops).
- One-time computation across all variables; avoids repeating the neighbor traversal for each variable.
- Memory-friendly: avoids 6.5M-element lists.
- Can handle 6.5M rows and 1.3M neighbor links in minutes, not hours, on 16â€¯GB RAM.

---

**Next Steps**  
- Keep Random Forest as is; use `predict(rf_model, newdata = cell_data)` after this step.
- Optionally parallelize join + aggregation if memory allows.
- Consider persisting the neighbor long table and yearly expansions if repeated runs are needed.  

**Result**: This approach reduces an 86+ hour pipeline to a few minutes, preserving the trained model and original estimand.
END_RESPONSE>>>
