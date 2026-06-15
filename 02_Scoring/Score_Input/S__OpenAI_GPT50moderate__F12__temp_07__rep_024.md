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
run_id: OpenAI_F12_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list-binding inside `compute_neighbor_stats()` are the main bottlenecks is incorrect. The true performance issue lies in the **neighbor lookup structure** and the repeated R-level interpretation of lists across millions of rows. Each call to `compute_neighbor_stats()` iterates over ~6.46M rows, performing per-row neighbor indexing and vector operations in pure R, five times (once per variable). This is orders of magnitude more expensive than a single `do.call(rbind, ...)`. The bottleneck is therefore the per-row `lapply` logic, combined with repeated filtering and indexing.

---

### **Correct Optimization**
- Precompute a **long-form neighbor table** (row â†’ neighbor rows) once.
- Join this with the data and compute **aggregates using vectorized `data.table`** or `dplyr` instead of looping.
- Compute all 5 variables in one pass over the neighbor table.
- Preserve the trained Random Forest model and original numerical estimand.

---

### **Optimized Strategy**
1. Build a neighbor mapping as a two-column integer matrix: `(row_id, neighbor_id)`.
2. Convert `cell_data` into a keyed `data.table`.
3. Melt and aggregate using `data.table` to compute `max`, `min`, `mean` per row for all required variables in one grouped operation.
4. Merge results back into `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)
dt[, row_id := .I]  # add unique row index

# Build neighbor mapping once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# Flatten to long form: row_id -> neighbor_id
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Join neighbor values
setkey(dt, row_id)
setkey(neighbor_dt, neighbor_id)
neighbor_long <- neighbor_dt[dt, on = .(neighbor_id = row_id), 
                              nomatch = 0L, allow.cartesian = TRUE]

# Keep only needed variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
keep_cols <- c("row_id", vars)
neighbor_long <- neighbor_long[, c("row_id", vars), with = FALSE]

# Melt and aggregate
neighbor_long <- melt(neighbor_long, id.vars = "row_id", 
                      variable.name = "var", value.name = "val", 
                      variable.factor = FALSE)
neighbor_long <- neighbor_long[!is.na(val)]

agg <- neighbor_long[, .(
  neigh_max = max(val),
  neigh_min = min(val),
  neigh_mean = mean(val)
), by = .(row_id, var)]

# Reshape wide
agg_wide <- dcast(agg, row_id ~ var, 
                  value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# Merge back to main data
setkey(agg_wide, row_id)
dt <- dt[agg_wide]

# Drop helper column
dt[, row_id := NULL]

# Final optimized cell_data
cell_data <- as.data.frame(dt)
```

---

### **Why This Works**
- Eliminates 6.46M Ã— 5 R-level loops â†’ single vectorized aggregation pass.
- Replaces repeated indexing and filtering with efficient `data.table` joins and grouping.
- Scales to tens of millions of relationships in memory on a 16 GB laptop.

**Expected Performance:** Hours reduced from 86+ to under 1â€“2 hours (or minutes with parallelization).
END_RESPONSE>>>
