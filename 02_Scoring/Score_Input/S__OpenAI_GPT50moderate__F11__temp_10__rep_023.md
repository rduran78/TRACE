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
run_id: OpenAI_F11_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim is incorrect: Random Forest inference (loading models, `predict()`, writing predictions) is negligible compared to the massive cost of constructing neighbor-based features repeatedly. Evidence:

- The outer loop computes **neighbor statistics for 5 variables over 6.46M rows**, each requiring `lapply` across **all rows**, repeatedly accessing `neighbor_lookup`.  
- `build_neighbor_lookup` returns a list of length 6.46M; later, every `compute_neighbor_stats` iterates through it again and performs vector filtering and aggregation.  
- This results in heavy R-level interpretation overhead and memory churn, which dominates runtime.  

Random Forest inference on even millions of rows usually takes minutes, while this nested `lapply` structure across ~32M neighbor-stat computations easily accounts for **86+ hours**.

---

**Correct Bottleneck**: The repeated *neighbor feature computation*, not RF prediction.  
**Optimization Strategy**:  
- Precompute and store neighbor indices once (done already), then use vectorized or compiled operations instead of large `lapply`.  
- Use **`data.table`** or **matrix-based aggregation**.  
- Compute all neighbor features in one pass over neighbor pairs, instead of looping variable-by-variable.

---

### Optimized Approach

1. Reshape `cell_data` to a `data.table` with keys `(id, year)`.
2. Melt neighbor relationships into a two-column `data.table` (`src`, `nbr`).
3. Join on `nbr` to fetch values for all variables, then aggregate by `src` and year in *one grouped query* using `max`, `min`, `mean`.
4. Merge back aggregated features.

---

### Working R Code

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Source IDs and neighbor relationships precomputed
neighbors_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand neighbor relationships across all years
years <- unique(dt$year)
neighbors_dt[, key := 1]
year_dt <- data.table(year = years, key = 1)
neighbors_dt <- neighbors_dt[year_dt, on = "key"][, key := NULL]

# Join neighbor values
setkey(dt, id, year)
setkey(neighbors_dt, nbr, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# For efficiency, melt dt to long for selected vars
long_dt <- melt(dt, id.vars = c("id", "year"), measure.vars = neighbor_source_vars,
                variable.name = "feature", value.name = "val")

setkey(long_dt, id, year)
neighbors_long <- neighbors_dt[long_dt, on = .(nbr = id, year)]

# Aggregate neighbor stats
agg_dt <- neighbors_long[
  !is.na(val),
  .(max_val = max(val), min_val = min(val), mean_val = mean(val)),
  by = .(src, year, feature)
]

# Cast back to wide format
agg_wide <- dcast(
  agg_dt,
  src + year ~ feature,
  value.var = c("max_val", "min_val", "mean_val")
)

# Merge back with original
setnames(agg_wide, "src", "id")
dt <- merge(dt, agg_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- Handles all 5 variables and ~1.37M neighbor links across 28 years in **one pipeline** using vectorized `data.table` operations.
- Eliminates per-row `lapply` calls (6.46M Ã— 5 iterations) that caused the 86-hour runtime.
- Memory footprint: ~tens of MB for neighbor link table, feasible on 16 GB RAM.

**Expected Speed-Up**: From 86+ hours to minutes on a 16 GB laptop.

**Random Forest Step**: Keep using `predict()` after this. Its cost is tiny compared to the previous R loop overhead.

**Final Answer**: The real bottleneck is the neighbor feature construction; optimize by converting it into a join-and-aggregate workflow using `data.table` rather than millions of R-level loops.
END_RESPONSE>>>
