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
run_id: OpenAI_F11_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence shows the main bottleneck is **neighbor feature computation**, not Random Forest inference. The `build_neighbor_lookup` and `compute_neighbor_stats` functions use `lapply` over 6.46M rows Ã— 5 variables with heavy list manipulations and repeated vector indexing, which is extremely slow in R. Random Forest prediction on 6.46M rows with 110 predictors generally completes in minutes on a modern laptop, so the colleagueâ€™s claim is incorrect.

---

### **Diagnosis**
- `build_neighbor_lookup()` creates a list of length 6.46M, each element containing integer vectors. This is memory-heavy and slow to construct.
- `compute_neighbor_stats()` repeatedly traverses these lists and subsets `vals` for each row using `idx`. This is an O(N Ã— K) loop in pure R, which is highly inefficient.

---

### **Optimization Strategy**
- Avoid per-row `lapply`. Restructure the neighbor relationships into a **long table** and compute statistics with **vectorized operations** (using `data.table`).
- Precompute neighbor stats once per year and variable, then join back to the main table.
- Keep Random Forest model as is; just replace feature engineering step.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
cell_dt <- as.data.table(cell_data)

# Extract neighbor pairs
# rook_neighbors_unique: list of neighbors for each id in id_order
neighbor_pairs <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique)
)

# Add year expansion for panel
years <- unique(cell_dt$year)
neighbor_dt <- neighbor_pairs[, .(id = from_id, neighbor_id = to_id)]
neighbor_dt <- neighbor_dt[rep(seq_len(.N), each = length(years))]
neighbor_dt[, year := rep(years, times = nrow(neighbor_pairs))]

# Merge neighbor values
setkey(cell_dt, id, year)
setkey(neighbor_dt, neighbor_id, year)

neighbor_dt <- neighbor_dt[cell_dt, on = .(neighbor_id = id, year), nomatch = 0]

# Function to compute stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  neighbor_dt[, val := get(var_name)]
  stats <- neighbor_dt[!is.na(val),
                       .(max_val = max(val),
                         min_val = min(val),
                         mean_val = mean(val)),
                       by = .(id, year)]
  setnames(stats, c("id", "year", 
                    paste0(var_name, "_nbr_max"),
                    paste0(var_name, "_nbr_min"),
                    paste0(var_name, "_nbr_mean")))
  stats
}

# Compute for all variables and merge
stats_list <- lapply(c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
                     compute_neighbor_stats_fast)

final_stats <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE),
                      stats_list)

# Merge back to main dataset
cell_dt <- merge(cell_dt, final_stats, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- Eliminates 6.46M `lapply` iterations.
- Uses `data.table` grouping and aggregation in **C-optimized code**.
- Computes each variableâ€™s neighbor stats in one grouped pass rather than per-row list lookups.
- Memory footprint is controlled via long format instead of massive nested lists.

---

**Expected Performance:** Hours reduced to **minutes** for neighbor feature computation, making Random Forest prediction the true final step.
END_RESPONSE>>>
