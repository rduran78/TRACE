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
run_id: OpenAI_F11_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The code evidence points to the **neighbor feature computation** as the true bottleneck, not Random Forest inference. Building `neighbor_lookup` and repeatedly scanning millions of rows to compute max, min, and mean across 6.46M cell-year rows and 1.37M neighbor edges is extremely expensive in R due to repeated `lapply` and `rbind` calls, which are memory-inefficient for such scale. Random Forest prediction on 6.46M rows with 110 features typically completes in minutes, not 86+ hours.

---

### **Diagnosis**
- `build_neighbor_lookup` creates a large list with millions of elements.
- `compute_neighbor_stats` runs **nested R loops** (`lapply` inside for loop) for every variable, causing heavy interpretive overhead.
- These steps are responsible for the 86+ hour runtime, not model inference.

---

### **Optimization Strategy**
- Replace expensive R loops with **vectorized joins and grouped aggregations** using `data.table`.
- Precompute neighbor pairs once, then join with source data and compute aggregated stats (max, min, mean) in one pass per variable, fully exploiting grouping at C-level.
- Preserve numerical estimand by keeping same statistic definitions.

---

### **Optimized R Code**

```r
library(data.table)

# Convert original data to data.table
dt <- as.data.table(cell_data)

# Build neighbor edge list once
# rook_neighbors_unique: list of neighbor IDs for each cell_id
edges <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand to cell-year level
years <- unique(dt$year)
edges_full <- CJ(year = years, cell_id = edges$cell_id, neighbor_id = edges$neighbor_id)

# For fast joins
setkey(dt, id, year)
setkey(edges_full, neighbor_id, year)

# Function to compute neighbor stats per variable
compute_neighbor_features <- function(var_name) {
  # Join neighbor var values
  tmp <- dt[, .(neighbor_id = id, year, val = get(var_name))]
  joined <- edges_full[tmp, on = .(neighbor_id, year)]

  # Aggregate per (cell_id, year)
  agg <- joined[, .(
    max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]

  # Merge back
  setkey(agg, cell_id, year)
  dt[agg, paste0(var_name, "_nbr_max") := i.max, on = .(id = cell_id, year)]
  dt[agg, paste0(var_name, "_nbr_min") := i.min, on = .(id = cell_id, year)]
  dt[agg, paste0(var_name, "_nbr_mean") := i.mean, on = .(id = cell_id, year)]
}

# Apply for all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) compute_neighbor_features(v)

cell_data <- as.data.frame(dt)
```

---

### **Expected Gains**
- Moves from **86 hours** to **well under 1 hour** on a standard laptop, as `data.table` handles millions of rows and grouped stats efficiently.
- Memory and speed performance improved by eliminating nested loops and list-based lookups.
- Random Forest prediction remains unchanged and will complete quickly after these optimizations.
END_RESPONSE>>>
