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
run_id: OpenAI_F04_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated nested `lapply` loops across 6.46M rows Ã— 5 variables, each performing string concatenation, list lookup, and numeric aggregation in pure R. This results in massive overhead.  
- `build_neighbor_lookup` constructs a 6.46M-length list with string pastes and index lookups.
- `compute_neighbor_stats` runs millions of small loops across that huge list for every variable.  
The algorithm is memory-inefficient, not vectorized, and repeatedly processes the same neighbor indices per year.

---

**Optimization Strategy**  
1. **Precompute neighbor relationships once in index form** (eliminate string concatenations).
2. **Vectorize neighbor feature computation**: flatten data into matrix and use fast aggregation.
3. Use **`data.table`** or **`collapse`** for fast joins and grouped computations.
4. Compute all neighbor variables in a **single pass** rather than looping five times.
5. Minimize list overhead: store neighbors as integer indices and process via matrix operations.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute index mapping: cell_id -> row indices
# Assuming data has columns: id, year, and all variables
setkey(cell_data, id, year)

# Expand neighbor relationships across years in an efficient way
years <- sort(unique(cell_data$year))
n_years <- length(years)
cell_ids <- sort(unique(cell_data$id))
n_cells <- length(cell_ids)

# rook_neighbors_unique assumed: list where rook_neighbors_unique[[i]] are neighbor IDs of cell_ids[i]
neighbor_map <- rook_neighbors_unique   # already in memory

# Build a long neighbor table: (focal_id, neighbor_id, year)
focal_ids <- rep(cell_ids, sapply(neighbor_map, length))
neighbor_ids <- unlist(neighbor_map)
neighbor_dt <- data.table(focal_id = rep(focal_ids, each = n_years),
                           neighbor_id = rep(neighbor_ids, each = n_years),
                           year = rep(years, times = length(focal_ids)))

# Join neighbor values for needed vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a long table: neighbor values per var
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
# neighbor_dt now has columns: focal_id, neighbor_id, year, ntl, ec, ...

# Aggregate neighbor stats in one go
agg_dt <- neighbor_dt[, lapply(.SD, function(x) {
  .(max = max(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    mean = mean(x, na.rm = TRUE))
}), by = .(focal_id, year), .SDcols = vars]

# Reshape wide: combine max/min/mean suffixes
setnames(agg_dt, old = names(agg_dt)[-(1:2)],
         new = paste(rep(vars, each = 3), c("max", "min", "mean"), sep = "_"))

# Merge back to cell_data
cell_data <- agg_dt[cell_data, on = .(focal_id = id, year)]

# cell_data now has neighbor-derived features appended
```

---

**Key Improvements**  
- Eliminated 6.46M Ã— 5 repeated loops; now a vectorized aggregation via `data.table`.
- Computed neighbor stats for all variables in **one pass**.
- Avoided costly string operations inside loops.
- Memory-efficient merges instead of large nested lists.
- Still returns identical estimand because logic is equivalent: neighbor max, min, mean per variable/year.

**Expected Runtime Drop**  
From 86+ hours down to a few minutes on 6.5M rows with 16 GB RAM using `data.table`.
END_RESPONSE>>>
