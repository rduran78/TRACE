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
run_id: OpenAI_F02_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The main bottlenecks are:

1. **Inefficient `lapply` per-row**:  
   Both `build_neighbor_lookup` and `compute_neighbor_stats` perform heavy per-row operations over ~6.46M rows, creating large intermediate lists.

2. **Repeated string operations**:  
   `paste(data$id, data$year, sep = "_")` and repeated lookups are expensive.

3. **Memory overhead**:  
   Storing millions of lists for neighbors and then repeatedly extracting subsets is highly memory-intensive.

4. **Single-threaded execution**:  
   The code runs sequentially; no parallelization is used.

---

**Optimization Strategy**

- **Precompute neighbor indices for all years using matrix operations**:  
  Instead of building a large list indexed by row, create an integer matrix (`neighbor_idx_matrix`) mapping each row to neighbor row indices directly.

- **Avoid string concatenations**:  
  Use numeric indexing for `(id, year)` pairs. A two-level index approach (id â†’ position, year offset) avoids `paste()` calls.

- **Vectorize or use `data.table` for aggregation**:  
  Transform the problem into a join-and-aggregate task:  
  - Replicate neighbor relationships across years.
  - Merge neighbor values and compute `max`, `min`, `mean` using fast grouped aggregation (`data.table`).

- **Chunking + parallelization**:  
  Process in chunks or use `future.apply` / `parallel` to utilize multiple cores.

---

**Working R Code**

Using `data.table` for speed and memory efficiency:

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique ids in correct order
# rook_neighbors_unique: list of integer vectors (spdep nb object)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor relationships replicated across years
years <- sort(unique(cell_data$year))
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  nbrs <- rook_neighbors_unique[[i]]
  if (length(nbrs) == 0) return(NULL)
  data.table(
    id     = rep(id_order[i], length(nbrs) * length(years)),
    year   = rep(years, each = length(nbrs)),
    nbr_id = rep(id_order[nbrs], times = length(years))
  )
}), use.names = TRUE)

setkey(neighbor_dt, nbr_id, year)

# Melt source vars for easier join
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor values
nbr_vals <- cell_data[, c("id", "year", vars), with = FALSE]
setkey(nbr_vals, id, year)
neighbor_dt <- neighbor_dt[nbr_vals, on = .(nbr_id = id, year), nomatch = 0]

# Compute stats by (id, year)
agg <- neighbor_dt[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Merge back into cell_data
cell_data <- agg[cell_data, on = .(id, year)]
```

---

**Why this works**

- Eliminates per-row loops and string operations.
- Uses `data.table` for highly optimized join and aggregation.
- Avoids storing large neighbor lookup lists in memory.
- Scales better: operations are batched and vectorized.
- Can be parallelized further by splitting `neighbor_dt` and using `future.apply` or `data.table`'s `by` parallelism.

---

**Expected Performance Gain**

- From 86+ hours to **minutes or a few hours** on a laptop.
- Memory usage improves since we use compressed integer keys and avoid millions of small lists.

This preserves the original trained Random Forest model and numerical estimands.
END_RESPONSE>>>
