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
run_id: OpenAI_F04_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The computational bottleneck lies in `build_neighbor_lookup` and `compute_neighbor_stats` due to heavy `lapply` use over 6.46M rows for each of 5 variables. The neighbor lookup is repeatedly computed and row-wise operations prevent vectorization. Memory churn (building millions of small integer vectors and pasting keys) and repeated filtering for NAs add overhead. The Random Forest inference is not the major cost.

---

**Optimization Strategy**  
1. **Pre-expand neighbor relationships once** as a (cell_id, neighbor_id) mapping joined with yearsâ€”avoid per-row `lapply`.
2. **Use `data.table` for join-based aggregation**:
   - Melt/pivot the dataset so neighbor stats can be computed with fast group-by instead of R loops.
3. Compute all 5 variables in a single grouped operation to minimize passes.
4. Eliminate repeated `do.call(rbind, ...)` which copies large objects repeatedly.
5. Ensure keys and joins are on integers instead of pasted strings.

Memory fits if processed sequentially using `data.table` with 16GB RAM.

---

**Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# cell_data columns assumed: id, year, ntl, ec, pop_density, def, usd_est_n2
vars <- c("ntl","ec","pop_density","def","usd_est_n2")

# Build neighbor relationships as a long data.table
edges <- data.table(from_id = rep(id_order, lengths(rook_neighbors_unique)),
                    to_id   = unlist(rook_neighbors_unique, use.names = FALSE))

# Expand by years
years <- unique(cell_data$year)
edges <- edges[CJ(year = years), on = .(dummy = NULL)]
setnames(edges, "year", "year")  # edges now from_id,to_id,year

# Join to neighbor values
neighbor_values <- cell_data[, c("id","year",vars), with = FALSE]
setkey(neighbor_values, id, year)
setkey(edges, to_id, year)
edges <- neighbor_values[edges, on = .(id = to_id, year)]

# edges now: from_id, to_id, year, ntl, ec, pop_density, def, usd_est_n2

# Compute stats per (from_id, year)
agg <- edges[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean= mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean= mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean= mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean= mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean= mean(usd_est_n2, na.rm = TRUE)
), by = .(id = from_id, year)]

# Merge back to cell_data
setkey(cell_data, id, year)
cell_data <- agg[cell_data]

# Replace NaN with NA after aggregation
for (j in names(cell_data)) set(cell_data, which(is.nan(cell_data[[j]])), j, NA)
```

---

**Why this is Faster**
- Eliminates 6.46M `lapply` calls.
- Vectorized aggregation using `data.table` in ~ minutes, not 86 hours.
- Single join for all variables avoids repeated scanning.
- Preserves original estimand (stats per row-id-year, same neighbors, same summary).

This should reduce runtime from days to **well under an hour** on 16GB RAM.
END_RESPONSE>>>
