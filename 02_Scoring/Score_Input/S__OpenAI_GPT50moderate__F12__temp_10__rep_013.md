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
run_id: OpenAI_F12_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence shows the true bottleneck is **not** `do.call(rbind, result)`. That combines ~6.46â€¯M rows Ã— 3 columns only once per variable â€” negligible compared to the **neighbor aggregation within `compute_neighbor_stats()`**, which repeatedly scans potentially millions of neighbor indices in R loops across 5 variables. The dominant cost is running 6.46â€¯M `lapply` calls with nested indexing and vector operations in Râ€™s interpreter.

### **Diagnosis**
- `compute_neighbor_stats()` iterates over every row (~6.46â€¯M).
- Each row performs:
  - Lookup into `vals`.
  - Filtering `NA`s.
  - Recomputing `max`, `min`, `mean`.
- Done 5 times (for 5 variables), so ~32â€¯M high-overhead operations.
- `do.call(rbind, result)` is minor (few seconds vs. hours), so the colleagueâ€™s diagnosis is wrong.

**Source of slowness:** Pure R row-wise operations with millions of neighbor-index evaluations.

---

### **Optimization Strategy**
- **Vectorize neighbor aggregation**: Pre-expand to long format (`cell_id`, `year`, `neighbor_id`, `var_value`), then group and aggregate using `data.table` or `dplyr`.
- This collapses loops into efficient native C-level aggregation.
- Preserve existing `neighbor_lookup` by representing edges once and joining efficiently.

---

### **Optimized Approach with `data.table`**
```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
dt <- as.data.table(cell_data)

# Build an edge list from neighbor_lookup:
# neighbor_lookup is list of integer vectors (indices into dt rows)
edges <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(
      src = i,
      nbr = neighbor_lookup[[i]]
    )
  })
)

# Join year to match correctly
edges[, year := dt$year[src]]

# Expand neighbor info
edges[, c("id", "nbr_year") := .(dt$id[src], dt$year[nbr])]

# Only keep neighbors in same year
edges <- edges[year == nbr_year]

# Long join of variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long <- melt(dt, id.vars = c("id", "year"), measure.vars = vars,
             variable.name = "var", value.name = "val")

# Join edges to long neighbor values
edges_long <- merge(edges, long, by.x = c("nbr", "year"), by.y = c("id", "year"),
                    allow.cartesian = TRUE)

# Aggregate by source cell-year and variable
agg <- edges_long[, .(
  max = max(val, na.rm = TRUE),
  min = min(val, na.rm = TRUE),
  mean = mean(val, na.rm = TRUE)
), by = .(src, year, var)]

# Reshape to wide for joining back
agg_wide <- dcast(agg, src + year ~ var, value.var = c("max", "min", "mean"))

# Merge back to original data
result <- cbind(dt, agg_wide[order(src)][, -"src"])
```

---

### **Why This Fix Works**
- Replaces **6.46â€¯M Ã— 5 row-wise loops** with a few vectorized joins and aggregations in optimized C code (`data.table`).
- Avoids repeated subset calls and redundant NA filtering.
- Memory footprint is manageable because the neighbor edge list (~1.37â€¯M rows) is much smaller than 6.46â€¯M Ã— neighbors.

---

**Expected performance gain:** Likely hours â†’ minutes on a 16â€¯GB laptop.

**Model integrity:** Preserves existing trained RF model and estimands by computing the same 3 summary stats per neighbor set.

---

**Summary:** The colleagueâ€™s claim is incorrect; `do.call(rbind, ...)` cost is negligible. The true bottleneck is millions of interpreted R loops in `compute_neighbor_stats()`. Use a vectorized joinâ€“aggregate pipeline with `data.table` for a 100Ã—+ speedup.
END_RESPONSE>>>
