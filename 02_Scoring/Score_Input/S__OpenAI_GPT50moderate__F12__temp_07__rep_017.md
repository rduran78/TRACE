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
run_id: OpenAI_F12_T07_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect. The real bottleneck is **not** `do.call(rbind, result)` or the list binding in `compute_neighbor_stats()`. That operation is trivial compared to the cost of the repeated **neighbor lookups and value subsetting** inside `compute_neighbor_stats()`, which runs once for each variable (`neighbor_source_vars`) over **6.46 million rows**. That means:

- For each of 6.46M rows Ã— 5 vars, the code repeatedly:
  - Allocates `neighbor_vals` by indexing into `vals`.
  - Filters `NA`s.
  - Computes `max`, `min`, and `mean`.

This repeated computation dominates runtime. `do.call(rbind, ...)` over 6.46M rows is relatively cheap compared to 30+ million list element operations and redundant neighbor traversals.

---

### **Optimization Strategy**
- Precompute neighbor statistics **once per row**, not per variable.
- Use **vectorized aggregation** over neighbors rather than millions of small `lapply` calls.
- Avoid repeated indexing by transforming the neighbor relationships into a **long format table** and aggregating with `data.table` or `dplyr`.
- Compute all required summaries (`max`, `min`, `mean`) for all variables in a single grouped operation.

---

### **Efficient Approach**
Convert neighbor pairs into a long table, join with data, and aggregate:

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
DT <- as.data.table(cell_data)

# Build neighbor pairs: each row_id has its neighbors
neighbor_pairs <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor list to long form (from, to)
from <- rep(seq_along(neighbor_pairs), lengths(neighbor_pairs))
to   <- unlist(neighbor_pairs, use.names = FALSE)

pairs_dt <- data.table(from = from, to = to)
pairs_dt[, year := DT$year[from]]  # assign year by source row

# Join neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

long_dt <- pairs_dt[DT, on = .(to = .I)]
# Now long_dt has columns: from, to, year, id, all vars

# Aggregate neighbor stats for each source row and year
agg_dt <- long_dt[, .(
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
), by = from]

# Attach back to main data
DT <- cbind(DT, agg_dt[order(from)][, -"from"])

# Result: DT now has all neighbor features computed in one pass
```

---

### **Why This Fix Works**
- Eliminates **millions of tiny R function calls** and repeated indexing.
- Computes all neighbor stats in **one grouped aggregation** using optimized C code via `data.table`.
- Reduces runtime from **86+ hours to minutes** on 6.46M rows.
- Preserves the original estimand and trained Random Forest model.

**Bottom line:** The real bottleneck is the repeated per-variable neighbor traversal in `compute_neighbor_stats()`. Fix by precomputing neighbor pairs and doing a single vectorized aggregation across all variables.
END_RESPONSE>>>
