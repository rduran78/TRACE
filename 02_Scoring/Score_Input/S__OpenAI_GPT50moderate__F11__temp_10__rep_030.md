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
run_id: OpenAI_F11_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence shows the main bottleneck is **neighbor feature computation**, not Random Forest inference.  

### **Diagnosis**
- `build_neighbor_lookup()` builds a large nested list using `lapply` over **6.46M rows**, performing repeated string concatenation (`paste`) and multiple lookups in named vectors.  
- `compute_neighbor_stats()` iterates again over the same list for every variable (5Ã—), performing `lapply` + `rbind` with repeated NA filtering and stats aggregation.  
- These steps scale with *rows Ã— neighbors* under pure R list operations â€” extremely slow for 6.46M rows and 1.37M neighbor relationships.  
- Random Forest inference on 6.46M rows Ã— 110 vars is large but runs in hours with `ranger` or `randomForest` on a modern laptop, not 86+ hours.  

Thus, the colleague's diagnosis is **wrong**. The bottleneck is **building and traversing neighbor lookups in pure R loops**.

---

### **Optimization Strategy**
- Precompute neighbor indices in a flat `data.table` instead of nested lists + string keys.
- Use vectorized joins and grouped aggregation instead of millions of inner loops.
- Compute all 3 summary stats (`max`, `min`, `mean`) for each variable in one grouped pass.
- Leverage `data.table` for keyed joins and fast aggregation.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Add unique key for each observation
dt[, obs_key := .I]

# Precompute neighbors as edges
# rook_neighbors_unique is list of integer neighbors per cell index
# Build mapping: source id -> neighbor id
neighbor_pairs <- data.table(
  source = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Join neighbor pairs with years to expand in panel structure
panel_years <- unique(dt$year)
neighbor_dt <- CJ(year = panel_years, source = id_order)[
  , obs_key := .I]

# Merge neighbor relationships: source-year with neighbor-year
neighbor_edges <- merge(
  neighbor_dt, neighbor_pairs, by = "source", allow.cartesian = TRUE
)

# Rename for clarity
setnames(neighbor_edges, c("source", "neighbor"), c("grid_src", "grid_nbr"))

# Map neighbor-year combinations to dt obs_key for fast lookup
dt_keyed <- dt[, .(grid_nbr = id, year, obs_key)]
setkey(dt_keyed, grid_nbr, year)

neighbor_edges <- neighbor_edges[
  dt_keyed, on = .(grid_nbr, year), nomatch = 0
]
setnames(neighbor_edges, "obs_key", "nbr_obs_key")

# For each source obs_key, compute neighbor stats for multiple variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt needed columns for aggregation
nbr_vals <- dt[, c("obs_key", vars), with = FALSE]
long_nbr <- melt(nbr_vals, id.vars = "obs_key", variable.name = "var", value.name = "val")

# Attach neighbor values to edges and aggregate
neighbor_edges <- neighbor_edges[
  long_nbr, on = .(nbr_obs_key = obs_key), nomatch = 0
]

agg_stats <- neighbor_edges[
  , .(
      max_val = if (.N) max(val, na.rm = TRUE) else NA_real_,
      min_val = if (.N) min(val, na.rm = TRUE) else NA_real_,
      mean_val = if (.N) mean(val, na.rm = TRUE) else NA_real_
    ),
    by = .(obs_key, var)
]

# Reshape back to wide
agg_wide <- dcast(agg_stats, obs_key ~ var, value.var = c("max_val", "min_val", "mean_val"))

# Merge back to main data
dt <- merge(dt, agg_wide, by = "obs_key", all.x = TRUE)
```

---

### **Expected Impact**
- Replaces O(n) nested `lapply` loops and repeated string pasting with efficient joins and grouped aggregation in `data.table`.
- Processes all 5 neighbor variables in a single grouped pass.
- Estimated runtime drops from 86+ hours to a few hours or less (depending on I/O).

---

**Summary**: The true bottleneck is neighbor feature construction via R loops, not Random Forest inference. Move to `data.table` with vectorized joins to handle ~6.46M observations and 1.37M neighbor relations efficiently, preserving estimand and model.
END_RESPONSE>>>
