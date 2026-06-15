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
run_id: OpenAI_F04_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The computational bottleneck is `build_neighbor_lookup` and repeated per-variable lapply calls in `compute_neighbor_stats`. Issues:

- `build_neighbor_lookup` constructs a list of length = number of rows (â‰ˆ6.46M), duplicating neighbor lookup for each year â†’ massive memory and time cost.
- Nested character operations (`paste`) and repeated indexing inside `lapply` are expensive.
- For each variable, `compute_neighbor_stats` iterates again over all rows â†’ full pass for every variable (5 passes).
- Random Forest prediction cost is negligible compared to this preprocessing step.

---

**Optimization Strategy**

- **Precompute neighbor indices once per-cell, not per cell-year**; then reuse across years.
- **Avoid building a list of length 6.46M**. Instead, store neighbors as row indices grouped by cell, and then apply across all years.
- Use **matrix/vectorized operations** for stats computation (aggregate functions over slices).
- Optionally leverage **data.table** for fast grouping, keyed merges, and numeric operations.
- Keep starting memory footprint low by avoiding large character vectors.

---

### **Optimized Algorithm**

1. Map each cell to its neighbors once (using `rook_neighbors_unique`).
2. For each year:
   - Subset the yearâ€™s rows and compute neighbor stats via joins rather than lapply.
3. Bind results back to full dataset.
4. Use `data.table` for speed.

---

**Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Facts
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_ids <- unique(cell_data$id)
years <- sort(unique(cell_data$year))

# Precompute neighbor mapping: list of neighbors for each id
neighbor_map <- rook_neighbors_unique

# Build long table of (id, neighbor_id) pairs
neighbor_pairs <- data.table(
  id = rep(cell_ids, lengths(neighbor_map)),
  neighbor_id = unlist(neighbor_map)
)

# Function to compute all neighbor stats for a given year
compute_year_features <- function(yr) {
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join neighbors
  setkey(dt_year, id)
  joined <- neighbor_pairs[dt_year, on = .(id = neighbor_id), allow.cartesian = TRUE]
  # joined has columns: id (cell), neighbor_id, year vars from dt_year (neighbor's values)
  
  # Compute stats grouped by target id
  features <- joined[, lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(rep(NA_real_, 3))
    c(max(x), min(x), mean(x))
  }), by = id, .SDcols = neighbor_source_vars]
  
  # Reshape columns
  cols_new <- unlist(lapply(neighbor_source_vars, function(v)
    paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  ))
  feature_mat <- do.call(cbind, features[, -1])
  features_out <- data.table(id = features$id)
  features_out[, (cols_new) := as.data.table(t(feature_mat))]
  
  features_out[, year := yr]
  features_out
}

# Compute for all years efficiently
result_list <- lapply(years, compute_year_features)
neighbor_features <- rbindlist(result_list)

# Merge back to main data
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data, on = .(id, year)]
```

---

### **Why This is Faster**

- Avoids 6.46M-length lapply; instead, computes per-year using grouped operations.
- Minimizes string concatenation; works on numeric IDs and merges.
- Uses `data.table` for fast joins and aggregations.
- One pass per year (28 passes), not per variable (previously 5 Ã— 6.46M iterations).
- Memory footprint is controlled via joins rather than large pre-built lists.

---

**Expected Time Reduction**

From multi-day runtime (86+ hours) to **well under a few hours** on a 16â€¯GB laptop, thanks to vectorization and data.table optimizations, preserving trained Random Forest model and original estimand.
END_RESPONSE>>>
