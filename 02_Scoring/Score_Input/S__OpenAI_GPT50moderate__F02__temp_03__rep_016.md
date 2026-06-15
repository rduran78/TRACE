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
run_id: OpenAI_F02_T03_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length â‰ˆ 6.46M (one per cell-year), each storing neighbor indices. This is highly inefficient in R.  
- `compute_neighbor_stats` iterates over this large list multiple times (once per variable), performing repeated lookups and allocations.  
- The entire process is single-threaded and uses `lapply` on millions of elements.  
- Memory overhead from large lists and repeated `paste` operations is huge.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lookups**: Instead of building a massive list, use a **long-format join** approach:  
   - Expand neighbor relationships across years once.  
   - Join on `year` to map neighbors efficiently.  
2. **Vectorize aggregation**: Use `data.table` for fast grouping and aggregation instead of millions of `lapply` calls.  
3. **Compute all neighbor stats in one pass**: Melt the data for the 5 variables, join with neighbor pairs, and compute `max`, `min`, `mean` grouped by `(id, year, var)`.  
4. **Memory efficiency**: Work in chunks if necessary, but `data.table` should handle 6.5M rows on 16 GB RAM if optimized.  
5. **Preserve model and estimand**: Only feature engineering changes; no retraining.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 1. Prepare neighbor pairs (id, neighbor_id)
# rook_neighbors_unique: list of integer vectors, names correspond to id_order
id_order <- as.integer(id_order)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# 2. Expand across years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# 3. Join neighbor values
# Melt cell_data for the 5 variables
long_data <- melt(
  cell_data,
  id.vars = c("id", "year"),
  measure.vars = neighbor_source_vars,
  variable.name = "var",
  value.name = "val"
)

# Join neighbor values
setkey(long_data, id, year)
neighbor_pairs_long <- neighbor_pairs[long_data, on = .(neighbor_id = id, year), allow.cartesian = TRUE]

# 4. Aggregate neighbor stats
neighbor_stats <- neighbor_pairs_long[
  !is.na(val),
  .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ),
  by = .(id, year, var)
]

# 5. Reshape back to wide format
neighbor_stats_wide <- dcast(
  neighbor_stats,
  id + year ~ var,
  value.var = c("max_val", "min_val", "mean_val")
)

# 6. Merge back to original data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why this is faster:**  
- Eliminates per-row `lapply` and repeated string operations.  
- Uses `data.table` joins and groupings (optimized in C).  
- Computes all neighbor stats in a single pass.  
- Memory footprint is reduced by avoiding massive lists.  

**Expected runtime:**  
On 6.5M rows and ~1.3M neighbor links, this approach should finish in **minutes to a few hours** on a 16 GB laptop, not 86+ hours.  

**Preserves:**  
- Original estimand.  
- Trained Random Forest model (no retraining).  
- All required neighbor-based features.
END_RESPONSE>>>
