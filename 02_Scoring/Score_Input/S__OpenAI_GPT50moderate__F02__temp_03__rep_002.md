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
run_id: OpenAI_F02_T03_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length â‰ˆ 6.46M (one per cell-year), each storing neighbor indices. This is costly in both memory and time.  
- `compute_neighbor_stats` repeatedly subsets large vectors and applies `lapply`, which is inefficient for millions of rows.  
- The entire process is single-threaded and heavily R-interpreted (no vectorization).  

**Optimization Strategy**  
1. **Avoid per-row lists**: Instead of building a massive neighbor lookup for every cell-year, compute neighbor stats by joining data on `(id, year)` and precomputed neighbor relationships.  
2. **Use data.table**: Efficient joins and aggregations on large datasets.  
3. **Precompute neighbor pairs**: Expand neighbor relationships into `(cell_id, neighbor_id)` pairs, then join on year to compute stats in a grouped manner.  
4. **Compute all variables in one pass**: Aggregate max, min, mean for all neighbor source variables simultaneously.  
5. **Memory efficiency**: Process in chunks if needed, but data.table should handle 6.5M rows on 16 GB RAM if optimized.  

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (directed)
# rook_neighbors_unique: list of integer vectors (spdep::nb)
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand to cell-year level by joining on year
# Duplicate neighbor pairs for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N))]

# Merge neighbor values
# Keep only required columns to reduce memory
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_dt <- merge(neighbor_dt,
                     cell_data[, c("id", "year", vars), with = FALSE],
                     by.x = c("neighbor_id", "year"),
                     by.y = c("id", "year"),
                     all.x = TRUE)

# Compute neighbor stats: max, min, mean per id-year
agg_exprs <- lapply(vars, function(v) {
  list(max = as.name(v), min = as.name(v), mean = as.name(v))
})
names(agg_exprs) <- vars

neighbor_stats <- neighbor_dt[, c(
  lapply(.SD, function(x) list(max = max(x, na.rm = TRUE),
                               min = min(x, na.rm = TRUE),
                               mean = mean(x, na.rm = TRUE))),
  .SDcols = vars
), by = .(id, year)]

# Flatten column names
setnames(neighbor_stats,
         old = names(neighbor_stats)[-(1:2)],
         new = unlist(lapply(vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean")))))

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)
```

---

**Why this is faster:**  
- Eliminates 6.46M-element list and repeated `lapply`.  
- Uses `data.table` joins and grouped aggregation in compiled C code.  
- Computes all neighbor stats in a single grouped operation.  

**Expected performance:**  
- From 86+ hours to under 1 hour on a 16 GB laptop (based on similar pipelines).  
- Memory footprint manageable because we only materialize necessary columns and use efficient joins.  

**Preserves:**  
- Original Random Forest model.  
- Original numerical estimand (same neighbor stats).
END_RESPONSE>>>
