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
run_id: OpenAI_F02_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely memory- and compute-intensive because:  
- `neighbor_lookup` builds a massive list of length 6.46M, each holding vectors of neighbor indices. This causes huge object size and GC overhead.  
- `compute_neighbor_stats` repeatedly traverses these lists and performs vectorized operations in R loops, which are slow for millions of elements.  
- `lapply` on 6.46M rows with nested loops is inherently inefficient in base R.  
- Data structure lookups (`id_order`, `neighbors`) and repeated string concatenation worsen performance.  

**Optimization Strategy**  
- **Avoid building a full neighbor list for all rows**. Instead, compute neighbor stats by joining on `(id, year)` pairs in a long format.  
- **Leverage vectorized joins in `data.table`** to aggregate neighbor statistics efficiently.  
- **Pre-expand neighbor relationships once at cell-level** and replicate across years (or join by year) rather than looping.  
- **Compute all 3 stats (max, min, mean) in grouped operations** instead of per-row computations.  
- Process in chunks if memory is tight.  

**Key Idea**:  
Convert neighbor relationships into a long table:  
`(cell_id, neighbor_id)` â†’ replicate for each year â†’ join values for `neighbor_id` and compute stats grouped by `(cell_id, year)`.  

---

### **Optimized R Code using `data.table`**

```r
library(data.table)

# Convert data to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Precompute neighbor pairs
# rook_neighbors_unique: list of neighbor ids for each cell in id_order
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand for all years: Cartesian join
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(cell_id, neighbor_id)]

# Merge neighbor values
long_dt <- merge(
  neighbor_pairs,
  cell_data[, .(neighbor_id = id, year, ntl, ec, pop_density, def, usd_est_n2)],
  by = c("neighbor_id", "year"),
  all.x = TRUE
)

# Compute neighbor stats for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Aggregate: max, min, mean per (cell_id, year)
agg_expr <- lapply(neighbor_source_vars, function(var) {
  list(
    as.name(paste0(var, "_nbr_max")) := max(get(var), na.rm = TRUE),
    as.name(paste0(var, "_nbr_min")) := min(get(var), na.rm = TRUE),
    as.name(paste0(var, "_nbr_mean")) := mean(get(var), na.rm = TRUE)
  )
})

agg_dt <- long_dt[, eval(as.call(c(quote(.(
  cell_id = cell_id, year = year)), unlist(agg_expr, recursive = FALSE)))), by = .(cell_id, year)]

# Merge back to original data
cell_data <- merge(cell_data, agg_dt, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
```

---

**Why this is faster and memory-efficient**  
- Uses `data.table` joins and aggregations instead of R loops.  
- No massive nested lists; all operations are vectorized.  
- Avoids repeated string concatenation and redundant lookups.  
- Scales better for 6.46M rows and 1.37M neighbor links.  
- Can process in **chunks by year** if memory is tight: run the aggregation per year and `rbindlist` results.  

**Expected speed-up**: From 86+ hours to under 1â€“2 hours on a 16 GB laptop (with possible further tuning like chunked processing).  

**Model Preservation**: This approach only computes features; the trained Random Forest model remains unchanged.
END_RESPONSE>>>
