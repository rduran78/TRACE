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
run_id: OpenAI_F06_T00_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F06_OpenAI_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly performing vector lookups and allocations.  
- Neighbor stats are recomputed row-by-row instead of leveraging vectorized or grouped operations.  
- Memory overhead from millions of small lists and repeated `paste` operations.  

**Optimization Strategy**  
- Precompute a long-format table of `(cell_id, year, var, value)` and join with neighbor relationships using `data.table`.  
- Use `data.table` grouping to compute `max`, `min`, and `mean` in bulk rather than per-row loops.  
- Avoid repeated string concatenation; use integer keys for joins.  
- Keep everything in memory-efficient `data.table` form.  
- Append results back to `cell_data` by keyed join.  

This approach preserves the original estimand and avoids retraining the Random Forest model.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in order
# rook_neighbors_unique: spdep::nb object

# Build neighbor pairs (directed)
neighbor_pairs <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand neighbor pairs across years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[, .(id = from, neighbor_id = to)][
  , .(id = rep(id, each = length(years)),
      neighbor_id = rep(neighbor_id, each = length(years)),
      year = rep(years, times = .N))
]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Join neighbor values for all variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(v))]
  setkey(tmp, neighbor_id, year)
  
  # Join neighbor values
  neighbor_dt <- tmp[neighbor_dt, on = .(neighbor_id, year)]
  
  # Compute stats by (id, year)
  stats <- neighbor_dt[!is.na(val),
                        .(max = max(val), min = min(val), mean = mean(val)),
                        by = .(id, year)]
  
  # Rename columns
  setnames(stats, c("id", "year", "max", "min", "mean"),
           c("id", "year",
             paste0(v, "_nbr_max"),
             paste0(v, "_nbr_min"),
             paste0(v, "_nbr_mean")))
  
  # Merge back to cell_data
  cell_data <- stats[cell_data, on = .(id, year)]
  
  # Drop val column for next iteration
  neighbor_dt[, val := NULL]
}

# cell_data now has neighbor features appended
```

---

**Why This Works**  
- Uses `data.table` joins and grouped aggregation instead of millions of `lapply` calls.  
- Processes all rows in bulk, leveraging efficient C-level operations.  
- Avoids repeated string concatenation and redundant lookups.  

**Expected Performance**  
- Reduces runtime from 86+ hours to a few minutes (depending on disk I/O and CPU).  
- Memory footprint remains manageable on 16 GB RAM because operations are vectorized and intermediate tables are reused.  

This preserves the trained Random Forest model and the original numerical estimand while making the computation feasible.
END_RESPONSE>>>
