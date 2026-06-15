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
run_id: OpenAI_F02_T00_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F02_OpenAI_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length 6.46M, each element being a vector of neighbor indices. This is expensive in both memory and time.  
- `compute_neighbor_stats` repeatedly traverses large lists and performs many small operations in R loops, which are inherently slow.  
- The entire process is done in pure R with `lapply` over millions of rows, which is not scalable on a laptop.  

**Optimization Strategy**  
1. **Avoid building a massive neighbor list for every row**. Instead, work with a long-format edge list (cell-year â†’ neighbor-year) and join it to the data.  
2. **Vectorize aggregation** using `data.table` or `dplyr` rather than looping.  
3. **Process by year** to reduce memory footprint (28 chunks instead of all at once).  
4. **Precompute neighbor relationships at the cell level** and then expand by year.  
5. **Compute max, min, mean in one grouped aggregation** rather than per-row loops.  

This approach reduces complexity from O(N Ã— neighbors) in R loops to efficient grouped operations in C-backed data.table.  

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor edge list at cell level
# rook_neighbors_unique: list of integer vectors (spdep nb object)
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Process year by year to keep memory manageable
years <- sort(unique(cell_data$year))

# Function to compute neighbor stats for one year
compute_year_stats <- function(yr, data, edges, vars) {
  dt <- data[year == yr, .(id, year, (vars)), with = FALSE]
  setkey(dt, id)

  # Join edges to get neighbor values
  edge_dt <- edges[, .(id = from, neighbor = to)]
  edge_dt <- edge_dt[dt, on = .(id), nomatch = 0]
  edge_dt <- edge_dt[dt, on = .(neighbor = id), nomatch = 0, 
                     allow.cartesian = TRUE]

  # edge_dt now has: id, neighbor, year, vars...
  # Compute stats by id
  res_list <- lapply(vars, function(v) {
    edge_dt[, .(
      max = max(get(v), na.rm = TRUE),
      min = min(get(v), na.rm = TRUE),
      mean = mean(get(v), na.rm = TRUE)
    ), by = id]
  })

  # Merge all results
  out <- Reduce(function(x, y) merge(x, y, by = "id"), res_list)
  out[, year := yr]
  out
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute in chunks
result_list <- lapply(years, compute_year_stats, data = cell_data, 
                      edges = edges, vars = neighbor_source_vars)

neighbor_stats <- rbindlist(result_list)

# Merge back to main data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works better:**  
- No giant list of 6.46M elements.  
- Uses `data.table` joins and grouped aggregation in C for speed.  
- Processes one year at a time to keep memory under control.  
- Preserves original estimand and trained model (only feature engineering changes).  

**Expected performance:**  
- From 86+ hours to a few hours (or less) on a 16 GB laptop.  
- Memory footprint reduced drastically because we never materialize a full neighbor lookup list.  

**Further improvements:**  
- Parallelize `lapply(years, ...)` with `future.apply` or `parallel::mclapply`.  
- If still too large, split by year and variable separately.  
- Consider writing intermediate results to disk if RAM is tight.
END_RESPONSE>>>
