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
run_id: OpenAI_F07_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the double nested loop:  
- `build_neighbor_lookup()` creates a year-specific neighbor index for each of 6.46M rows, which is extremely memory- and time-intensive.  
- Then `compute_neighbor_stats()` iterates over all rows again for each of 5 variables, repeatedly scanning subsets.  
This results in ~32M row-level computations and heavy list operations in R, which are slow for millions of elements. The entire process is single-threaded and not vectorized.

---

**Optimization Strategy**  
1. **Avoid building a huge `neighbor_lookup` list**: Instead, precompute neighbor pairs (cell-year â†’ neighbor-year) as a flat `data.frame` using vectorized joins.  
2. **Exploit `data.table` for speed and memory efficiency**: Join once, aggregate neighbor values with `max`, `min`, `mean` using fast group operations.  
3. **Batch by year**: The panel structure allows splitting into 28 yearly slices to keep memory within 16 GB.  
4. **Parallelize across years** with `future.apply` or `parallel::mclapply`.  
5. **Preserve original estimand**: Use the same neighbor relationships and years; only the computational method changes.  

---

**Working R Code**

```r
library(data.table)
library(parallel)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2, ...)
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of IDs corresponding to rook_neighbors_unique

# Precompute neighbor pairs (directed)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

setkey(neighbor_dt, src)

# Convert main data to data.table
setDT(cell_data)

# Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute stats for one year
compute_year <- function(yr) {
  dt_year <- cell_data[year == yr, .SD, .SDcols = c("id", "year", neighbor_source_vars)]
  
  # Join neighbors for this year: src-year â†’ nbr-year
  pairs <- neighbor_dt[dt_year, on = .(src = id), allow.cartesian = TRUE]
  setnames(pairs, "nbr", "id")  # rename for join
  
  # Join to get neighbor values
  pairs <- pairs[dt_year, on = .(id), nomatch = 0]
  
  # Aggregate neighbor statistics by source cell
  stats <- pairs[, lapply(.SD, function(x) {
    c(max = max(x, na.rm = TRUE),
      min = min(x, na.rm = TRUE),
      mean = mean(x, na.rm = TRUE))
  }), by = src, .SDcols = neighbor_source_vars]
  
  # Reshape result: wide format with suffixes
  out <- dcast(melt(stats, id.vars = "src"), src ~ variable + L1)
  
  # Merge back into dt_year
  setnames(out, "src", "id")
  merge(dt_year, out, by = "id", all.x = TRUE)
}

# Parallel execution by year
years <- sort(unique(cell_data$year))
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("cell_data", "neighbor_dt", "neighbor_source_vars"))
clusterEvalQ(cl, library(data.table))

results_list <- parLapply(cl, years, compute_year)
stopCluster(cl)

# Combine all years
cell_data_enriched <- rbindlist(results_list)
```

---

**Expected Gains**  
- Eliminates 6.46M-list overhead â†’ uses vectorized joins and aggregation.  
- Processes one year at a time in memory (<250K rows/year).  
- Parallelizes across 28 years on multiple cores.  
- Should reduce runtime from **86+ hours to well under 2 hours** on a 16 GB laptop.

**Preserves**:  
- Original rook-neighbor topology.  
- Same numerical estimand for max, min, mean.  
- Existing Random Forest model remains untouched.
END_RESPONSE>>>
