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
run_id: OpenAI_F06_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the `lapply` over ~6.46M rows in `compute_neighbor_stats`, which repeatedly subsets large vectors and performs aggregation. This results in massive R-level overhead and memory churn. The nested loops (outer loop Ã— 5 variables) exacerbate the problem. With 344k cells Ã— 28 years and ~1.37M neighbor links, the naive approach becomes prohibitively slow.

**Optimization Strategy**  
- **Precompute neighbor indices once** and reuse them (already done).
- **Vectorize aggregation**: Instead of looping over rows, convert neighbor relationships into a long edge list and use `data.table` or `dplyr` group operations to compute `max`, `min`, and `mean` in bulk.
- **Leverage fast joins**: Use `data.table` keyed joins for speed and memory efficiency.
- Avoid raster focal analogy since we must exactly preserve neighbor relationships across panel years, which raster kernels can't guarantee for irregular IDs.

**Optimized Approach**  
1. Expand neighbor relationships into `(cell_id, neighbor_id)` pairs for *each year*.
2. Join this long edge table with the value column for each neighbor variable.
3. Aggregate per `(cell_id, year)` using `data.table::frollapply`-like grouping to compute max, min, mean.
4. Merge results back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table for speed
setDT(cell_data)

# Build edge list of neighbors for all years
# id_order: vector of cell IDs in reference order
# rook_neighbors_unique: list of neighbor indices

edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    src = id_order[i],
    nbr = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand edge list to all years
years <- sort(unique(cell_data$year))
edge_dt <- edge_list[, .(year = years), by = .(src, nbr)]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(edge_dt, nbr, year)

compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  joined <- edge_dt[cell_data, on = .(nbr = id, year), nomatch = 0]
  setnames(joined, var_name, "nbr_val")

  # Drop NAs
  joined <- joined[!is.na(nbr_val)]

  # Aggregate max, min, mean by src + year
  agg <- joined[, .(
    max_val = max(nbr_val),
    min_val = min(nbr_val),
    mean_val = mean(nbr_val)
  ), by = .(id = src, year)]

  # Merge back to cell_data
  setnames(agg,
           c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  agg
}

# Process all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge all aggregated features
result_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE),
                    agg_list)

# Join with original data
cell_data <- merge(cell_data, result_dt, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works Faster**  
- Single pass per variable instead of millions of tiny loops.
- `data.table` uses highly optimized C-level aggregation.
- Memory footprint is reduced by working with a long edge table rather than repeatedly slicing a huge data.frame.

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to under 1 hour (varies by I/O overhead and CPU threads).

**Preserves**  
- Original Random Forest model.
- Exact neighbor-based statistics (max, min, mean).
- Panel structure across years.
END_RESPONSE>>>
