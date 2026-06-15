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
run_id: OpenAI_F06_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck is the `lapply` over ~6.46M rows where for each row, the neighbor indices are looked up and statistics (max, min, mean) are computed. This results in **O(N Ã— k)** complexity (N = 6.46M rows, k â‰ˆ number of neighbors), with heavy R list processing and repeated indexing. Additionally, the code repeatedly allocates vectors and performs `NA` filtering for every row. This design is not cache-friendly and cannot scale on a laptop with 16 GB RAM.

---

**Optimization Strategy**  
1. **Avoid per-row `lapply` loops**: Use a **vectorized aggregation** by reshaping the data into a data.table and joining neighbor relationships.
2. **Precompute neighbor relationships for all years**: Expand rook neighbors into a long edge list replicated across years.  
3. **Use `data.table` grouped aggregation**: Compute max, min, mean in one pass instead of looping.
4. **Chunk processing or memory-efficient join**: Handle the 6.46M rows efficiently without exploding memory.
5. **Preserve estimand**: Do not alter numerical computations; same max/min/mean logic applies after filtering NAs.

---

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert your data to data.table
setDT(cell_data)

# 1. Build neighbor edge list (cell_id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors (spdep::nb)
# id_order: vector mapping index to cell_id
# Create long edge table
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# 2. Expand to all years
years <- unique(cell_data$year)
edges_expanded <- edges[, .(id = from, neighbor = to), by = years]
setnames(edges_expanded, "years", "year")

# 3. Create keys for fast joins
setkey(cell_data, id, year)
setkey(edges_expanded, neighbor, year)

# 4. Join neighbor values for each variable and aggregate
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  joined <- cell_data[edges_expanded, on = .(id = neighbor, year)]
  # joined columns: id (from edges_expanded), neighbor, year, var_name
  # Aggregate by focal cell (edges_expanded$id) and year
  agg <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  # Replace Inf/-Inf if all NA
  agg[is.infinite(max_val), c("max_val","min_val","mean_val") := NA]
  # Return table for merging
  setnames(agg, c("id","year","max_val","min_val","mean_val"),
           c("id","year",
             paste0(var_name,"_nbr_max"),
             paste0(var_name,"_nbr_min"),
             paste0(var_name,"_nbr_mean")))
  agg
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute for all variables and merge back
results_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id","year"), all = TRUE), results_list)

# Merge neighbor features into original data
cell_data <- merge(cell_data, neighbor_features, by = c("id","year"), all.x = TRUE)
```

---

**Why This Is Faster**  
- Eliminates 6.46M `lapply` calls â†’ replaced by a few highly optimized C-level `data.table` group operations.
- Single pass per variable with grouping instead of millions of small index lookups.
- No repeated vector filtering inside loops.

**Expected Performance Gain**  
From 86+ hours to **minutes or a few hours** (depending on disk speed and CPU), as the main cost is now a few large joins and group aggregations rather than millions of R-level calls.

**Preserves**  
- Original estimand (max, min, mean among rook neighbors per cell-year).
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
