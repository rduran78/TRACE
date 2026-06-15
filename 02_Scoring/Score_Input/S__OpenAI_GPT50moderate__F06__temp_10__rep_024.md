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
run_id: OpenAI_F06_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested structure of `lapply` over ~6.46M rows, computing neighbor statistics naively for each cell-year. Each iteration recomputes values with random memory access and R function calls, which is prohibitively slow. The task is embarrassingly parallelizable, but also cache-inefficient. The neighbor set (rook neighbors) is constant per cell, while panel years create many repeated lookups â€” doing this via R loops is inefficient. Current complexity is roughly *O(N Ã— avg_neighbors)* with `N â‰ˆ 6.46M`.

---

**Optimization Strategy**  
1. **Precompute reusable structures**:  
   - Create a mapping of neighbors per cell once. Avoid repeated string concatenation.
2. **Use `data.table` for fast grouping and joins**, leveraging ID and year as keys.  
3. **Vectorize neighbor aggregation**:  
   - Reshape data to *wide by year* only if needed, but here keep long format and compute aggregates through a self-join keyed by `(cell_id, year)`.
4. **Optional parallelization** using `future.apply` or `data.table` multithreading.
5. **Preserve estimand**: results of `max`, `min`, `mean` over neighbor values must be identical.

---

**Working R Code** *(Efficient data.table solution)*  

```r
library(data.table)

# Assume cell_data as data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Convert neighbors list to long format edge table: cell_id -> neighbor_id
edges <- data.table(from = rep(id_order, lengths(rook_neighbors_unique)),
                    to   = unlist(rook_neighbors_unique))
# You only need one direction? Here it's directed as per original
setnames(edges, c("cell_id", "neighbor_id"))

# Prepare keys for fast join
setkey(cell_data, id, year)

compute_neighbor_stats_dt <- function(cell_dt, edges, var_name) {
  # Extract pairings with year
  # Join left: edges$neighbor_id -> cell_data$id to get neighbor values
  neighbor_vals <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)

  # Expand: join edges on neighbor_id
  joined <- neighbor_vals[edges, on = .(neighbor_id), allow.cartesian = TRUE]
  # joined now has columns: neighbor_id, year, value, cell_id (from edges)
  
  # Group by focal cell and year
  stats <- joined[, .(
    max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]

  # Join back to main table
  out <- cell_dt[stats, on = .(id = cell_id, year)]
  setnames(out, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  out[]
}

# Apply for all variables
for (var in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  cell_data <- compute_neighbor_stats_dt(cell_data, edges, var)
}

# cell_data now enriched with neighbor stats, ready for RF prediction
```

---

**Why this works faster**  
- No per-row lapply: operations are aggregated by `data.table` group statistics in C.
- String operations eliminated; joins driven by integer IDs.
- Handles all rows in parallel internally.
- Complexity reduces to linear in relation/row count instead of nested R calls.
- Memory-friendly and works within 16 GB RAM for given dataset (~6.5M rows + ~1.37M edges).

**Expected Performance**  
From 86+ hours â†’ typically **minutes to low hours** on standard hardware. Further speedup possible with `setDTthreads(parallel::detectCores())`.  

**Model Preservation**  
The Random Forest model remains untouched; only feature engineering is accelerated while preserving original statistical outputs.
END_RESPONSE>>>
