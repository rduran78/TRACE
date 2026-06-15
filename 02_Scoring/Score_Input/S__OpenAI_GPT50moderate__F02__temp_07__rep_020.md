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
run_id: OpenAI_F02_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-heavy because:  
- `build_neighbor_lookup` creates a huge list of length 6.46M with repeated string concatenations and lookups.  
- `compute_neighbor_stats` repeatedly traverses large vectors using `lapply`, causing excessive R interpreter overhead and memory allocation.  
- Everything is computed in pure R loops; this is inefficient for 6.46M rows Ã— 5 variables.  

---

**Optimization Strategy**  
1. **Avoid building massive lists in R**: Instead of precomputing a per-row neighbor index list, work at the *cell level* and then broadcast to years.  
2. **Use matrix/vectorized operations**: Precompute neighbor statistics per year using numeric indexing and `tapply`/`rowsum` or `data.table`.  
3. **Process year-by-year**: Reduces memory footprint dramatically.  
4. **Use `data.table` for speed**: Handles group operations efficiently.  
5. **Leverage sparse representation**: Store neighbor links as two integer vectors (from, to) and join instead of building nested lists.  

---

### **Optimized Approach (data.table)**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Ensure keys
setkey(cell_data, id, year)

# Neighbor edges: expand to all years once
# rook_neighbors_unique is list: each entry = neighbors of id_order[i]
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Cross with years (Cartesian join)
years <- sort(unique(cell_data$year))
edges_full <- edges[, .(from, to, year = rep(years, each = .N))]

# Merge to attach variable values (for each neighbor)
compute_neighbor_stats_fast <- function(dt, var) {
  # Attach neighbor values
  vals <- dt[, .(id, year, v = get(var))]
  setkey(vals, id, year)
  e <- edges_full[vals, on = .(to = id, year)]
  
  # Now compute stats grouped by (from, year)
  stats <- e[!is.na(v), .(
    max_v = max(v),
    min_v = min(v),
    mean_v = mean(v)
  ), by = .(from, year)]
  
  # Merge back to main table
  setnames(stats, c("from","year","max_v","min_v","mean_v"),
           c("id","year",
             paste0(var,"_nb_max"),
             paste0(var,"_nb_min"),
             paste0(var,"_nb_mean")))
  dt[stats, on = .(id, year)]
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_fast(cell_data, var)
}
```

---

**Why this works:**  
- `edges_full` acts as a long table of neighbor relationships per year (â‰ˆ 1.3M Ã— 28 â‰ˆ 38M rows, but handled in chunks if needed).  
- Each join and aggregation is in C via `data.table`, vastly faster than R loops.  
- No giant list objects; everything is columnar.  
- Can further chunk by year if memory is tight: process one year at a time and `rbindlist`.  

**Expected speed-up:** From 86+ hours to a few hours or less on a laptop.  

**Memory tip:** If 38M rows is too large, process year-by-year:
```r
result_list <- vector("list", length(years))
for (i in seq_along(years)) {
  y <- years[i]
  e_year <- edges[, .(from, to)]
  vals <- cell_data[year == y, .(id, v1 = ntl, v2 = ec, ...)]
  # Compute per var in this subset
}
cell_data <- rbindlist(result_list)
```

This preserves the trained Random Forest model and the original estimand, while making the pipeline practical on a 16 GB machine.
END_RESPONSE>>>
