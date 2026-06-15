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
run_id: OpenAI_F07_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is very slow because it repeatedly computes neighbor statistics in R using inefficient `lapply` loops across ~6.46M rows and reconstructs neighbor relationships for each variable. The main bottlenecks:  
- Pure R loops over millions of entries (`lapply` on large lists).  
- Repeated string concatenation and lookups inside loops.  
- No vectorization or memory-efficient joins.  
- No parallelization, despite embarrassingly parallel structure.  

Given 6.46M rows and ~1.37M neighbor links, the current method (building and traversing lookup lists) scales poorly. Current runtime (86+ hrs) confirms this.

---

**Optimization Strategy**  
1. **Precompute neighbor index mapping ONCE** as integer vectors per row.  
2. **Use `data.table` + parallelization** for fast indexing and aggregation.  
3. **Avoid repeated string operations** by joining integer IDs instead of character keys.  
4. **Flatten neighbor pairs** into a long table `(i, neighbor_id, year)`, then join on data values, allowing a single grouped aggregation per variable.  
5. **Compute max, min, mean with `data.table` aggregation** (vectorized and multithreaded with `setDTthreads()`).  

Outcome: reduces from hours to minutes on 16 GB laptop.

---

**Working R Code**

```r
library(data.table)
setDTthreads(parallel::detectCores())  # Parallelize data.table ops

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
dt <- as.data.table(cell_data)

# Precompute neighbor long table
# rook_neighbors_unique: list of integer vectors, same order as id_order
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i],
               neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand neighbor pairs over all years in the panel
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N))]

# Create keys for fast joins
setkey(dt, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Join neighbor values for all source vars
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_long <- dt[neighbor_dt, on = .(id = neighbor_id, year), nomatch = 0,
                    .(id = i.id, year, value = get(var)), by = .EACHI,
                    allow.cartesian = TRUE]

# Compute stats in a loop for each variable and merge back
for (var in neighbor_vars) {
  neighbor_long <- dt[neighbor_dt, on = .(id = neighbor_id, year), nomatch = 0,
                      .(id = i.id, year, value = get(var))]

  stats <- neighbor_long[
    !is.na(value),
    .(max_val = max(value), min_val = min(value), mean_val = mean(value)),
    by = .(id, year)
  ]

  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste(var, c("nmax", "nmin", "nmean"), sep = "_"))

  dt <- merge(dt, stats, by = c("id", "year"), all.x = TRUE)
}

# Result: dt now has neighbor stats added efficiently
```

---

**Why This Works**  
- Eliminates repeated lookups and string concatenation by flattening neighbors into a relational structure.  
- Utilizes `data.table` for memory efficiency and speed (~100x faster than `lapply`).  
- Aggregates in parallel using optimized C-level loops under the hood.  

**Expected Runtime**  
On a 16 GB laptop: a few minutes per variable instead of hours, total <1 hour for all 5 variables.

**Trained Model** remains **unchanged** because we only enriched `cell_data` with new features; the pipeline downstream can reuse these features without retraining if predictions are already model-defined.
END_RESPONSE>>>
