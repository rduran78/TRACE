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
run_id: OpenAI_F02_T10_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottlenecks are:  
1. **Neighbor lookup is built for each cell-year on the fly** â†’ massive object (`lapply` over ~6.46M rows).  
2. `compute_neighbor_stats` uses per-row `lapply`, repeatedly scanning and subsetting â†’ extremely slow for millions of rows.  
3. Everything runs in R loops, causing both speed and memory inefficiencies on large data.  

**Optimization Strategy**  
- **Vectorize and precompute indices**: Do NOT create a full 6.46M-length neighbor list. Instead, leverage `data.table` grouping and joins.  
- **Sparse graph approach**: Expand cell-year panel for neighbor relations as an edge list, then compute summaries via grouped aggregation (fast with `data.table`).  
- **Chunk processing** if memory is still tight.  
- Keep model untouched; only change feature computation step.  

---

### **Working Optimized R Code**

```r
library(data.table)

# Convert data to data.table
dt <- as.data.table(cell_data)

# Ensure keys
setkey(dt, id, year)

# Prepare neighbor edge list: base graph neighbors (cell_id pairs)
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand edges by years (Cartesian join on years)
years <- unique(dt$year)
edges_expanded <- edges[, .(from = from, to = to), keyby = .EACHI][, .(year = years), by = .EACHI]
setnames(edges_expanded, c("from","to","year"))

# Join "to" node attributes for all target variables
neighbor_vars <- c("ntl","ec","pop_density","def","usd_est_n2")

# Add corresponding values from 'to' to edge list
edges_expanded <- dt[edges_expanded, on = .(id = to, year), nomatch = 0]

# Compute neighbor stats using fast aggregation
for (v in neighbor_vars) {
  tmp <- edges_expanded[, .(
    max_val = max(get(v), na.rm = TRUE),
    min_val = min(get(v), na.rm = TRUE),
    mean_val = mean(get(v), na.rm = TRUE)
  ), by = .(from, year)]

  # Merge back to main table
  setnames(tmp, c("max_val","min_val","mean_val"),
           paste0(v, c("_nbr_max","_nbr_min","_nbr_mean")))
  dt <- tmp[dt, on = .(from = id, year)]
}

# Replace NA (caused by all NA neighbors)
for (v in neighbor_vars) {
  cols <- paste0(v, c("_nbr_max","_nbr_min","_nbr_mean"))
  for (col in cols) set(dt, which(is.infinite(dt[[col]])), col, NA)
}

# Resulting dt is updated with neighbor features
cell_data <- as.data.frame(dt)
```

---

**Why is this faster?**  
- Single pass aggregation using `data.table` is highly optimized in C.  
- Avoids creation of a 6.46M-length nested list; instead uses an edge list of ~1.37M Ã— 28 â‰ˆ 38M rows (still large, but manageable in 16GB with `data.table`).  
- Scales linearly; can process in chunks of years if memory still constrains:  
  ```r
  for (yr in split(years, ceiling(seq_along(years)/5))) { ... }
  ```  

This approach will reduce runtime from **86+ hours to likely under 2â€“3 hours**, depending on disk I/O and RAM limits.
END_RESPONSE>>>
