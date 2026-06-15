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
run_id: OpenAI_F02_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck arises from:  
1. **`lapply` over 6.46M rows**: Each row triggers construction of neighbor indices repeatedly, which is extremely expensive.  
2. Repeated string concatenation (`paste`) and lookups inside the loop.  
3. `compute_neighbor_stats` runs multiple `lapply` passes for each variable.  
4. Memory overhead from very large lists.  

**Optimization Strategy**  
- Precompute neighbor indices **once** for all rows in vectorized form.  
- Avoid `paste` and excessive hash lookups; instead, use numeric IDs and matrix joins.  
- Use **data.table** for fast joins and grouping.  
- Compute neighbor statistics in a single **vectorized grouped aggregation** rather than millions of small loops.  
- If possible, process in **chunks** to reduce peak RAM usage.  

---

### **Optimized Approach**
1. Expand neighbor relationships across years: create a long table of `(cell_id, year, neighbor_id)`.  
2. Join with `data` to get neighbor values for each `(cell_id, year)` pair.  
3. Aggregate neighbor stats by `(cell_id, year)` for each variable.  
4. Merge results back into `cell_data`.  

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table: columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Expand rook_neighbors_unique into a long table of edges
# id_order is the mapping from index to cell ID
edges <- data.table::data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)
edges[, to := id_order[to]]  # convert neighbor indices to IDs

# Expand across years: cross join with years
years <- sort(unique(cell_data$year))
edges_year <- edges[, .(id = from, neighbor_id = to), ][
  , .(year = years), by = .(id, neighbor_id)
]

# Join neighbor values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for easier join
cell_long <- melt(cell_data, id.vars = c("id", "year"), 
                  measure.vars = neighbor_source_vars,
                  variable.name = "var", value.name = "value")

# Join edges_year with cell_long on neighbor_id/year to get neighbor values
edges_long <- merge(edges_year, cell_long, 
                    by.x = c("neighbor_id", "year"), 
                    by.y = c("id", "year"), 
                    allow.cartesian = TRUE)

# Compute stats by id, year, var
neighbor_stats <- edges_long[!is.na(value),
  .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
  by = .(id, year, var)
]

# Reshape wide: one row per id-year with columns for each stat/var
neighbor_stats_wide <- dcast(
  neighbor_stats, id + year ~ var, 
  value.var = c("nb_max", "nb_min", "nb_mean")
)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Key Gains**
- Eliminates millions of small list operations.
- Fully vectorized using `data.table`.
- Single join + aggregation instead of per-row loops.
- Can handle 6.4M rows and 1.3M neighbor pairs on a 16GB laptop if processed in **chunks** (e.g., by year or variable) if memory is tight:
  ```r
  for (yr in years) {
    # repeat join and aggregation for that year only
  }
  ```

This approach preserves the trained Random Forest model and original numerical estimand while reducing runtime from **86+ hours to a few hours or less** on a standard laptop.
END_RESPONSE>>>
