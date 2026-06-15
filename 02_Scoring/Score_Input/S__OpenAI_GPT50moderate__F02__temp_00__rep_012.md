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
run_id: OpenAI_F02_T00_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F02_OpenAI_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length â‰ˆ 6.46M (one per cell-year), each storing neighbor indices. This is highly inefficient in both memory and time.  
- `compute_neighbor_stats` repeatedly traverses these lists and subsets vectors, causing heavy R-level overhead.  
- The process is purely interpreted R loops (`lapply`), which is not scalable for millions of rows.  

**Optimization Strategy**  
- **Avoid per-row neighbor lookups**: Instead of building a 6.46M-element list, compute neighbor stats by joining on neighbor relationships in a long format.  
- **Vectorize using `data.table`**: Convert the dataset to `data.table` and perform grouped aggregations.  
- **Precompute neighbor relationships once**: Expand rook neighbors into a long table of `(cell_id, neighbor_id)` pairs, then join with data for each year.  
- **Compute all neighbor stats in one pass**: Melt the variables and aggregate by `(cell_id, year)`.  
- **Memory efficiency**: Use integer keys and avoid large nested lists.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# 1. Build neighbor pairs (cell_id -> neighbor_id)
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# 2. Expand neighbor pairs for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# 3. Join neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for easier aggregation
long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var_name", value.name = "value")

# Join neighbor values
neighbor_long <- merge(neighbor_dt, long_data, by.x = c("neighbor_id", "year"),
                       by.y = c("id", "year"), allow.cartesian = TRUE)

# 4. Compute neighbor stats: max, min, mean per (id, year, var_name)
neighbor_stats <- neighbor_long[
  !is.na(value),
  .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
  by = .(id, year, var_name)
]

# 5. Reshape back to wide format
neighbor_wide <- dcast(neighbor_stats, id + year ~ var_name,
                       value.var = c("nb_max", "nb_min", "nb_mean"))

# 6. Merge back into cell_data
cell_data <- merge(cell_data, neighbor_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- Eliminates 6.46M list elements â†’ uses a compact long table.
- Aggregation is done in C via `data.table`, drastically reducing overhead.
- Memory footprint is manageable because we only store neighbor relationships once and compute in chunks.
- Preserves original estimand and trained model (we only add features).

**Expected Performance**:  
On a 16 GB laptop, this approach should complete in **minutes to a few hours** instead of 86+ hours, depending on disk speed and CPU.  

**Further Tips**:  
- If memory is still tight, process one variable at a time instead of melting all.  
- Use `setkey` on `data.table` for faster joins.  
- Optionally parallelize with `future.apply` or `data.table`â€™s `by` + `multithread`.
END_RESPONSE>>>
