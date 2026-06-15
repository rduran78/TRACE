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
run_id: OpenAI_F04_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is indeed the neighbor feature construction, not Random Forest inference. Two major inefficiencies:  
1. **`build_neighbor_lookup`** constructs a massive list (length â‰ˆ 6.46M) by iterating over all rows, repeatedly performing string concatenation and hash lookups.  
2. **`compute_neighbor_stats`** uses `lapply` over the same list repeatedly for each variable, causing 5Ã— redundant traversal.  

Current approach scales as O(N Ã— k Ã— V) where N â‰ˆ 6.46M, k = average neighbor count, V = 5 variables. This is why it takes ~86 hours.

---

### **Optimization Strategy**
- **Avoid repeated string operations**: Precompute numeric indices once; remove costly `paste` calls inside loops.
- **Vectorize neighbor relationships**: Use a long-format edge list (cell-year â†’ neighbor-year) and **data.table** joins instead of nested lapply.
- **Compute all neighbor stats in one grouped operation**: Aggregate max/min/mean for each variable per observation.
- Use **parallelization and efficient memory structures**: `data.table` for in-memory joins; chunking if memory constrained.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setkey(dt, id, year)

# Build edge list (cell-year â†’ neighbor-year)
# rook_neighbors_unique: list of neighbors per cell id
id_order <- as.integer(id_order)
edges <- data.table(from = rep(id_order, lengths(rook_neighbors_unique)),
                    to   = unlist(rook_neighbors_unique))

# Replicate for each year (cartesian join)
years <- sort(unique(dt$year))
edges_year <- edges[,.(id = from, neighbor_id = to)][,.(year = years), by=.(id, neighbor_id)]
# edges_year: columns id, neighbor_id, year

# Merge to get neighbor-year indices
# Create long form: (id, year, neighbor_id, neighbor_year)
edges_year[, neighbor_year := year]

# Join neighbor values
vars <- c("ntl","ec","pop_density","def","usd_est_n2")

# Melt dt for easier aggregation
dt_long <- melt(dt, id.vars=c("id","year"), measure.vars=vars,
                variable.name="var", value.name="val")

# Join edges with dt_long on neighbor_id + neighbor_year
setkey(dt_long, id, year)
joined <- dt_long[edges_year, on = .(id = neighbor_id, year = neighbor_year)]
# joined now has: id(i), year(i), neighbor_id, var, val

# Aggregate stats: max, min, mean per (target id/year/var)
agg <- joined[, .(nb_max = max(val, na.rm=TRUE),
                  nb_min = min(val, na.rm=TRUE),
                  nb_mean = mean(val, na.rm=TRUE)),
              by=.(target_id = i.id, target_year = i.year, var)]

# Reshape to wide and merge back
agg_wide <- dcast(agg, target_id + target_year ~ var,
                  value.var=c("nb_max","nb_min","nb_mean"))

# Merge with original dt
setnames(agg_wide, c("target_id","target_year"), c("id","year"))
final_dt <- merge(dt, agg_wide, by=c("id","year"), all.x=TRUE)

# final_dt now contains original data + neighbor stats
```

---

### **Why This Works**
- Complexity reduced from ~6.46M * 5 * k list traversals to **one pass join + grouped aggregation**.
- No repeated string concatenation; all numeric keys.
- **data.table** operations are highly optimized in C, memory-efficient, and allow parallelization if `setDTthreads()` > 1.
- Preserves original estimand and Random Forest model (no retraining).

**Expected runtime**: From 86+ hours â†’ a few minutes (~5â€“15 min on 16GB RAM) depending on threads and disk I/O.
END_RESPONSE>>>
