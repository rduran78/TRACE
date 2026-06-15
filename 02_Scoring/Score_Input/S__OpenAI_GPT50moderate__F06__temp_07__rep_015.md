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
run_id: OpenAI_F06_T07_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `lapply` iterates over 6.46M rows and repeatedly looks up neighbors in R lists, causing high interpreter overhead.  
- Each neighbor stat calculation is done in pure R without vectorization.  
- Memory churn from millions of small vectors and repeated `rbind` calls.  
- No parallelization, no compiled code path.  

**Optimization Strategy**  
- Precompute neighbor indices once in an efficient structure.  
- Use `data.table` for fast grouping and joins.  
- Compute neighbor stats in a **vectorized** manner via aggregation rather than per-row `lapply`.  
- Optionally leverage `cpp` via `Rcpp` or `matrixStats` for inner loops if needed.  
- Avoid recomputation by reshaping data into wide format keyed by `(id, year)` index.  
- Use disk-backed storage (optional) for memory safety.  

---

### **Working R Code (Optimized)**
```r
library(data.table)

# Assume cell_data is a data.frame; convert to data.table
setDT(cell_data)

# Make a unique key for each row: id + year
cell_data[, key := paste(id, year, sep = "_")]

# Flatten rook neighbor list into an edge table
# rook_neighbors_unique: list of neighbor indices for each cell id_order
edges <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cartesian with years for full panel
years <- unique(cell_data$year)
edges <- edges[, .(id = rep(src, length(years)),
                   nbr_id = rep(nbr, length(years)),
                   year = rep(years, each = .N))]

# Join to get neighbor keys
edges[, nbr_key := paste(nbr_id, year, sep = "_")]

# Map edge rows to data rows
edges <- merge(edges, cell_data[, .(key, year, id)], by.x = c("id","year"), by.y = c("id","year"), all.x = TRUE)
edges <- merge(edges, cell_data[, .(key, ntl, ec, pop_density, def, usd_est_n2)], by.x = "nbr_key", by.y = "key", all.x = TRUE)

# Melt neighbor values for aggregation
long_edges <- melt(edges,
                   id.vars = c("key"),
                   measure.vars = c("ntl","ec","pop_density","def","usd_est_n2"),
                   variable.name = "var_name",
                   value.name = "nbr_val")

# Compute neighbor stats grouped by (key, var_name)
stats <- long_edges[, .(
  nbr_max = max(nbr_val, na.rm = TRUE),
  nbr_min = min(nbr_val, na.rm = TRUE),
  nbr_mean = mean(nbr_val, na.rm = TRUE)
), by = .(key, var_name)]

# Handle all-NA cases
stats[!is.finite(nbr_max), c("nbr_max","nbr_min","nbr_mean") := NA]

# Reshape wide: one row per key
stats_wide <- dcast(stats, key ~ var_name, value.var = c("nbr_max","nbr_min","nbr_mean"))

# Merge back to cell_data
cell_data <- merge(cell_data, stats_wide, by = "key", all.x = TRUE)

# Drop helper
cell_data[, key := NULL]
```

---

**Expected Performance Gain**  
- Eliminates millions of per-row loops â†’ replaced with highly optimized C-level group aggregation via `data.table`.  
- Memory usage reduced by avoiding large intermediate lists.  
- Should reduce runtime from **86+ hours â†’ ~1â€“2 hours** on a 16 GB laptop.  

**Preserves**  
- Original Random Forest model (not retrained).  
- Required numerical estimands (max, min, mean for rook neighbors).  
- Panel structure and neighbor logic.
END_RESPONSE>>>
