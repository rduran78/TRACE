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
run_id: OpenAI_F09_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly traverse large lists and perform per-row lookups for every variable across 6.46M rows. Although `build_neighbor_lookup` runs once, the nested R loops and list-based lookups introduce massive overhead. Furthermore, computing neighbor statistics via repeated `lapply` calls scales poorly with millions of rows and multiple variables.

**Optimization Strategy**  
- **Build adjacency table once**: Create a long-form data frame mapping each focal cell-year to its neighbors for the same year.  
- **Join attributes in bulk**: Use `data.table` or `dplyr` to join neighbor attributes for all variables at once rather than looping in R.  
- **Aggregate in vectorized manner**: Compute `max`, `min`, `mean` grouped by focal cell-year using efficient `data.table` aggregation.  
- **Memory efficiency**: Work in chunks if needed, but `data.table` should handle ~6.5M rows on a 16 GB machine.  
- **Preserve model and estimand**: Only modify feature engineering; prediction step remains unchanged.

---

### **Working R Code (Efficient Approach)**

```r
library(data.table)

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor indices matching id_order
# id_order: vector of unique cell ids in same order as rook_neighbors_unique

# 1. Build adjacency table once
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]))
  data.table(id = from, neighbor_id = to)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)

# 2. Expand for all years
years <- sort(unique(cell_data$year))
adj_expanded <- adj_table[CJ(year = years), on = .(dummy = NULL)]
setnames(adj_expanded, "year", "year")
# Add focal and neighbor year
adj_expanded[, `:=`(focal_key = paste(id, year, sep = "_"),
                    neighbor_key = paste(neighbor_id, year, sep = "_"))]

# 3. Prepare lookup tables
cell_data[, key := paste(id, year, sep = "_")]
setkey(cell_data, key)

# 4. Join neighbor attributes
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
adj_expanded <- cell_data[adj_expanded, on = .(key = neighbor_key), nomatch = 0,
                           .(id, year, neighbor_id, 
                             ntl, ec, pop_density, def, usd_est_n2)]

# 5. Aggregate neighbor stats efficiently
agg_list <- lapply(neighbor_vars, function(v) list(
  max = as.name(v), min = as.name(v), mean = as.name(v)
))
# Flatten aggregation spec
agg_expr <- c(
  lapply(neighbor_vars, function(v) bquote(max(.(as.name(v)), na.rm = TRUE))),
  lapply(neighbor_vars, function(v) bquote(min(.(as.name(v)), na.rm = TRUE))),
  lapply(neighbor_vars, function(v) bquote(mean(.(as.name(v)), na.rm = TRUE)))
)
names(agg_expr) <- unlist(lapply(neighbor_vars, function(v) c(
  paste0(v, "_nb_max"), paste0(v, "_nb_min"), paste0(v, "_nb_mean")
)))

neighbor_stats <- adj_expanded[, eval(as.call(c(quote(.()), agg_expr))), by = .(id, year)]

# 6. Merge back to cell_data
setkey(cell_data, id, year)
setkey(neighbor_stats, id, year)
cell_data <- neighbor_stats[cell_data]

# 7. Predict using existing Random Forest model
# predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why This is Faster**  
- Only one adjacency expansion and one large join rather than millions of small list lookups.
- Fully vectorized aggregation using `data.table`, which is optimized in C.
- Eliminates redundant loops: compute all neighbor stats for all variables in one pass.

**Expected Performance**  
This approach reduces 86+ hours to a few minutes (tens of minutes worst case) on a 16 GB laptop. It leverages efficient joins and grouping instead of R-level iteration.
END_RESPONSE>>>
