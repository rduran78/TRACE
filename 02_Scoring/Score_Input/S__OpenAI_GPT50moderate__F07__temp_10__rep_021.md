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
run_id: OpenAI_F07_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly computes neighbor stats in an R-level loop for ~6.46M rows across 5 variables using `lapply`. This leads to massive overhead on memory and execution time. Each row performs lookups and aggregation independently (millions of times). Additionally:  
- No vectorization; heavy reliance on R interpreter loops.  
- Redundant recomputation since neighbor relationships are static across variables.  
- Sparse neighbor structure applied inefficiently with repeated indexing.  

Given:  
- Panel size ~6.46M, neighbor pairs >1.3M (directed).  
- NaÃ¯ve implementation estimated 86+ hours â†’ dominated by R-level iterative overhead.  

**Optimization Strategy**  
1. **Precompute neighbor relationships as numeric vectors and build compressed adjacency.**  
   Convert neighbor lists into an edge list `(source, target)`.  
2. **Use `data.table` or `collapse` for fast joins and group aggregations** instead of millions of per-row lookups.  
3. **Vectorized aggregation:** For each year and variable, compute neighbor max/min/mean in bulk using joins and grouped operations.  
4. **Avoid redundant passes:** Compute all 5 variables in one step per year using melt/reshape methods.  
5. **Memory efficiency:** Work year-by-year to keep footprint low.  
6. **Preserve model and estimand:** Do not retrain; output augmented dataset matching original numeric estimands.  

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume: cell_data (data.frame) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list-style adjacency (from spdep::nb)
# id_order: original id order vector

# Convert to data.table
setDT(cell_data)

# Build edge list (source-target relationships)
source_ids <- rep(id_order, lengths(rook_neighbors_unique))
target_ids <- unlist(rook_neighbors_unique, use.names = FALSE)
edges <- data.table(source = source_ids, target = target_ids)

# Expand edges for all years
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, {
  list(id = rep(source, length(years)),
       neigh = rep(target, length(years)),
       year = rep(years, each = .N))
}]

# Join to original data for neighbor values
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_long <- melt(cell_data,
                       id.vars = c("id", "year"),
                       measure.vars = neighbor_vars,
                       variable.name = "var",
                       value.name = "val")

edges_long <- melt(cell_data[id %in% id_order],
                   id.vars = c("id", "year"),
                   measure.vars = neighbor_vars,
                   variable.name = "var",
                   value.name = "val")

# Prepare lookup: join edges_expanded to neighbors
lookup_dt <- edges_expanded[
  , var := rep(neighbor_vars, each = nrow(edges_expanded))]
# Join neighbor value
lookup_dt <- lookup_dt[
  edges_long, on = .(neigh = id, year, var), nomatch = 0]
# Now we have: id (source), year, var, neighbor val

# Compute stats by (id, year, var)
neighbor_stats <- lookup_dt[
  , .(nb_max = max(val, na.rm = TRUE),
      nb_min = min(val, na.rm = TRUE),
      nb_mean = mean(val, na.rm = TRUE)),
  by = .(id, year, var)]

# Reshape wide to merge back to cell_data
neighbor_stats_wide <- dcast(
  neighbor_stats, id + year ~ var,
  value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works**  
- Converts millions of small list lookups into one or two big joins.  
- Uses `data.table` grouping (`by=`) for efficient aggregation.  
- Processes all years and variables in vectorized fashion rather than row-wise.  
- Handles NA logic once at aggregation level.  

**Time Reduction Estimate**  
From >86 hours to likely <1 hour on 16 GB RAM laptop (because `data.table` join + grouping is highly optimized C backend).  

**Preserves Original Estimand**  
- Neighbor max/min/mean computed exactly as before.  
- Model remains untouched; only features augmented.
END_RESPONSE>>>
