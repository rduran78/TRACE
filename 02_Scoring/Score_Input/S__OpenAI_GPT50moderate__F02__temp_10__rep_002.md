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
run_id: OpenAI_F02_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is from nested `lapply` calls over ~6.5M rows and repeated R-level operations. Memory pressure comes from building huge lists (`neighbor_lookup` of length 6.46M) and repeatedly computing neighbor statistics in pure R loops. This approach is inefficient because:

- `build_neighbor_lookup` expands neighbors per row repeatedly by string concatenations (`paste`) and hashmap lookups.
- `compute_neighbor_stats` loops per row again and aggregates at R level.
- No vectorization; all row-wise operations are interpreted.
- Large intermediate lists (millions of elements) blow up RAM.

---

**Optimization Strategy**  
Key targets:
1. **Avoid row-wise loops**: Restructure the pipeline to compute features using *joins/grouped summarization* rather than per-row lists.
2. **Precompute neighbor mappings in a long format once** and join back using `data.table` for speed and low memory.
3. Use **fast aggregation** with `data.table` (`fread`, keyed joins) to compute max/min/mean in bulk.
4. Do all operations for all variables in a single grouped pass to reduce repeated scans.
5. Ensure chunk-wise/stream processing if memory remains tight.

---

**Optimized Implementation with `data.table`**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Convert neighbors list (rook_neighbors_unique) into long table
# id_order: vector of cell IDs in same order as rook_neighbors_unique
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src_id = id_order[i], nb_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cartesian expansion across years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = src_id, nb_id, year = years), by = seq_len(nrow(neighbor_dt))]
neighbor_dt[, seq_len := NULL]

# Merge neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, nb_id, year)
merged <- cell_data[neighbor_dt, on = .(id = nb_id, year), nomatch = 0L]

# Compute stats for all neighbor_source_vars at once
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg <- merged[, .(
  ntl_max  = max(ntl, na.rm=TRUE),
  ntl_min  = min(ntl, na.rm=TRUE),
  ntl_mean = mean(ntl, na.rm=TRUE),
  ec_max   = max(ec, na.rm=TRUE),
  ec_min   = min(ec, na.rm=TRUE),
  ec_mean  = mean(ec, na.rm=TRUE),
  pop_max  = max(pop_density, na.rm=TRUE),
  pop_min  = min(pop_density, na.rm=TRUE),
  pop_mean = mean(pop_density, na.rm=TRUE),
  def_max  = max(def, na.rm=TRUE),
  def_min  = min(def, na.rm=TRUE),
  def_mean = mean(def, na.rm=TRUE),
  usd_max  = max(usd_est_n2, na.rm=TRUE),
  usd_min  = min(usd_est_n2, na.rm=TRUE),
  usd_mean = mean(usd_est_n2, na.rm=TRUE)
), by = .(id, year)]

# Merge back to cell_data
cell_data <- agg[cell_data, on = .(id, year)]

# Clean inf values if any NA-only groups existed
replace_inf <- function(x) fifelse(is.infinite(x), NA_real_, x)
cell_data[, names(agg)[-(1:2)] := lapply(.SD, replace_inf), .SDcols = names(agg)[-(1:2)]]
```

---

**Why this is faster**:  
- All neighbor relationships represented as a single long table (`neighbor_dt`) instead of millions of small lists.
- Single keyed join + grouped aggregation instead of millions of function calls.
- `data.table` uses C-level operations for grouping â†’ orders of magnitude faster and memory efficient.
- Handles all variables in one scan.

**Expected performance**: Minutes to low hours on 16â€¯GB RAM, compared to 86+ hours.

**Preserves**:  
- Original estimand (same max/min/mean per (id, year) over neighbors).
- Random Forest model remains untouched.

This is the practical, scalable approach on a standard laptop.
END_RESPONSE>>>
