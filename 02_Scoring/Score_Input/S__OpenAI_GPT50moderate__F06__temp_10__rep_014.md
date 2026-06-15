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
run_id: OpenAI_F06_T10_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is excessively slow because it repeatedly computes neighbor statistics in pure R loops (`lapply` inside `compute_neighbor_stats`) for ~6.46M rows and 5 variables. Each lookup involves character-based key matching and dynamic vector allocation. This results in massive overhead (memory thrashing and millions of function calls).

**Optimization Strategy**  
1. **Avoid per-row `lapply` lookups and character keys** by materializing neighbor indices as integer vectors in advance, and process in matrix form.  
2. **Use vectorized / compiled aggregation** by leveraging `data.table` joins or `rowsum` aggregation, or **prebuild long format neighbor pairs once and aggregate with fast group operations**.  
3. **Precompute neighbor stats for all variables in a single pass** rather than looping one variable at a time.  
4. Maintain the estimand exactly: compute *max*, *min*, and *mean* among **same-year** rook neighbors per cell-year.  

The core trick: expand the panel into an edge list (cell-year to neighbor-year), join values, and compute group summaries using `data.table` grouping by `cell_id` and `year`.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data (data.frame) columns: id, year, ntl, ec, pop_density, def, usd_est_n2
#          id_order: vector of unique cell IDs
#          rook_neighbors_unique: spdep::nb object
# Convert to data.table for speed
DT <- as.data.table(cell_data)
setkey(DT, id, year)

# 1. Build neighbor edge table (id -> neighbor_id)
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))
# Repeat neighbors for all 28 years
years <- sort(unique(DT$year))
neighbor_dt <- neighbor_dt[, .(id = rep(id, each = length(years)),
                                neighbor_id = rep(neighbor_id, each = length(years)),
                                year = rep(years, times = .N)), by = .I][, .I := NULL]

# Merge neighbor values from DT
merge_dt <- merge(neighbor_dt, DT, by.x = c("neighbor_id", "year"), by.y = c("id", "year"),
                  all.x = TRUE, suffixes = c("", "_nbr"))

# 2. Compute stats for each source variable per (id,year)
vars <- c("ntl","ec","pop_density","def","usd_est_n2")

agg_list <- lapply(vars, function(v) {
  merge_dt[, .(
    max = max(get(v), na.rm = TRUE),
    min = min(get(v), na.rm = TRUE),
    mean = mean(get(v), na.rm = TRUE)
  ), by = .(id, year)]
})

# 3. Combine back into DT
for (i in seq_along(vars)) {
  nm <- vars[i]
  stats <- agg_list[[i]]
  setnames(stats, c("id","year","max","min","mean"),
           c("id","year", paste0(nm,"_nbr_max"), paste0(nm,"_nbr_min"), paste0(nm,"_nbr_mean")))
  DT <- merge(DT, stats, by = c("id","year"), all.x = TRUE)
}

# DT now has neighbor stats appended; proceed to Random Forest prediction
```

---

**Why This is Faster**  
- Expands neighbors once into a long form (`neighbor_dt`), then applies **data.table grouping** (highly optimized in C).  
- Replaces per-row `lapply` loops with bulk operations.  
- Avoids expensive character concatenations.  
- Leverages `J`-optimized keyed merges.  

**Expected Performance**  
- Memory: ~150â€“250 MB for neighbor pair structure + merge overhead (fits in 16 GB).  
- Runtime: minutes to a few hours instead of 86+ hours on a laptop.  

**Preserved Requirements**  
- Identical max/min/mean logic per same-year neighbors.  
- No model retraining; original Random Forest can directly consume the augmented table.  
- Numeric estimand unchanged.
END_RESPONSE>>>
