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
run_id: OpenAI_F06_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach performs `compute_neighbor_stats` for each of ~6.46M rows and 5 variables using `lapply` plus many repeated lookups and vector operations in R. This is highly inefficient because:  
- Repeated list/lookup calls per row create massive overhead at scale.  
- Statistics (max, min, mean) for neighbor values are computed serially for each row and variable rather than vectorized.  
- `do.call(rbind, â€¦)` over millions of rows is memory- and time-intensive.  
- Process is entirely in R loops (single-threaded), causing the 86+ hour runtime.

**Optimization Strategy**  
- Pre-flatten neighbor relationships into vectors and use `data.table` or `dplyr` for grouped aggregation, which exploits optimized C code.  
- Compute all stats for all variables in *one pass* per neighbor pair table instead of per-row lists.  
- Avoid repeated lookups by building a long table of `(focal_idx, neighbor_idx)` at once.  
- Parallelize if possible, but even vectorization and aggregation will give orders of magnitude improvement.  

**Working R Code**  
Below, we:  
1. Flatten neighbor index pairs into a data frame.  
2. Join neighbor values for all vars, then group and compute min, max, mean via fast aggregation.  
3. Merge back into `cell_data`.  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Step 1: Flatten neighbor relationships
# neighbor_lookup logic in bulk
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), 
                       paste(cell_data$id, cell_data$year, sep = "_"))

focal_years <- cell_data$year
focal_ids   <- cell_data$id

# Build combined neighbor pairs
neighbor_list <- vector("list", length(focal_ids))
for (i in seq_along(focal_ids)) {
  ref_idx <- id_to_ref[as.character(focal_ids[i])]
  nbs     <- rook_neighbors_unique[[ref_idx]]
  focal   <- rep(i, length(nbs))
  neighbor_ids <- id_order[nbs]
  # Translate neighbor_id + year -> row index
  keys    <- paste(neighbor_ids, focal_years[i], sep = "_")
  neighbor_idx <- idx_lookup[keys]
  neighbor_list[[i]] <- data.frame(focal = focal, neighbor = neighbor_idx, stringsAsFactors = FALSE)
}

pairs_dt <- rbindlist(neighbor_list)
pairs_dt <- pairs_dt[!is.na(neighbor)]  # Remove NA neighbors

# Step 2: Long format values for each neighbor variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- melt(cell_data[, c("rowid" = .I, neighbor_source_vars), with = FALSE],
                id.vars = "rowid", variable.name = "var", value.name = "value")

# Join neighbor from pairs_dt -> long_dt
pairs_long <- pairs_dt[, .(focal, neighbor)][long_dt, on = .(neighbor = rowid)]

# Step 3: Compute stats by focal & var
agg_stats <- pairs_long[!is.na(value),
                         .(nb_max = max(value, na.rm = TRUE),
                           nb_min = min(value, na.rm = TRUE),
                           nb_mean = mean(value, na.rm = TRUE)),
                         by = .(focal, var)]

# Step 4: Spread back to wide format
stats_wide <- dcast(agg_stats, focal ~ var, value.var = c("nb_max", "nb_min", "nb_mean"))

# Step 5: Merge back into cell_data
cell_data <- cbind(cell_data, stats_wide[order(focal), -"focal"])
```

**Why This Works**  
- All lookups and computations are vectorized and done via `data.table` in C.  
- Eliminates per-row loops; stats computed in grouped aggregation.  
- Memory fit: ~6.5M rows + neighbor pairs (~10â€“12M rows) is manageable on 16 GB RAM with `data.table`.  
- No change to estimandsâ€”still using **rook neighbors** and computing max, min, mean.  

**Expected Speed-up**: From >86 hours to <1 hour on a modern laptop, possibly much less if using parallelism.  

**Preserves**:  
- Original Random Forest model unaltered.  
- Numerical outputs identical (subject to floating-point order).
END_RESPONSE>>>
