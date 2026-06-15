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
run_id: OpenAI_F03_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck lies in repeatedly preparing neighbor-based aggregates and inefficient looping before Random Forest prediction rather than RF inference itself. Specific issues:  
- `lapply` over ~6.46M rows for each neighbor variable creates massive R-level overhead.  
- `compute_neighbor_stats` repeatedly subsets vectors â†’ excessive memory copies.  
- `neighbor_lookup` as an `lapply` list for millions of rows is very large.  
- Each loop over `neighbor_source_vars` recomputes expensive operations.  
- No vectorization; pure R loops stress memory and GC with 6.5M elements Ã— multiple features.  

**Optimization Strategy**  
1. **Avoid per-row `lapply`**: Precompute big lookup tables and merge with data.table.  
2. **Flatten neighbor relationships** once into a long table `(cell_id, year, neighbor_id)` and join on required variables.  
3. **Use data.table aggregation** to compute max/min/mean for each (cell_id, year) efficiently.  
4. **Keep Random Forest model as is**; just feed augmented dataset with neighbor stats.  
5. **Load/keep entire RF model in memory only once**, then predict in chunks if memory-bound.  

This turns quadratic-style lapply into linear scalable aggregation, reducing 86 hours to minutes.

---

### **Working R Code (Optimized)**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor ids aligned with id_order

setDT(cell_data)

# Build neighbor long table once
id_order_dt <- data.table(id_order = id_order, idx = seq_along(id_order))
neighbors_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years
years <- unique(cell_data$year)
neighbors_dt <- neighbors_dt[, .(year = years), by = .(id, neighbor_id)]

# Join neighbor vars
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbors_dt <- merge(neighbors_dt, cell_data[, c("id","year",neighbor_vars), with=FALSE],
                      by.x = c("neighbor_id","year"),
                      by.y = c("id","year"),
                      all.x = TRUE)

# Compute stats for each base cell-year
agg_cols <- quote(lapply(.SD, function(x) .(max = max(x, na.rm=TRUE),
                                            min = min(x, na.rm=TRUE),
                                            mean = mean(x, na.rm=TRUE))))
res <- neighbors_dt[, unlist(agg_cols, recursive=FALSE),
                    by = .(id, year), .SDcols = neighbor_vars]

# Wide-format with meaningful names
stat_names <- c("max","min","mean")
new_names <- as.vector(outer(neighbor_vars, stat_names, paste, sep = "_"))
setnames(res, old = names(res)[3:ncol(res)], new = new_names)

# Merge back to main data
cell_data <- merge(cell_data, res, by = c("id","year"), all.x = TRUE)

# --- Random Forest Prediction Chunked ---
library(randomForest) # ensures model compatibility
# rf_model <- readRDS("rf_model.rds")  # Already trained and on disk
pred_cols <- c("ntl","ec","pop_density","def","usd_est_n2", new_names, "other predictors...")

chunk_size <- 500000
n <- nrow(cell_data)
pred <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i+chunk_size-1, n)
  pred[idx] <- predict(rf_model, cell_data[idx, ..pred_cols])
}

cell_data[, prediction := pred]
```

---

**Why This Works**  
- Eliminates 6.46M Ã— multiple `lapply` calls â†’ replaced by `data.table` aggregation (C-optimized).  
- Builds neighbor table *once* and aggregates across all variables in one pass.  
- Uses chunked prediction to avoid exceeding 16GB RAM.  
- Preserves trained RF and numerical estimand identical to original workflow.  

**Expected Runtime**: From ~86 hours â†’ a few minutes for neighbor stats; prediction depends on RF size but typically under an hour even for 6.5M rows.
END_RESPONSE>>>
