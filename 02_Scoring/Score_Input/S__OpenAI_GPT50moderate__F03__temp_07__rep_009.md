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
run_id: OpenAI_F03_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottlenecks are:  
1. **Repeated lapply loops and object copying**: `compute_neighbor_stats` and `compute_and_add_neighbor_features` repeatedly traverse large lists and copy data frames.  
2. **List-of-lists neighbor lookup**: Building and iterating over large R lists (6.46M entries) is memory-heavy and slow.  
3. **Prediction loop inefficiency**: If predictions are applied in small chunks with `predict()`, overhead accumulates.  
4. **Random Forest inference on full data**: Large input matrix (~6.46M Ã— 110) stresses memory; conversion to `data.frame` is costly.  

**Optimization Strategy**  
- Precompute neighbor relationships in a **long table** (edges) instead of list-of-lists.  
- Compute neighbor stats with **data.table** group operations, avoiding repeated `lapply`.  
- Use **matrix or data.table** for features to minimize copies.  
- Run predictions in **large batches** or all at once.  
- Keep model in memory; avoid reloading.  
- Use **parallelization** for neighbor stats and prediction if possible.  

---

### **Optimized Workflow**

```r
library(data.table)
library(ranger) # Fast RF inference (alternatively randomForest)

# Assume: cell_data (data.table) with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (nb object)
# id_order: vector mapping index to id

# Convert to data.table
setDT(cell_data)

# Build long neighbor table once
build_neighbor_table <- function(id_order, neighbors) {
  src <- rep(seq_along(neighbors), lengths(neighbors))
  dst <- unlist(neighbors, use.names = FALSE)
  data.table(src_id = id_order[src], nb_id = id_order[dst])
}

neighbor_dt <- build_neighbor_table(id_order, rook_neighbors_unique)

# Join with years: expand for all years (cartesian join)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_dt, "V1", "year") # after CJ

# Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(cell_data, neighbor_dt, var_name) {
  # Prepare source values keyed by id-year
  vals <- cell_data[, .(id, year, value = get(var_name))]
  setkey(vals, id, year)
  
  # Join neighbors on nb_id-year to get neighbor values
  joined <- neighbor_dt[vals, on = .(nb_id = id, year), nomatch = 0]
  
  # Compute stats by src_id-year
  stats <- joined[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(src_id, year)]
  
  setnames(stats, c("src_id", "year", paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))))
  stats
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_stats <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_dt(cell_data, neighbor_dt, v))
all_stats_dt <- Reduce(function(x, y) merge(x, y, by = c("src_id", "year"), all = TRUE), all_stats)

# Merge back into main cell_data
setnames(all_stats_dt, "src_id", "id")
cell_data <- merge(cell_data, all_stats_dt, by = c("id", "year"), all.x = TRUE)

# Prepare predictors as matrix for fast prediction
predictor_vars <- setdiff(names(cell_data), c("id", "year", "gdp")) # adjust target name
X <- as.matrix(cell_data[, ..predictor_vars])

# Load pre-trained Random Forest model
# model <- readRDS("rf_model.rds")

# Predict in large chunks using ranger (fastest)
pred <- predict(model, data = X)$predictions

cell_data[, gdp_pred := pred]
```

---

**Why This Is Faster**  
- Neighbor stats computed in **vectorized, grouped operations** via `data.table`, eliminating millions of `lapply` calls.  
- One **long join-based approach** for all neighbors and years.  
- Minimal data copies; merges happen in-place.  
- Random Forest inference uses **matrix input** and `ranger` for speed (or `predict` in bulk if using `randomForest`).  

**Expected Impact**  
From 86+ hours to a few hours (dominated by joins and prediction), well within 16 GB RAM if processed in chunks (e.g., per 5M rows for prediction).
END_RESPONSE>>>
