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
run_id: OpenAI_F03_T07_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is in neighbor feature computation and repeated Random Forest predictions on 6.46M rows. Issues:  
- `lapply` over millions of rows is extremely inefficient in R.  
- `compute_neighbor_stats` repeatedly allocates vectors and binds them (`do.call(rbind, ...)`), causing heavy copying.  
- `neighbor_lookup` is a list of length nrow(data) with variable-length elements â†’ poor memory locality.  
- Random Forest predictions on millions of rows can be slow if done in small chunks or loops.  
- Model loading repeatedly or predicting row-by-row is catastrophic for speed.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer matrix or compressed storage.  
2. Replace `lapply` with **vectorized or compiled operations** using `data.table` or `vctrs`.  
3. Compute neighbor stats in bulk using efficient aggregation, not per-row loops.  
4. Use **data.table for panel data** (fast grouping, joins).  
5. Load the Random Forest model once; predict on large chunks (e.g., 100k rows) to avoid memory blowup.  
6. If possible, use **parallel processing** and packages like `ranger` for fast RF inference (threaded).  

---

### **Working Optimized R Code**

```r
library(data.table)
library(ranger)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup as a named integer vector for fast access
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))

# Flatten neighbor lookup into a data.table rather than list-of-lists
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
  cell_id <- id_order[ref_idx]
  nbs <- rook_neighbors_unique[[ref_idx]]
  if (length(nbs) == 0) return(NULL)
  data.table(
    id = cell_id,
    neighbor_id = id_order[nbs]
  )
}))

# Expand to include year for panel structure
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(neighbor_id = neighbor_id, id_year = paste(id, years, sep = "_")), by = years]
neighbor_dt[, neighbor_key := paste(neighbor_id, years, sep = "_")]

# Map neighbor_key to row index
neighbor_dt[, neighbor_row := idx_lookup[neighbor_key]]

# Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, value := vals[neighbor_row]]
  # Aggregate by id_year
  stats <- neighbor_dt[!is.na(value),
    .(
      nb_max = max(value),
      nb_min = min(value),
      nb_mean = mean(value)
    ),
    by = id_year
  ]
  stats
}

# Apply for all variables and merge back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  cell_data <- merge(cell_data, stats, by.x = c("id", "year"),
                     by.y = c("tstrsplit(stats$id_year, '_')[1]", "tstrsplit(stats$id_year, '_')[2]"),
                     all.x = TRUE)
}

# Random Forest inference optimization
# Load model once
rf_model <- readRDS("trained_rf_model.rds")

# Predict in large chunks
chunk_size <- 100000
n <- nrow(cell_data)
preds <- numeric(n)
for (i in seq(1, n, by = chunk_size)) {
  j <- min(i + chunk_size - 1, n)
  preds[i:j] <- predict(rf_model, data = cell_data[i:j])$predictions
}

cell_data[, gdp_pred := preds]
```

---

**Why This Is Faster**  
- **No per-row lapply** â†’ replaced with `data.table` joins and group aggregation.  
- **Chunked RF prediction** â†’ avoids repeated model loading and memory thrashing.  
- `ranger` uses multi-threaded prediction internally (set `num.threads` for speed).  
- Avoids repeated object copying with `rbind` in loops.  

This approach should reduce runtime from 86+ hours to a few hours or less on a 16GB laptop.
END_RESPONSE>>>
