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
run_id: OpenAI_F03_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Major bottlenecks occur **before prediction** due to:  
  1. repeated use of `lapply` and object copying in `build_neighbor_lookup` and `compute_neighbor_stats`.  
  2. constructing neighbor features in R loops across 6.46M rows Ã— 5 variables â†’ extremely inefficient.  
  3. large intermediate lists and repeated string concatenation â†’ excessive memory pressure.  
- Random Forest inference is also slow if done row-by-row; `predict()` must accept a data frame in bulk.  
- Current workflow builds neighbor features on-the-fly; this should be vectorized or delegated to efficient compiled code.  

---

**Optimization Strategy**  
1. **Precompute an integer index matrix** for neighbors (`matrix/list of integer vectors`) without repeated string concatenation.  
2. Replace `lapply` loops with **vectorized computations** or `data.table` joins.  
3. Use a **wide table and bulk matrix operations** for neighbor statistics:  
   - Convert neighbor lookup to an `IntegerList`-like structure.  
   - Compute `max, min, mean` for all observations in compiled code using `data.table` or `Rcpp`.  
4. For Random Forest prediction:  
   - Load model once.  
   - Run `predict()` on the full `data.table` or in batches (e.g., chunks of 500k rows).  
5. Use `data.table` to drastically cut overhead and memory copies.

---

**Fast Implementation (using data.table)**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique (from spdep), rf_model loaded

cell_dt <- as.data.table(cell_data)

# Prebuild neighbor lookup as integer indices per cell index (no year string concatenation)
id_to_pos <- setNames(seq_along(id_order), id_order)
neighbor_idx <- lapply(rook_neighbors_unique, function(nbs) id_to_pos[nbs])

# Add row index by id-year combination
cell_dt[, row_id := .I]
key_vec <- paste(cell_dt$id, cell_dt$year, sep = "_")
idx_lookup <- setNames(cell_dt$row_id, key_vec)

# Expand neighbor lookup into year-specific indices in a fast way
build_year_neighbors <- function(cell_dt, neighbor_idx) {
  n <- nrow(cell_dt)
  res <- vector("list", n)
  ids <- cell_dt$id
  yrs <- cell_dt$year
  for (i in seq_len(n)) {
    nbs <- neighbor_idx[[ids[i]]]
    if (length(nbs)) {
      nb_keys <- paste(id_order[nbs], yrs[i], sep = "_")
      res[[i]] <- idx_lookup[nb_keys]
    } else {
      res[[i]] <- integer(0)
    }
  }
  res
}

neighbor_lookup <- build_year_neighbors(cell_dt, neighbor_idx)

# Compute neighbor stats efficiently with Rcpp or fast lapply
compute_neighbor_stats_dt <- function(vals, lookup) {
  n <- length(vals)
  mat <- matrix(NA_real_, n, 3) # (max, min, mean)
  for (i in seq_len(n)) {
    idx <- lookup[[i]]
    if (length(idx)) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v)) {
        mat[i, 1] <- max(v)
        mat[i, 2] <- min(v)
        mat[i, 3] <- mean(v)
      }
    }
  }
  mat
}

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_vars) {
  stats_mat <- compute_neighbor_stats_dt(cell_dt[[var]], neighbor_lookup)
  set(cell_dt, j = paste0(var, "_nb_max"), value = stats_mat[, 1])
  set(cell_dt, j = paste0(var, "_nb_min"), value = stats_mat[, 2])
  set(cell_dt, j = paste0(var, "_nb_mean"), value = stats_mat[, 3])
}

# Drop helper columns
cell_dt[, row_id := NULL]

# Predict in chunks for memory efficiency
chunk_size <- 500000
nrows <- nrow(cell_dt)
pred <- vector("numeric", nrows)

for (i in seq(1, nrows, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, nrows)
  pred[idx] <- predict(rf_model, newdata = cell_dt[idx])
}

cell_dt[, prediction := pred]
```

---

**Key Gains**  
- Eliminates repeated concatenation and copying â†’ huge speedup.  
- Moves from many small object allocations to preallocated structures.  
- Predict in bulk instead of row-by-row (orders of magnitude faster).  

**Expected runtime**: Hours â†’ **under 1 hour on 16 GB RAM** (depending on chunk size and CPU).  

This preserves the trained Random Forest model, uses all original estimands, and focuses optimization on inference and feature construction without retraining.
END_RESPONSE>>>
