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
run_id: OpenAI_F03_T00_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects. This is extremely slow and memory-inefficient.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: `lapply` with string concatenation (`paste`) for every row is costly.  
4. **Random Forest prediction**: If predictions are done in small chunks or repeatedly loading the model, this adds overhead.  
5. **Memory pressure**: 6.46M rows Ã— 110+ variables is large; repeated intermediate objects exacerbate RAM usage.  

---

**Optimization Strategy**  
- **Precompute neighbor indices as integer vectors** (avoid string keys).  
- **Vectorize neighbor stats computation** using `data.table` or `matrix` operations instead of millions of `lapply` calls.  
- **Avoid repeated copies**: compute all neighbor features in one pass.  
- **Batch Random Forest predictions**: load model once, predict in large chunks.  
- **Use `data.table` for fast joins and memory efficiency**.  

---

**Optimized R Code**  

```r
library(data.table)
library(randomForest)

# Convert to data.table for speed
setDT(cell_data)

# Precompute a fast lookup: map (id, year) -> row index
cell_data[, key := .I]  # row index
id_year_key <- cell_data[, .(id, year, key)]
setkey(id_year_key, id)

# Build neighbor lookup as integer indices (no string concatenation)
build_neighbor_lookup_fast <- function(id_order, neighbors, id_year_key) {
  # neighbors: list of integer vectors (rook neighbors)
  # id_order: vector of ids in same order as neighbors
  id_to_idx <- setNames(seq_along(id_order), id_order)
  
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    # Return neighbor ids (not row indices yet)
    neighbor_ids
  })
}

neighbor_lookup_ids <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, id_year_key)

# Compute all neighbor stats in one pass
compute_all_neighbor_stats <- function(cell_data, neighbor_lookup_ids, vars) {
  n <- nrow(cell_data)
  result_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    var_name <- vars[v]
    vals <- cell_data[[var_name]]
    
    # Preallocate matrix: rows = n, cols = 3 (max, min, mean)
    mat <- matrix(NA_real_, n, 3)
    
    for (i in seq_along(neighbor_lookup_ids)) {
      # Get all rows for this id across years
      id_rows <- which(cell_data$id == id_order[i])
      for (r in id_rows) {
        year <- cell_data$year[r]
        # Find neighbors for same year
        neighbor_ids <- neighbor_lookup_ids[[i]]
        neighbor_rows <- id_year_key[J(neighbor_ids)][year == year]$key
        if (length(neighbor_rows) > 0) {
          neighbor_vals <- vals[neighbor_rows]
          neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
          if (length(neighbor_vals) > 0) {
            mat[r, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
          }
        }
      }
    }
    result_list[[v]] <- mat
  }
  result_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup_ids, neighbor_source_vars)

# Bind results efficiently
for (i in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[i]
  cell_data[, paste0(var, "_nbr_max") := neighbor_stats[[i]][, 1]]
  cell_data[, paste0(var, "_nbr_min") := neighbor_stats[[i]][, 2]]
  cell_data[, paste0(var, "_nbr_mean") := neighbor_stats[[i]][, 3]]
}

# Random Forest prediction in batches
load("trained_rf_model.RData")  # Assume model is in rf_model
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminated millions of `lapply` calls â†’ replaced with indexed loops and preallocation.  
- Avoided repeated `paste` and string lookups.  
- Used `data.table` for fast joins and memory efficiency.  
- Batched predictions to avoid memory spikes.  

**Expected Impact**: From 86+ hours to a few hours (dominated by neighbor stats computation). Further speedup possible by parallelizing inner loops or using `Rcpp`.
END_RESPONSE>>>
