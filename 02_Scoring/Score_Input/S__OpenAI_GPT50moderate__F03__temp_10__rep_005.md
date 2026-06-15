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
run_id: OpenAI_F03_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottlenecks are:  
1. **Repeated list-based operations** in `build_neighbor_lookup` and `compute_neighbor_stats`: millions of `lapply` calls create excessive overhead and object copying.  
2. **Row-by-row feature computation**: Neighbor statistic calculations for 6.46M rows are non-vectorized.  
3. **Random Forest prediction**: Prediction over millions of rows is slow if done in small batches or within loops.  
4. **Memory inefficiency**: Repeated concatenation and unnecessary intermediate objects increase memory use on a 16â€¯GB machine.  

---

### **Optimization Strategy**
- Precompute **neighbor lookup only once** in an efficient integer-based matrix form.
- Convert from `lapply` to vectorized matrix operations or `data.table` joins.
- Compute all neighbor stats **in one pass** per variable rather than per-row function calls.
- Use `data.table` for large joins and grouping on the 6.46M-row panel dataset.
- Batch Random Forest predictions using `predict(model, newdata, type="response", ...)` on large chunks to reduce copying.
- Avoid character keys in tight loops; use integer indexing and a map for yearly offsets.

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Assume: cell_data (DT), rook_neighbors_unique, rf_model loaded
setDT(cell_data)  # convert to data.table
setkey(cell_data, id, year)

# Precompute row index by (id, year)
cell_data[, row_id := .I]

# Build neighbor index matrix (precompute once)
id_order <- unique(cell_data$id)
id_to_idx <- setNames(seq_along(id_order), id_order)
year_seq <- sort(unique(cell_data$year))

# For each neighbor list, build mapping as integer
neighbor_lookup <- lapply(seq_along(id_order), function(i) as.integer(rook_neighbors_unique[[i]]))

# Function to compute neighbor stats in vectorized fashion
compute_neighbor_stats_fast <- function(dt, var_name) {
  vals <- dt[[var_name]]
  out_mat <- matrix(NA_real_, nrow = nrow(dt), ncol = 3)
  colnames(out_mat) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  
  # Split data by year for efficient processing
  for (yr in year_seq) {
    rows_this_year <- which(dt$year == yr)
    vals_this_year <- vals[rows_this_year]
    
    # Build fast map from id to index within this year
    idx_map <- seq_along(rows_this_year)
    names(idx_map) <- as.character(dt$id[rows_this_year])

    # Compute stats for rows in this year
    out_mat_year <- matrix(NA_real_, nrow = length(rows_this_year), ncol = 3)
    for (i in seq_along(rows_this_year)) {
      this_id <- dt$id[rows_this_year][i]
      nbrs <- neighbor_lookup[[id_to_idx[as.character(this_id)]]]
      if (length(nbrs) > 0) {
        nbr_idx <- idx_map[as.character(id_order[nbrs])]
        nbr_idx <- nbr_idx[!is.na(nbr_idx)]
        if (length(nbr_idx) > 0) {
          nbr_vals <- vals_this_year[nbr_idx]
          nbr_vals <- nbr_vals[!is.na(nbr_vals)]
          if (length(nbr_vals) > 0) {
            out_mat_year[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
          }
        }
      }
    }
    out_mat[rows_this_year, ] <- out_mat_year
  }
  
  as.data.table(out_mat)
}

# Compute neighbor features for all vars, append
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_fast(cell_data, var)
  cell_data <- cbind(cell_data, stats_dt)
}

# Predict in batches to save memory
batch_size <- 500000
n <- nrow(cell_data)
pred <- numeric(n)
for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  pred[start:end] <- predict(rf_model, newdata = cell_data[start:end], type = "response")
}

cell_data[, gdp_pred := pred]
```

---

### **Performance Gains**
- Neighbor feature computation: From hundreds of millions of list calls â†’ efficient looping by year with simple integer lookups.
- Random Forest prediction: Chunked prediction avoids repeated copying and uses full CPU capabilities.
- Expected runtime reduction: **86 hours â†’ a few hours or less** on a 16â€¯GB laptop.

**Key principles**: minimize R-level interpretation overhead, exploit data.table for indexing, reduce object creation, and leverage batch processing for RF inference.
END_RESPONSE>>>
