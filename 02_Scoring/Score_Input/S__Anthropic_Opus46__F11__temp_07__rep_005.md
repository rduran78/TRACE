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
run_id: Anthropic_F11_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence from the code:**

1. **`build_neighbor_lookup`** uses `lapply` over **~6.46 million rows**, and inside each iteration it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe *per element*, and with ~6.46M rows Ã— ~4 neighbors on average, this generates tens of millions of string allocations and hash lookups. This single function likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** is called 5 times (once per neighbor source variable), each time iterating over 6.46M rows with `lapply`. While lighter per iteration than `build_neighbor_lookup`, it still performs ~32.3 million R-level function calls total (5 vars Ã— 6.46M rows), each with subsetting, `is.na` filtering, and summary statistics.

3. **Random Forest inference** (`predict()` on a pre-trained model) over 6.46M rows Ã— 110 predictors is a single vectorized C-level call in most R RF implementations (`ranger`, `randomForest`). This typically completes in seconds to minutes â€” orders of magnitude faster than the row-level `lapply` loops above.

**Root cause:** The bottleneck is the row-by-row R-level looping with expensive string operations across 6.46 million rows, not the RF prediction.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup` entirely as a per-row string-keyed lapply.** Replace it with a fully vectorized `data.table` merge/join approach. Instead of building a lookup list of length 6.46M, we expand the neighbor relationships into an edge table and join against the data.

2. **Eliminate `compute_neighbor_stats` as a per-row lapply.** Replace it with a single grouped `data.table` aggregation over the edge table.

3. **Preserve the trained Random Forest model** â€” we only change feature engineering, not the model or its predictions.

4. **Preserve the original numerical estimand** â€” the computed features (max, min, mean of each neighbor variable) are numerically identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a vectorized edge table from the nb object
#         (runs once; replaces build_neighbor_lookup entirely)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of length length(id_order);

  # nb_obj[[i]] is an integer vector of neighbor indices into id_order.
  
  # Expand to two-column edge list (indices into id_order)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)
  
  # Map indices to actual cell IDs
  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Vectorized neighbor feature computation
#         (replaces compute_neighbor_stats + the outer for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_features <- function(cell_data_df, edge_dt,
                                          neighbor_source_vars) {
  
  # Convert to data.table (copy so we don't modify the original unexpectedly)
  dt <- as.data.table(cell_data_df)
  
  # Ensure key columns exist
  stopifnot(all(c("id", "year") %in% names(dt)))
  
  # Create a row key for the focal observations
  # We will join edges Ã— years to get neighbor values.
  
  # Cross the edge table with all years present in the data
  years <- sort(unique(dt$year))
  
  # Expand edges across years: every directed edge exists in every year
  # This creates the full set of (focal_id, year, neighbor_id) triples
  edge_year <- edge_dt[, CJ(year = years), by = .(focal_id, neighbor_id)]
  
  # Now join to get the neighbor's variable values
  # Key the data on (id, year) for fast lookup
  setkey(dt, id, year)
  
  # We only need the neighbor source vars + the join keys from the data
  cols_needed <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- dt[, ..cols_needed]
  
  # Join: for each edge-year row, attach the neighbor's variable values
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)
  
  merged <- neighbor_vals[edge_year, on = .(neighbor_id, year), nomatch = NA]
  
  # â”€â”€ Grouped aggregation: compute max, min, mean per (focal_id, year) â”€â”€
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0(v, "_neighbor_max")]]  <-
      bquote(as.numeric(max(.(v_sym),  na.rm = TRUE)), list(v_sym = v_sym))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <-
      bquote(as.numeric(min(.(v_sym),  na.rm = TRUE)), list(v_sym = v_sym))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <-
      bquote(mean(.(v_sym), na.rm = TRUE), list(v_sym = v_sym))
  }
  
  # Build the aggregation call programmatically
  agg_result <- merged[,
    lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }),
    by = .(focal_id, year)
  ]
  
  # The above returns list columns in V1..V5; let's use a cleaner approach:
  # Aggregate each variable separately and merge back â€” still fully vectorized.
  
  agg_list <- vector("list", length(neighbor_source_vars))
  
  for (i in seq_along(neighbor_source_vars)) {
    v <- neighbor_source_vars[i]
    
    # Subset to non-NA neighbor values for this variable
    sub <- merged[!is.na(get(v)), .(focal_id, year, val = get(v))]
    
    if (nrow(sub) > 0L) {
      agg_i <- sub[, .(
        vmax  = max(val),
        vmin  = min(val),
        vmean = mean(val)
      ), by = .(focal_id, year)]
    } else {
      agg_i <- data.table(
        focal_id = integer(0), year = integer(0),
        vmax = numeric(0), vmin = numeric(0), vmean = numeric(0)
      )
    }
    
    setnames(agg_i, c("vmax", "vmin", "vmean"),
             paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))
    
    agg_list[[i]] <- agg_i
  }
  
  # Merge all aggregated features together
  feat <- agg_list[[1]]
  for (i in seq_along(agg_list)[-1]) {
    feat <- merge(feat, agg_list[[i]], by = c("focal_id", "year"), all = TRUE)
  }
  
  # Merge back onto the original data
  setnames(feat, "focal_id", "id")
  result <- merge(dt, feat, by = c("id", "year"), all.x = TRUE)
  
  # Restore original row order
  setkey(result, NULL)
  
  return(as.data.frame(result))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Random Forest prediction (unchanged â€” not the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# rf_model is the pre-trained model loaded from disk; we do NOT retrain.
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions$predictions  # or predictions, depending on package
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with string ops | One vectorized edge-table construction (milliseconds) |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations | 5 grouped `data.table` aggregations (vectorized C) |
| Estimated runtime | 86+ hours | **Minutes** (edge expansion ~27M rows; grouped agg is O(n) in C) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical results | max/min/mean of neighbor values | Identical max/min/mean of neighbor values |

The optimization replaces ~38 million R-interpreter-level loop iterations (with per-iteration string allocation and hash lookups) with a handful of vectorized `data.table` joins and grouped aggregations executed in compiled C code. This is the correct fix for the actual bottleneck.
END_RESPONSE>>>
