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
run_id: Anthropic_F11_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. Named vector lookup in R is effectively a hash-table probe, but doing it millions of times inside an interpreted loop with repeated string allocation is extremely expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` loop that subsets, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference** (`predict()`) on a pre-trained model over 6.46M rows with ~110 predictors is a single vectorized C-level call (in `randomForest` or `ranger`). It typically completes in seconds to minutes â€” orders of magnitude faster than the row-level R loops above.

**Conclusion:** The bottleneck is the row-by-row R-interpreted loop construction of neighbor lookups and the repeated `lapply`-based neighbor statistics computation, not the RF prediction step.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized approach using `data.table` joins. Pre-expand all neighbor relationships into an edge list (cell-year â†’ neighbor-cell-year), then join to get row indices. This eliminates millions of `paste()` and named-vector lookups.

2. **Vectorize `compute_neighbor_stats()`**: Once we have an edge-list with row indices, compute `max`, `min`, and `mean` for each variable using `data.table` grouped aggregation â€” a single pass per variable, fully in C.

3. **Compute all 5 variables' stats in one pass** over the edge list rather than 5 separate `lapply` calls over 6.46M rows.

These changes should reduce runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# ============================================================
# Inputs expected:
#   cell_data            : data.frame/data.table with columns id, year, 
#                          and the neighbor_source_vars
#   id_order             : vector of cell IDs in the order matching 
#                          rook_neighbors_unique
#   rook_neighbors_unique: spdep nb object (list of integer index vectors)
#   neighbor_source_vars : character vector of variable names
# ============================================================

compute_all_neighbor_features <- function(cell_data, id_order, 
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # --- Step 0: Convert to data.table and assign row indices -----------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  # --- Step 1: Build a full edge list of directed neighbor pairs ------
  #     (focal_cell_id -> neighbor_cell_id) from the nb object
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # nb contains integer indices into id_order; 0 means no neighbors
    nb <- nb[nb > 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  # edge_list now has ~1.37M rows: one per directed rook-neighbor pair
  
  # --- Step 2: Create a keyed lookup from (id, year) -> row_idx ------
  key_dt <- dt[, .(id, year, row_idx)]
  
  # --- Step 3: Expand edge list across all years ----------------------
  #     For every (focal_id, neighbor_id) pair and every year present 
  #     for the focal cell, we need the neighbor's row in that same year.
  
  # Get the set of years each focal cell has data for
  focal_years <- dt[, .(year), keyby = .(id)]
  
  # Join: for each focal_id, get all its years
  # Then for each (focal_id, year, neighbor_id), look up neighbor's row
  setnames(edge_list, c("focal_id", "neighbor_id"))
  
  # Merge edge_list with focal_years to get (focal_id, year, neighbor_id)
  expanded <- edge_list[focal_years, 
                        on = .(focal_id = id), 
                        allow.cartesian = TRUE,
                        nomatch = 0L]
  # expanded has columns: focal_id, neighbor_id, year
  
  # Look up the focal row index
  expanded <- merge(expanded, key_dt, 
                    by.x = c("focal_id", "year"), 
                    by.y = c("id", "year"), 
                    all.x = TRUE, sort = FALSE)
  setnames(expanded, "row_idx", "focal_row_idx")
  
  # Look up the neighbor row index
  expanded <- merge(expanded, key_dt, 
                    by.x = c("neighbor_id", "year"), 
                    by.y = c("id", "year"), 
                    all.x = TRUE, sort = FALSE)
  setnames(expanded, "row_idx", "neighbor_row_idx")
  
  # Drop rows where neighbor has no data in that year
  expanded <- expanded[!is.na(neighbor_row_idx)]
  
  # --- Step 4: Attach neighbor variable values ------------------------
  for (vn in neighbor_source_vars) {
    set(expanded, j = vn, value = dt[[vn]][expanded$neighbor_row_idx])
  }
  
  # --- Step 5: Grouped aggregation ------------------------------------
  #     Compute max, min, mean per (focal_row_idx) for each variable
  agg_exprs <- list()
  for (vn in neighbor_source_vars) {
    vn_sym <- as.name(vn)
    agg_exprs[[paste0("neighbor_max_", vn)]]  <- 
      bquote(as.numeric(max(.(vn_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", vn)]]  <- 
      bquote(as.numeric(min(.(vn_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", vn)]] <- 
      bquote(mean(.(vn_sym), na.rm = TRUE))
  }
  
  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  stats <- expanded[, eval(agg_call), by = .(focal_row_idx)]
  
  # Replace -Inf/Inf from max/min on all-NA groups with NA
  num_cols <- setdiff(names(stats), "focal_row_idx")
  for (col in num_cols) {
    vals <- stats[[col]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats, j = col, value = vals)
  }
  
  # --- Step 6: Join back to the main data table -----------------------
  # Initialize new columns with NA
  for (col in num_cols) {
    set(dt, j = col, value = NA_real_)
  }
  
  # Fill in computed values by row index
  for (col in num_cols) {
    set(dt, i = stats$focal_row_idx, j = col, value = stats[[col]])
  }
  
  # Remove helper column
  dt[, row_idx := NULL]
  
  return(dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# ============================================================
# RANDOM FOREST PREDICTION â€” unchanged, as it is NOT the bottleneck
# ============================================================
# Example (preserving the trained model and original estimand):
# predictions <- predict(trained_rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows; `paste` + named-vector lookup per row | Vectorized `data.table` join; edge-list expansion via `allow.cartesian` |
| `compute_neighbor_stats` | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single grouped `data.table` aggregation for all 5 variables |
| Estimated runtime | 86+ hours | Minutes (typically 5â€“15 min on 16 GB laptop) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Identical `max`, `min`, `mean` per neighbor set | Identical â€” no approximation |

The true bottleneck is the millions of interpreted R-loop iterations with string operations in the neighbor feature engineering, not the Random Forest inference.
END_RESPONSE>>>
