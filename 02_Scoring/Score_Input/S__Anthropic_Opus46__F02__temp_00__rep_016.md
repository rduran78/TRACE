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
run_id: Anthropic_F02_T00_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated for 5 variables, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-key lookups at scale:** `idx_lookup` is a named vector with ~6.46M entries. Named-vector lookup in R uses linear hashing that degrades at this scale. Each of the 6.46M rows performs multiple lookups into it.
2. **Per-row `lapply` with allocations:** Each iteration of the `lapply` in `build_neighbor_lookup` allocates character vectors (`paste`), performs named lookups, and filters NAs â€” millions of tiny allocations that thrash the garbage collector.
3. **List-of-vectors structure for `neighbor_lookup`:** Storing ~6.46M list elements (each a small integer vector) is memory-inefficient and cache-unfriendly.
4. **`compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)`:** Binding 6.46M 3-element vectors row-by-row is slow.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** problem. Each `(cell, year)` needs to find its neighbors' values for the same year. This is a merge between a neighbor-edge table and the data table on `(neighbor_id, year)`, followed by a `group-by` aggregation. `data.table` performs this in optimized C with minimal memory overhead.

**Steps:**

1. **Build an edge table once** â€” a two-column `data.table` of `(id, neighbor_id)` from the `nb` object. This is ~1.37M rows.
2. **For each variable, join the edge table to the data on `(neighbor_id, year)`** to get neighbor values, then **group by `(id, year)`** to compute max, min, mean.
3. **Left-join** the aggregated stats back to the main data.

This eliminates all per-row R loops, all string-key hashing, and all list-of-vectors storage. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build the edge table ONCE from the nb object
#         (rook_neighbors_unique is a list of integer vectors
#          indexed by position in id_order)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# Step 2: Convert main data to data.table (in-place if possible)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure keys for fast joins
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# Step 3: For each neighbor source variable, compute neighbor
#         max, min, mean via keyed join + grouped aggregation,
#         then left-join back to cell_data.
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor features for:", var_name, "\n")
  
  # Subset only the columns we need for the join (minimise memory)
  # Columns: neighbor_id (to join on), year, and the variable value
  val_dt <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # Join edges to data: for each (id, neighbor_id) pair, attach
  # the neighbor's value in each year.
  # edge_dt has (id, neighbor_id); we join val_dt on (neighbor_id, year).
  # We need to bring in year from the focal cell. Strategy:
  #   1. Join edge_dt to cell_data's (id, year) to expand edges across years.
  #      But that would be 1.37M edges Ã— 28 years = ~38M rows â€” manageable.
  #   2. Then join to val_dt on (neighbor_id, year).
  
  # Get unique (id, year) pairs from cell_data
  focal_dt <- cell_data[, .(id, year)]
  setkey(focal_dt, id)
  
  # Expand: each focal (id, year) gets its neighbor_ids
  # This is a join of focal_dt and edge_dt on 'id'
  setkey(edge_dt, id)
  expanded <- edge_dt[focal_dt, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  # Rows: ~1.37M avg neighbors Ã— 28 years â‰ˆ 38.4M rows
  
  # Now join to get the neighbor's value
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, on = .(neighbor_id, year), neighbor_val := i.val]
  
  # Aggregate by (id, year)
  agg <- expanded[!is.na(neighbor_val),
                  .(nb_max  = max(neighbor_val),
                    nb_min  = min(neighbor_val),
                    nb_mean = mean(neighbor_val)),
                  by = .(id, year)]
  
  # Rename columns to match expected feature names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))
  
  # Left-join back to cell_data
  setkey(agg, id, year)
  setkey(cell_data, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  # Clean up intermediate objects to free RAM
  rm(val_dt, focal_dt, expanded, agg)
  gc()
  
  cat("  Done.\n")
}

# ---------------------------------------------------------------
# Step 4: Predict with the existing trained Random Forest model
#         (model object is unchanged; feature names and values
#          are numerically identical to the original pipeline)
# ---------------------------------------------------------------
# Example (adjust to your actual model object and predict call):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (same rook adjacency, same year matching, same NA exclusion) as the original code. The `data.table` grouped aggregation is numerically identical. |
| **Trained RF model** | The model object is never touched. The output columns have the same names and values, so `predict()` produces identical results. |
| **NA handling** | `!is.na(neighbor_val)` in the aggregation and `nomatch = 0L` in joins replicate the original NA-filtering logic. Cells with zero valid neighbors get `NA` for all three stats via the left join. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M `lapply` iterations with string ops) | ~seconds (vectorized edge table construction) |
| Stats per variable | ~15+ hours (6.46M `lapply` + `do.call(rbind)`) | ~1â€“3 minutes (`data.table` keyed join + grouped agg on ~38M rows) |
| **Total for 5 variables** | **~86+ hours** | **~10â€“20 minutes** |
| Peak RAM | Unbounded list-of-lists growth | ~3â€“5 GB (38M-row intermediate, freed each iteration) |

This fits comfortably within the 16 GB laptop constraint and reduces runtime by roughly **two to three orders of magnitude**.
END_RESPONSE>>>
