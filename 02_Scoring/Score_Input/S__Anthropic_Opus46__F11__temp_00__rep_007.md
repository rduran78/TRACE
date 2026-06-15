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
run_id: Anthropic_F11_T00_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model is a single vectorized C/C++ call over the dataset. Even with 110 predictors and 6.46M rows, this is orders of magnitude faster than tens of millions of interpreted R loop iterations with per-element string operations.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not RF inference.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices via millions of `paste`/lookup operations, we expand the neighbor relationships into an edge table and merge with the data using keyed `data.table` joins â€” all in compiled C code under the hood.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable. Instead of 6.46M `lapply` iterations per variable, we compute `max`, `min`, and `mean` via `data.table`'s `[, .(â€¦), by=]` which runs in optimized C.

3. **Preserve the trained Random Forest model** â€” no changes to the model or prediction step.

4. **Preserve the original numerical estimand** â€” the same `max`, `min`, `mean` of neighbor values are computed; only the implementation mechanism changes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build an edge table from the nb object (vectorized)
# ---------------------------------------------------------------
# id_order: vector of cell IDs (length = 344,208)
# rook_neighbors_unique: an nb object (list of length 344,208)
#   where each element contains integer indices into id_order

build_neighbor_edges <- function(id_order, neighbors) {
  # Compute lengths of each neighbor list element
  n_neighbors <- lengths(neighbors)
  
  # Source index (into id_order) repeated for each neighbor
  src_idx <- rep(seq_along(neighbors), times = n_neighbors)
  
  # Destination indices (into id_order), unlisted
  dst_idx <- unlist(neighbors, use.names = FALSE)
  
  # Map to actual cell IDs
  data.table(
    focal_id    = id_order[src_idx],
    neighbor_id = id_order[dst_idx]
  )
}

# Build edge table once (~1.37M rows)
edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# Step 2: Convert cell_data to data.table and key it
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Create a unique row identifier to map results back
cell_dt[, .row_id := .I]

# ---------------------------------------------------------------
# Step 3: Vectorized neighbor feature computation
# ---------------------------------------------------------------
# For each focal cell-year, we need to find all neighbor cell-years
# (same year, neighbor cell) and compute stats on each variable.

# Build the join: focal (id, year) -> neighbor (neighbor_id, year)
# We join edge_dt with cell_dt twice:
#   - First to get (focal_id, year, neighbor_id) for every focal row
#   - Then to get the neighbor's variable values

compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  
  # Focal table: just id, year, and row_id
  focal <- cell_dt[, .(focal_id = id, year, .row_id)]
  
  # Join focal rows to their neighbors: focal_id -> neighbor_id
  # Result: one row per (focal_row, neighbor) combination
  setkey(edge_dt, focal_id)
  focal_neighbors <- edge_dt[focal, on = .(focal_id), allow.cartesian = TRUE,
                             nomatch = NULL]
  # focal_neighbors has columns: focal_id, neighbor_id, year, .row_id
  
  # Now join to get neighbor variable values (neighbor_id + year -> row in cell_dt)
  # Prepare a lookup keyed on (id, year)
  neighbor_vals <- cell_dt[, c("id", "year", var_names), with = FALSE]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  
  # Merge: get variable values for each neighbor cell-year
  merged <- neighbor_vals[focal_neighbors, on = .(neighbor_id, year),
                          nomatch = NULL]
  # merged has: neighbor_id, year, var columns, focal_id, .row_id
  
  # Compute grouped stats per focal row
  for (vn in var_names) {
    cat("Computing neighbor stats for:", vn, "\n")
    
    # Aggregate by .row_id
    stats <- merged[!is.na(get(vn)),
                    .(nb_max  = max(get(vn)),
                      nb_min  = min(get(vn)),
                      nb_mean = mean(get(vn))),
                    by = .row_id]
    
    # Set proper column names
    max_col  <- paste0("nb_max_", vn)
    min_col  <- paste0("nb_min_", vn)
    mean_col <- paste0("nb_mean_", vn)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    
    # Join back to cell_dt by .row_id
    cell_dt <- stats[cell_dt, on = .(.row_id)]
  }
  
  cell_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_dt <- compute_all_neighbor_features(cell_dt, edge_dt, neighbor_source_vars)

# ---------------------------------------------------------------
# Step 4: Clean up and convert back if needed
# ---------------------------------------------------------------
cell_dt[, .row_id := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# Step 5: Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ---------------------------------------------------------------
# library(randomForest)  # or ranger, etc.
# rf_model <- readRDS("trained_rf_model.rds")   # already trained
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# write.csv(cell_data, "predictions.csv", row.names = FALSE)
```

---

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | Vectorized `rep`/`unlist` â†’ ~1.37M-row `data.table` (seconds) |
| `compute_neighbor_stats` | ~32.3M R-level `lapply` iterations (5 vars Ã— 6.46M) | 5 grouped `data.table` aggregations on a ~27M-row join table (minutes) |
| **Total neighbor engineering** | **~86+ hours** | **~5â€“20 minutes** (depending on hardware) |
| RF inference | Fast (unchanged) | Fast (unchanged) |

The optimization replaces interpreted R loops and per-element string operations with `data.table`'s compiled C-based joins and grouped aggregations, reducing runtime by roughly **three orders of magnitude** while producing numerically identical neighbor features and preserving the trained Random Forest model exactly as-is.
END_RESPONSE>>>
