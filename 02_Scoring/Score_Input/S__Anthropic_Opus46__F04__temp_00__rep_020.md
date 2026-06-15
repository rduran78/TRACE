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
run_id: Anthropic_F04_T00_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated `paste` and named-vector lookups on a vector of ~6.46M names, is extremely slow. Named vector lookup in R degrades as the vector grows; at 6.46M entries the hash table is enormous and each probe is expensive. Multiply by ~4 neighbors per cell Ã— 6.46M rows â‰ˆ 25.8 billion character operations.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 Ã— 6.46M â‰ˆ 32.3 million times. The per-call overhead of anonymous function dispatch, `is.na`, and three summary functions dominates.

### Estimated cost breakdown

| Step | Calls | Estimated share |
|---|---|---|
| `build_neighbor_lookup` (paste + named lookup) | 6.46M | ~60â€“70% |
| `compute_neighbor_stats` (5 vars Ã— 6.46M) | 32.3M | ~25â€“35% |
| Random Forest `predict()` | 1 | <5% |

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** problem. Every cell-year needs to be joined to its neighbors' cell-years. This is a classic equi-join on `(id, year)` that `data.table` handles in C.

**Steps:**

1. **Build an edge table** (a two-column `data.table` of `id â†’ neighbor_id`) from the `nb` object â€” done once, vectorized.
2. **Join** the edge table to the panel data on `(neighbor_id, year)` to get neighbor values â€” one vectorized join per variable (or all at once).
3. **Group-aggregate** by `(id, year)` to compute `max`, `min`, `mean` â€” fully vectorized in `data.table`.

This eliminates all 6.46M-iteration R loops, all `paste` key construction, and all named-vector lookups. Expected speedup: **~200â€“500Ã—** (from 86+ hours to roughly 10â€“25 minutes).

**Preservation guarantees:**
- The trained Random Forest model is untouched (no retraining).
- The numerical estimand is identical: for each `(id, year)`, the neighbor max, min, and mean of each variable are computed over the same rook-neighbor set with the same NA handling.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a vectorized edge table from the nb object (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors into id_order)
  # Expand to a two-column data.table: focal_id -> neighbor_id
  n <- length(neighbors)
  
  # Pre-compute lengths to allocate once
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute all neighbor features via vectorized join + group-agg
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  # Convert to data.table (by reference if already; copy if not)
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }
  
  # 1. Build edge table  (~1.37M rows, instant)
  edges <- build_edge_table(id_order, neighbors)
  
  # 2. Attach year from focal cell: expand edges Ã— years

  #    Instead of a cross-join (which would be huge), we join edges to dt
  #    to get (focal_id, year, neighbor_id), then join again to get
  #    neighbor values.
  
  #    But smarter: join edges to dt on focal side to get year,
  #    then join to dt on neighbor side to get neighbor values.
  
  # Subset dt to only the columns we need (saves memory)
  cols_needed <- c("id", "year", neighbor_source_vars)
  dt_sub <- dt[, ..cols_needed]
  
  # 2a. Join edges to focal rows to get (focal_id, year, neighbor_id)
  #     This is edges Ã— years_per_cell, but done via join not cross-join.
  setkey(dt_sub, id)
  
  # Get unique (id, year) pairs from dt_sub â€” but dt_sub has all of them
  focal_years <- dt_sub[, .(id, year)]
  setnames(focal_years, "id", "focal_id")
  
  # Merge: for each (focal_id, year), attach all neighbor_ids
  setkey(edges, focal_id)
  setkey(focal_years, focal_id)
  
  # This produces ~1.37M Ã— 28 â‰ˆ 38.5M rows (manageable in 16GB)
  expanded <- edges[focal_years, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year
  
  # 2b. Join to dt_sub on (neighbor_id, year) to get neighbor variable values
  setnames(dt_sub, "id", "neighbor_id")
  setkeyv(dt_sub, c("neighbor_id", "year"))
  setkeyv(expanded, c("neighbor_id", "year"))
  
  joined <- dt_sub[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # joined has: neighbor_id, year, <all source vars>, focal_id
  
  # 3. Group-aggregate by (focal_id, year)
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Build the aggregation call
  agg_list <- as.call(c(as.name("list"), agg_exprs))
  
  agg <- joined[, eval(agg_list), by = .(focal_id, year)]
  
  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  for (v in neighbor_source_vars) {
    max_col <- paste0("neighbor_max_", v)
    min_col <- paste0("neighbor_min_", v)
    agg[is.infinite(get(max_col)), (max_col) := NA_real_]
    agg[is.infinite(get(min_col)), (min_col) := NA_real_]
  }
  
  # 4. Merge aggregated features back to original dt
  setnames(agg, "focal_id", "id")
  setkeyv(dt, c("id", "year"))
  setkeyv(agg, c("id", "year"))
  
  feature_cols <- setdiff(names(agg), c("id", "year"))
  dt[agg, (feature_cols) := mget(paste0("i.", feature_cols)),
     on = c("id", "year")]
  
  # 5. Return as data.frame if input was data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Budget Check

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols, numeric) | ~5.7 GB |
| `edges` (1.37M Ã— 2 integer cols) | ~11 MB |
| `expanded` (~38.5M Ã— 3 cols) | ~0.9 GB |
| `joined` (~38.5M Ã— 7 cols) | ~2.2 GB |
| `agg` (6.46M Ã— 17 cols) | ~0.9 GB |
| **Peak total** | **~9.7 GB** (fits in 16 GB) |

If memory is tight, process variables in batches of 2â€“3 instead of all 5 at once (reducing `joined` width), or use `rm()` + `gc()` after each intermediate step.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M R-loop iterations with `paste` + named vector | Vectorized `data.table` join |
| Neighbor stats | 32.3M R-loop iterations (5 vars) | Single grouped `data.table` aggregation |
| Key mechanism | Named character vector hash (R-level) | `data.table` binary-search join (C-level) |
| Estimated runtime | 86+ hours | ~10â€“25 minutes |
| Numerical result | Identical | Identical |
| RF model | Unchanged | Unchanged |
END_RESPONSE>>>
