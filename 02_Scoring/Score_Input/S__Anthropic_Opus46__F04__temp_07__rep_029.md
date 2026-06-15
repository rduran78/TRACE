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
run_id: Anthropic_F04_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector â€” **O(1) amortized but with constant overhead from `as.character()` and named-vector hashing, repeated 6.46M times**.
2. Constructs `paste(..., sep="_")` key strings for every neighbor of every row â€” millions of string allocations.
3. Indexes into `idx_lookup` (a named character vector) â€” named vector lookup in R is hash-based but still has per-call overhead, and this is done for every neighbor of every row.

The total number of string constructions and hash lookups is approximately **6.46M rows Ã— ~4 avg neighbors â‰ˆ 25.8M paste + lookup operations**, all inside a sequential `lapply` with R-level overhead.

**`compute_neighbor_stats`** then loops over 6.46M entries again, computing `max/min/mean` in pure R â€” slow but less catastrophic since the inner operations are cheap. However, it is called 5 times (once per variable), totaling ~32.3M R-level function calls.

**Root cause summary:**
- Millions of R-level string allocations and hash lookups in a sequential loop.
- No vectorization or use of `data.table` merge/join semantics.
- `compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)` over millions of small vectors.

## Optimization Strategy

1. **Replace `build_neighbor_lookup` entirely** with a vectorized `data.table` equi-join approach: expand the neighbor graph into an edge list, join on `(neighbor_id, year)` to get row indices, then group by source row.
2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable â€” `max`, `min`, `mean` computed in C-level `data.table` internals, not R-level loops.
3. **Avoid materializing the full neighbor_lookup list at all** â€” go directly from edge list to aggregated statistics.

This reduces the problem to a merge + grouped aggregation, which `data.table` handles in seconds-to-minutes on data of this size.

## Optimized Working R Code

```r
library(data.table)

#' Build a directed edge list from the spdep nb object.
#' Returns a data.table with columns: source_id, neighbor_id
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  n <- length(neighbors)
  # Pre-allocate by computing total edges
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  source_id   <- integer(total)
  neighbor_id <- integer(total)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    k <- length(nb_i)
    if (k > 0L) {
      idx <- pos:(pos + k - 1L)
      source_id[idx]   <- id_order[i]
      neighbor_id[idx] <- id_order[nb_i]
      pos <- pos + k
    }
  }
  
  data.table(source_id = source_id, neighbor_id = neighbor_id)
}

#' Compute neighbor summary statistics for one variable using data.table joins.
#' Returns a data.table with columns: id, year, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(dt, edge_dt, var_name) {
  # dt must be a data.table with columns: id, year, row_idx, and <var_name>
  # edge_dt has columns: source_id, neighbor_id
  
  # Step 1: Cross edge list with years via join on neighbor side
  # We need: for each (source_id, year), find all (neighbor_id, year) rows and
  # aggregate var_name.
  
  # Create a keyed version for joining
  neighbor_vals <- dt[, .(neighbor_id = id, year, nval = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)
  
  # Expand edges Ã— years: join edge_dt to dt to get (source_id, year) pairs,
  # then join to neighbor values.
  # More efficient: join edges to neighbor_vals, then join back source identity.
  
  # edges_with_vals: for each (source_id, neighbor_id), for each year,
  # get the neighbor's value
  edges_expanded <- edge_dt[neighbor_vals, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # edges_expanded now has: source_id, neighbor_id, year, nval
  
  # Step 2: Aggregate by (source_id, year)
  max_name  <- paste0(var_name, "_max")
  min_name  <- paste0(var_name, "_min")
  mean_name <- paste0(var_name, "_mean")
  
  agg <- edges_expanded[
    !is.na(nval),
    .(V_max = max(nval), V_min = min(nval), V_mean = mean(nval)),
    by = .(source_id, year)
  ]
  
  setnames(agg, c("source_id", "V_max", "V_min", "V_mean"),
           c("id", max_name, min_name, mean_name))
  
  agg
}

#' Main optimized pipeline: compute all neighbor features and merge into cell_data.
#' Preserves the original data and adds neighbor feature columns.
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  # 1. Build edge list once (fast, ~1.37M rows)
  message("Building edge list...")
  edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
  
  # 2. For each variable, compute neighbor stats and merge
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor features for: %s", var_name))
    
    agg <- compute_neighbor_stats_fast(dt, edge_dt, var_name)
    
    # Merge back onto dt by (id, year); unmatched rows get NA (preserving original behavior)
    max_name  <- paste0(var_name, "_max")
    min_name  <- paste0(var_name, "_min")
    mean_name <- paste0(var_name, "_mean")
    
    # Remove columns if they already exist (idempotency)
    for (col in c(max_name, min_name, mean_name)) {
      if (col %in% names(dt)) dt[, (col) := NULL]
    }
    
    dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  }
  
  # 3. Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# â”€â”€ Usage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has the 15 new neighbor columns (_max, _min, _mean Ã— 5 vars).
# Feed directly into the existing trained Random Forest for prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over the identical set of non-NA neighbor values for each `(id, year)` as the original code. The `data.table` join `on = .(neighbor_id), allow.cartesian = TRUE` replicates the same edge Ã— year expansion, and the `!is.na(nval)` filter matches the original `neighbor_vals[!is.na(neighbor_vals)]`. Unmatched `(id, year)` pairs receive `NA` via `all.x = TRUE`, identical to the original fallback. |
| **Trained RF model** | No model retraining occurs. The code only constructs predictor columns. |
| **Column names** | Identical `<var>_max`, `<var>_min`, `<var>_mean` naming convention. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | Eliminated; replaced by vectorized edge list build (~1.37M rows, <1 sec) |
| `compute_neighbor_stats` Ã— 5 vars | ~32.3M R-level `lapply` calls + `do.call(rbind)` | 5 `data.table` keyed joins + grouped aggregations (C-level, parallelized internally) |
| **Total estimated time** | **86+ hours** | **~5â€“15 minutes** on 16 GB RAM laptop |

The `data.table` join expands ~1.37M edges Ã— 28 years â‰ˆ ~38.4M rows per variable, which fits comfortably in memory and is aggregated at C level. Peak memory usage is approximately 2â€“3 GB above the base dataset.
END_RESPONSE>>>
