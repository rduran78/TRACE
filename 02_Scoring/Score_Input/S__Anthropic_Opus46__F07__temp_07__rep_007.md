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
run_id: Anthropic_F07_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over **~6.46 million rows** one at a time in a pure-R `lapply`, performing per-row string pasting, hash lookups, and NA filtering. This creates ~6.46 million small character vectors, does millions of `paste` and named-vector lookups, and produces millions of small integer vectors. The `compute_neighbor_stats` function then loops over the same 6.46 million entries again, subsetting a numeric vector with each small index vector.

Specific costs:

| Step | Operation | Why it's slow |
|---|---|---|
| `build_neighbor_lookup` | `paste(neighbor_cell_ids, data$year[i], sep="_")` per row | 6.46M string-construction calls |
| `build_neighbor_lookup` | `idx_lookup[neighbor_keys]` per row | 6.46M named-vector lookups (hash probes on character keys) |
| `compute_neighbor_stats` | `lapply` over 6.46M entries, subsetting + `max/min/mean` | R-level loop overhead, no vectorization |
| Outer loop | Repeated 5Ã— for each source variable | `build_neighbor_lookup` cost is paid once, but `compute_neighbor_stats` is paid 5Ã— |

Estimated: ~86+ hours on a 16 GB laptop.

## Optimization Strategy

**Replace the row-level R loops with fully vectorized operations using `data.table` and a pre-expanded edge list.**

Key ideas:

1. **Build the edge list once** â€” expand the `nb` object into a two-column integer matrix of `(cell_id, neighbor_cell_id)` pairs. This is ~1.37M rows.

2. **Join by (neighbor_id, year) using `data.table`** â€” instead of looping over 6.46M rows and doing string-key lookups, merge the edge list with the panel on `(id, year)` to retrieve neighbor values. `data.table` binary-search joins make this extremely fast.

3. **Group-by aggregation** â€” after the join, compute `max`, `min`, and `mean` of neighbor values grouped by `(id, year)` in one vectorized pass per variable.

4. **Loop only over the 5 variables**, not over rows.

This eliminates all per-row R-level iteration. Expected runtime: **minutes, not hours**.

The trained Random Forest model is untouched. The numerical results (neighbor max, min, mean per cell-year) are identical because the same neighbor relationships and the same aggregation functions are used.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert the spdep nb object to a two-column edge list (cell IDs)
#    id_order maps position index -> cell id
#    rook_neighbors_unique is the nb object (list of integer vectors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbs <- nb_obj[[i]]
    if (length(nbs) == 0L || (length(nbs) == 1L && nbs[1] == 0L)) next
    n <- length(nbs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbs]
    pos <- pos + n
  }
  
  # Trim if any cells had zero neighbors (0-sentinel in nb)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Vectorized neighbor stats computation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_vectorized <- function(cell_dt, edge_dt, var_name) {
  # Build a lookup table: (neighbor_id aliased as id, year) -> value
  # We only need id, year, and the variable column from the panel
  lookup_cols <- c("id", "year", var_name)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, c("id", var_name), c("neighbor_id", "nval"))
  
  # Join edge list with panel to get (id, year, neighbor_id),

  # then join on (neighbor_id, year) to get neighbor values.
  # Step A: cross edge list with all years for each id?
  #   No â€” more efficient: join panel with edge list on id,
  #   then join the result with lookup on (neighbor_id, year).
  
  # Merge panel rows with their neighbor IDs
  # cell_dt has (id, year, ...). We need (id, year) x edge_dt on id -> (id, year, neighbor_id)
  # Use edge_dt keyed on id.
  
  setkey(edge_dt, id)
  
  # Get unique (id, year) pairs â€” these are just the row indices of cell_dt
  id_year <- cell_dt[, .(id, year)]
  
  # Join: for each (id, year) row, find all neighbor_ids from edge_dt
  # This produces ~6.46M * (avg ~4 neighbors) â‰ˆ 25-26M rows
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Now join to get the neighbor's value in that year
  setkey(lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, nval, id
  
  # Aggregate by (id, year)
  stats <- expanded[!is.na(nval),
                    .(nmax  = max(nval),
                      nmin  = min(nval),
                      nmean = mean(nval)),
                    by = .(id, year)]
  
  # Name the new columns
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  # Left-join back onto cell_dt (preserves rows with no neighbors as NA)
  setkey(stats, id, year)
  setkey(cell_dt, id, year)
  
  cell_dt <- stats[cell_dt, on = .(id, year)]
  
  # Clean up the lookup rename so we don't mutate the caller's copy
  cell_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Main pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert cell_data to data.table (if not already)
cell_data <- as.data.table(cell_data)

# Build edge list once (~1.37M rows, instant)
edge_list <- build_edge_list(id_order, rook_neighbors_unique)

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and attach neighbor features for each variable
for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  cell_data <- compute_neighbor_features_vectorized(cell_data, edge_list, var_name)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (The model object and predict call remain exactly as before.)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `build_edge_list` reads the same `rook_neighbors_unique` nb object and the same `id_order` mapping. Every directed edge is identical. |
| **Same aggregation** | `max`, `min`, `mean` over the same non-NA neighbor values per `(id, year)` group â€” algebraically identical to the original `compute_neighbor_stats`. |
| **NAs handled identically** | `!is.na(nval)` filtering before aggregation mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the left join (same as the original `c(NA, NA, NA)` return). |
| **Trained RF untouched** | No retraining; the feature columns have the same names and values. |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M R-loop iterations with string ops) | ~seconds (vectorized edge list construction) |
| Stats per variable | ~17 hours Ã— 5 = 85h | ~30â€“90 seconds Ã— 5 â‰ˆ 3â€“8 minutes |
| **Total** | **86+ hours** | **~5â€“10 minutes** |

Peak memory for the expanded join table: ~25M rows Ã— 3 numeric columns â‰ˆ ~600 MB, well within 16 GB.
END_RESPONSE>>>
