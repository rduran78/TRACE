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
run_id: Anthropic_F09_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` constructs a row-level lookup across all 6.46 million cell-year rows using `lapply` with string-pasting and named-vector indexing.** For each of the 6.46M rows, it pastes neighbor cell IDs with the row's year, then looks them up in a named character vector of length 6.46M. Named vector lookup in R is O(n) per query in the worst case; doing this ~6.46M Ã— ~4 neighbors â‰ˆ 26M string lookups against a 6.46M-length named vector is catastrophically slow.

2. **The lookup is *year-redundant*.** The spatial neighbor topology is identical across all 28 years. A cell's rook neighbors in 1992 are the same cells as in 2019. Yet `build_neighbor_lookup` rebuilds the full mapping for every cell-year row, embedding the year into the key and re-resolving it. This inflates the problem from ~344K cells Ã— ~4 neighbors to ~6.46M rows Ã— ~4 neighbors.

3. **`compute_neighbor_stats` then iterates over all 6.46M entries in `neighbor_lookup` with `lapply`, computing max/min/mean per row.** This is pure R-level loopingâ€”no vectorization.

**Core insight:** The neighbor *topology* is a static spatial property. It should be built **once** as a simple cell-to-cell adjacency table (~1.37M rows), then **joined** to the panel data by year to resolve neighbor attribute values, then **aggregated** with vectorized grouped operations. This converts the entire pipeline from O(rows Ã— neighbors Ã— string-ops) to a few fast data.table joins and grouped aggregations.

---

## Optimization Strategy

1. **Build a static directed-edge table once** from `rook_neighbors_unique` (the `nb` object): a two-column data.table with `focal_id` and `neighbor_id`. This has ~1.37M rows and never changes.

2. **For each year and each variable**, join the edge table to the cell-year attribute data on `(neighbor_id, year)` to attach each neighbor's variable value to each directed edge. Then group by `(focal_id, year)` and compute `max`, `min`, `mean` in a single vectorized pass.

3. **Join the resulting summary statistics back** onto `cell_data` by `(id, year)`.

4. This replaces 6.46M-iteration `lapply` calls with a handful of keyed data.table merges and `[, .(â€¦), by=â€¦]` aggregationsâ€”orders of magnitude faster.

**Expected speedup:** The entire neighbor-feature computation should drop from ~86+ hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static cell-neighbor edge table ONCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inputs:
#   id_order             â€” vector of cell IDs (length 344,208), in the
#                          same order as rook_neighbors_unique
#   rook_neighbors_unique â€” an nb object (list of integer index vectors)
#
# Output:
#   edge_dt â€” a data.table with columns: focal_id, neighbor_id
#             (~1,373,394 rows)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate vectors
  n_edges <- sum(lengths(neighbors_nb))
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_id[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_id[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  # Trim if any 0-neighbor cells reduced the count
  data.table(focal_id = focal_id[1:(pos - 1L)],
             neighbor_id = neighbor_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (non-destructive)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Preserve original row order for downstream prediction
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_order := .I]   # bookmark original order

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: For each neighbor source variable, compute neighbor
#         max / min / mean via join + grouped aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the attribute table for fast join on (id, year)
# We create a slim lookup: just id, year, and the five source vars
attr_cols <- c("id", "year", neighbor_source_vars)
attr_dt   <- cell_dt[, ..attr_cols]
setkey(attr_dt, id, year)

# We will accumulate new columns in a results table keyed by (id, year)
results_dt <- cell_dt[, .(id, year, .row_order)]
setkey(results_dt, id, year)

for (var_name in neighbor_source_vars) {

  message("Computing neighbor stats for: ", var_name)

  # Slim neighbor-attribute table: just id (as neighbor_id), year, value
  nb_attr <- attr_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(nb_attr, neighbor_id, year)

  # Expand edges Ã— years: join neighbor attributes onto every edge for

  # every year.
  #
  # Strategy: cross-join edge_dt with unique years, then look up the
  # neighbor's value.  But that would create edge_dt Ã— 28 rows first.
  # More memory-efficient: join edges onto the long cell-year table.

  # For each row in cell_dt, we need that row's focal_id's neighbors.
  # Approach: join cell_dt (as focal) â†’ edge_dt â†’ nb_attr in two steps.

  # Step A: attach neighbors to each focal cell-year
  #   focal_rows has one row per (focal_id, year, neighbor_id)
  focal_key <- cell_dt[, .(focal_id = id, year)]
  setkey(focal_key, focal_id)
  setkey(edge_dt, focal_id)

  # This is an equi-join: every focal cell-year gets replicated for each

  # of its neighbors.  ~6.46M Ã— avg_neighbors â‰ˆ 26M rows â€” fits in 16 GB.
  expanded <- edge_dt[focal_key, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: focal_id, neighbor_id, year

  # Step B: attach the neighbor's variable value
  setkey(expanded, neighbor_id, year)
  expanded[nb_attr, on = .(neighbor_id, year), value := i.value]

  # Step C: aggregate by (focal_id, year)
  stats <- expanded[!is.na(value),
                    .(nb_max  = max(value),
                      nb_min  = min(value),
                      nb_mean = mean(value)),
                    by = .(focal_id, year)]

  # Rename columns to match original pipeline naming convention
  #   e.g. ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col, min_col, mean_col))

  # Join back to results
  setkey(stats, focal_id, year)
  results_dt <- stats[results_dt, on = .(focal_id = id, year)]
  setnames(results_dt, "focal_id", "id")
  setkey(results_dt, id, year)

  # Clean up per-iteration large objects
  rm(nb_attr, expanded, stats)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Merge neighbor features back into cell_data (original frame)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Restore original row order
setorder(results_dt, .row_order)

# Identify the new neighbor-stat columns
new_cols <- setdiff(names(results_dt), c("id", "year", ".row_order"))

# Bind new columns onto the original cell_data (preserves class: data.frame / tibble)
for (col in new_cols) {
  cell_data[[col]] <- results_dt[[col]]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g. `rf_model`) is used as-is.
# cell_data now contains the same neighbor feature columns with the
# same numerical values as the original pipeline would have produced.
#
# Example (adjust to your actual model object name):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The join-then-aggregate approach computes `max`, `min`, `mean` over exactly the same set of non-NA neighbor values as the original `compute_neighbor_stats`. No rounding or approximation is introduced. |
| **Trained RF model preserved** | The model object is never touched. Only the input feature columns are recomputed (identically). |
| **Column names preserved** | The `setnames` call produces the same `{var}_neighbor_max/min/mean` columns the original `compute_and_add_neighbor_features` would have created. |
| **NA handling** | Rows with no neighbors (or all-NA neighbor values) receive `NA` for all three stats via the `nomatch` / left-join semantics, matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| `build_edge_table` (once) | < 5 seconds |
| Per-variable join + aggregation | ~30â€“90 seconds each |
| Total for 5 variables | **~3â€“8 minutes** |
| **Original implementation** | **~86+ hours** |

The speedup factor is roughly **600â€“1,700Ã—**, achieved entirely by eliminating per-row string operations and replacing R-level `lapply` loops with vectorized data.table keyed joins and grouped aggregations.
END_RESPONSE>>>
