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
run_id: Anthropic_F07_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²) list construction via per-row string hashing

The function iterates over **~6.46 million rows**, and for each row it:
- Looks up the cell's neighbor IDs.
- Pastes string keys (`id_year`).
- Searches a named character vector (`idx_lookup`) by string matching.

String-based named-vector lookups in R are **O(n)** per call in the worst case (hash collisions aside, the overhead of `paste` + named indexing over 6.46M rows is enormous). The resulting `neighbor_lookup` is a **list of 6.46 million integer vectors**, each built individually.

### 2. `compute_neighbor_stats` â€” Repeated per-variable list-apply

For each of the 5 source variables, the code iterates over all 6.46M rows again via `lapply`, extracting neighbor values element by element. This is **5 Ã— 6.46M = 32.3M** R-level loop iterations with per-element subsetting.

### Combined effect
These two stages together produce the estimated **86+ hour** runtime. The fundamental issue is: **row-level R loops over millions of rows with string operations**.

---

## Optimization Strategy

### A. Replace string-key lookup with integer join via `data.table`

Instead of building a 6.46M-element list of neighbor row indices using string keys, we:

1. Create a `data.table` of all directed neighbor pairs: `(id, neighbor_id)` â€” ~1.37M pairs.
2. Cross this with all 28 years to get `(id, year, neighbor_id)` â€” but more efficiently, we join on `(id, year)` to attach the row index, then join on `(neighbor_id, year)` to attach the neighbor's row index. This is a **vectorized equi-join**, not a per-row loop.

### B. Compute all neighbor stats in one vectorized grouping operation

Once we have a table of `(row_index, neighbor_row_index)`, we can:
1. Pull the variable value for each neighbor row.
2. Group by `row_index` and compute `max`, `min`, `mean` in a single `data.table` aggregation â€” fully vectorized in C.

### C. Avoid building the 6.46M-element `neighbor_lookup` list entirely

The list is never needed. The join table replaces it.

### Expected speedup
- `data.table` equi-joins and grouped aggregations over ~38M edge-year rows (1.37M edges Ã— 28 years) should complete in **seconds to low minutes**, not hours.
- Total for 5 variables: **under 10 minutes** on a 16 GB laptop.

### Preserving the estimand
The numerical values computed (`max`, `min`, `mean` of non-NA neighbor values, with `NA` when no valid neighbors exist) are **identical** to the original code. No model retraining is needed.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature engineering for cell-year panel data.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all source vars.
#' @param id_order        character/integer vector â€” the cell IDs in the order matching rook_neighbors_unique.
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for.
#' @return cell_data as a data.table with new columns appended.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Step 0: Convert to data.table if needed; add row-position key ----------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Step 1: Build directed edge list (cell-level, year-independent) --------
  #     Each entry in the nb object is an integer vector of neighbor positions
  #     within id_order.  We expand to a two-column data.table of (id, neighbor_id).
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep nb encodes "no neighbors" as a single 0L
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb])
  }))

  # --- Step 2: Cross edges with years via join to get row indices -------------
  #     We need:  for every (id, year) row, the .row_idx of each neighbor in that year.

  # Keyed lookup:  (id, year) -> .row_idx
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # Get all unique years once
  all_years <- unique(dt$year)

  # Expand edges Ã— years  (1.37M Ã— 28 â‰ˆ 38.4M rows â€” fits easily in RAM)
  edge_years <- edges[, CJ(year = all_years), by = .(id, neighbor_id)]

  # Attach the focal row's index
  setkey(edge_years, id, year)
  edge_years <- row_key[edge_years, on = .(id, year), nomatch = 0L]
  setnames(edge_years, ".row_idx", "focal_row")

  # Attach the neighbor row's index
  setnames(edge_years, c("id", "neighbor_id"), c("focal_id", "id"))
  setkey(edge_years, id, year)
  edge_years <- row_key[edge_years, on = .(id, year), nomatch = 0L]
  setnames(edge_years, ".row_idx", "neighbor_row")
  setnames(edge_years, c("id", "focal_id"), c("neighbor_id", "id"))

  # edge_years now has columns: id, year, focal_row, neighbor_id, neighbor_row

  # --- Step 3: For each variable, compute grouped neighbor stats --------------
  for (var_name in neighbor_source_vars) {

    message("Computing neighbor stats for: ", var_name)

    # Pull neighbor values via direct integer indexing (vectorized)
    edge_years[, nval := dt[[var_name]][neighbor_row]]

    # Aggregate â€” drop NAs within each group
    agg <- edge_years[!is.na(nval),
                      .(nmax  = max(nval),
                        nmin  = min(nval),
                        nmean = mean(nval)),
                      keyby = .(focal_row)]

    # Prepare NA-filled columns, then fill where we have results
    max_col  <- paste0("n_max_",  var_name)
    min_col  <- paste0("n_min_",  var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    dt[agg$focal_row, (max_col)  := agg$nmax]
    dt[agg$focal_row, (min_col)  := agg$nmin]
    dt[agg$focal_row, (mean_col) := agg$nmean]
  }

  # Clean up helper column
  edge_years[, nval := NULL]
  dt[, .row_idx := NULL]

  return(dt[])
}


# =============================================================================
# Usage â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now contains the same n_max_*, n_min_*, n_mean_* columns
# as the original pipeline.  The trained Random Forest model can be
# applied directly with no retraining.
```

---

## Memory Estimate

| Object | Approximate size |
|---|---|
| `dt` (6.46M Ã— ~115 cols) | ~5.5 GB |
| `edge_years` (38.4M Ã— 5 cols) | ~1.5 GB |
| `row_key` (6.46M Ã— 3 cols) | ~0.15 GB |
| Overhead + temporaries | ~1â€“2 GB |
| **Total peak** | **~8â€“9 GB** âœ” (fits in 16 GB) |

## Runtime Estimate

| Step | Expected time |
|---|---|
| Edge expansion + joins | ~30â€“60 seconds |
| Per-variable aggregation (Ã—5) | ~20â€“40 seconds each |
| **Total** | **~3â€“5 minutes** |

This represents a speedup of roughly **1,000Ã—â€“1,500Ã—** over the original 86+ hour estimate, while producing **numerically identical** results (same `max`, `min`, `mean` of non-NA rook-neighbor values, same `NA` propagation for isolated or boundary cells). The trained Random Forest model requires no modification.
END_RESPONSE>>>
