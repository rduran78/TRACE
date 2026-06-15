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
run_id: Anthropic_F04_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For every row, it:

1. Performs repeated character coercion and hash-table lookups (`id_to_ref`, `idx_lookup`) via named-vector indexing.
2. Constructs paste-based composite keys (`paste(id, year, sep="_")`) 6.46M times, each touching a variable-length neighbor set.
3. Returns ragged lists of integer indices.

A secondary bottleneck is **`compute_neighbor_stats`**, which iterates over the same 6.46M-element list five times (once per variable), computing `max`/`min`/`mean` in pure R loops.

**Root causes:**
- **O(N Ã— k) character key construction and named-vector lookup** where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4 for rook). Named-vector lookup in R is hash-based but carries per-call overhead; doing it ~25.8M times is devastating.
- **Ragged list-of-vectors representation** prevents vectorization.
- **`compute_neighbor_stats` is called in a loop over 5 variables**, each re-traversing the 6.46M-element list.

## Optimization Strategy

**Core idea:** Replace the row-level `lapply` with fully vectorized operations using `data.table` joins and grouped aggregations. Instead of building a lookup list, create a flat edge table `(row_i, neighbor_row_j)` via a single merge, then compute all neighbor statistics with `data.table` grouped operations in one pass.

**Key steps:**

1. **Flat edge table construction (vectorized):** Expand the `nb` object into an edge data.frame `(id, neighbor_id)`. Join with the panel data on `(neighbor_id, year)` to get `(row_index, neighbor_row_index)` pairs â€” one `data.table` merge, no per-row R loop.

2. **Grouped aggregation:** For each source variable, join neighbor row indices to their values, group by the focal row, and compute `max`/`min`/`mean` in `data.table` â€” fully vectorized C-level computation.

3. **All 5 variables in a single pass** over the edge table to avoid redundant traversals.

This reduces the estimated runtime from 86+ hours to **minutes** on a 16 GB laptop.

## Optimized R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {

  # â”€â”€ Step 1: Convert cell_data to data.table and assign row indices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # â”€â”€ Step 2: Build flat edge table from the nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  #   rook_neighbors_unique is a list of integer vectors (spdep nb object).
  #   Element i contains the indices (into id_order) of neighbors of id_order[i].
  n_ids <- length(id_order)
  from_ref <- rep(seq_len(n_ids), lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )
  # Remove self-neighbors and the 0-coded "no neighbor" entries if any
  edges <- edges[neighbor_id != 0L & focal_id != neighbor_id]

  # â”€â”€ Step 3: Map (focal_id, year) â†’ row index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # Create a keyed lookup: for each (id, year) â†’ .row_idx
  id_year_key <- dt[, .(id, year, .row_idx)]
  setkey(id_year_key, id, year)

  # â”€â”€ Step 4: Expand edges across all years via merge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # Get unique years
  years <- unique(dt$year)

  # Cross-join edges with years â†’ one row per (focal_id, neighbor_id, year)
  # This produces ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB easily
  edge_year <- CJ_edges_years(edges, years)

  # Attach focal row index
  setkey(edge_year, focal_id, year)
  edge_year <- id_year_key[edge_year, on = .(id = focal_id, year),
                            nomatch = NULL]
  setnames(edge_year, ".row_idx", "focal_row")

  # Attach neighbor row index
  edge_year <- id_year_key[edge_year, on = .(id = neighbor_id, year),
                            nomatch = NULL]
  setnames(edge_year, ".row_idx", "neighbor_row")

  # Keep only what we need
  edge_year <- edge_year[, .(focal_row, neighbor_row)]
  setkey(edge_year, focal_row)

  # â”€â”€ Step 5: Compute neighbor stats for each variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    edge_year[, nval := vals[neighbor_row]]

    stats <- edge_year[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       by = focal_row]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]

    edge_year[, nval := NULL]
  }

  dt[, .row_idx := NULL]
  return(as.data.frame(dt))
}

# Helper: cross-join edges Ã— years without full CJ (memory-friendly)
CJ_edges_years <- function(edges, years) {
  n_edges <- nrow(edges)
  n_years <- length(years)
  data.table(
    focal_id    = rep(edges$focal_id,    times = n_years),
    neighbor_id = rep(edges$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# â”€â”€ Usage (drop-in replacement for the original outer loop) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” apply predict() as before.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor sets (rook contiguity within the same year) and the same underlying values. The `data.table` grouped aggregation uses the same arithmetic; results are identical to the original to floating-point precision. |
| **Trained RF model** | The code only modifies the feature columns on the prediction data. The model object is never touched or retrained. Column names are preserved. |
| **Column naming** | The helper produces identically named columns (adjust the naming pattern to match your `compute_and_add_neighbor_features` convention if it differs). |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60+ hrs (6.46M R-level iterations, paste, hash lookup) | ~30 sec (one `data.table` merge) |
| Neighbor stats (5 vars) | ~26 hrs (5 Ã— 6.46M list traversals) | ~2-3 min (5 grouped aggregations on ~38.5M rows) |
| **Total** | **~86+ hrs** | **~3-5 minutes** |

Peak memory for the edge-year table: ~38.5M rows Ã— 2 integer columns â‰ˆ 0.6 GB, well within the 16 GB budget.
END_RESPONSE>>>
