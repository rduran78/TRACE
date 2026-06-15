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
run_id: Anthropic_F04_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically the combination of:

1. **`build_neighbor_lookup`**: An `lapply` over ~6.46 million rows that, for each row, performs character coercion, string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe *per element*, but constructing ~6.46 million string keys and doing millions of named lookups is extremely slow in an interpreted row-wise loop.

2. **`compute_neighbor_stats`**: Another `lapply` over ~6.46 million rows computing `max`, `min`, `mean` on small vectors. This is called **5 times** (once per source variable), yielding ~32.3 million R-level function calls to summary statistics.

**Quantified cost of `build_neighbor_lookup`:**
- 6.46M iterations Ã— (string paste + named vector subset) â‰ˆ tens of hours alone.
- The string-keyed lookup `idx_lookup[neighbor_keys]` is O(k) per row where k = number of neighbors (â‰ˆ4 for rook), but the constant factor of R-level string operations is enormous at this scale.

**Quantified cost of `compute_neighbor_stats`:**
- 5 variables Ã— 6.46M rows Ã— 3 summary stats = ~97M scalar computations wrapped in R `lapply` overhead.

**Root cause summary:** Row-level R loops with string manipulation and named-vector lookups over 6.46 million rows, repeated across 5 variables.

---

## Optimization Strategy

The strategy is to **eliminate all row-level R loops** by converting to **vectorized join and grouped aggregation** using `data.table`:

1. **Replace `build_neighbor_lookup`** with a single `data.table` equi-join. Pre-build an edge table (`cell_id` â†’ `neighbor_cell_id`) and join it against the data keyed on `(id, year)`. This produces a tall table of (row_index, neighbor_row_index) pairs â€” no string keys, no `lapply`.

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation: group by the focal row index, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass.

3. **Process all 5 variables** in a single pass over the neighbor-joined table (or 5 fast vectorized passes), eliminating redundant joins.

**Expected speedup:** From ~86+ hours to **minutes** (typically 5â€“15 minutes on a 16 GB laptop), because `data.table` joins and grouped aggregations are implemented in C and operate on integer keys without string construction.

**Numerical equivalence:** The operations (`max`, `min`, `mean` of the same neighbor sets) are identical, preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all neighbor_source_vars columns.
#' @param id_order         integer vector of cell IDs in the order matching the
#'                         nb object (i.e., id_order[i] is the cell ID for the
#'                         i-th element of rook_neighbors_unique).
#' @param neighbors        spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return cell_data (data.table) with new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
#'         for each var in neighbor_source_vars.

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      neighbors,
                                      neighbor_source_vars) {

  # --- Step 0: Convert to data.table (by reference if already one) -----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order for downstream RF prediction
  cell_data[, .row_idx := .I]

  # --- Step 1: Build directed edge table (focal_id -> neighbor_id) -----------
  #
  # This replaces the per-row string-paste + named-vector lookup in

  # build_neighbor_lookup with a single vectorized construction.

  # Number of neighbors per node (0 for isolates encoded as list(0L) by spdep)
  n_neighbors <- vapply(neighbors, function(x) {
    if (length(x) == 1L && x[0 + 1] == 0L) 0L else length(x)
  }, integer(1))
  # Handle spdep convention: a neighbor list entry of integer(0) or 0L means

  # no neighbors.
  n_neighbors_safe <- vapply(neighbors, function(x) {
    nx <- x[x != 0L]
    length(nx)
  }, integer(1))

  focal_indices <- rep(seq_along(id_order), times = n_neighbors_safe)
  neighbor_indices <- unlist(lapply(neighbors, function(x) x[x != 0L]),
                             use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )

  # --- Step 2: Join edges with data to get neighbor row indices --------------
  #
  # Key idea: for every (focal_id, year) we need the data-row indices of

  # (neighbor_id, year).  We achieve this with two keyed joins â€” no strings.

  # Minimal keyed reference for focal rows: maps (id, year) -> .row_idx
  focal_ref <- cell_data[, .(id, year, .row_idx)]
  setkey(focal_ref, id, year)

  # Expand edges by year: each edge applies to every year present for the

  # focal cell.  Instead of a full cross-join (expensive in memory), we join
  # edges onto the data.


  # Join 1: attach focal .row_idx and year to each edge
  setkey(edges, focal_id)
  focal_years <- cell_data[, .(focal_id = id, year, focal_row = .row_idx)]
  setkey(focal_years, focal_id)

  # This is the big join: for every (edge Ã— year-of-focal), get focal_row

  edge_year <- edges[focal_years, on = .(focal_id), allow.cartesian = TRUE,
                     nomatch = 0L]
  # edge_year columns: focal_id, neighbor_id, year, focal_row

  # Join 2: look up the neighbor's row index for the same year
  neighbor_ref <- cell_data[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_ref, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)

  edge_year <- neighbor_ref[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # Keep only rows where the neighbor actually exists in that year
  edge_year <- edge_year[!is.na(neighbor_row)]

  # --- Step 3: Vectorized grouped aggregation per variable -------------------
  #
  # For each source variable, pull neighbor values via integer indexing
  # (vectorized), then group-by focal_row to compute max/min/mean.

  for (var_name in neighbor_source_vars) {
    vals_vec <- cell_data[[var_name]]

    # Attach neighbor values by integer index â€” fully vectorized
    edge_year[, nval := vals_vec[neighbor_row]]

    # Remove NAs in neighbor values before aggregation
    valid <- edge_year[!is.na(nval)]

    agg <- valid[, .(
      nmax  = max(nval),
      nmin  = min(nval),
      nmean = mean(nval)
    ), keyby = focal_row]

    # Initialize new columns to NA (handles rows with no valid neighbors)
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")

    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    # Write aggregated values back by integer index â€” vectorized
    set(cell_data, i = agg$focal_row, j = col_max,  value = agg$nmax)
    set(cell_data, i = agg$focal_row, j = col_min,  value = agg$nmin)
    set(cell_data, i = agg$focal_row, j = col_mean, value = agg$nmean)
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]
  edge_year[, nval := NULL]  # free memory

  return(cell_data)
}


# ============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Budget Check

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols, numeric) | ~5.4 GB |
| `edge_year` (â‰ˆ6.46M Ã— 4 neighbors Ã— 5 cols, integer) | ~1.5 GB |
| `agg` (temporary, â‰¤6.46M Ã— 4 cols) | ~0.2 GB |
| Overhead / copies | ~2 GB |
| **Total peak** | **~9â€“10 GB** |

This fits within 16 GB RAM. If memory is tight, the `for` loop over the 5 variables already reuses the `nval` column in place, and `agg` is overwritten each iteration.

---

## Why This Preserves the Original Estimand

- The edge table encodes exactly the same directed rook-neighbor relationships as the original `nb` object.
- The `(focal_id, year)` â†’ `(neighbor_id, year)` join is logically identical to the original string-key lookup.
- `max`, `min`, `mean` are computed on exactly the same neighbor value sets (after the same `!is.na` filter).
- The trained Random Forest model is never retrained â€” only the input feature matrix is reconstructed faster.
- Column names follow the same `<var>_neighbor_{max,min,mean}` convention assumed by the downstream model.
END_RESPONSE>>>
