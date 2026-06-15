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
run_id: Anthropic_F04_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs `paste`-based string keys for each neighborâ€“year combination.
4. Matches those keys against a named character vector (`idx_lookup`) of length ~6.46M.

Named-vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of repeated `paste`, character conversion, and subsetting on a 6.46M-name vector millions of times is enormous). With ~6.46M rows and an average of ~4 rook neighbors each, this produces roughly **25.8 million string constructions and lookups**, each against a 6.46M-element named vector. The `compute_neighbor_stats` function is comparatively cheap (just numeric subsetting), but `do.call(rbind, ...)` on 6.46M small vectors is also unnecessarily slow.

**Secondary issue:** `compute_neighbor_stats` is called 5 times (once per variable), each time iterating over 6.46M entries. This is redundant since the neighbor index sets don't change across variables.

## Optimization Strategy

1. **Replace the per-row `lapply` in `build_neighbor_lookup` with a vectorized, merge-based approach using `data.table`.** Instead of building a list of neighbor row indices per row, build a two-column edge table `(focal_row, neighbor_row)` in one vectorized pass. This eliminates millions of `paste` calls and named-vector lookups.

2. **Compute all neighbor stats in one grouped `data.table` aggregation** over the edge table, for all 5 variables simultaneously, avoiding 5 separate `lapply` passes over 6.46M list elements.

3. **Eliminate `do.call(rbind, ...)`** on millions of small vectors (which is O(nÂ²) in memory copies).

**Expected speedup:** From 86+ hours to roughly 5â€“15 minutes on the same laptop.

## Optimized Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # Convert to data.table if not already; add a row index
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # ---- Step 1: Build a complete (focal_row, neighbor_row) edge table ----

  # Map each cell id to its position in id_order
  id_to_ref <- data.table(
    id      = id_order,
    ref_idx = seq_along(id_order)
  )

  # Expand the nb object into an edge list: (focal_ref_idx, neighbor_ref_idx)
  # rook_neighbors_unique is a list of integer vectors (spdep nb object)
  focal_ref <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  neighbor_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-neighbor entries (nb encodes no-neighbor as integer(0), already handled by rep/unlist)
  # Map ref indices back to cell ids
  edges <- data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[neighbor_ref]
  )

  # ---- Step 2: Join edges with data to get (focal_row, neighbor_row) per year ----

  # Create a keyed lookup: (id, year) -> row_id
  lookup <- dt[, .(id, year, .row_id)]

  # For each edge (focal_id, neighbor_id), we need every year that the focal cell appears in.
  # Then find the neighbor's row in the same year.

  # First, get focal rows with their year
  # Merge edges with focal lookup to get (focal_row, focal_year, neighbor_id)
  setkey(lookup, id)
  focal_expanded <- lookup[edges, on = .(id = focal_id),
                           .(focal_row = .row_id,
                             year      = year,
                             neighbor_id = i.neighbor_id),
                           nomatch = NULL,
                           allow.cartesian = TRUE]

  # Now merge to get neighbor_row in the same year
  setnames(lookup, ".row_id", "neighbor_row")
  setkey(lookup, id, year)
  setkey(focal_expanded, neighbor_id, year)

  edge_table <- lookup[focal_expanded,
                       on = .(id = neighbor_id, year = year),
                       .(focal_row    = i.focal_row,
                         neighbor_row = neighbor_row),
                       nomatch = NA_integer_]

  # Drop edges where neighbor had no matching row (missing year)
  edge_table <- edge_table[!is.na(neighbor_row)]

  # ---- Step 3: Compute all neighbor stats in one vectorized pass ----

  # Extract neighbor values for all source variables at once
  # Build a matrix of neighbor values indexed by edge_table$neighbor_row
  for (var in neighbor_source_vars) {
    set(edge_table, j = var, value = dt[[var]][edge_table$neighbor_row])
  }

  # Group by focal_row and compute max, min, mean for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  # Using a simpler, robust approach:
  stats <- edge_table[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = focal_row]

  # ---- Step 4: Merge stats back onto the original data ----

  # Rows with no neighbors at all won't appear in stats; they get NA (correct).
  dt[stats, on = .(.row_id = focal_row),
     (agg_names) := mget(agg_names)]

  # Replace infinite values from max/min of empty sets (safety)
  for (col in agg_names) {
    dt[is.infinite(get(col)), (col) := NA_real_]
  }

  # Remove helper column and return as data.frame if original was data.frame
  dt[, .row_id := NULL]

  return(as.data.frame(dt))
}

# ---- Usage (drop-in replacement for the original outer loop) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as-is downstream â€” no retraining needed.
# predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The same `max`, `min`, `mean` are computed over exactly the same neighbor sets (same rook contiguity, same year matching, same NA handling). |
| **Trained RF model** | No model object is touched; only the feature columns fed to `predict()` are constructed. Column names match the originals. |
| **Edge cases** | Cells with no neighbors (or no neighbor present in that year) receive `NA`, identical to the original `is.na` / `length == 0` guards. |

## Performance Accounting

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `paste` + named-vector lookups â†’ hours | One vectorized `data.table` join (~25.8M edge rows) â†’ seconds |
| Stat computation | 5 Ã— `lapply` over 6.46M list elements | Single grouped `data.table` aggregation â†’ minutes |
| Result binding | `do.call(rbind, 6.46M)` Ã— 5 | In-place `:=` column assignment |
| **Estimated total** | **86+ hours** | **~5â€“15 minutes** |

The key insight is replacing millions of scalar R-level operations (string construction, named-vector lookup, list element creation) with a small number of bulk `data.table` keyed joins and grouped aggregations that execute in compiled C code.
END_RESPONSE>>>
