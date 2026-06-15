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
run_id: Anthropic_F04_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs character key strings via `paste()`,
4. Looks them up in `idx_lookup` (a named character vector of length ~6.46M),
5. Filters `NA`s.

**Key problems:**

- **`paste()` called ~6.46M times**, each generating a vector of neighbor keys â€” this is extremely expensive string allocation.
- **Named-vector lookup (`idx_lookup[neighbor_keys]`)** on a 6.46M-element named vector is O(n) hashing per call, repeated billions of times in aggregate (~6.46M rows Ã— ~4 neighbors average â‰ˆ 26M lookups, but the overhead per call is dominated by string construction and matching).
- **`lapply` over 6.46M elements in R** with non-trivial per-element work is inherently slow; R's interpreted loop overhead is massive here.
- **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements per variable (Ã—5 variables = ~32.3M iterations), each extracting, filtering, and summarizing â€” again slow in pure R.

**Estimated cost:** ~6.46M Ã— (string ops + hash lookups) for the lookup build, then ~32.3M summary operations. This easily accounts for the 86+ hour runtime on a laptop.

## Optimization Strategy

### Core Insight
The neighbor topology is **time-invariant** â€” a cell's spatial neighbors are the same in every year. We should:

1. **Build the neighbor lookup at the cell level (344K cells), not the cell-year level (6.46M rows).**
2. **Vectorize the stats computation using `data.table` grouping and matrix operations** instead of per-row `lapply`.
3. **Avoid all `paste()`-based key construction** â€” use integer indexing throughout.

### Approach

- Convert data to `data.table`, keyed by `(id, year)`.
- Explode the neighbor list into an edge table: `(cell_row, neighbor_id)`.
- Join to get neighbor row indices per year in a fully vectorized manner.
- Compute `max`, `min`, `mean` via `data.table` grouped aggregation â€” one pass per variable.

This replaces ~6.46M R-level iterations with a few vectorized joins and group-bys.

## Optimized R Code

```r
library(data.table)

#' Build a vectorized neighbor edge table and compute all neighbor features.
#' Preserves the original numerical estimand exactly (max, min, mean of
#' non-NA neighbor values; NA when no valid neighbors exist).
#'
#' @param cell_data       data.frame / data.table with columns `id`, `year`, and all source vars
#' @param id_order        integer vector of cell IDs in the same order as the nb object
#' @param neighbors       spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to summarize
#' @return data.table with original columns plus neighbor feature columns appended

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Assign a row index to every row (preserves original order) ---
  dt[, .row_idx := .I]

  # --- Step 2: Build cell-level edge list (time-invariant) ---
  #     For each cell index i in id_order, get its neighbor cell IDs.
  #     This is only 344,208 cells, very fast.
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1.37M rows (directed rook edges)

  # --- Step 3: Create a keyed lookup from (id, year) -> row_idx ---
  setkey(dt, id, year)

  # --- Step 4: Expand edges by year via a join ---
  #     For every (focal_id, year) row, we need the row indices of its neighbors
  #     in the same year.
  #
  #     Strategy: join edge_list to dt twice â€”

  #       (a) get focal row index + year
  #       (b) get neighbor row index for that (neighbor_id, year)

  # 4a. Get all (focal_id, year, focal_row_idx) combinations
  focal_dt <- dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

  # 4b. Join edges to focal rows to get (focal_row_idx, neighbor_id, year)
  #     This is an equi-join on focal_id.
  setkey(edge_list, focal_id)
  setkey(focal_dt, focal_id)
  expanded <- edge_list[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: focal_id, neighbor_id, year, focal_row_idx
  # Rows: ~1.37M edges Ã— 28 years â‰ˆ 38.5M (fits in 16 GB easily as integer columns)

  # 4c. Join to dt again to get neighbor_row_idx for (neighbor_id, year)
  neighbor_idx_lookup <- dt[, .(neighbor_id = id, year, neighbor_row_idx = .row_idx)]
  setkey(neighbor_idx_lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_idx_lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Now expanded has: focal_row_idx, neighbor_row_idx (NA if neighbor missing that year)

  # Drop rows where neighbor is missing (no data for that cell-year)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Keep only the columns we need to minimize memory
  expanded <- expanded[, .(focal_row_idx, neighbor_row_idx)]

  # --- Step 5: For each source variable, compute grouped stats vectorized ---
  for (var_name in neighbor_source_vars) {

    vals <- dt[[var_name]]

    # Attach neighbor values
    expanded[, nval := vals[neighbor_row_idx]]

    # Drop NAs in neighbor values for stats (matches original logic)
    valid <- expanded[!is.na(nval)]

    # Grouped aggregation â€” one vectorized pass
    stats <- valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row_idx]

    # Initialize result columns with NA (matches original behavior for cells with no valid neighbors)
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values by row index
    dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]

    # Clean up temp column
    expanded[, nval := NULL]
  }

  # --- Step 6: Clean up and return in original row order ---
  setorder(dt, .row_idx)
  dt[, .row_idx := NULL]

  return(dt)
}

# ============================================================
# USAGE (drop-in replacement for the original outer loop)
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Works and Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| Lookup build | 6.46M `lapply` iterations with `paste` + named-vector lookup | 344K-iteration edge list + two `data.table` keyed joins |
| Stats computation | 6.46M Ã— 5 = 32.3M `lapply` iterations with per-element subsetting | 5 vectorized `data.table` group-by aggregations over ~38.5M rows |
| String operations | ~6.46M `paste()` calls + ~26M character hash lookups | **Zero** â€” all integer indexing |
| Estimated runtime | **86+ hours** | **~5â€“15 minutes** on the same laptop |
| Peak memory | Lower per-step but slow | ~38.5M rows Ã— 2 int cols â‰ˆ 0.6 GB for the expanded edge table; fits in 16 GB |
| Numerical result | max/min/mean of non-NA neighbor values, NA if none | **Identical** â€” same aggregation functions, same NA handling |

The ~1000Ã— speedup comes from eliminating millions of interpreted R loop iterations and string operations in favor of `data.table`'s C-level keyed joins and grouped aggregation.
END_RESPONSE>>>
