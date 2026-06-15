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
run_id: Anthropic_F04_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character matching (`id_to_ref[as.character(...)]`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character key strings by pasting cell ID and year (`paste(..., sep="_")`).
4. Looks up row indices in a named character vector (`idx_lookup[neighbor_keys]`).

This means ~6.46 million iterations, each performing **character coercion, string pasting, and named-vector lookup** (which is O(n) hash-table probing on character keys). With ~1.37M neighbor relationships spread across 28 years, that's tens of millions of string operations. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M entries five times (once per variable), each time extracting, filtering NAs, and computing summary stats in pure R.

**Root causes:**
- Row-level `lapply` in R (interpreted loop over millions of rows).
- Repeated `paste()`/character key construction and named-vector lookups (slow hashing).
- `compute_neighbor_stats` uses per-row `lapply` with R-level `max/min/mean` calls instead of vectorized operations.
- The lookup is rebuilt monolithically instead of exploiting the panel structure (same neighbor topology repeats every year).

## Optimization Strategy

**Key insight:** The spatial neighbor graph is *time-invariant*. Cell `i`'s neighbors are the same in every year. Therefore, we should:

1. **Separate spatial topology from temporal indexing.** Build a cell-to-cell neighbor edge list once (~1.37M edges), then join it to the panel by year using `data.table` equi-joins â€” fully vectorized, no per-row loop.
2. **Replace `build_neighbor_lookup` entirely** with a vectorized `data.table` merge approach that expands the neighbor edge list across years.
3. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation (vectorized C-level `max`, `min`, `mean`) â€” no `lapply`.
4. **Process all 5 variables in one pass** per aggregation to avoid redundant joins.

This reduces the runtime from ~86+ hours to minutes.

## Optimized R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a time-invariant edge list from the nb object (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ---------------------------------------------------------------
# 2. Compute all neighbor features in a vectorized fashion
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  # Convert to data.table if not already; work on a copy to be safe
  dt <- as.data.table(copy(cell_data))

  # Ensure a row-order key so we can restore original order later
  dt[, .row_order := .I]

  # Step 1: build the edge list (time-invariant, ~1.37M rows)
  edges <- build_edge_list(id_order, neighbors)

  # Step 2: expand edges across years via a merge with the panel.
  #
  # We need, for every (focal_id, year) row, the values of each
  # source variable at every (neighbor_id, year) row.
  #
  # Strategy:

  #   a) Create a slim table: id, year, + source vars.
  #   b) Join edges to that table on neighbor_id == id to get
  #      neighbor values; this is keyed by (focal_id, year).
  #   c) Aggregate (max, min, mean) grouped by (focal_id, year).
  #   d) Join aggregated stats back to dt.

  keep_cols <- c("id", "year", neighbor_source_vars)
  slim <- dt[, ..keep_cols]

  # Keyed join: for every edge, attach neighbor values per year
  # Result has one row per (focal_id, neighbor_id, year) combination
  setkey(slim, id, year)
  neighbor_vals <- edges[slim,
                         on = .(neighbor_id = id),
                         allow.cartesian = TRUE,
                         nomatch = NULL]
  # neighbor_vals now has columns: focal_id, neighbor_id, year, + source vars

  # Step 3: aggregate by (focal_id, year)
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <-
      bquote(fifelse(all(is.na(.(v_sym))), NA_real_, max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_min_", v)]]  <-
      bquote(fifelse(all(is.na(.(v_sym))), NA_real_, min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_mean_", v)]] <-
      bquote(fifelse(all(is.na(.(v_sym))), NA_real_, mean(.(v_sym), na.rm = TRUE)))
  }
  # Build the call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  agg <- neighbor_vals[, eval(agg_call), by = .(focal_id, year)]

  # Step 4: merge aggregated stats back onto the main table
  setkey(agg, focal_id, year)
  setkey(dt, id, year)
  dt <- agg[dt, on = .(focal_id = id, year = year)]

  # Restore original row order and clean up
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  # Rename focal_id back to id (the join put it as focal_id)
  setnames(dt, "focal_id", "id")

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# 3. Drop-in replacement for the outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains nb_max_ntl, nb_min_ntl, nb_mean_ntl, â€¦ for all 5 vars.
# The trained Random Forest model can be applied directly to cell_data as before,
# since the column names and numerical values are identical to the original pipeline.
```

## Why This Works and What Changes

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M interpreted R iterations with string ops | Eliminated; replaced by vectorized edge expansion |
| Neighbor stats | 5 Ã— 6.46M `lapply` iterations | Single `data.table` grouped aggregation (C-level) |
| String key creation | ~30M+ `paste()` calls | Zero; uses integer/numeric join keys |
| Passes over data | 5 (one per variable) | 1 (all variables aggregated together) |
| Memory pattern | Named character vector of 6.46M entries | Integer-keyed `data.table` joins (more cache-friendly) |
| Expected runtime | ~86+ hours | ~2â€“10 minutes on 16 GB laptop |
| Numerical output | Original max/min/mean | Identical max/min/mean (preserves estimand) |
| Trained RF model | N/A | Untouched; column names match |

**Memory note:** The Cartesian expansion `edges Ã— years` produces ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows, each carrying 5 numeric columns (~1.5 GB). This fits within 16 GB RAM. If memory is tight, process years in batches:

```r
# Optional: batch by year to reduce peak memory
year_list <- unique(dt$year)
agg_list <- lapply(year_list, function(yr) {
  slim_yr <- slim[year == yr]
  setkey(slim_yr, id)
  nv <- edges[slim_yr, on = .(neighbor_id = id),
              allow.cartesian = TRUE, nomatch = NULL]
  nv[, eval(agg_call), by = .(focal_id, year)]
})
agg <- rbindlist(agg_list)
```

This caps peak memory at ~1/28th of the full expansion (~55 MB per batch) while remaining fully vectorized within each year.
END_RESPONSE>>>
