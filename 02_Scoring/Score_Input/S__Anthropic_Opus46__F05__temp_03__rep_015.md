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
run_id: Anthropic_F05_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: Row-wise `lapply` over 6.46M rows with string operations

1. **String-key construction is O(N) per row inside an O(N) loop â†’ O(NÂ²) string work.** The `idx_lookup` is built once (good), but inside the `lapply`, for every one of the ~6.46M rows, `paste()` is called to construct neighbor keys. Each row has ~4 rook neighbors on average (interior cells), so that's ~25.8M `paste()` calls plus ~25.8M named-vector lookups by string. The named-vector lookup itself is hash-based (O(1) amortized), but the string construction and R-level loop overhead dominate.

2. **The neighbor lookup is year-invariant but recomputed per cell-year.** The rook-neighbor topology is purely spatial â€” it doesn't change across years. Yet `build_neighbor_lookup` produces one entry per cell-year row (6.46M entries), each time re-discovering the same spatial neighbors and just filtering to those present in the same year. If the panel is balanced (344,208 cells Ã— 28 years â‰ˆ 9.64M, with 6.46M present), the neighbor structure only needs to be resolved once per cell, then broadcast across years.

3. **`compute_neighbor_stats` is efficient but called 5 times sequentially.** Each call iterates over the 6.46M-element neighbor lookup list. This is fine algorithmically but can be vectorized.

**Summary:** The string-keyed lookup is the visible symptom; the real disease is that a **year-invariant spatial topology** is being resolved row-by-row across all cell-years via expensive string operations, when it should be resolved once per cell and then joined via integer indexing.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor resolution | Per cell-year, string paste + hash lookup (6.46M iterations) | Per cell only (344K iterations), integer-indexed |
| Year matching | Implicit via string keys | Vectorized merge via `data.table` equi-join |
| Neighbor stats | R-level `lapply` over 6.46M list elements, once per variable | Vectorized `data.table` grouped aggregation, all variables at once |
| Complexity | ~O(N Ã— avg_neighbors) string ops in R loop | ~O(N Ã— avg_neighbors) integer ops, vectorized in C via `data.table` |

**Expected speedup:** From ~86+ hours to **minutes** (typically 2â€“10 minutes depending on I/O and RAM pressure).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec,
#                pop_density, def, usd_est_n2 (and others)
#   - rook_neighbors_unique: spdep nb object (list of integer index vectors)
#   - id_order: vector of cell IDs in the order matching rook_neighbors_unique
#
# Preserves: trained Random Forest model (no retraining), original numerical
#            estimand (max, min, mean of each neighbor variable).
# =============================================================================

library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                 "def", "usd_est_n2")) {

  # --- Step 1: Build spatial edge list ONCE (year-invariant) -----------------
  # rook_neighbors_unique[[i]] contains integer indices into id_order
  # for the neighbors of id_order[i].

  message("Step 1: Building spatial edge list...")

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L for cells with no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(
      focal_id    = id_order[i],
      neighbor_id = id_order[nb_idx]
    )
  }))

  message(sprintf("  Edge list: %s directed neighbor pairs", format(nrow(edge_list), big.mark = ",")))

  # --- Step 2: Convert cell_data to data.table if needed --------------------
  message("Step 2: Preparing data.table...")

  dt <- as.data.table(cell_data)

  # Create a minimal neighbor-value table with only the columns we need
  # for the join: id, year, and the source variables.
  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_vals <- dt[, ..keep_cols]

  # --- Step 3: Join edge list with panel data to get neighbor values ---------
  # For each (focal_id, year), find all neighbors present in that year and
  # retrieve their variable values.
  #
  # This replaces the entire build_neighbor_lookup + compute_neighbor_stats
  # pipeline with a single vectorized merge + grouped aggregation.

  message("Step 3: Joining neighbors with panel data...")

  # Merge edge list with focal-year combinations to get (focal_id, year, neighbor_id)
  # Then merge with dt_vals on (neighbor_id, year) to get neighbor values.

  # First, get the unique (focal_id, year) combinations from the data
  focal_years <- dt[, .(focal_id = id, year)]

  # Join: for each focal cell-year, expand to all spatial neighbors
  # focal_years Ã— edge_list on focal_id
  setkey(edge_list, focal_id)
  setkey(focal_years, focal_id)

  # This is the big join: each focal cell-year gets its neighbor list
  expanded <- edge_list[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # Result columns: focal_id, neighbor_id, year

  message(sprintf("  Expanded neighbor-year pairs: %s rows",
                  format(nrow(expanded), big.mark = ",")))

  # Now join to get neighbor variable values
  setnames(dt_vals, "id", "neighbor_id")
  setkey(dt_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  expanded_vals <- dt_vals[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Result: focal_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2

  # --- Step 4: Compute grouped aggregation (max, min, mean) -----------------
  message("Step 4: Computing neighbor statistics...")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  names(agg_exprs) <- agg_names

  # Perform grouped aggregation
  stats <- expanded_vals[,
    lapply(agg_exprs, eval, envir = .SD),
    by = .(focal_id, year),
    .SDcols = neighbor_source_vars
  ]

  # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen if nomatch=NULL

  # filtered them, but be safe)
  for (col_name in agg_names) {
    vals <- stats[[col_name]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats, j = col_name, value = vals)
  }

  # --- Step 5: Merge statistics back to original data -----------------------
  message("Step 5: Merging results back to cell_data...")

  # Rename focal_id back to id for the merge
  setnames(stats, "focal_id", "id")

  # Remove any pre-existing neighbor columns from dt to avoid duplication
  existing_neighbor_cols <- intersect(names(dt), agg_names)
  if (length(existing_neighbor_cols) > 0) {
    dt[, (existing_neighbor_cols) := NULL]
  }

  # Left join: preserves all original rows, NAs for cells with no neighbors
  setkey(dt, id, year)
  setkey(stats, id, year)
  dt <- stats[dt, on = c("id", "year")]

  message("Done. Returning augmented data.table.")
  return(dt)
}

# =============================================================================
# USAGE
# =============================================================================
# cell_data <- optimize_neighbor_features(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched â€” just use cell_data for
# # predict(rf_model, newdata = cell_data) as before.
```

---

## Why This Preserves the Original Numerical Estimand

| Original | Optimized | Equivalence |
|----------|-----------|-------------|
| `max(neighbor_vals)` | `max(v, na.rm=TRUE)` grouped by `(focal_id, year)` | Identical: same set of neighbor values, same function |
| `min(neighbor_vals)` | `min(v, na.rm=TRUE)` grouped by `(focal_id, year)` | Identical |
| `mean(neighbor_vals)` | `mean(v, na.rm=TRUE)` grouped by `(focal_id, year)` | Identical |
| NAs when no neighbors present | `nomatch=NULL` drops missing â†’ left join reintroduces `NA` | Identical |

The neighbor set for each `(cell, year)` is determined by the same spatial topology (`rook_neighbors_unique`) intersected with the same year's available data â€” exactly as the original code does via string-key filtering.

---

## Performance Estimate

| Operation | Estimated Time |
|-----------|---------------|
| Edge list construction (344K cells) | ~2 seconds |
| Cartesian join (~6.46M Ã— ~4 neighbors) | ~10â€“30 seconds |
| Equi-join on (neighbor_id, year) | ~10â€“30 seconds |
| Grouped aggregation (5 vars Ã— 3 stats) | ~30â€“90 seconds |
| Final merge | ~10 seconds |
| **Total** | **~2â€“5 minutes** |

This represents a speedup of roughly **1,000â€“2,500Ã—** over the original 86+ hour estimate, achieved by eliminating the R-level row-wise loop, all string operations, and replacing them with vectorized `data.table` joins and grouped aggregations executed in C.
END_RESPONSE>>>
