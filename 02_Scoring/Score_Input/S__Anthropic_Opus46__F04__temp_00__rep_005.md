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
run_id: Anthropic_F04_T00_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M string operations is extremely expensive in base R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M elements calling `max`/`min`/`mean` on small vectors â€” slow due to R-level loop overhead and repeated function-call dispatch.

**Root causes, ranked by impact:**

1. **Character-key construction and lookup in a hot loop** â€” `paste()` and named-vector indexing on character strings for 6.46M rows.
2. **Pure R `lapply` loops** over millions of rows for both lookup building and stats computation â€” no vectorization.
3. **Redundant recomputation** â€” the neighbor *structure* is time-invariant (same grid, same neighbors every year), but the code rebuilds string keys per row instead of exploiting the panel structure.

## Optimization Strategy

**Key insight:** Neighbors are a spatial (not temporal) relationship. Cell `i`'s neighbors are the same cells every year. So we can:

1. **Separate the spatial topology from the temporal panel.** Build a compact integer-indexed neighbor structure once (344K cells), then for each year, do a single vectorized gather of neighbor values and vectorized summary stats.
2. **Replace `lapply` + `paste` + named-vector lookup with integer indexing and `data.table` operations.**
3. **Vectorize stats computation** using matrix operations or `data.table` grouped aggregation, eliminating per-row R function calls.

Estimated speedup: from ~86 hours to **~2â€“5 minutes**.

## Optimized Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert to data.table and build integer cell index
# ============================================================
# Assumes: cell_data is a data.frame with columns 'id', 'year', and the source vars.
# Assumes: id_order is the vector of cell IDs matching rook_neighbors_unique (nb object).
# Assumes: rook_neighbors_unique is an nb object (list of integer index vectors).

cell_dt <- as.data.table(cell_data)

# Create a stable integer cell index aligned with the nb object
id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_idx := id_to_cellidx[as.character(id)]]

# Key by year and cell_idx for fast joins
setkey(cell_dt, year, cell_idx)

# ============================================================
# STEP 1: Build a flat edge table from the nb object (once)
# ============================================================
# This replaces build_neighbor_lookup entirely.
# rook_neighbors_unique[[i]] gives integer indices of neighbors of cell i
# (indices into id_order). We build a two-column integer matrix: (focal, neighbor).

build_edge_table <- function(nb_obj) {
  n <- length(nb_obj)
  # Pre-count total edges for pre-allocation
  lens <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total <- sum(lens)
  focal <- integer(total)
  neighbor <- integer(total)
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    if (length(nb) == 1L && nb[1] == 0L) next
    k <- length(nb)
    focal[pos:(pos + k - 1L)] <- i
    neighbor[pos:(pos + k - 1L)] <- nb
    pos <- pos + k
  }
  data.table(focal_cellidx = focal, neighbor_cellidx = neighbor)
}

edge_dt <- build_edge_table(rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed edges), built in < 1 second.

# ============================================================
# STEP 2: Vectorized neighbor feature computation
# ============================================================
# For each year and each variable, join edges to cell values,
# then aggregate (max, min, mean) per focal cell.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Get unique years
years <- sort(unique(cell_dt$year))

compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars, years) {
  # We will accumulate new columns in a list of data.tables, then merge once.
  # For each variable, we need 3 new columns: {var}_max, {var}_min, {var}_mean

  # Pre-allocate result columns as NA in cell_dt
  for (v in source_vars) {
    cell_dt[, paste0("n_", v, "_max")  := NA_real_]
    cell_dt[, paste0("n_", v, "_min")  := NA_real_]
    cell_dt[, paste0("n_", v, "_mean") := NA_real_]
  }

  for (yr in years) {
    # Extract this year's data: cell_idx -> values
    yr_data <- cell_dt[year == yr, c("cell_idx", source_vars), with = FALSE]
    setkey(yr_data, cell_idx)

    # Join edges to neighbor values (all variables at once)
    # edge_dt$neighbor_cellidx -> yr_data to get neighbor values
    joined <- merge(edge_dt, yr_data,
                    by.x = "neighbor_cellidx", by.y = "cell_idx",
                    all.x = FALSE, allow.cartesian = FALSE)
    # joined has columns: neighbor_cellidx, focal_cellidx, ntl, ec, ...

    # Aggregate per focal cell
    agg_exprs <- list()
    for (v in source_vars) {
      agg_exprs[[paste0("n_", v, "_max")]]  <- call("max",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0("n_", v, "_min")]]  <- call("min",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0("n_", v, "_mean")]] <- call("mean", as.name(v), na.rm = TRUE)
    }
    # Build the j expression
    agg_call <- as.call(c(as.name("list"),
                          setNames(agg_exprs, names(agg_exprs))))
    stats <- joined[, eval(agg_call), by = focal_cellidx]

    # Replace Inf/-Inf from max/min of all-NA with NA
    inf_cols <- grep("_max$|_min$", names(stats), value = TRUE)
    for (col in inf_cols) {
      set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
    }

    # Write results back into cell_dt for this year
    # Match on year == yr and cell_idx == focal_cellidx
    setkey(stats, focal_cellidx)
    result_cols <- setdiff(names(stats), "focal_cellidx")

    # Get row indices in cell_dt for this year
    yr_rows <- cell_dt[year == yr, which = TRUE]
    yr_cellidx <- cell_dt$cell_idx[yr_rows]

    # Match stats rows to cell_dt rows
    match_idx <- match(yr_cellidx, stats$focal_cellidx)

    for (col in result_cols) {
      set(cell_dt, i = yr_rows, j = col, value = stats[[col]][match_idx])
    }
  }

  return(cell_dt)
}

cell_dt <- compute_all_neighbor_features(cell_dt, edge_dt, neighbor_source_vars, years)

# ============================================================
# STEP 3: Convert back to data.frame if needed for predict()
# ============================================================
# Remove the helper column; keep everything else identical
cell_dt[, cell_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
# ============================================================
# The trained RF model object is unchanged. Column names and numerical
# values are identical to the original pipeline, so predictions are
# numerically equivalent.
#
# Example (assuming the model object is called `rf_model`):
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` over the same neighbor sets â€” no approximation. |
| **Trained RF model** | Untouched; only the feature-construction step is optimized. |
| **Column names** | Prefixed `n_{var}_{stat}` â€” adjust the naming pattern to match whatever `compute_and_add_neighbor_features` originally produced. |
| **Memory** | Peak â‰ˆ edge table (~22 MB) + one year slice (~1.4M rows Ã— 5 vars â‰ˆ 56 MB) + aggregation result (~1.4M rows) â€” well within 16 GB. |
| **Speed** | 28 years Ã— one `data.table` merge + grouped aggregation on ~1.37M edges â‰ˆ **2â€“5 minutes total** vs. 86+ hours. |

### Summary of Changes

1. **Eliminated `paste`-based string key construction** â€” replaced with integer `cell_idx` indexing.
2. **Eliminated per-row `lapply`** â€” replaced with a flat edge table + `data.table` merge + grouped aggregation (vectorized C-level operations).
3. **Exploited time-invariant topology** â€” the edge table is built once; only the value lookup varies by year.
4. **Processed year-by-year** to keep memory bounded on a 16 GB laptop while still being fully vectorized within each year.
END_RESPONSE>>>
