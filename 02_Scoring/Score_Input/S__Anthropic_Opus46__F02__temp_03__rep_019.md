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
run_id: Anthropic_F02_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` conversion and named-vector lookup (`id_to_ref`).
- `paste()` to build neighbor keys for that row's year.
- Named-vector lookup into `idx_lookup` (a 6.46M-length named character vector â€” each lookup is O(n) hash probe on a very large table).

This means roughly **6.46M Ã— k** hash lookups on a multi-million-entry named vector (where k â‰ˆ average neighbor count ~4 for rook contiguity). Named vectors in R use linear-probe hashing that degrades badly at this scale. The result is a list of 6.46M integer vectors â€” itself a large, fragmented memory object.

### 2. `compute_neighbor_stats` â€” repeated `lapply` over the 6.46M-element lookup list, once per variable

Each call iterates over 6.46M list elements, subsetting and computing `max/min/mean`. With 5 variables this is 5 Ã— 6.46M iterations. The overhead of R-level `lapply` with anonymous functions, per-element `is.na` filtering, and `c()` allocation is substantial.

### Combined effect

~86+ hours is consistent with billions of R-level interpreted operations on large named vectors. Memory pressure comes from the 6.46M-element list of integer vectors (`neighbor_lookup`) plus the repeated `do.call(rbind, ...)` on a 6.46M-row matrix.

---

## Optimization Strategy

The key insight: **replace the per-row, per-year neighbor lookup with a vectorized merge/join, and replace the per-row stat computation with a grouped `data.table` aggregation.**

| Current approach | Optimized approach |
|---|---|
| Named-vector hash lookup per row | `data.table` keyed equi-join (C-level binary search) |
| 6.46M-element R list for neighbor_lookup | No list â€” neighbor stats computed directly via join + group-by |
| `lapply` + anonymous function per row per variable | Single vectorized `data.table` grouped aggregation per variable |
| `paste` keys | Integer compound key (`id`, `year`) â€” no string allocation |
| 5 separate passes over the lookup list | Can be combined into one pass or remain 5 fast passes |

**Expected speedup:** from ~86 hours to roughly 5â€“15 minutes, well within 16 GB RAM.

**What is preserved:**
- The trained Random Forest model (untouched).
- The original numerical estimand: for each cell-year row and each neighbor variable, the max, min, and mean of that variable across rook neighbors are identical to the original code's output.

---

## Working R Code

```r
# ============================================================
# Optimized neighbor-feature pipeline using data.table
# Drop-in replacement for build_neighbor_lookup +
# compute_neighbor_stats + outer loop
# ============================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ----------------------------------------------------------
  # 1. Build an edge table from the nb object (once)
  #    Each entry in rook_neighbors_unique[[i]] gives the

  #    indices (into id_order) of neighbors of id_order[i].
  # ----------------------------------------------------------
  from_idx <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0L)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  # edges now has ~1.37M rows (directed rook pairs)

  # ----------------------------------------------------------
  # 2. Convert cell_data to data.table (in-place if possible)
  # ----------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Ensure original row order is recoverable
  cell_data[, .row_order := .I]

  # ----------------------------------------------------------
  # 3. For each variable, join neighbors and aggregate
  # ----------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    message("Computing neighbor features for: ", var_name)

    # Subset to only the columns we need for the join
    # (keeps memory low â€” only one numeric column at a time)
    sub <- cell_data[, .(id, year, val = get(var_name))]

    # Key the subset for fast join
    setkey(sub, id)

    # Join edges â†’ neighbor rows:
    #   For every (focal_id, neighbor_id) pair, attach the
    #   neighbor's year and value.
    #   Then join back to focal rows to match on year.
    # Strategy:
    #   neighbor_data = edges â‹ˆ sub  on neighbor_id = id
    #   This gives (focal_id, neighbor_id, year, val)
    #   Then group by (focal_id, year) â†’ max, min, mean

    neighbor_data <- merge(
      edges,
      sub,
      by.x = "neighbor_id",
      by.y = "id",
      allow.cartesian = TRUE   # each neighbor has 28 year-rows
    )
    # neighbor_data columns: neighbor_id, focal_id, year, val
    # rows â‰ˆ 1.37M edges Ã— 28 years â‰ˆ 38.5M (fits in RAM)

    # Drop NA values before aggregation (matches original logic)
    neighbor_data <- neighbor_data[!is.na(val)]

    # Aggregate
    stats <- neighbor_data[,
      .(nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)),
      keyby = .(focal_id, year)
    ]

    # Rename to match the convention used downstream
    # (adjust naming convention to match your pipeline)
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)
    setnames(stats,
             c("nb_max", "nb_min", "nb_mean"),
             c(max_col,  min_col,  mean_col))

    # Merge stats back into cell_data
    cell_data <- merge(
      cell_data,
      stats,
      by.x = c("id", "year"),
      by.y = c("focal_id", "year"),
      all.x = TRUE,
      sort = FALSE
    )

    # Free memory
    rm(neighbor_data, stats, sub)
    gc()
  }

  # ----------------------------------------------------------
  # 4. Restore original row order and clean up
  # ----------------------------------------------------------
  setorder(cell_data, .row_order)
  cell_data[, .row_order := NULL]

  return(cell_data)
}


# ============================================================
# Usage â€” replaces the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains the new columns:
#   nb_max_ntl, nb_min_ntl, nb_mean_ntl,
#   nb_max_ec,  nb_min_ec,  nb_mean_ec,
#   ... etc.
#
# Feed cell_data (with all ~110 predictors + new neighbor features)
# directly into predict(trained_rf_model, cell_data) as before.
# The trained Random Forest model is unchanged.
```

---

## Why This Works and What Changed

| Concern | Explanation |
|---|---|
| **Correctness / estimand preservation** | For every (cell, year) pair, the set of neighbor values considered is identical: rook neighbors present in the data for that same year, with NAs removed. `max`, `min`, `mean` are computed on the same sets â†’ numerically identical results. |
| **No model retraining** | Only the feature-engineering step is optimized. The RF model object is never touched. The output columns have the same values, so `predict()` produces the same predictions. |
| **Speed** | `data.table` keyed merges and grouped aggregations run in C with radix-sort joins. The ~38.5M-row intermediate table per variable is processed in seconds, not hours. Total wall time: ~5â€“15 minutes for all 5 variables. |
| **Memory** | Peak memory â‰ˆ `cell_data` (~5.4 GB for 6.46M Ã— 110 float64 cols) + one 38.5M Ã— 4-column intermediate (~1.2 GB) â‰ˆ **~7 GB peak**, well within 16 GB. The intermediate is freed after each variable. |
| **`allow.cartesian = TRUE`** | Required because one edge maps to 28 year-rows. This is intentional and controlled. |

This replaces the estimated 86+ hour runtime with a pipeline that completes in minutes on the same 16 GB laptop, while producing bit-identical neighbor feature columns.
END_RESPONSE>>>
