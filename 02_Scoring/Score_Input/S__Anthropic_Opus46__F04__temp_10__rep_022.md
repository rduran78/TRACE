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
run_id: Anthropic_F04_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction stages:

### 1. `build_neighbor_lookup` â€” O(N) row-level `lapply` with per-row string pasting and named-vector lookups

For each of the **~6.46 million rows**, this function:
- Converts the cell ID to a reference index via a named character vector lookup.
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes each neighbor cell ID with the current row's year to form string keys (one `paste` call per neighbor per row â€” roughly **1.37 million Ã— 2 directions Ã— 28 years â‰ˆ tens of millions of string operations**).
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) on a **6.46 million-element named character vector** for every row. Named vector lookup in R is hash-based but still carries per-call overhead that is devastating at this scale.
- Returns an integer vector, per row.

**Cost:** ~6.46M iterations Ã— (string construction + hash lookups per neighbor) â‰ˆ many hours alone.

### 2. `compute_neighbor_stats` â€” O(N) row-level `lapply` with per-row subsetting, NA removal, and summary stats

For each of the ~6.46M rows, this extracts neighbor values, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), yielding ~32.3M R-level iterations.

**Cost:** The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive (repeated memory allocation).

### Summary of root causes

| Cause | Impact |
|---|---|
| Row-level `lapply` over 6.46M rows (interpreted R loop) | Catastrophic |
| Per-row `paste()` string key construction | Very high |
| Per-row named-vector hash lookup on a 6.46M-key table | Very high |
| `do.call(rbind, ...)` on a 6.46M-element list | High |
| Repeated per-variable scan (5Ã—) over the full lookup | High |
| No vectorization or join-based approach | Fundamental design issue |

**Estimated serial time:** 86+ hours is consistent with this analysis.

---

## Optimization Strategy

**Replace all row-level R loops with a single vectorized `data.table` equi-join + grouped aggregation.**

The key insight: the neighbor lookup is conceptually a **join**. For every `(cell_id, year)` row, we want the values of neighboring cells *in the same year*. This is a standard equi-join on `(neighbor_id, year)`, followed by a grouped aggregation (`max`, `min`, `mean`).

### Steps

1. **Build an edge table** (a two-column data.table of `id â†’ neighbor_id`) from the `nb` object â€” done once, vectorized, ~1.37M rows.
2. **Join** the edge table with the panel data on `(neighbor_id, year)` to fetch neighbor values â€” a single `data.table` merge, fully vectorized in C.
3. **Aggregate** by `(id, year)` to get `max`, `min`, `mean` â€” a single `data.table` grouped operation per variable.
4. **Merge** the results back into the main data.

This eliminates all 6.46M-iteration R loops, all string key construction, and all per-row hash lookups.

**Expected speedup:** From 86+ hours to **minutes** (roughly 2â€“10 minutes depending on disk I/O and RAM pressure). All numerical results are preserved exactly.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0 â€” Convert the nb object to a vectorized edge list (done once)
# ==============================================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a spdep::nb object: list of integer index vectors
  # id_order is the vector mapping positional index -> cell ID
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ==============================================================================
# STEP 1 â€” Compute all neighbor features in a vectorized fashion
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Convert to data.table if needed (modifies in place for speed) ----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Build edge table once --------------------------------------------------
  edges <- build_edge_table(id_order, rook_neighbors_unique)

  # --- Ensure keys for fast joins ---------------------------------------------
  #     We will join edges with cell_data on (neighbor_id == id, year == year)
  #     so we need cell_data keyed by (id, year).
  setkey(cell_data, id, year)

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Subset only the columns we need for the join target
    # (id, year, and the current variable)
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # --- Join: for every (id, year) get all neighbor values -------------------
    #     edges has (id, neighbor_id).
    #     We add year via merge with cell_data's (id, year) universe,
    #     then look up neighbor values.
    #
    #     Efficient approach: expand edges Ã— years using cell_data's own rows
    #     as the driver.

    # Driver: every (id, year) row with its neighbors
    # Merge cell_data's (id, year) with edges on id -> gives (id, year, neighbor_id)
    driver <- cell_data[, .(id, year)]
    setkey(driver, id)
    setkey(edges, id)
    expanded <- edges[driver, on = "id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded columns: id, neighbor_id, year

    # Now look up the neighbor's value for that year
    setkey(expanded, neighbor_id, year)
    expanded[val_dt, on = c(neighbor_id = "id", "year"), neighbor_val := i.val]

    # --- Aggregate by (id, year) -----------------------------------------------
    stats <- expanded[!is.na(neighbor_val),
                      .(nbr_max  = max(neighbor_val),
                        nbr_min  = min(neighbor_val),
                        nbr_mean = mean(neighbor_val)),
                      by = .(id, year)]

    # Rename columns to match original naming convention
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")
    setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
             c(max_col, min_col, mean_col))

    # --- Merge back into cell_data ---------------------------------------------
    # Remove old columns if they exist (idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_data)) cell_data[, (col) := NULL]
    }

    setkey(stats, id, year)
    setkey(cell_data, id, year)
    cell_data <- stats[cell_data, on = c("id", "year")]
    # Left join: NAs where a cell has no valid neighbors (matches original behavior)

    # Clean up to free RAM
    rm(expanded, stats, val_dt, driver)
    gc()
  }

  return(cell_data)
}

# ==============================================================================
# STEP 2 â€” Drop-in replacement for the original outer loop
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now has the same neighbor feature columns as the original pipeline.
# The trained Random Forest model can be applied directly to cell_data for prediction.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per `(id, year)` as the original code. The `data.table` join replicates the same `(cell_id, year) â†’ neighbor_ids â†’ same-year values` logic. |
| **NA handling** | `expanded[!is.na(neighbor_val), ...]` mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Rows with no valid neighbors receive `NA` via the left join, matching `return(c(NA, NA, NA))`. |
| **Trained RF model** | No model retraining. The code only reconstructs the same feature columns that the model expects. |

---

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M `lapply` + paste + hash) | ~seconds (vectorized `rep`/`unlist`, 1.37M rows) |
| Compute stats (Ã—5 vars) | ~80+ hours (32.3M R-loop iterations) | ~2â€“8 minutes (5 Ã— data.table join + grouped agg in C) |
| **Total** | **86+ hours** | **~3â€“10 minutes** |

Peak RAM usage will spike during the `allow.cartesian = TRUE` join (the expanded table has ~6.46M Ã— avg. neighbors â‰ˆ 26M rows per variable), but with 16 GB this is well within budget (~2â€“3 GB per iteration, freed each loop).
END_RESPONSE>>>
