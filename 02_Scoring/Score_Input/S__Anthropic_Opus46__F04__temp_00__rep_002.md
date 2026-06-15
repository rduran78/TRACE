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
run_id: Anthropic_F04_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated `paste` and named-vector lookups on a vector of ~6.46M names, is extremely slow. Named vector lookup in R degrades as the vector grows; at 6.46M entries, each probe is expensive, and you do ~1.37M neighbor probes per year Ã— 28 years.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each of the 6.46M iterations calls `vals[idx]`, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable). That is ~32.3 million R-level function invocations with per-row overhead.

### 3. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors

This is a well-known R anti-pattern. Binding millions of small vectors is very slow.

**Estimated cost breakdown:**
| Step | Calls | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | `paste`, named-vector hash lookups on 6.46M keys |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M iterations | per-row subsetting, NA removal, summary stats |
| `do.call(rbind, ...)` | 5 calls binding 6.46M rows | memory allocation / copying |

---

## Optimization Strategy

**Principle:** Replace row-level R loops with vectorized, column-level operations using `data.table` joins and grouped aggregation.

| Original approach | Optimized approach |
|---|---|
| Build a 6.46M-element named lookup vector, probe it row-by-row | Build an edge-list `data.table` and do a keyed equi-join |
| `lapply` over every row to gather neighbor indices | A single merge produces all (focal-row, neighbor-row) pairs |
| Per-row `max`/`min`/`mean` in R | Grouped `data.table` aggregation: `[, .(max, min, mean), by=focal_row]` |
| `do.call(rbind, ...)` on millions of tiny vectors | Result is already a `data.table`; assign columns directly |

**Expected speedup:** From ~86+ hours to roughly **5â€“15 minutes** on the same laptop, because:
- The join is O(E) where E â‰ˆ 1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” large but handled in C by `data.table`.
- Grouped aggregation over 3 statistics Ã— 5 variables is extremely fast in `data.table`.
- No R-level per-row interpretation overhead.

**Memory:** The edge table is ~38.4M rows Ã— 2 integer columns â‰ˆ 0.6 GB. Joined with one numeric variable at a time, peak overhead is ~1 GB, well within 16 GB.

**Numerical equivalence:** The optimized code computes exactly the same `max`, `min`, `mean` of the same neighbor values, preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # -----------------------------------------------------------
  # Step 1: Build a directed edge list from the nb object

  #         (done once; ~1.37M edges)
  # -----------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  # -----------------------------------------------------------
  # Step 2: Convert cell_data to data.table (if not already)
  #         and create a row-index column for later assignment
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # Minimal keyed table for joining: (id, year) -> row index + variable values
  # We will join variable-by-variable to limit peak memory.

  # -----------------------------------------------------------
  # Step 3: For each source variable, compute neighbor stats
  #         via a single keyed join + grouped aggregation
  # -----------------------------------------------------------
  # Prepare a small focal table: for every row, its (id, year, .row_idx)
  focal <- dt[, .(focal_id = id, year, .row_idx)]

  # Join focal rows to their neighbor cell IDs (cross-year broadcast)
  # focal_id -> neighbor_id via edges, keeping year from focal
  setkey(edges, focal_id)
  setkey(focal, focal_id)

  # Merge: each focal row gets its neighbor_ids (same year implied)
  # Result: one row per (focal_row, neighbor_cell) pair, carrying the year
  focal_neighbors <- edges[focal, on = .(focal_id),
                           allow.cartesian = TRUE,
                           nomatch = NULL]
  # Columns: focal_id, neighbor_id, year, .row_idx
  # .row_idx refers to the focal row in dt

  for (var_name in neighbor_source_vars) {

    message("Computing neighbor features for: ", var_name)

    # Build a lookup: (id, year) -> value
    val_table <- dt[, .(neighbor_id = id, year, .val = get(var_name))]
    setkey(val_table, neighbor_id, year)

    # Join neighbor values onto the edge table
    joined <- val_table[focal_neighbors, on = .(neighbor_id, year),
                        nomatch = NA]
    # Columns: neighbor_id, year, .val, focal_id, .row_idx

    # Drop rows where the neighbor value is NA (matches original logic)
    joined <- joined[!is.na(.val)]

    # Grouped aggregation by focal row
    stats <- joined[, .(nb_max  = max(.val),
                        nb_min  = min(.val),
                        nb_mean = mean(.val)),
                    by = .(.row_idx)]

    # Initialise new columns to NA (rows with no valid neighbors stay NA)
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed stats back by row index
    dt[stats$.row_idx, (max_col)  := stats$nb_max]
    dt[stats$.row_idx, (min_col)  := stats$nb_min]
    dt[stats$.row_idx, (mean_col) := stats$nb_mean]

    # Free memory
    rm(val_table, joined, stats)
  }

  # Clean up helper column
  dt[, .row_idx := NULL]

  # Return as data.frame if the downstream RF predict() expects one
  return(as.data.frame(dt))
}

# -----------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# -----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with `paste` + named-vector probes | One `rbindlist` to build ~1.37M-row edge `data.table` |
| **Neighbor stat computation** | 6.46M-iteration `lapply` Ã— 5 vars | One keyed `data.table` join (cartesian, ~38.4M rows) + grouped `[, .(max,min,mean), by=.row_idx]` per variable |
| **Result assembly** | `do.call(rbind, <6.46M-element list>)` Ã— 5 | Direct column assignment by integer index |
| **Estimated wall time** | 86+ hours | ~5â€“15 minutes |
| **Peak RAM delta** | Moderate (millions of small R objects + GC pressure) | ~1â€“2 GB (dense edge table + one joined column at a time) |
| **Numerical result** | `max`, `min`, `mean` of non-NA rook-neighbor values | Identical |
| **Trained RF model** | â€” | Untouched |
END_RESPONSE>>>
