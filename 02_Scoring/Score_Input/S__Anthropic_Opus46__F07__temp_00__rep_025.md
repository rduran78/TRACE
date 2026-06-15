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
run_id: Anthropic_F07_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of R's interpreted loop is enormous at this scale.
3. **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements, extracting subsets of a vector by index. This is lighter but still slow in a pure-R loop.

**Estimated cost:** ~6.46M iterations Ã— (string ops + named-vector lookup in a 6.46M-entry table) Ã— 5 variables = the 86+ hour runtime.

**Key insight:** The neighbor topology is *time-invariant*. There are only 344,208 spatial cells with ~1.37M directed rook-neighbor pairs. The year dimension simply replicates this structure. We should:

- Work at the **cell level** (344K) not the **cell-year level** (6.46M).
- Use **vectorized joins** (via `data.table`) instead of named-vector lookups.
- Compute neighbor stats via a **sparse adjacency edge-list join**, not per-row `lapply`.

## Optimization Strategy

1. **Convert the `nb` object to a directed edge-list** (from_id, to_id) â€” ~1.37M rows, done once.
2. **Convert `cell_data` to a `data.table`**, keyed on `(id, year)`.
3. **For each source variable**, do a single vectorized merge of the edge-list with the data to get all neighbor values, then **group-by `(from_id, year)`** to compute `max`, `min`, `mean` â€” all in `data.table`, fully vectorized in C.
4. **Left-join** the results back to `cell_data`.

This replaces 6.46M Ã— 5 interpreted R iterations with ~5 vectorized `data.table` group-by operations over ~1.37M Ã— 28 â‰ˆ 38.5M edge-year rows. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert nb object to directed edge-list (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique is an nb object; id_order maps position â†’ cell id
nb_to_edge_list <- function(nb_obj, id_order) {
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor placeholders (spdep uses integer(0) but be safe)
  valid <- to > 0L
  data.table(
    from_id = id_order[from[valid]],
    to_id   = id_order[to[valid]]
  )
}

edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows: (from_id, to_id)

# â”€â”€ Step 1: Convert panel to data.table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# â”€â”€ Step 2: Vectorized neighbor-stat computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(dt, edges, var_name) {
  # Build a slim table: every cell-year's value for this variable
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # Expand edges Ã— years: for each (from_id, to_id) pair, join the

  # neighbor's (to_id) value in each year.
  # This is a keyed join â€” very fast in data.table.
  edge_vals <- edges[val_dt, on = "to_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_vals columns: from_id, to_id, year, val

  # Aggregate: for each (from_id, year), compute max/min/mean of neighbor vals
  agg <- edge_vals[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(from_id, year)
  ]

  # Rename columns to match original convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))
  setnames(agg, "from_id", "id")
  setkey(agg, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(dt)) dt[, (col) := NULL]
  }

  # Left-join back to the main table
  dt <- agg[dt, on = .(id, year)]
  setkey(dt, id, year)
  dt
}

# â”€â”€ Step 3: Loop over the 5 source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edges, var_name)
  gc()
}

# â”€â”€ Step 4: Predict with the existing (untouched) Random Forest model â”€â”€â”€â”€â”€â”€â”€
# The trained model object is unchanged; the new columns are numerically
# identical to the originals (same max/min/mean over the same rook neighbors),
# so predictions are reproducible.
# preds <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves the Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor set** | `nb_to_edge_list` reads the identical `rook_neighbors_unique` nb object and `id_order` mapping used by the original code. |
| **Same statistics** | `max`, `min`, `mean` are computed over exactly the same non-NA neighbor values per (cell, year). |
| **Same column names** | The naming convention (`neighbor_max_ntl`, etc.) matches the original `compute_and_add_neighbor_features` output. |
| **RF model untouched** | No retraining; the model object is never modified. |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | ~1 sec (vectorized edge-list) |
| Neighbor stats (Ã—5 vars) | ~80+ hours (lapply) | ~5â€“15 min (data.table group-by) |
| **Total** | **86+ hours** | **< 20 minutes** |

### Memory Note (16 GB laptop)

The largest intermediate object is `edge_vals` at ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows Ã— 4 columns â‰ˆ 1.2 GB. With `cell_data` itself at ~6.46M Ã— 110 cols â‰ˆ 5â€“6 GB, peak usage stays under 12 GB. The `gc()` call between variables reclaims temporaries.
END_RESPONSE>>>
