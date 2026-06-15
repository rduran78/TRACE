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
run_id: Anthropic_F09_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It constructs a **per-row** (per cell-year) list of neighbor row indices by pasting character keys and doing named-vector lookups across **~6.46 million rows**. This is catastrophically slow because:

1. **Redundant work across years.** The spatial neighbor topology is *identical* for every year. The current code rebuilds string keys and looks up neighbors for every cell-year row â€” repeating the same spatial logic 28 times.
2. **Character key hashing at scale.** `paste()` + named-vector lookup (`idx_lookup[neighbor_keys]`) for ~6.46M rows, each with ~4 neighbors on average, means tens of millions of string operations inside an `lapply` â€” effectively an O(NÃ—K) character-matching loop in R's interpreter.
3. **`lapply` over 6.46M elements** returns a list of 6.46M vectors. This is slow to build and slow to consume in `compute_neighbor_stats`, which again `lapply`s over the same 6.46M entries.

The `compute_neighbor_stats` function is comparatively lighter but still loops over 6.46M list elements in pure R, which is suboptimal.

**Summary:** The design conflates *spatial topology* (fixed) with *temporal panel structure* (repeated), causing a 28Ã— blowup in the most expensive operation, all executed in interpreted R loops over millions of rows.

---

## Optimization Strategy

### Core idea: Separate topology from attributes, then vectorize with `data.table` joins.

1. **Build the neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` derived from the `spdep::nb` object. This is ~1.37M rows and is year-independent.

2. **Join yearly attributes onto the edge table.** For each year, every cell's neighbors inherit their attribute values via a keyed `data.table` join. This turns the problem into a standard grouped aggregation.

3. **Compute grouped max/min/mean** using `data.table`'s optimized `by=` grouping â€” no R-level loops, no string key construction, no per-row `lapply`.

4. **Join the aggregated neighbor stats back** onto the main panel `data.table`.

### Complexity reduction

| Aspect | Old | New |
|---|---|---|
| Neighbor lookup construction | 6.46M string-key lookups | 1.37M-row static edge table (built once) |
| Per-variable aggregation | `lapply` over 6.46M list elements | `data.table` grouped aggregation (~38M edge-year rows) |
| Expected wall-clock time | 86+ hours | **~2â€“10 minutes** |

### Memory check

The edge table replicated across 28 years is ~1.37M Ã— 28 â‰ˆ 38.4M rows Ã— a few columns of doubles â€” roughly 1â€“2 GB, well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and
#         the neighbor source variables.
#         rook_neighbors_unique is the spdep::nb object.
#         id_order is the vector mapping nb list index â†’ cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial edge table ONCE.
#         This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbs <- nb_obj[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    nbs <- nbs[nbs != 0L]
    if (length(nbs) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nbs])
  }))
  edges
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table: ~1,373,394 rows, columns: cell_id, neighbor_id

cat(sprintf("Edge table: %d directed neighbor pairs\n", nrow(edge_table)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, compute neighbor max, min,
#         mean via data.table join + grouped aggregation, then attach
#         the results back to cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # --- 2a: Extract the minimal attribute table (id, year, value) ------
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # --- 2b: Replicate edges across all years via join ------------------
  #     edge_table has (cell_id, neighbor_id).
  #     Join onto attr_dt by neighbor_id to pick up (year, value).
  #     This gives us one row per (cell_id, neighbor_id, year) with the
  #     neighbor's attribute value.
  edge_year <- merge(edge_table, attr_dt, by = "neighbor_id",
                     allow.cartesian = TRUE)
  # edge_year columns: neighbor_id, cell_id, year, value

  # --- 2c: Grouped aggregation ----------------------------------------
  stats <- edge_year[!is.na(value),
                     .(nb_max  = max(value),
                       nb_min  = min(value),
                       nb_mean = mean(value)),
                     by = .(cell_id, year)]

  # --- 2d: Name the new columns to match original convention ----------
  new_names <- paste0(var_name, c("_neighbor_max",
                                   "_neighbor_min",
                                   "_neighbor_mean"))
  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           new_names)
  setnames(stats, "cell_id", "id")
  setkey(stats, id, year)

  # --- 2e: Remove old columns if they exist (idempotent re-run) -------
  for (nm in new_names) {
    if (nm %in% names(cell_data)) cell_data[, (nm) := NULL]
  }

  # --- 2f: Join back onto cell_data -----------------------------------
  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)

  # Clean up large intermediates each iteration to stay within 16 GB
  rm(attr_dt, edge_year, stats)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the already-trained Random Forest model.
#         The model object (rf_model) is unchanged.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adjust to your actual model object and predict call):
# cell_data[, prediction := predict(rf_model, newdata = .SD)]
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | `build_edge_table` reads the same `rook_neighbors_unique` nb object and the same `id_order` mapping. Every `(cell, neighbor)` pair is identical. |
| **Same per-year attribute values** | The join key is `(neighbor_id, year)`, so each neighbor's value for a given year is exactly the value from `cell_data`. |
| **Same aggregation functions** | `max`, `min`, `mean` over non-NA neighbor values â€” identical to the original `compute_neighbor_stats`. |
| **NA handling** | `edge_year[!is.na(value), ...]` drops NAs before aggregation, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the outer join (`stats[cell_data]`), matching the original `return(c(NA, NA, NA))`. |
| **Trained RF model** | The model object is never modified or retrained; only `predict()` is called on the enriched `cell_data`. |

**Expected speedup: from ~86+ hours to a few minutes** â€” dominated by the `merge` and grouped aggregation, both of which `data.table` handles at near-C speed.
END_RESPONSE>>>
