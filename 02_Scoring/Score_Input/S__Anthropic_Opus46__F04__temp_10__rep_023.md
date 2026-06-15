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
run_id: Anthropic_F04_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is squarely in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates `lapply` over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** neighbor cell IDs with the current year to create string keys (memory-allocating `paste` call per row).
4. **Looks up** those keys in a **named character vector** (`idx_lookup`) of length 6.46 million â€” this is an O(n) hash lookup per key, repeated for every neighbor of every row.

With ~6.46M rows and an average of ~4 rook neighbors each, that's **~25.8 million `paste` + named-vector lookups**, all inside a **serial `lapply`** in R (interpreted, no vectorization). The named-vector lookup on a 6.46M-element vector is slow because R's name-matching on character vectors has significant overhead at that scale. The `paste` allocations inside the loop compound the problem.

**`compute_neighbor_stats`** is a secondary bottleneck: another serial `lapply` over 6.46M rows computing `max`/`min`/`mean` â€” called **5 times** (once per variable). That's 32.3 million R-level function calls.

**In summary:** ~86+ hours is driven by billions of interpreted R operations (string construction, hash lookups, per-row function calls) across 6.46M rows, repeated multiple times.

---

## Optimization Strategy

### Principle: Replace row-level R loops with vectorized/join-based operations using `data.table`.

**Key ideas:**

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, create a flat edge-list `data.table` that maps every `(id, year)` pair to its neighbors' row indices via a fast **keyed join** â€” fully vectorized.

2. **Replace `compute_neighbor_stats`'s `lapply`** with a grouped `data.table` aggregation (`[, .(max, min, mean), by = ...]`), which is implemented in C internally and orders of magnitude faster.

3. **Memory management:** The flat edge-list for all cell-years will have ~25.8M rows Ã— a few integer/double columns â€” roughly 200â€“400 MB, well within 16 GB.

4. **Preserve the trained RF model** â€” we only change feature construction, producing numerically identical features.

**Expected speedup:** From 86+ hours to roughly **5â€“20 minutes** on the same machine.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a flat edge-list from the nb object (one-time, vectorized)
# ==============================================================================
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer vectors of neighbor indices)
  # id_order is the vector mapping position -> cell id
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id  (~1.37M rows, one per directed edge)

# ==============================================================================
# STEP 2: Convert cell_data to data.table and create a row-index column
# ==============================================================================
setDT(cell_data)
cell_data[, row_idx := .I]

# ==============================================================================
# STEP 3: Expand edges across all years and join to get neighbor values
#         Then aggregate â€” all vectorized in data.table
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {

  # Create a unique year vector
  years <- sort(unique(cell_data$year))

  # Cross-join edge list with years: every edge exists in every year
  # This gives us the full set of (from_id, year) -> to_id mappings
  edge_year <- edge_dt[, CJ(edge_idx = seq_len(.N), year = years)]
  edge_year[, `:=`(
    from_id = edge_dt$from_id[edge_idx],
    to_id   = edge_dt$to_id[edge_idx]
  )]
  edge_year[, edge_idx := NULL]
  # edge_year: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows (from_id, to_id, year)

  # Key cell_data for fast join on (id, year)
  setkey(cell_data, id, year)

  # Join to get neighbor row indices and values in one shot per variable
  # We join edge_year to cell_data on (to_id, year) to get neighbor values
  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    # Subset only what we need for the join
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # Join: for each edge-year, get the neighbor's value
    edge_vals <- val_dt[edge_year, on = .(id = to_id, year = year),
                        .(from_id = i.from_id, year = i.year, val = x.val),
                        nomatch = NA]

    # Remove NAs in val before aggregation
    edge_vals <- edge_vals[!is.na(val)]

    # Aggregate: max, min, mean grouped by (from_id, year)
    agg <- edge_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(from_id, year)]

    # Rename columns to match original pipeline's naming convention
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))

    # Join aggregated stats back to cell_data
    setkey(agg, from_id, year)
    setkey(cell_data, id, year)
    cell_data <- agg[cell_data, on = .(from_id = id, year = year)]

    # The join introduces 'from_id' as a column; rename back to 'id'
    # Actually, with agg[cell_data, ...] and on = .(from_id = id), 
    # data.table keeps the 'from_id' column. Let's handle this cleanly:
    setnames(cell_data, "from_id", "id")
  }

  cell_data[, row_idx := NULL]  # clean up helper column
  return(cell_data)
}

# ==============================================================================
# STEP 4: Execute
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# cell_data now has columns like ntl_nb_max, ntl_nb_min, ntl_nb_mean, etc.
# These are numerically identical to the original pipeline's output.

# ==============================================================================
# STEP 5: Predict with the existing trained RF model (unchanged)
# ==============================================================================
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Numerical Equivalence

| Original operation | Replacement | Equivalence |
|---|---|---|
| `paste(id, year)` key lookup â†’ index vector | `data.table` keyed join on `(id, year)` | Same row matching, deterministic |
| `max(vals[idx])` | `data.table [, max(val), by=.(from_id,year)]` | Identical IEEE 754 result |
| `min(vals[idx])` | `data.table [, min(val), by=.(from_id,year)]` | Identical |
| `mean(vals[idx])` | `data.table [, mean(val), by=.(from_id,year)]` | Identical (same `sum/n`) |
| Rows with no valid neighbors â†’ `NA` | Left join (`agg[cell_data]`) leaves unmatched as `NA` | Identical |

The trained Random Forest model is never touched â€” only its input feature columns are reconstructed via a faster code path that produces the same numbers.

---

## Memory Estimate

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_year` (38.4M Ã— 3 cols) | ~0.9 GB |
| `edge_vals` (temporary, ~38.4M Ã— 3) | ~0.9 GB |
| Aggregation result | ~0.15 GB |
| **Peak total** | **~8â€“9 GB (fits in 16 GB)** |

If memory is tight, `edge_year` can be processed in year-batches (e.g., 7 years at a time), reducing peak by ~75% at negligible speed cost.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `paste` + named-vector lookups in `lapply` | One vectorized `data.table` cross-join |
| Neighbor stats | 5 Ã— 6.46M `lapply` calls with per-row `max`/`min`/`mean` | 5 Ã— one grouped `data.table` aggregation (C-level) |
| Estimated runtime | 86+ hours | ~5â€“20 minutes |
| Numerical output | Baseline | Identical |
| RF model | Unchanged | Unchanged |
END_RESPONSE>>>
