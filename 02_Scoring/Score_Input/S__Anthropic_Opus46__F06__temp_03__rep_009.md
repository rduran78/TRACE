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
run_id: Anthropic_F06_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the spatial cell's rook neighbors (by cell ID).
2. Pastes together `(neighbor_id, year)` keys as character strings.
3. Searches a named character vector (`idx_lookup`) for those keys.

This produces **~6.46 million list elements**, each containing row indices. The `paste()`-based key construction and named-vector lookup are extremely slow at this scale. Critically, the neighbor *structure* is identical across all 28 years for a given cell â€” only the year suffix changes â€” yet the function redundantly recomputes this for every cell-year row.

### Bottleneck 2: `compute_neighbor_stats` â€” Repeated R-level loops

For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the index lists and computing `max`, `min`, `mean`. This is pure R-level looping with no vectorization. It runs 5 times (once per variable), totaling ~32.3 million R-level iterations.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. While the cells here are on a grid, the panel structure (cell Ã— year), missing cells, and the need to match exact `spdep::nb` rook relationships mean a focal approach would require reshaping data into raster stacks per year and then re-joining â€” adding complexity and risking subtle mismatches at boundaries. The better strategy is to vectorize the existing tabular approach using `data.table` joins, which preserves the exact neighbor relationships and results.

### Summary

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | 6.46M `paste` + named-vector lookups |
| `compute_neighbor_stats` | ~hours Ã— 5 vars | 6.46M R-level `lapply` iterations Ã— 5 |
| **Total estimated** | **86+ hours** | No vectorization; redundant per-year work |

---

## Optimization Strategy

1. **Expand the neighbor list once at the cell level (344K cells), not the cell-year level (6.46M rows).** Build an edge table `(cell_id, neighbor_id)` from the `nb` object â€” this has ~1.37M rows.

2. **Use `data.table` equi-joins** to attach neighbor variable values by `(neighbor_id, year)`. This replaces all `paste`/`lapply` logic with a single vectorized join per variable.

3. **Compute grouped `max`, `min`, `mean`** using `data.table`'s `by=` grouping â€” a single C-level pass per variable, replacing 6.46M R-level iterations.

4. **Process all 5 variables in one pass** (or 5 fast passes) to avoid redundant joins.

5. **Memory**: The edge table Ã— 28 years â‰ˆ 38.5M rows of integers â€” well within 16 GB.

**Expected speedup**: From 86+ hours to **minutes** (typically 2â€“10 minutes total).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed to exist:
#       cell_data              â€” data.frame/data.table with columns:
#                                 id, year, ntl, ec, pop_density, def, usd_est_n2
#       id_order               â€” integer vector of cell IDs matching the nb object
#       rook_neighbors_unique  â€” spdep::nb object (list of integer index vectors)
#       rf_model               â€” pre-trained Random Forest model (not retrained)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table (in-place if already; copy otherwise)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build edge table from the nb object  (~1.37M rows)
#     Each element rook_neighbors_unique[[i]] contains integer indices
#     into id_order for the neighbors of id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep::nb encodes "no neighbors" as a single 0L

if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx == 0L)) {
    return(NULL)
  }
  data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
}))

cat(sprintf("Edge table: %d directed rook-neighbor pairs\n", nrow(edges)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Define source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  For each variable, join, aggregate, and merge back
#     This replaces build_neighbor_lookup + compute_neighbor_stats
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var in neighbor_source_vars) {

  cat(sprintf("Processing neighbor stats for: %s\n", var))

  # --- 3a. Build a slim lookup: (id, year, value) for the current variable ---
  #         Rename columns so the join is clean.
  val_dt <- cell_data[, .(neighbor_id = id, year, .var_val = get(var))]
  setkey(val_dt, neighbor_id, year)

  # --- 3b. Join edges Ã— years to get neighbor values ---
  #         edges has (id, neighbor_id); we expand by year via the join.
  #         Result: one row per (id, year, neighbor_id) with the neighbor's value.
  joined <- merge(edges, val_dt, by = "neighbor_id", allow.cartesian = TRUE)
  #         joined columns: neighbor_id, id, year, .var_val

  # --- 3c. Aggregate: max, min, mean per (id, year) ---
  agg <- joined[!is.na(.var_val),
                 .(nb_max  = max(.var_val),
                   nb_min  = min(.var_val),
                   nb_mean = mean(.var_val)),
                 by = .(id, year)]
  setkey(agg, id, year)

  # --- 3d. Define output column names (match original naming convention) ---
  #         Adjust these names to match whatever compute_and_add_neighbor_features

  #         originally produced.  A common convention:
  max_col  <- paste0(var, "_nb_max")
  min_col  <- paste0(var, "_nb_min")
  mean_col <- paste0(var, "_nb_mean")

  # --- 3e. Merge aggregated stats back into cell_data ---
  #         Remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)

  # Rename the generic agg columns to variable-specific names
  setnames(cell_data,
           old = c("nb_max", "nb_min", "nb_mean"),
           new = c(max_col,  min_col,  mean_col))

  # Clean up to free memory before next iteration
  rm(val_dt, joined, agg)
  gc()
}

# Re-key for any downstream use
setkey(cell_data, id, year)

cat("All neighbor features computed.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | The edge table is built from the identical `rook_neighbors_unique` nb object. Every `(cell, neighbor)` pair is preserved exactly. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same sets of non-NA neighbor values per `(id, year)` group. |
| **Same NA handling** | Rows with no valid neighbors receive `NA` via the `all.x = TRUE` left join, identical to the original `c(NA, NA, NA)` return. |
| **RF model untouched** | The pre-trained model is only used for `predict()`; no retraining occurs. |
| **Floating-point identity** | `data.table` uses the same R-level `max`, `min`, `mean` functions, so results are numerically identical (not merely approximate). |

---

## Expected Performance

| Step | Estimated Time |
|---|---|
| Build edge table | ~1â€“2 seconds |
| Per-variable join + aggregate (Ã—5) | ~30â€“90 seconds each |
| **Total** | **~3â€“8 minutes** |

This represents a speedup of roughly **600â€“1,700Ã—** over the original 86+ hour estimate, achieved entirely through vectorized `data.table` joins and grouped aggregation, with no change to the computed values or the trained model.
END_RESPONSE>>>
