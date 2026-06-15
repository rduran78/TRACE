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
run_id: Anthropic_F06_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) hash lookup).
2. Constructs character key strings via `paste()` for every neighbor Ã— year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching on a 6.46M-length named vector.

Character-keyed lookups on a vector of length 6.46M are **O(n)** per probe in the worst case (R's named vector lookup uses hashing, but the sheer volume â€” ~6.46M Ã— ~4 neighbors average â€” means billions of character operations). This alone likely accounts for the majority of the 86+ hour estimate.

### Bottleneck 2: `compute_neighbor_stats` â€” Row-wise `lapply` over 6.46M rows

For each of 5 variables, another `lapply` iterates over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. The per-row overhead of R function calls, subsetting, and `is.na` checks across 6.46M iterations Ã— 5 variables = ~32.3M R-level loop iterations is substantial.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics on regular grids extremely efficiently using compiled C code with sliding windows. The panel data here **is** on a regular spatial grid, so conceptually each year-slice is a raster and rook-neighbor max/min/mean are exactly a 3Ã—3 cross-shaped (rook) focal operation. However, the data is stored as a **long panel data.frame**, not as a raster stack, and cells may have missing years or irregular coverage. The optimization strategy below uses the **same logic** as focal operations â€” vectorized, year-sliced, matrix-indexed computation â€” while preserving exact numerical results for all edge cases (boundary cells, missing data).

---

## Optimization Strategy

| Step | What changes | Why it's faster |
|---|---|---|
| **1. Replace character-key lookup with integer join** | Use `data.table` keyed join on `(id, year)` instead of `paste()`-based named vector lookup. | Eliminates ~6.46M `paste()` calls and character hash lookups. `data.table` binary search join is O(log n). |
| **2. Build an edge list once, then vectorized merge** | Expand the `nb` object into a two-column integer edge list of `(id, neighbor_id)`. Join to the panel on `(neighbor_id, year)` to get all neighbor values in one vectorized operation per variable. | Replaces 6.46M R-level loop iterations with a single `data.table` merge (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows, handled in compiled C). |
| **3. Grouped aggregation** | `data.table` grouped `max`, `min`, `mean` by `(id, year)` on the joined result. | Replaces 6.46M per-row `lapply` calls with a single compiled grouped aggregation. |
| **4. Repeat for each variable (or batch)** | Loop over the 5 variables, or pivot and do all at once. | Each variable takes seconds, not hours. |

**Expected speedup**: From ~86+ hours to **~2â€“10 minutes** on 16 GB RAM.

**Numerical equivalence**: The `max`, `min`, `mean` are computed on exactly the same neighbor sets with the same NA handling, so the trained Random Forest model receives identical inputs and need not be retrained.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed to exist:
#     - cell_data        : data.frame/data.table with columns id, year, 
#                          ntl, ec, pop_density, def, usd_est_n2, ...
#     - rook_neighbors_unique : spdep nb object (list of integer vectors)
#     - id_order          : integer/character vector mapping nb list 
#                           positions to cell ids
#     - rf_model           : pre-trained Random Forest (untouched)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table (by reference if already a data.table)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build directed rook-neighbor edge list (once)
#     From the nb object: for each cell i, neighbors[[i]] gives the
#     positions in id_order of its rook neighbors.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  
  # Remove the 0-length (no-neighbor) entries naturally (they contribute

  # nothing to from_idx / to_idx via rep + unlist).
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394 (directed rook edges among 344,208 cells)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Key the panel for fast joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  For each source variable, compute neighbor max, min, mean
#     and join back to cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_fast <- function(dt, edge_dt, var_name) {
  
  # --- 3a. Build a slim lookup: (id, year, value) ----------------------
  lookup <- dt[, .(id, year, value = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)
  
  # --- 3b. Cross join edges Ã— years -----------------------------------
  #   For every (id -> neighbor_id) edge and every year present for `id`,
  #   retrieve the neighbor's value.
  #
  #   Strategy: get the years each cell appears in, cross with edges,
  #   then join to lookup.  But it's simpler and equally fast to:
  #     (i)  join edge_dt to lookup on neighbor_id (adds year & value), 
  #     (ii) then ensure the focal cell also exists in that year.
  
  # Expand: every edge Ã— every year the *neighbor* has data
  expanded <- merge(edge_dt, lookup, by = "neighbor_id", allow.cartesian = TRUE)
  # expanded columns: neighbor_id, id, year, value
  
  # Keep only rows where the focal cell (id) also exists in that year.
  # We use a semi-join via keyed existence check.
  focal_keys <- unique(dt[, .(id, year)])
  setkey(focal_keys, id, year)
  setkey(expanded, id, year)
  expanded <- expanded[focal_keys, nomatch = 0L]
  
  # --- 3c. Aggregate: max, min, mean per (id, year) -------------------
  #   Drop NA values in the variable before aggregating (matches original code).
  agg <- expanded[!is.na(value), 
                  .(nb_max  = max(value),
                    nb_min  = min(value),
                    nb_mean = mean(value)),
                  by = .(id, year)]
  
  # --- 3d. Name the new columns to match original convention -----------
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))
  
  # --- 3e. Left-join aggregates back to the main table -----------------
  #   Cells with no valid neighbors in a given year get NA (correct).
  setkey(agg, id, year)
  setkey(dt, id, year)
  
  # Remove old columns if re-running
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(dt)) dt[, (col) := NULL]
  }
  
  dt <- agg[dt, on = .(id, year)]
  setkey(dt, id, year)
  
  return(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Run for all 5 variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edge_dt, var_name)
  cat("  Done in", round((proc.time() - t0)[3], 1), "seconds.\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the existing Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory Note for 16 GB RAM

The largest intermediate object is `expanded` in step 3b. With ~1.37M edges Ã— 28 years = ~38.4M rows Ã— 3 columns (two integer IDs + one double value) â‰ˆ **~920 MB**. This is well within 16 GB. Each variable iteration creates and discards this object, so peak usage stays under ~4â€“5 GB including `cell_data` itself (~6.46M rows Ã— 110 columns â‰ˆ 5.7 GB at 8 bytes/double).

If memory is tight, the loop already processes one variable at a time and `expanded` is garbage-collected between iterations. You can add an explicit `gc()` call after each iteration if needed.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | `paste()` + named vector (character hashing, 6.46M iterations) | `data.table` integer-keyed binary search join |
| Stats computation | `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| Estimated runtime | 86+ hours | ~2â€“10 minutes |
| Numerical results | Baseline | **Identical** (same neighbor sets, same `max`/`min`/`mean`, same NA handling) |
| RF model | Untouched | **Untouched** |
END_RESPONSE>>>
