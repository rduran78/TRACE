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
run_id: Anthropic_F02_T00_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated 5 times, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-key lookups at scale:** `idx_lookup` is a named vector of length ~6.46M. Named-vector lookup in R uses hashing, but constructing 6.46M paste keys and looking up variable-length neighbor sets against this hash is extremely slow in an interpreted `lapply` loop.
2. **Per-row `lapply` overhead:** 6.46M iterations of anonymous R functions, each allocating small vectors, causes massive interpreter and GC overhead.
3. **Redundant computation:** The neighbor *structure* is static across years â€” cell A's rook neighbors are the same cells every year. But the lookup is rebuilt as if the topology could change per row.
4. **Memory pressure:** Storing 6.46M list elements in `neighbor_lookup` (each a variable-length integer vector) is memory-inefficient and cache-unfriendly.

---

## Optimization Strategy

**Key insight:** The spatial neighbor topology is year-invariant. Cell `i`'s neighbors are the same cells regardless of year. We can therefore separate the spatial structure from the temporal indexing and use vectorized, column-oriented operations.

**Strategy (3 pillars):**

### 1. Vectorized Neighbor Lookup via `data.table` Joins (eliminate `lapply` entirely)

Instead of building a per-row list, we:
- Expand the `rook_neighbors_unique` nb object into an edge list `(focal_id, neighbor_id)`.
- Join this edge list to the panel data by `(neighbor_id, year)` to retrieve neighbor values.
- Group by `(focal_id, year)` and compute `max`, `min`, `mean` in one vectorized pass.

This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with a single `data.table` merge + grouped aggregation â€” no R-level loops at all.

### 2. Process One Variable at a Time (control peak memory)

With 6.46M rows and ~1.37M directed edges, the expanded join table is ~6.46M Ã— (avg ~4 neighbors) â‰ˆ 26M rows, but only needs 3 columns at a time (`focal_id`, `year`, `value`). At ~26M rows Ã— 3 columns Ã— 8 bytes â‰ˆ 0.6 GB per variable, this fits comfortably in 16 GB alongside the original data (~5.7 GB for 6.46M Ã— 110 cols).

### 3. Preserve the Trained Model and Numerical Estimand

We only change *how* the features are computed, not *what* is computed. The `max`, `min`, `mean` aggregations over the identical neighbor sets produce bit-identical results. The Random Forest model is never touched.

**Expected speedup:** Each variable's join + aggregation should take ~30â€“90 seconds on a modern laptop. Total for 5 variables: **~3â€“8 minutes** (vs. 86+ hours).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert the spdep nb object to a data.table edge list (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# id_order is the vector of cell IDs aligned with rook_neighbors_unique
# (i.e., id_order[k] is the cell ID for the k-th element of the nb list)

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  n <- length(neighbors)
  # Pre-allocate by counting total edges
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  focal_id    <- integer(total)
  neighbor_id <- integer(total)
  
  pos <- 1L
  for (k in seq_len(n)) {
    nb <- neighbors[[k]]
    m  <- lens[k]
    if (m == 0L) next
    idx <- pos:(pos + m - 1L)
    focal_id[idx]    <- id_order[k]
    neighbor_id[idx] <- id_order[nb]
    pos <- pos + m
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” trivial memory

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor features for each variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_dt) {
  # Extract only the columns we need for the join (minimise memory)
  # Columns: neighbor_id (to join on), year, and the variable value
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)
  
  # Join edge list with values: for each (focal_id, neighbor_id) pair,
  # look up the neighbor's value in every year
  # Result: one row per (focal_id, neighbor_id, year) with the neighbor's value
  joined <- edge_dt[val_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # joined has columns: focal_id, neighbor_id, year, value
  
  # Aggregate by (focal_id, year)
  agg <- joined[!is.na(value),
                 .(nb_max  = max(value),
                   nb_min  = min(value),
                   nb_mean = mean(value)),
                 by = .(focal_id, year)]
  
  # Build target column names (must match what the trained RF model expects)
  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(col_max,  col_min,  col_mean))
  setnames(agg, "focal_id", "id")
  setkey(agg, id, year)
  
  # Merge back into cell_data (left join to preserve all rows, NAs where no neighbors)
  # Remove columns if they already exist (idempotent re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
  setkey(cell_data, id, year)
  
  # Clean up intermediate objects to free memory
  rm(val_dt, joined, agg)
  gc()
  
  return(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Outer loop â€” identical interface, vastly faster
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  message("  Done. Columns: ", ncol(cell_data))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (model object and predict call are untouched)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with string hashing | One-time edge-list build (simple `for` over 344K cells) |
| **Stat computation** | 6.46M-iteration `lapply` per variable Ã— 5 | `data.table` equi-join + grouped aggregation per variable Ã— 5 |
| **Estimated runtime** | 86+ hours | ~3â€“8 minutes |
| **Peak memory** | Uncontrolled (millions of list elements + named vectors) | ~2â€“3 GB transient per variable, freed between iterations |
| **Numerical output** | max / min / mean of rook-neighbor values | Identical max / min / mean of identical neighbor sets |
| **Trained RF model** | â€” | Untouched; same feature names, same values |
END_RESPONSE>>>
