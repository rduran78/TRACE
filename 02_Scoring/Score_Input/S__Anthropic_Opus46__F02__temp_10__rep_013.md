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
run_id: Anthropic_F02_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core performance bottleneck is **`build_neighbor_lookup`**, which runs an `lapply` over **~6.46 million rows**, performing per-row character coercion, string pasting, and named-vector lookups. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. The result is a list of 6.46 million integer vectors, consuming both enormous time and memory. Then `compute_neighbor_stats` iterates over that 6.46M-element list again for *each* of the 5 variables, performing per-element subsetting, NA removal, and summary statsâ€”adding another 5 Ã— 6.46M â‰ˆ 32.3M R-level loop iterations.

**Specific problems:**

1. **String-key lookups at scale:** `paste()` + named-vector indexing on 6.46M rows is O(n) with a large constant factor. Each call to `id_to_ref[as.character(...)]` and `idx_lookup[neighbor_keys]` involves hashing millions of strings.
2. **Ragged list of 6.46M elements:** The `neighbor_lookup` list stores ~6.46M integer vectors. This is ~1.37M directed edges Ã— years replicated, consuming several GB of RAM just for the list overhead (each R vector has a ~56-byte header).
3. **Row-level R loops:** `lapply` over millions of rows in pure R is inherently slowâ€”no vectorization, no SIMD, no parallelism.
4. **Redundant computation:** The neighbor *graph* is year-invariant (a cell's rook neighbors don't change over time), but the lookup is rebuilt as if each cell-year is unique.

---

## Optimization Strategy

**Key insight:** The neighbor topology is **time-invariant**. Cell `i`'s neighbors in 1992 are the same cells as in 2019. We only need to look up neighbor *values* for the matching year. This can be fully vectorized using `data.table` joins, eliminating the 6.46M-element list entirely.

**Approach â€” "Edge Table + data.table grouped join":**

1. **Expand the neighbor graph into an edge table** (~1.37M rows of `(cell_id, neighbor_id)` pairs). This is done once.
2. **Join the edge table to the panel data** on `(neighbor_id, year)` to pull each neighbor's variable value. This is a single keyed `data.table` mergeâ€”highly optimized in C.
3. **Group by `(cell_id, year)`** and compute `max`, `min`, `mean` in one pass using `data.table`'s grouped aggregation (also C-level, vectorized).
4. **Repeat for each of the 5 variables** (or batch them in one join).

This replaces billions of R-level operations with a handful of C-level `data.table` operations. Expected speedup: **~100â€“500Ã—** (minutes instead of days). Memory stays well within 16 GB because the edge table is only ~1.37M rows, and the join result is ~1.37M Ã— 28 â‰ˆ 38.4M rows of numeric data (~1â€“2 GB).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure id and year are proper types for joining
cell_dt[, id := as.integer(id)]
cell_dt[, year := as.integer(year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the edge table from the nb object (done once)
#
# rook_neighbors_unique is a list of length 344,208 where element
# [[i]] is an integer vector of indices into id_order.
# id_order maps position -> cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # Pre-compute lengths for pre-allocation
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    n  <- length(nb)
    if (n == 0L) next
    idx_range <- pos:(pos + n - 1L)
    from_id[idx_range] <- id_order[i]
    to_id[idx_range]   <- id_order[nb]
    pos <- pos + n
  }
  
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor features for all variables via join
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We join edge_dt Ã— years to cell_dt to get neighbor values.
# To avoid a massive cross-join, we join per-variable in a memory-
# efficient loop.

# Key cell_dt for fast joining
setkey(cell_dt, id, year)

# Get the unique years vector once
all_years <- sort(unique(cell_dt$year))

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Extract only the columns we need for the lookup side
  # This keeps memory lean
  lookup_cols <- c("id", "year", var_name)
  neighbor_vals_dt <- cell_dt[, ..lookup_cols]
  setnames(neighbor_vals_dt, c("id", var_name), c("neighbor_id", "nb_val"))
  setkey(neighbor_vals_dt, neighbor_id, year)
  
  # Join: for every (cell_id, neighbor_id) edge and every year,
  # get the neighbor's value.
  # We do this by joining edge_dt to the panel on neighbor_id,
  # allowing year from neighbor_vals_dt to carry through.
  
  # Expand edges Ã— years efficiently:
  # Rather than a literal cross-join (which would be 1.37M Ã— 28 = 38.4M rows),
  # we merge edge_dt with neighbor_vals_dt on neighbor_id,
  # which naturally gives us one row per (edge, year) combination
  # wherever the neighbor has data.
  
  joined <- merge(edge_dt, neighbor_vals_dt,
                  by = "neighbor_id",
                  allow.cartesian = TRUE)
  # joined has columns: neighbor_id, cell_id, year, nb_val
  # ~38.4M rows (1.37M edges Ã— 28 years)
  
  # Aggregate by (cell_id, year)
  agg <- joined[!is.na(nb_val),
                 .(nb_max  = max(nb_val),
                   nb_min  = min(nb_val),
                   nb_mean = mean(nb_val)),
                 by = .(cell_id, year)]
  
  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # Merge back into cell_dt
  setkey(agg, cell_id, year)
  cell_dt <- merge(cell_dt, agg, by.x = c("id", "year"),
                   by.y = c("cell_id", "year"), all.x = TRUE)
  
  # Free intermediate objects
  rm(joined, agg, neighbor_vals_dt)
  gc()
  
  cat(sprintf("    Done â€” added %s, %s, %s\n", max_col, min_col, mean_col))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Convert back to data.frame if downstream code expects it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)
rm(cell_dt, edge_dt)
gc()

cat("Neighbor feature computation complete.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained Random Forest model is untouched. Proceed to
# prediction using the existing model object as before:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The numerical values of max/min/mean neighbor stats are
# identical to the original implementation (same arithmetic,
# same neighbor graph), preserving the original estimand.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `build_edge_table` reads the identical `rook_neighbors_unique` nb object and `id_order` mapping. Every directed edge is preserved. |
| **Same per-year matching** | The join on `(neighbor_id, year)` ensures each cell-year row picks up its neighbors' values from the *same* yearâ€”identical to the original `paste(id, year)` key logic. |
| **Same summary statistics** | `max`, `min`, `mean` after `!is.na` filtering are the same arithmetic operations as the original `compute_neighbor_stats`. |
| **Trained RF model untouched** | No model retraining occurs. The output column names match the original convention, so `predict(rf_model, newdata = ...)` works unchanged. |

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M R-level string ops â†’ hours | Replaced by `build_edge_table`: ~1.37M iterations in a simple for-loop â†’ **< 5 seconds** |
| **`compute_neighbor_stats`** (Ã—5 vars) | ~32.3M R-level list iterations â†’ tens of hours | Replaced by keyed `data.table` merge + grouped agg â†’ **~2â€“5 minutes per variable** |
| **Total wall time** | 86+ hours estimated | **~15â€“30 minutes total** |
| **Peak RAM** | Multiple GB for the 6.46M-element list | ~3â€“5 GB peak (38.4M-row join intermediate) â€” fits in 16 GB |
END_RESPONSE>>>
