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
run_id: Anthropic_F02_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which performs ~6.46 million `lapply` iterations, each doing character-based key construction and named-vector lookups. Named vector lookups via character keys in R are hash-table operations, but doing millions of `paste()` calls and millions of named-vector subscripts inside a sequential `lapply` is extremely slow. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again, extracting and summarizing neighbor values per row. This is repeated **5 times** (once per variable), compounding the cost.

**Specific problems:**

1. **`build_neighbor_lookup`:** Character key construction (`paste`) and named-vector indexing (`idx_lookup[neighbor_keys]`) inside a per-row `lapply` over 6.46M rows. This is O(n Ã— k) string operations where k is the average neighbor count, and R's overhead per iteration is high.

2. **`compute_neighbor_stats`:** Returns a list of 6.46M three-element vectors, then `do.call(rbind, ...)` on that list â€” a notoriously slow pattern in R for large lists.

3. **Memory:** Storing a list of 6.46M integer vectors (the neighbor lookup) plus intermediate character vectors consumes significant RAM on a 16 GB machine.

4. **Redundant structure:** The neighbor relationships are **time-invariant** (the same cell has the same rook neighbors every year), but the lookup is built by expanding neighbors into every cell-year combination, duplicating the spatial topology 28 times.

---

## Optimization Strategy

### Key Insight: Separate Spatial Topology from Temporal Expansion

Rook neighbors are a property of **cells**, not cell-years. There are only 344,208 cells. The neighbor graph should be stored as a sparse structure over cells, and temporal expansion should be handled via vectorized joins, not per-row iteration.

### Strategy Summary

| Step | Technique | Speedup Source |
|---|---|---|
| 1 | Represent the neighbor graph as a two-column **edge list** (integer cell indices), not a per-row list. | Eliminates 6.46M list elements; enables vectorized ops. |
| 2 | Use `data.table` keyed joins instead of named-vector lookups. | O(n log n) merge vs. O(n Ã— k) hash lookups in a loop. |
| 3 | Compute all 5 variables' neighbor stats in a **single pass** via `data.table` grouped aggregation over the edge list. | Eliminates 5 separate `lapply` passes over 6.46M rows. |
| 4 | Avoid `do.call(rbind, list_of_vectors)`. | `data.table` returns a single data.table directly. |
| 5 | Use integer cell indices and integer year throughout. | Eliminates all `paste`/character operations. |

**Expected runtime:** Minutes instead of 86+ hours. Memory stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Convert spdep nb object to an integer edge-list (one-time, fast)
# ===========================================================================
# rook_neighbors_unique is a list of length 344,208.
# id_order is the vector of cell IDs in the same order as the nb object.

build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order of i's neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Map positional indices to actual cell IDs
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows: (from_id, to_id)

# ===========================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ===========================================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns are proper types for fast joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ===========================================================================
# STEP 3: Compute all neighbor features in one vectorized pass
# ===========================================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, var_names) {
  # We need: for every (id, year) row, find its neighbors' values and compute
  # max, min, mean for each variable.
  #
  # Approach:
  #   1. Join edge_dt Ã— cell_data on (to_id = id, year) to get neighbor values.
  #   2. Group by (from_id, year) and compute stats.
  #   3. Join results back to cell_data on (id = from_id, year).

  # Subset cell_data to only the columns we need for the neighbor lookup
  # to minimise memory during the join.
  lookup_cols <- c("id", "year", var_names)
  neighbor_vals <- cell_data[, ..lookup_cols]
  setnames(neighbor_vals, "id", "to_id")
  setkey(neighbor_vals, to_id, year)

  # Expand edge list by year: every edge exists in every year.
  # Instead of a full cross-join (which would be 1.37M Ã— 28 = 38.4M rows),
  # we do a keyed join which is efficient.
  # Join: for each edge (from_id, to_id) and each year that from_id appears,
  # pull the neighbor (to_id) values for that year.
  #
  # Efficient approach: join edge_dt with cell_data to get (from_id, year)
  # combinations, then join to get neighbor values.

  # Get all (from_id, year) combinations that exist in the data
  from_years <- cell_data[, .(from_id = id, year)]
  setkey(from_years, from_id)

  # Add neighbor IDs: (from_id, year, to_id)
  setkey(edge_dt, from_id)
  expanded <- edge_dt[from_years, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: from_id, to_id, year
  # Approx rows: 6.46M Ã— avg_neighbors â‰ˆ 6.46M Ã— 4 â‰ˆ 25.8M (manageable)

  # Join neighbor values: attach variable values for (to_id, year)
  setkey(expanded, to_id, year)
  expanded <- neighbor_vals[expanded, on = .(to_id, year), nomatch = NA]
  # Now expanded has: to_id, year, from_id, and all var_name columns

  # Compute grouped stats
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Construct the call
  agg_list <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))

  stats <- expanded[, eval(agg_list), by = .(from_id, year)]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  setnames(stats, "from_id", "id")
  setkey(stats, id, year)

  return(stats)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# ===========================================================================
# STEP 4: Join the neighbor features back onto cell_data
# ===========================================================================
setkey(cell_data, id, year)

# Identify new columns to add (avoid overwriting if re-running)
new_cols <- setdiff(names(neighbor_stats), c("id", "year"))
existing  <- intersect(new_cols, names(cell_data))
if (length(existing) > 0) {
  cell_data[, (existing) := NULL]
}

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# ===========================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ===========================================================================
# The model object (e.g., rf_model) is already in memory or loaded from disk.
# The feature names it expects now exist in cell_data.
# Predict exactly as before â€” the numerical estimand is preserved because
# we computed identical max/min/mean statistics, just more efficiently.
#
# Example (adjust to your actual model object and feature matrix):
# cell_data[, predicted_gdp := predict(rf_model, newdata = cell_data)]
```

---

## Why This Preserves the Original Numerical Estimand

The optimized code computes **exactly the same three summary statistics** (max, min, mean of non-NA neighbor values) for each cell-year and each variable. The only difference is the computational pathway:

- **Original:** Per-row `lapply` â†’ extract neighbor indices â†’ subset values â†’ compute stats.
- **Optimized:** Vectorized edge-list join â†’ grouped `data.table` aggregation â†’ same stats.

The resulting feature columns are numerically identical (up to floating-point associativity, which is negligible here since `mean` is over small neighbor sets of typically 2â€“4 values). The trained Random Forest model is never modified; it simply receives the same feature matrix it was trained on.

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with `paste`/hash lookup | Single vectorized edge-list construction (<1 sec) |
| Temporal expansion + join | Implicit in per-row loop | `data.table` keyed join (~25.8M rows, ~10â€“30 sec) |
| Neighbor stats (5 vars) | 5 Ã— 6.46M `lapply` + `do.call(rbind)` | Single grouped aggregation (~30â€“60 sec) |
| **Total estimated time** | **86+ hours** | **~2â€“5 minutes** |
| Peak memory | Multiple large intermediate lists | ~3â€“5 GB (edge expansion table) |
END_RESPONSE>>>
