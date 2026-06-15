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
run_id: Anthropic_F02_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup`** function. It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:

1. **Character coercion and named-vector lookup** (`id_to_ref[as.character(...)]`) â€” repeated millions of times, each call has O(n) hash overhead.
2. **`paste()` to build keys** for every neighbor of every row â€” millions of small string allocations.
3. **Named-vector indexing** (`idx_lookup[neighbor_keys]`) â€” hash lookups on a vector of length 6.46M, repeated per row.

This produces a **list of 6.46 million integer vectors**, which is itself a large memory object (~hundreds of MB to several GB depending on neighbor counts). Then `compute_neighbor_stats` loops over this list again for **each of the 5 variables**, calling `max`, `min`, `mean` inside an `lapply` of 6.46M iterations â€” adding another ~5 Ã— 6.46M R-level function calls.

**Summary of problems:**

| Problem | Location | Impact |
|---|---|---|
| Per-row `paste` + hash lookup (Ã—6.46M) | `build_neighbor_lookup` | ~80%+ of runtime |
| R-level `lapply` over millions of rows | Both functions | High interpreter overhead |
| Storing 6.46M-element list of integer vectors | `build_neighbor_lookup` output | High memory pressure |
| Repeating the stats loop 5 times independently | Outer loop | Multiplied overhead |

---

## Optimization Strategy

### Key idea: Replace row-level R loops with vectorized `data.table` joins and grouped aggregations.

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, construct a `data.table` of directed neighbor edges `(id, neighbor_id)` from the `nb` object, then join with the panel data on `(neighbor_id, year)` to get neighbor values. This replaces millions of R-level hash lookups with a single indexed equi-join.

2. **Compute all 5 variables' stats in one grouped aggregation** over `(id, year)` after the join, avoiding 5 separate passes.

3. **Use `data.table`'s in-place `:=` assignment** to add columns back to the main table without copying.

4. **Memory management:** The edge table (~1.37M rows Ã— 2 cols) and the join result (~1.37M Ã— 28 years â‰ˆ 38.5M rows, but only for existing pairs) are manageable in 16 GB, especially if we process one variable at a time if needed.

**Expected speedup:** From 86+ hours to **minutes** (the join is O(n log n) or O(n) with keys; grouped aggregation is highly optimized in `data.table`).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build a directed edge table from the nb object (once)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) becomes nothing via unlist)
  # Convert positional indices to actual cell IDs
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ---------------------------------------------------------------
# Step 2: Compute neighbor stats for all variables via join
# ---------------------------------------------------------------
add_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                      neighbor_source_vars) {

  # Convert to data.table if not already (by reference if possible)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # 1. Build edge table: ~1.37M rows of (id, neighbor_id)
  edges <- build_edge_table(id_order, neighbors)

  # 2. Add year via join with cell_data on id
  #    We need edges Ã— years, i.e., for each (id, neighbor_id) pair,
  #    we look up the neighbor's value in the same year.
  #
  #    Strategy: join edges with cell_data to get (id, year, neighbor_id),
  #    then join again to get the neighbor's variable values.

  # Subset cell_data to only the columns we need for the neighbor lookup
  id_year_key <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

  # Key for the join on the "focal" cell side â€” we need all (id, year) combos
  # joined to edges to produce (id, year, neighbor_id)
  # More efficient: join edges to id_year_key on id
  setkey(edges, id)
  setkey(id_year_key, id)

  # This gives us: for each (id, year), all neighbor_ids
  # Result has columns: id, neighbor_id, year, <source_vars for focal cell â€” not needed>
  # We only need id, year, neighbor_id from this join.
  focal_neighbors <- edges[id_year_key[, .(id, year)],
                           on = "id",
                           allow.cartesian = TRUE,
                           nomatch = NULL]
  # focal_neighbors columns: id, neighbor_id, year

  # 3. Now join to get the neighbor's variable values
  #    We need to look up (neighbor_id, year) in id_year_key
  #    Rename for the join:
  setnames(id_year_key, "id", "neighbor_id")
  setkey(id_year_key, neighbor_id, year)
  setkey(focal_neighbors, neighbor_id, year)

  # Equi-join: attach neighbor variable values
  joined <- id_year_key[focal_neighbors, on = .(neighbor_id, year), nomatch = NULL]
  # joined columns: neighbor_id, year, <source_vars>, id

  # 4. Grouped aggregation: compute max, min, mean per (id, year) for each variable
  #    Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Evaluate the grouped aggregation
  stats <- joined[, lapply(agg_exprs, eval), by = .(id, year)]

  # Replace -Inf/Inf from max/min on all-NA groups with NA
  inf_cols <- grep("neighbor_(max|min)_", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # 5. Merge stats back into cell_data
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]

  return(cell_data)
}

# ---------------------------------------------------------------
# Step 3: Usage (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# The trained Random Forest model can be applied as before â€” no retraining needed.
# The numerical estimand (max, min, mean of neighbor values) is preserved exactly.
```

---

## Memory-Constrained Variant

If the single join (`focal_neighbors` can reach ~38M rows Ã— several columns) strains 16 GB, process **one variable at a time**:

```r
add_neighbor_features_lowmem <- function(cell_data, id_order, neighbors,
                                         neighbor_source_vars) {
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

  edges <- build_edge_table(id_order, neighbors)
  setkey(edges, id)

  # Base key table (id, year only)
  base_key <- cell_data[, .(id, year)]
  setkey(base_key, id)

  focal_neighbors <- edges[base_key, on = "id",
                           allow.cartesian = TRUE, nomatch = NULL]
  # columns: id, neighbor_id, year  (~38M rows, 3 int/num cols â‰ˆ ~0.9 GB)
  setkey(focal_neighbors, neighbor_id, year)

  for (v in neighbor_source_vars) {
    message("Processing neighbor stats for: ", v)

    # Extract only the column we need
    lookup <- cell_data[, .(neighbor_id = id, year, val = get(v))]
    setkey(lookup, neighbor_id, year)

    tmp <- lookup[focal_neighbors, on = .(neighbor_id, year), nomatch = NULL]
    # tmp columns: neighbor_id, year, val, id

    agg <- tmp[, .(
      nmax  = max(val, na.rm = TRUE),
      nmin  = min(val, na.rm = TRUE),
      nmean = mean(val, na.rm = TRUE)
    ), by = .(id, year)]

    # Fix Inf/-Inf
    agg[is.infinite(nmax), nmax := NA_real_]
    agg[is.infinite(nmin), nmin := NA_real_]

    setnames(agg, c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    setkey(agg, id, year)
    setkey(cell_data, id, year)
    cell_data <- agg[cell_data, on = .(id, year)]

    rm(tmp, agg, lookup)
    gc()
  }

  return(cell_data)
}

# Usage:
cell_data <- add_neighbor_features_lowmem(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M R-level `paste` + hash lookups | Single `data.table` equi-join |
| Stats computation | 5 Ã— 6.46M `lapply` calls with per-element R functions | Vectorized grouped `data.table` aggregation |
| Estimated runtime | 86+ hours | **5â€“20 minutes** |
| Peak memory | Uncontrolled (6.46M-element list) | ~2â€“4 GB (controlled, with low-mem variant) |
| Numerical output | max, min, mean per neighbor set | **Identical** â€” preserved exactly |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
