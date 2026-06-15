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
run_id: Anthropic_F12_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague is **partially right but mostly wrong** about the main bottleneck. Let me audit each claim:

**Claim 1: `do.call(rbind, result)` is a bottleneck.**
This is a minor concern. `do.call(rbind, ...)` on a list of ~6.46M fixed-length numeric vectors (each length 3) is not trivial, but it completes in seconds-to-minutes, not hours. It's a single matrix allocation and copy operation.

**Claim 2: "Repeated list binding inside `compute_neighbor_stats()`".**
There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses `lapply` to produce a list in one pass, then binds once. This claim is factually wrong about the code.

**The actual deep bottleneck is `build_neighbor_lookup()`.**

Specifically, inside the `lapply` over **6.46 million rows**:

1. **`as.character(data$id[i])` and `id_to_ref[as.character(...)]`** â€” 6.46M individual character conversions and named-vector lookups.
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” 6.46M calls to `paste()` constructing key vectors (each with ~4 neighbors on average for rook contiguity, so ~25.8M string constructions).
3. **`idx_lookup[neighbor_keys]`** â€” 6.46M named-vector lookups, where `idx_lookup` itself has **6.46M entries**. Named vector lookup in R is **O(n)** per query (linear scan or hash with overhead), making the total complexity approximately **O(n Ã— k Ã— m)** where n = 6.46M rows, k = avg neighbors (~4), and m = lookup overhead on a 6.46M-length named vector.

The named-vector `idx_lookup` with 6.46M elements is the **critical bottleneck**. Each lookup into this vector inside the row-level `lapply` is extremely expensive at scale. With ~6.46M iterations Ã— ~4 lookups each, this is what produces the 86+ hour runtime.

`compute_neighbor_stats()`, by contrast, does only **integer indexing** (`vals[idx]`) which is O(1) per element â€” it is fast.

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup()` with a vectorized approach** using `data.table` hash joins or environment-based hash lookups instead of named-vector lookups on a 6.46M-length vector.
2. **Pre-vectorize the entire neighbor-lookup construction** by expanding neighbors at the cell level (344K cells Ã— ~4 neighbors = ~1.37M pairs), then joining with year in a single merge operation â€” eliminating the per-row loop entirely.
3. **Represent the lookup as a sparse adjacency structure** (list of integer row indices) built via `data.table` merge, not per-row string matching.
4. **Vectorize `compute_neighbor_stats()`** using the pre-built integer index lists, which is already reasonably fast but can be improved with `vapply` or direct matrix operations.

The key insight: instead of iterating 6.46M rows and doing string-based lookups each time, we:
- Build a `data.table` of all (cell_id, neighbor_id) pairs (~1.37M rows).
- Cross-join with years to get (cell_id, year, neighbor_id, year) â†’ (row_index, neighbor_row_index) pairs (~1.37M Ã— 28 â‰ˆ 38.5M rows).
- Group by row_index to build the lookup list.

This replaces 6.46M sequential R-level iterations with a single vectorized `data.table` merge.

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup using data.table hash joins
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  # Convert data to data.table if not already (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build a mapping from id to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )

  # Expand the nb object into a data.table of (focal_id, neighbor_id) pairs
  # neighbors is an nb object: a list of integer vectors (indices into id_order)
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    # nb objects use 0 for no-neighbor islands; filter those out
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  # Build keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # For each focal_id and year, find the row_idx of the focal cell
  # Then find row_idx of each neighbor_id in the same year

  # Step 1: Get all unique (focal_id, year) combinations present in data
  focal_dt <- dt[, .(id, year, focal_row_idx = row_idx)]

  # Step 2: Join edge_list with focal_dt to get (focal_row_idx, neighbor_id, year)
  # For each row in the data that has a focal_id in the edge list,
  # we expand to its neighbors
  setnames(focal_dt, "id", "focal_id")
  setkey(edge_list, focal_id)
  setkey(focal_dt, focal_id)

  # Merge: for each focal row, get all its neighbors
  expanded <- merge(focal_dt, edge_list, by = "focal_id", allow.cartesian = TRUE)
  # expanded has columns: focal_id, year, focal_row_idx, neighbor_id

  # Step 3: Look up the row index of each (neighbor_id, year) in the data
  neighbor_key <- dt[, .(neighbor_id = id, year, neighbor_row_idx = row_idx)]
  setkey(neighbor_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  matched <- merge(expanded, neighbor_key, by = c("neighbor_id", "year"), nomatch = 0L)
  # matched has: focal_row_idx, neighbor_row_idx (and other cols)

  # Step 4: Build the lookup list â€” group neighbor_row_idx by focal_row_idx
  setkey(matched, focal_row_idx)
  lookup_dt <- matched[, .(neighbors = list(neighbor_row_idx)), by = focal_row_idx]

  # Initialize result list (all rows, default to empty integer)
  n_rows <- nrow(dt)
  result <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    result[[i]] <- integer(0L)
  }

  # Fill in the non-empty entries
  result[lookup_dt$focal_row_idx] <- lookup_dt$neighbors

  return(result)
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats using vapply
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Use vapply for type-stable output, returns 3 x n matrix
  stats_mat <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))

  # vapply returns 3 x n matrix; transpose to n x 3
  t(stats_mat)
}


# =============================================================================
# Wrapper: compute and add neighbor features (preserves original column naming)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# MAIN PIPELINE â€” drop-in replacement
# =============================================================================
# Build neighbor lookup ONCE (this is the step that was 86+ hours; now ~minutes)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched.
# The numerical estimand is preserved: same max/min/mean neighbor statistics,
# same column names, same row order.
```

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~80â€“86 hours (6.46M row-level string lookups into 6.46M named vector) | **~2â€“5 minutes** (single `data.table` merge of ~38.5M rows) |
| `compute_neighbor_stats` (Ã—5 vars) | ~minutes (integer indexing, already OK) | **~1â€“2 minutes** (`vapply` avoids `do.call(rbind,...)` overhead) |
| **Total** | **86+ hours** | **~5â€“10 minutes** |

## Summary

The colleague's diagnosis was **incorrect**. `do.call(rbind, result)` and the alleged "repeated list binding" (which doesn't exist in the code) are not the bottleneck. The true bottleneck is the **O(nÂ²)-behaving named-vector lookup** in `build_neighbor_lookup()`, where 6.46M rows each perform string-key lookups into a 6.46M-element named vector. Replacing this with `data.table` hash-based equi-joins eliminates the bottleneck entirely while preserving the exact same numerical output and leaving the trained Random Forest model untouched.
END_RESPONSE>>>
