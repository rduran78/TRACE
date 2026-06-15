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
run_id: Anthropic_F12_T00_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" growing a structure iteratively; `lapply` pre-allocates the list.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Per-row string pasting and named-vector lookup (O(n) Ã— constant but enormous n).** For each of the ~6.46 million rows, the function calls `as.character(data$id[i])`, performs a named-vector lookup in `id_to_ref`, pastes every neighbor ID with the current year (`paste(neighbor_cell_ids, data$year[i], sep = "_")`), and then looks each key up in `idx_lookup` â€” another named character vector of length 6.46 million. Named-vector lookup in R is hash-based but carries significant per-call overhead when invoked millions of times inside an `lapply` scalar loop.

2. **Multiplied by neighbor fan-out.** With ~1.37 million directed rook-neighbor relationships across 344,208 cells, the average cell has ~4 neighbors. For every one of the 6.46 million rows, the function constructs ~4 paste keys and performs ~4 hash lookups. That is **~25.8 million `paste` + hash-probe operations inside a scalar R loop** â€” the dominant cost.

3. **The lookup is rebuilt from scratch every run** even though the spatial topology is static across all 28 years. The function loops over every cell-year row, yet the neighbor *structure* is year-invariant; only the *row indices* change by year.

4. **`compute_neighbor_stats` is comparatively cheap.** It does only `vals[idx]` (integer subsetting â€” very fast), a few simple aggregations, and one `do.call(rbind, ...)` per variable. Profiling arithmetic: 5 variables Ã— 1 `do.call(rbind, 6.46M-element list)` â‰ˆ seconds. The lookup construction is hours.

**Verdict:** The bottleneck is the **~6.46 million iterations of scalar string manipulation and hash lookup inside `build_neighbor_lookup()`**, not the `rbind` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` completely** â€” eliminate the per-row `lapply`. Exploit the fact that the neighbor graph is year-invariant: build a cell-level neighbor map once (344K cells), then expand it to all 28 years using vectorized integer arithmetic instead of string hashing.

2. **Replace `do.call(rbind, ...)` with direct matrix construction** in `compute_neighbor_stats` â€” pre-allocate a matrix and fill it, or use `vapply` which returns a matrix directly. This is a minor but easy win.

3. **Use `data.table` for the row-index mapping** instead of named character vectors, giving O(1) keyed joins on integer columns rather than string hashing.

The optimized pipeline reduces the complexity from ~25.8 million scalar R-loop iterations with string operations to a handful of fully vectorized joins and integer operations, bringing runtime from 86+ hours down to minutes.

---

## Working R Code

```r
# ============================================================
# Optimized pipeline â€” preserves trained RF model and original
# numerical estimand (max, min, mean of neighbor values).
# ============================================================

library(data.table)

# ----------------------------------------------------------
# 1. Vectorized neighbor-lookup builder
#    Key insight: the rook-neighbor topology is YEAR-INVARIANT.
#    Build a cell-level edge list once, then map to row indices
#    for all years via a single keyed join.
# ----------------------------------------------------------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be ordered consistently)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- cell-level edge list (year-invariant) ---
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(focal_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # --- Map every (focal_id, year) row to its neighbor rows ---
  # Create a keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Expand: for each row in dt, find its neighbor cell IDs
  # Join dt with edge_list on id == focal_id to get neighbor_id per row
  focal <- dt[, .(focal_row = row_idx, focal_id = id, year)]
  setkey(edge_list, focal_id)

  # merge: each focal row gets its neighbor cell IDs
  expanded <- edge_list[focal, on = .(focal_id), allow.cartesian = TRUE,
                        nomatch = NA]
  # expanded has columns: focal_id, neighbor_id, focal_row, year

  # Drop rows with no neighbors (neighbor_id == NA)
  expanded <- expanded[!is.na(neighbor_id)]

  # Now join to get the ROW INDEX of each (neighbor_id, year) pair
  neighbor_rows <- dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_rows, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  merged <- neighbor_rows[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Keep only matched rows (neighbor exists in that year)
  merged <- merged[!is.na(neighbor_row)]

  # --- Build the lookup list (one element per row of data) ---
  # Order by focal_row so we can split efficiently
  setkey(merged, focal_row)

  n_rows <- nrow(dt)
  lookup <- vector("list", n_rows)

  # Split neighbor_row by focal_row
  split_idx <- split(merged$neighbor_row, merged$focal_row)

  # Assign into lookup (focal rows with no neighbors remain NULL â†’ handle later)
  focal_rows_with_neighbors <- as.integer(names(split_idx))
  for (j in seq_along(split_idx)) {
    lookup[[focal_rows_with_neighbors[j]]] <- split_idx[[j]]
  }

  # Replace NULLs with integer(0)
  null_mask <- vapply(lookup, is.null, logical(1))
  if (any(null_mask)) {
    lookup[null_mask] <- list(integer(0))
  }

  lookup
}

# ----------------------------------------------------------
# 2. Optimized compute_neighbor_stats using vapply
#    Returns an nÃ—3 matrix (max, min, mean) â€” identical
#    numerical output to the original.
# ----------------------------------------------------------

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]

  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, numeric(3))

  t(result)  # transpose to nÃ—3
}

# ----------------------------------------------------------
# 3. Wrapper that adds the three columns per variable
#    (preserves original column-naming convention)
# ----------------------------------------------------------

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]
  data
}

# ----------------------------------------------------------
# 4. Main pipeline execution
# ----------------------------------------------------------

# Build the lookup ONCE (vectorized, minutes not hours)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# cell_data now has the same 15 new columns (5 vars Ã— {max, min, mean})
# with identical numerical values as the original pipeline.
# The trained Random Forest model is untouched and can be used directly
# for prediction on the updated cell_data.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M scalar iterations with `paste` + named-vector hash lookups | Eliminated |
| **Lookup construction** | `lapply` over 6.46M rows, string keys | Vectorized `data.table` keyed join on integer columns; one cell-level edge list expanded by year |
| **`compute_neighbor_stats`** | `do.call(rbind, ...)` on 6.46M-element list (minor cost) | `vapply` â†’ direct matrix, `t()` transpose (marginal improvement) |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the `data.table` join on ~38M expanded edges) |
| **Trained RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical output** | max / min / mean per neighbor set | Identical âœ“ |
END_RESPONSE>>>
