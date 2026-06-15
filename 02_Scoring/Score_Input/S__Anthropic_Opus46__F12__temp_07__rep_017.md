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
run_id: Anthropic_F12_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” called only 5 times total. Each inner function returns a fixed-length numeric vector of length 3, so there is no "repeated list binding" growing an object; `lapply` pre-allocates the list.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here is why:

1. **Character coercion and `paste()` on 6.46M rows:** `paste(data$id, data$year, sep = "_")` creates 6.46 million character keys. Then, for every single row `i` (6.46M iterations), it calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` again â€” creating ~4 character keys per row on average (rook neighbors), totaling ~26 million `paste` + named-vector-lookup operations.

2. **Named vector lookup (`idx_lookup[neighbor_keys]`) is O(n) per probe in the worst case** for long named vectors. With 6.46M names in `idx_lookup` and ~26M lookups, this is catastrophically slow. R's named vector lookup uses linear hashing but with 6.46M entries the constant factor is enormous compared to a proper hash or, better, a direct integer index.

3. **`id_to_ref[as.character(data$id[i])]` is called 6.46M times** â€” each time converting a single integer to character and probing a named vector. This is row-level scalar R code in a hot loop.

4. **The function is called once but produces a list of 6.46M integer vectors**, each constructed through multiple character-key lookups. This single call dominates the entire 86+ hour runtime. `compute_neighbor_stats` is called 5 times and does only numeric indexing â€” it is comparatively fast.

**In summary:** The bottleneck is the row-level `lapply` over 6.46M rows in `build_neighbor_lookup`, driven by millions of `paste()` calls and named-vector character lookups. The fix is to eliminate character-key lookups entirely and replace them with direct integer-indexed operations using vectorized joins.

---

## Optimization Strategy

1. **Replace character key lookups with integer-indexed hash maps** (`data.table` keyed joins or `match()` on integer-pair keys) to build the neighbor lookup.
2. **Vectorize `build_neighbor_lookup`** by expanding the neighbor list into a flat edge table, joining on `(neighbor_id, year)` pairs in one vectorized operation, then splitting back into a list.
3. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation on the flat edge table instead of row-level `lapply`.
4. **Preserve the trained Random Forest model** â€” we only change feature-engineering code; the resulting columns are numerically identical.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 1. OPTIMIZED build_neighbor_lookup (vectorized via data.table)
#    Returns a list of length nrow(data), each element an
#    integer vector of row indices of that row's neighbors.
# ============================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and a natural row order)
  # id_order: vector mapping reference index -> cell id
  # neighbors: spdep nb object (list of integer vectors of neighbor ref indices)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- Step A: Build a mapping from (id, year) -> row_idx ---
  # Use integer keys; avoid all paste/character work.
  setkey(dt, id, year)

  # --- Step B: Build flat edge table of (focal_row, neighbor_id, year) ---
  # Map each cell id to its reference index
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Unique cell ids present in data
  unique_ids <- unique(dt$id)

  # For each unique cell id, find its neighbor cell ids (once per cell, not per row)
  # This loop is over 344,208 unique cells â€” fast.
  neighbor_cell_map <- lapply(as.character(unique_ids), function(cid) {
    ref <- id_to_ref[cid]
    if (is.na(ref) || length(neighbors[[ref]]) == 0) {
      return(integer(0))
    }
    id_order[neighbors[[ref]]]
  })
  names(neighbor_cell_map) <- as.character(unique_ids)

  # Expand: for each row in dt, cross its neighbor_cell_ids with its year.
  # Build this as a flat data.table.
  # First, get the neighbor cell ids per unique id as a data.table:
  edge_dt <- rbindlist(lapply(seq_along(unique_ids), function(j) {
    nids <- neighbor_cell_map[[j]]
    if (length(nids) == 0) return(NULL)
    data.table(focal_id = unique_ids[j], neighbor_id = nids)
  }))

  if (nrow(edge_dt) == 0) {
    return(vector("list", nrow(data)))
  }

  # Now cross with years: each focal_id appears in multiple years.
  # Instead of a full cross, join focal_id -> rows to get (focal_row_idx, neighbor_id, year)
  focal_rows <- dt[, .(focal_row_idx = row_idx, year), by = id]
  setnames(focal_rows, "id", "focal_id")
  setkey(focal_rows, focal_id)
  setkey(edge_dt, focal_id)

  # Merge: each edge (focal_id, neighbor_id) x each year that focal_id appears

  expanded <- edge_dt[focal_rows, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: focal_id, neighbor_id, focal_row_idx, year

  # --- Step C: Look up the row index of (neighbor_id, year) ---
  setkey(dt, id, year)
  neighbor_rows <- dt[, .(neighbor_row_idx = row_idx, id, year)]
  setnames(neighbor_rows, c("id", "year"), c("neighbor_id", "year"))
  setkey(neighbor_rows, neighbor_id, year)

  expanded_matched <- neighbor_rows[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched rows
  expanded_matched <- expanded_matched[!is.na(neighbor_row_idx)]

  # --- Step D: Split into list indexed by focal_row_idx ---
  setkey(expanded_matched, focal_row_idx)
  n_rows <- nrow(data)

  # Use split for speed
  lookup_list <- vector("list", n_rows)
  if (nrow(expanded_matched) > 0) {
    split_result <- split(expanded_matched$neighbor_row_idx, expanded_matched$focal_row_idx)
    idx <- as.integer(names(split_result))
    lookup_list[idx] <- split_result
  }

  # Fill NULLs with integer(0)
  lookup_list[vapply(lookup_list, is.null, logical(1))] <- list(integer(0))

  lookup_list
}


# ============================================================
# 2. OPTIMIZED compute_neighbor_stats (vectorized via data.table)
#    Operates on the flat edge table to avoid 6.46M lapply calls.
# ============================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  # Option A: If neighbor_lookup is already built as a list, we can still
  # vectorize the aggregation by unlisting.

  n <- length(neighbor_lookup)
  lens <- lengths(neighbor_lookup)

  # Focal row indices (repeated for each neighbor)
  focal_idx <- rep.int(seq_len(n), lens)
  # Neighbor row indices (flat)
  neighbor_idx <- unlist(neighbor_lookup, use.names = FALSE)

  vals <- data[[var_name]]
  neighbor_vals <- vals[neighbor_idx]

  # Build data.table for grouped aggregation
  agg_dt <- data.table(focal = focal_idx, nval = neighbor_vals)
  # Remove NAs in neighbor values
  agg_dt <- agg_dt[!is.na(nval)]

  stats <- agg_dt[, .(
    max_val  = max(nval),
    min_val  = min(nval),
    mean_val = mean(nval)
  ), by = focal]

  # Build output matrix (n x 3), fill with NA, then place computed values
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[stats$focal, 1] <- stats$max_val
  out[stats$focal, 2] <- stats$min_val
  out[stats$focal, 3] <- stats$mean_val

  out
}


# ============================================================
# 3. Helper: compute_and_add_neighbor_features (drop-in replacement)
# ============================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]

  data
}


# ============================================================
# 4. OUTER LOOP (drop-in replacement for the original pipeline)
#    Preserves trained RF model; only feature engineering changes.
# ============================================================

# Convert to data.table for speed (if not already)
cell_data <- as.data.table(cell_data)

# Build lookup once â€” this is now vectorized and fast
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# cell_data now has the same 15 new columns (5 vars Ã— {max, min, mean})
# with numerically identical values to the original implementation.
# The trained Random Forest model is untouched and can be used for prediction as-is.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M scalar `paste()` + named-vector probes | Eliminated |
| `build_neighbor_lookup` | O(6.46M) character key lookups in a 6.46M-entry named vector | One `data.table` keyed join on integer pairs |
| `compute_neighbor_stats` | `lapply` over 6.46M rows (acceptable but slow) | Flat `data.table` grouped aggregation |
| `do.call(rbind, ...)` | 5 calls, each binding 6.46M length-3 vectors â€” **not the bottleneck** | Replaced with pre-allocated matrix fill |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes on the same laptop |
| **Numerical output** | Baseline | Identical (same max/min/mean over same neighbor sets) |
| **Trained RF model** | Preserved | Preserved (no retraining) |

The colleague's diagnosis was wrong: `do.call(rbind, ...)` and the inner `lapply` of `compute_neighbor_stats` are minor costs. The dominant bottleneck is `build_neighbor_lookup`'s row-level character-key construction and named-vector probing over 6.46 million rows. Replacing these with vectorized `data.table` integer-keyed joins eliminates the bottleneck.
END_RESPONSE>>>
