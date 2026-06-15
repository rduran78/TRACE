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
run_id: Anthropic_F12_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable â€” only 5 calls total. Each inner function returns a fixed-length vector `c(NA, NA, NA)` or `c(max, min, mean)`, so there is no repeated list *growth* or binding inside the function. This is a standard `lapply` + `do.call(rbind, ...)` pattern that is reasonably efficient.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Character key construction and named-vector lookup over 6.46 million rows.** The function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then does named-vector indexing via `idx_lookup[neighbor_keys]` â€” **once per row, inside an `lapply` over 6.46 million iterations**. Named vector lookup in R is O(n) hash-table probing per call, and constructing millions of character keys is extremely expensive.

2. **`as.character(data$id[i])` and `id_to_ref[...]` are called 6.46 million times** â€” repeated scalar character conversion and named lookup.

3. **The neighbor lookup is year-invariant in structure** but is being redundantly expanded across all 28 years for every single cell-year row. There are only 344,208 unique cells, each with the same neighbors across all 28 years. Yet the function builds 6.46 million entries (344,208 Ã— 28), doing redundant string-pasting and hash lookups for every year.

4. **Estimated cost**: ~6.46M iterations Ã— ~4 neighbors on average Ã— (paste + hash lookup) = tens of billions of character operations. This dwarfs the cost of `do.call(rbind, ...)` on 5 pre-allocated matrices.

The `compute_neighbor_stats()` function, while improvable, is secondary: its inner `lapply` does only integer indexing into a numeric vector â€” fast operations. The `do.call(rbind, ...)` on a list of 6.46M length-3 vectors takes seconds, not hours.

## Optimization Strategy

1. **Build the neighbor lookup at the cell level (344,208 entries), not the cell-year level (6.46M entries).** The rook-neighbor structure is time-invariant.

2. **Use integer-indexed group mapping instead of character key hashing.** Map each `(id, year)` to a row index using a fast integer-keyed approach (e.g., `data.table`), then for each cell-year row, find neighbor rows by joining cell-level neighbor IDs with the same year â€” but do this in a vectorized/batch fashion, not row-by-row.

3. **Vectorize `compute_neighbor_stats()` entirely** using `data.table` grouped operations or matrix indexing, eliminating the per-row `lapply`.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build cell-level neighbor lookup (time-invariant)
#         Only 344,208 entries instead of 6.46 million.
# ============================================================

build_cell_neighbor_lookup <- function(id_order, rook_neighbors_unique) {
  # id_order[i] is the cell id for the i-th entry in the nb object.
  # rook_neighbors_unique[[i]] gives integer indices into id_order
  # for the neighbors of cell id_order[i].
  #
  # Returns a named list: cell_id (character) -> vector of neighbor cell_ids (integer/matching type)
  n <- length(id_order)
  lookup <- vector("list", n)
  names(lookup) <- as.character(id_order)
  for (i in seq_len(n)) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      lookup[[i]] <- integer(0)
    } else {
      lookup[[i]] <- id_order[nb_idx]
    }
  }
  lookup
}

# ============================================================
# STEP 2: Vectorized neighbor stats using data.table
#         Processes all 6.46M rows Ã— all neighbors in batch.
# ============================================================

compute_neighbor_features_fast <- function(cell_data, neighbor_source_vars,
                                           id_order, rook_neighbors_unique) {

  # Convert to data.table if not already; preserve row order
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # --- Cell-level neighbor lookup ---
  cell_nb <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

  # Build an edge list: (cell_id, neighbor_id) â€” one row per directed edge
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb <- cell_nb[[i]]
    if (length(nb) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = nb)
  }))

  # --- Row index by (id, year) for fast join ---
  # Add row index to dt
  dt[, .row_idx := .I]

  # Key columns for joining
  # We need: for each row (id, year), find all neighbor rows (neighbor_id, same year)

  # Create a mapping: (id, year) -> row_idx
  setkey(dt, id, year)

  # Expand edge_list by year: for each (cell, neighbor) pair, we need all years.
  # But instead of a massive cross join, we join edges onto the data.
  #
  # Strategy:
  #   1. Join dt with edge_list on dt$id == edge_list$id to get (row_idx, year, neighbor_id)
  #   2. Join result with dt on (neighbor_id == id, same year) to get neighbor row indices
  #   3. Extract neighbor values and aggregate

  # Step 2a: For each row in dt, get its neighbor cell IDs
  # This is a join: dt[, .(row_idx = .row_idx, id, year)] joined with edge_list on id
  cat("Building row-to-neighbor-row mapping...\n")

  row_info <- dt[, .(.row_idx, id, year)]
  setkey(row_info, id)
  setkey(edge_list, id)

  # Each row in dt gets expanded by its number of neighbors
  row_to_nb_cell <- edge_list[row_info, on = "id", allow.cartesian = TRUE,
                               nomatch = NULL]
  # Columns: id, neighbor_id, .row_idx, year
  # .row_idx is the index of the focal row; neighbor_id is the cell id of the neighbor

  # Step 2b: Find the row index of each (neighbor_id, year) in dt
  # Create a lookup: (id, year) -> .row_idx for neighbor side
  nb_row_lookup <- dt[, .(nb_row_idx = .row_idx, nb_id = id, year)]
  setkey(nb_row_lookup, nb_id, year)
  setkey(row_to_nb_cell, neighbor_id, year)

  row_to_nb_row <- nb_row_lookup[row_to_nb_cell,
                                  on = c("nb_id==neighbor_id", "year"),
                                  nomatch = NA,
                                  allow.cartesian = FALSE]
  # Columns: nb_row_idx, nb_id, year, id, .row_idx
  # .row_idx = focal row, nb_row_idx = neighbor's row in dt

  # Drop rows where neighbor row was not found (edge cells in certain years)
  row_to_nb_row <- row_to_nb_row[!is.na(nb_row_idx)]

  cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")

  # Step 3: For each variable, extract neighbor values, group by focal row, compute stats
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")

    # Extract neighbor values via integer indexing (very fast)
    vals_vec <- dt[[var_name]]
    row_to_nb_row[, nb_val := vals_vec[nb_row_idx]]

    # Remove NA neighbor values before aggregation
    valid <- row_to_nb_row[!is.na(nb_val)]

    # Grouped aggregation â€” data.table is highly optimized for this
    agg <- valid[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = .row_idx]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign aggregated values back by row index
    dt[agg$.row_idx, (max_col)  := agg$nb_max]
    dt[agg$.row_idx, (min_col)  := agg$nb_min]
    dt[agg$.row_idx, (mean_col) := agg$nb_mean]

    # Clean up temp column
    row_to_nb_row[, nb_val := NULL]
  }

  # Restore original order and return as data.frame
  setorder(dt, .row_order)
  dt[, c(".row_order", ".row_idx") := NULL]

  as.data.frame(dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# The trained Random Forest model is untouched.
# The output columns are numerically identical to the original implementation.
# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup construction** | 6.46M `lapply` iterations with `paste()` + named-vector hash lookup | One-time vectorized `data.table` join; cell-level lookup is only 344K entries |
| **Key construction** | ~6.46M `paste()` calls creating character keys per row | Integer-keyed `data.table` joins on `(id, year)` â€” no character keys |
| **Neighbor stats** | 6.46M `lapply` iterations per variable Ã— 5 variables = 32.3M R function calls | 5 vectorized `data.table` grouped aggregations over a pre-built edge table |
| **`do.call(rbind, ...)`** | 5 calls binding 6.46M tiny vectors (colleague's claimed bottleneck â€” actually minor) | Eliminated entirely; replaced by column assignment |
| **Estimated time** | 86+ hours | **Minutes** (dominated by the one-time join expansion, which `data.table` handles efficiently in memory) |

## Memory Consideration

The expanded edge table (`row_to_nb_row`) will have approximately 6.46M Ã— 4 (average neighbors) â‰ˆ 25.8M rows with a few integer columns â€” roughly 600 MBâ€“1 GB, well within the 16 GB laptop constraint. The original `neighbor_lookup` list of 6.46M entries with integer vectors likely consumed comparable memory.

## Summary

The colleague misidentified `do.call(rbind, ...)` as the bottleneck. The true bottleneck is `build_neighbor_lookup()`, which performs **6.46 million scalar character-key constructions and named-vector hash lookups** to redundantly expand a time-invariant spatial structure across 28 years. The fix is to (1) keep the neighbor topology at the cell level, (2) use `data.table` integer-keyed joins to resolve `(neighbor_id, year)` â†’ row mappings in batch, and (3) compute grouped aggregations vectorially instead of row-by-row `lapply`.
END_RESPONSE>>>
