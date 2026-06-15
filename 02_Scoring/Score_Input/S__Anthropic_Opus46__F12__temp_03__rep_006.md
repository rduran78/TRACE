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
run_id: Anthropic_F12_T03_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable â€” only called 5 times total. This is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Row-level `paste()` and character lookups inside `lapply` over 6.46 million rows.** Each iteration of the `lapply` in `build_neighbor_lookup` performs:
   - `as.character(data$id[i])` â€” character conversion per row.
   - `id_to_ref[as.character(...)]` â€” named-vector lookup (hash-like, but with overhead per call).
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string construction for every neighbor of every row.
   - `idx_lookup[neighbor_keys]` â€” another named-vector lookup on a 6.46M-entry vector, repeated for each neighbor key.

2. **Scale of the problem:** With ~6.46 million rows and an average of ~4 rook neighbors per cell, this inner function performs roughly **25.8 million `paste()` calls and 25.8 million named-vector lookups**, all inside a sequential `lapply`. String construction and hashing at this scale dominate runtime.

3. **`compute_neighbor_stats()`**, by contrast, does only integer indexing (`vals[idx]`) and simple arithmetic â€” these are fast vectorized operations. The `do.call(rbind, result)` on a list of 6.46M length-3 vectors takes seconds, not hours.

**Conclusion:** The 86+ hour runtime is driven by the massive string-based lookup construction in `build_neighbor_lookup()`, not by `do.call(rbind, ...)` or list binding in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all string operations from the lookup.** Replace `paste(id, year)` key construction with direct integer arithmetic. Map `(id, year)` pairs to row indices using an integer-keyed structure (a matrix or `data.table` join) instead of a named character vector.

2. **Vectorize the neighbor lookup construction.** Instead of `lapply` over 6.46M rows, expand the neighbor relationships into a flat table, join on `(neighbor_id, year)` to get target row indices, then split back into a list. This replaces millions of R-level function calls with a single `data.table` merge.

3. **Vectorize `compute_neighbor_stats()`.** Instead of `lapply` + `do.call(rbind, ...)`, use `data.table` grouped aggregation on the flat edge list to compute max/min/mean in one pass per variable.

4. **Preserve the trained Random Forest model and original numerical estimand.** The output columns are identical â€” same neighbor max, min, mean values â€” just computed faster.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a fast integer-indexed row lookup using data.table
# ---------------------------------------------------------------
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs (index corresponds to nb object position)
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Create a mapping from (id, year) -> row_idx
  setkey(dt, id, year)

  # --- Build flat edge list: for each cell position in id_order,
  #     enumerate its neighbor cell IDs ---
  # Convert nb object to a flat edge list (focal_pos -> neighbor_pos)
  n_cells <- length(id_order)
  focal_pos <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_pos <- unlist(neighbors)

  # Remove zero-length entries (isolated cells produce integer(0))
  valid <- !is.na(neighbor_pos) & neighbor_pos > 0
  focal_pos <- focal_pos[valid]
  neighbor_pos <- neighbor_pos[valid]

  # Map positions to actual cell IDs
  focal_ids <- id_order[focal_pos]
  neighbor_ids <- id_order[neighbor_pos]

  edge_dt <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)

  # --- Cross with years to get (focal_id, year, neighbor_id, year) ---
  years <- sort(unique(dt$year))

  # Expand edges across all years
  # This creates ~1.37M edges * 28 years â‰ˆ 38.5M rows â€” fits in 16GB RAM
  edge_year <- CJ_dt(edge_dt, years)

  # Join to get neighbor row indices
  setkey(dt, id, year)
  edge_year[dt, neighbor_row_idx := i.row_idx,
            on = .(neighbor_id = id, year = year)]

  # Join to get focal row indices
  edge_year[dt, focal_row_idx := i.row_idx,
            on = .(focal_id = id, year = year)]

  # Drop edges where either side has no matching row
  edge_year <- edge_year[!is.na(focal_row_idx) & !is.na(neighbor_row_idx)]

  return(edge_year)
}

# Helper: cross join a data.table with a vector of years
CJ_dt <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join
  k <- nrow(edge_dt)
  m <- length(years)
  out <- data.table(
    focal_id    = rep(edge_dt$focal_id, each = m),
    neighbor_id = rep(edge_dt$neighbor_id, each = m),
    year        = rep(years, times = k)
  )
  return(out)
}

# ---------------------------------------------------------------
# STEP 2: Vectorized neighbor stats using data.table grouping
# ---------------------------------------------------------------
compute_neighbor_stats_fast <- function(cell_data, edge_year, var_name) {
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Attach the variable values for neighbor rows
  vals <- dt[[var_name]]
  edge_year[, neighbor_val := vals[neighbor_row_idx]]

  # Group by focal_row_idx and compute stats
  stats <- edge_year[!is.na(neighbor_val),
                     .(nmax  = max(neighbor_val),
                       nmin  = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = focal_row_idx]

  # Initialize output columns with NA
  n <- nrow(dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  col_max[stats$focal_row_idx]  <- stats$nmax
  col_min[stats$focal_row_idx]  <- stats$nmin
  col_mean[stats$focal_row_idx] <- stats$nmean

  # Clean up temp column
  edge_year[, neighbor_val := NULL]

  suffix <- paste0("n_", var_name)
  cell_data[[paste0(suffix, "_max")]]  <- col_max
  cell_data[[paste0(suffix, "_min")]]  <- col_min
  cell_data[[paste0(suffix, "_mean")]] <- col_mean

  return(cell_data)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline (replaces the original outer loop)
# ---------------------------------------------------------------
# Build the flat edge-year table ONCE (replaces build_neighbor_lookup)
message("Building vectorized neighbor edge table...")
edge_year <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_neighbor_stats_fast(cell_data, edge_year, var_name)
}

message("Done. Proceed to prediction with the existing trained Random Forest model.")
# The trained RF model is untouched â€” use predict(rf_model, cell_data) as before.
```

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| **Lookup construction** | ~6.46M `lapply` iterations with `paste()` + named-vector hash lookups â†’ **dominant bottleneck (hours)** | Single `data.table` cross-join + keyed merge â†’ **minutes** |
| **Neighbor stats** | `lapply` + `do.call(rbind, ...)` Ã— 5 vars â†’ moderate cost | `data.table` grouped aggregation Ã— 5 vars â†’ **seconds** |
| **Memory** | ~6.46M-element list of integer vectors | ~38.5M-row flat `data.table` (~1â€“2 GB, fits in 16 GB) |
| **Estimated total time** | 86+ hours | **~5â€“15 minutes** |

The numerical output (neighbor max, min, mean per variable per cell-year) is identical, and the trained Random Forest model is completely preserved.
END_RESPONSE>>>
