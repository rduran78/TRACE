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
run_id: Anthropic_F12_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable â€” only 5 calls total. Each inner function returns a fixed-length vector `c(NA, NA, NA)` or `c(max, min, mean)`, so there is no repeated list growth or progressive binding inside the function. This is a standard `lapply` â†’ `rbind` pattern that handles ~6.46M rows in seconds to low minutes.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup` â€” a named character vector of length ~6.46 million. Named vector lookup in R is **O(n)** per query (linear scan or hash lookup with significant overhead at this scale).

2. **This is called ~6.46 million times**, and each call does multiple named-vector lookups (one per neighbor, typically 4 for rook contiguity). That means roughly **25+ million named-vector lookups** against a 6.46M-entry named vector, all wrapped in `lapply` with per-element R function call overhead.

3. **Character coercion and string operations dominate.** `as.character(data$id[i])`, `paste(...)`, and named indexing are all expensive string operations repeated per row.

4. The function is called **once**, but it takes the vast majority of the 86+ hour runtime. `compute_neighbor_stats` is called only 5 times and does simple numeric indexing (`vals[idx]`) â€” which is O(1) per element in R.

**Summary:** The bottleneck is the O(N Ã— K) string-based lookup architecture in `build_neighbor_lookup()`, not the `rbind` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all string operations from `build_neighbor_lookup()`.** Replace the `paste`-based key scheme with pure integer arithmetic. Encode each `(id, year)` pair as a single integer: `(id_index - 1) * n_years + year_index`. Build a direct integer-to-row mapping vector (not a named vector) so lookup is O(1) array indexing.

2. **Vectorize the neighbor lookup construction.** Instead of `lapply` over 6.46M rows with per-row R function calls, use vectorized operations: expand all neighbor pairs at once using `data.table` or vectorized integer math, then do a single merge/match.

3. **Vectorize `compute_neighbor_stats()`.** Replace the `lapply` + `do.call(rbind, ...)` with `data.table` grouped aggregation over a pre-built edge list, computing max/min/mean in one pass per variable.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup (integer-indexed, vectorized)
# ============================================================
build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order: vector of unique spatial cell IDs in the order matching 'neighbors'
  # neighbors: spdep nb object (list of integer neighbor index vectors)

  # Step 1: Map cell IDs to integer indices (1-based, matching id_order)
  n_cells <- length(id_order)
  id_to_idx <- integer(max(id_order))
  id_to_idx[id_order] <- seq_len(n_cells)
  # If IDs are not guaranteed to be small positive integers, use a hash:
  # id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  # But for typical grid cell IDs that are integers, direct indexing is fastest.

  # Step 2: Map years to integer indices
  years_unique <- sort(unique(data_dt$year))
  n_years <- length(years_unique)
  year_to_idx <- integer(max(years_unique))
  year_to_idx[years_unique] <- seq_len(n_years)

  # Step 3: Build a direct row-lookup array.
  # Encode (cell_idx, year_idx) -> row number in data_dt.
  # Composite key: (cell_idx - 1) * n_years + year_idx
  max_key <- n_cells * n_years
  key_to_row <- integer(max_key)  # 0 means "no row"

  cell_indices <- id_to_idx[data_dt$id]
  year_indices <- year_to_idx[data_dt$year]
  composite_keys <- (cell_indices - 1L) * n_years + year_indices
  key_to_row[composite_keys] <- seq_len(nrow(data_dt))

  # Step 4: Build the edge list (focal_row -> neighbor_row) fully vectorized.
  # Expand the nb object into an edge list of (focal_cell_idx, neighbor_cell_idx)
  focal_cell_idx <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_cell_idx <- unlist(neighbors)

  # Remove zero-length / self-referencing if any (spdep nb objects use 0 for no neighbors)
  valid <- neighbor_cell_idx > 0L
  focal_cell_idx <- focal_cell_idx[valid]
  neighbor_cell_idx <- neighbor_cell_idx[valid]

  # Now cross with all years to get (focal_cell_idx, neighbor_cell_idx, year_idx)
  # This gives the full directed edge list across all cell-years.
  # n_edges_spatial = length(focal_cell_idx)  (~1.37M)
  # Total edges = n_edges_spatial * n_years   (~1.37M * 28 â‰ˆ 38.5M) â€” manageable

  focal_cell_rep <- rep(focal_cell_idx, each = n_years)
  neighbor_cell_rep <- rep(neighbor_cell_idx, each = n_years)
  year_idx_rep <- rep(seq_len(n_years), times = length(focal_cell_idx))

  # Compute composite keys for focal and neighbor
  focal_keys <- (focal_cell_rep - 1L) * n_years + year_idx_rep
  neighbor_keys <- (neighbor_cell_rep - 1L) * n_years + year_idx_rep

  # Map to actual row numbers
  focal_rows <- key_to_row[focal_keys]
  neighbor_rows <- key_to_row[neighbor_keys]

  # Keep only edges where both focal and neighbor rows exist in the data
  valid_edges <- focal_rows > 0L & neighbor_rows > 0L
  edge_dt <- data.table(
    focal_row = focal_rows[valid_edges],
    neighbor_row = neighbor_rows[valid_edges]
  )

  return(edge_dt)
}

# ============================================================
# OPTIMIZED compute_neighbor_stats (data.table grouped agg)
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_dt, var_name) {
  # Extract neighbor values via integer indexing (vectorized, O(1) per element)
  neighbor_vals <- data_dt[[var_name]][edge_dt$neighbor_row]

  # Build a temporary table for grouped aggregation
  agg_dt <- data.table(
    focal_row = edge_dt$focal_row,
    val = neighbor_vals
  )

  # Remove NAs before aggregation
  agg_dt <- agg_dt[!is.na(val)]

  # Grouped aggregation â€” single pass, highly optimized in data.table
  stats <- agg_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Initialize output columns with NA
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  # Fill in computed values
  out_max[stats$focal_row]  <- stats$nb_max
  out_min[stats$focal_row]  <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  # Return as a data.table with the standard column naming convention
  setNames(
    data.table(out_max, out_min, out_mean),
    paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  )
}

# ============================================================
# MAIN PIPELINE (drop-in replacement)
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table for speed (non-destructive copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Building vectorized neighbor edge list...\n")
  t0 <- Sys.time()
  edge_dt <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
  cat("  Edge list built:", nrow(edge_dt), "directed edges in",
      round(difftime(Sys.time(), t0, units = "secs"), 1), "seconds.\n")

  # Neighbor source variables (same as original)
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    t1 <- Sys.time()
    stats_cols <- compute_neighbor_stats_fast(cell_data, edge_dt, var_name)
    # Bind new columns to cell_data (by reference)
    cell_data[, names(stats_cols) := stats_cols]
    cat("  Done in", round(difftime(Sys.time(), t1, units = "secs"), 1), "seconds.\n")
  }

  return(cell_data)
}

# ============================================================
# USAGE (preserves trained RF model â€” no retraining needed)
# ============================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # The resulting cell_data now has the same 15 neighbor feature columns
# # (5 vars Ã— 3 stats each) with numerically identical values.
# # Feed directly into the existing trained Random Forest for prediction:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Solves the Problem

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations, each doing string `paste` + named-vector lookup (~25M string matches against a 6.46M-entry named vector) | Single vectorized expansion: integer arithmetic, no strings, O(1) array indexing |
| **Neighbor stats** | 6.46M `lapply` iterations + `do.call(rbind, 6.46M-element list)` per variable | `data.table` grouped aggregation on a pre-built integer edge list â€” single pass, C-level |
| **Total string operations** | ~50+ million `paste` and named-vector lookups | **Zero** |
| **Estimated runtime** | 86+ hours | **Minutes** (edge list build: ~30â€“60s; stats per variable: ~10â€“30s; total: ~5 min) |
| **Memory** | Acceptable | ~38.5M-row edge table Ã— 2 integer columns â‰ˆ ~300 MB â€” fits in 16 GB |
| **Numerical output** | Original | **Identical** â€” same max/min/mean over same neighbor sets |
| **Trained RF model** | â€” | **Preserved** â€” no retraining required |

The core insight: the original code's catastrophic performance came from **millions of R-level function calls doing string construction and named-vector hash lookups**, not from `do.call(rbind, ...)`. The fix is to replace the entire string-keyed architecture with pure integer arithmetic and vectorized `data.table` aggregation.
END_RESPONSE>>>
