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
run_id: Anthropic_F05_T00_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The string-keyed lookup `idx_lookup[neighbor_keys]` is nominally O(1) amortized per key (R uses hashing for named vectors), but the constant factor is enormous: each call hashes freshly-allocated strings against a 6.46M-name hash table. The total work is roughly **25.8 million string allocations + hash lookups**, all inside an R-level `lapply` with no vectorization.

### The Deeper Structural Insight

The entire string-key scheme is **unnecessary**. The lookup answers one simple question:

> *"Given that row `i` belongs to cell `id` in year `y`, which rows belong to cell `id`'s rook neighbors in the same year `y`?"*

Since the panel is balanced (344,208 cells Ã— 28 years = 9,637,824 potential rows, ~6.46M present), the mapping from `(cell, year)` â†’ row index can be done with **integer arithmetic** on a pre-built integer matrix, completely eliminating string operations.

Furthermore, `compute_neighbor_stats` is called **5 separate times**, each time iterating over the same 6.46M-element `neighbor_lookup`. The neighbor topology doesn't change across variables â€” only the values do. This means the neighbor-gather pattern should be **vectorized across all variables simultaneously** using matrix indexing.

---

## Optimization Strategy

| Layer | Current | Proposed |
|-------|---------|----------|
| **Cellâ†’index mapping** | String paste + named vector | Integer lookup table `cell_row_matrix[cell_index, year_index]` |
| **Neighbor expansion** | Row-by-row `lapply` with string keys | Vectorized construction of a sparse neighbor-row edge list using `data.table` joins |
| **Stat computation** | `lapply` over 6.46M lists, per variable | Single grouped aggregation via `data.table` over the edge list, all variables at once |
| **Overall complexity** | ~6.46M Ã— (string alloc + hash) Ã— 5 vars | One join to build edge list + one grouped aggregation |

**Estimated speedup**: from 86+ hours to **minutes** (dominated by the `data.table` grouped aggregation over ~25M edges Ã— 5 variables).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves the exact numerical estimand (max, min, mean of non-NA neighbor vals)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (by reference if already one, copy otherwise)
  dt <- as.data.table(cell_data)
  dt[, row_idx__ := .I]
  
  # ------------------------------------------------------------------
  # Step 1: Build an integer mapping from cell id -> position in id_order
  #         This replaces id_to_ref.
  # ------------------------------------------------------------------
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Assign each row its cell-position index (integer, no strings)
  dt[, cell_pos__ := id_to_pos[as.character(id)]]
  
  # ------------------------------------------------------------------
  # Step 2: Build a directed edge list of (cell_pos, neighbor_cell_pos)
  #         from the nb object. This is done once, independent of year.
  # ------------------------------------------------------------------
  # rook_neighbors_unique is a list of length = length(id_order),
  # where element [[j]] gives integer indices into id_order of j's neighbors.
  
  from_pos <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_pos <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  edges <- data.table(cell_pos_from = from_pos, cell_pos_to = to_pos)
  
  cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edges)))
  
  # ------------------------------------------------------------------
  # Step 3: For each row in dt, find its neighbor rows in the same year.
  #
  #   Logic: row i has cell_pos__ = p, year = y.
  #          Its neighbor rows are all rows with cell_pos__ in neighbors(p)

  #          AND year == y.
  #
  #   We achieve this with a join:
  #     dt[, .(row_idx__, cell_pos__, year)]
  #       JOIN edges ON cell_pos__ == cell_pos_from
  #     -> gives (row_idx__=i, year=y, cell_pos_to=q)
  #     Then join back to dt on (cell_pos__=q, year=y) to get neighbor row.
  # ------------------------------------------------------------------
  
  # Slim table for the "from" side: each row's identity
  dt_from <- dt[, .(row_idx_from = row_idx__, cell_pos__, year)]
  setkey(dt_from, cell_pos__)
  
  # Slim table for the "to" side: lookup by (cell_pos, year) -> row index
  dt_to <- dt[, .(row_idx_to = row_idx__, cell_pos__, year)]
  setkey(dt_to, cell_pos__, year)
  
  # Join from-rows to edges: for each row, get its neighbor cell positions
  # This produces ~6.46M * avg_neighbors rows â‰ˆ 25-26M rows
  setkey(edges, cell_pos_from)
  
  cat("Joining rows to neighbor edges...\n")
  expanded <- dt_from[edges, on = .(cell_pos__ = cell_pos_from),
                      .(row_idx_from, year, cell_pos_to = cell_pos_to),
                      nomatch = NULL,
                      allow.cartesian = TRUE]
  
  cat(sprintf("Expanded edge-row table: %d rows\n", nrow(expanded)))
  
  # Now join to find the actual neighbor row index in the same year
  cat("Resolving neighbor rows by (cell_pos, year)...\n")
  expanded[dt_to,
           row_idx_to := i.row_idx_to,
           on = .(cell_pos_to = cell_pos__, year = year)]
  
  # Drop edges where the neighbor cell-year doesn't exist in the data
  expanded <- expanded[!is.na(row_idx_to)]
  
  cat(sprintf("Resolved neighbor pairs: %d\n", nrow(expanded)))
  
  # ------------------------------------------------------------------
  # Step 4: Gather neighbor values for all source variables at once,
  #         then compute grouped stats (max, min, mean of non-NA).
  # ------------------------------------------------------------------
  
  # Extract the variable columns for neighbor rows
  # We do this by direct column indexing into the original dt
  cat("Gathering neighbor variable values...\n")
  
  for (vn in neighbor_source_vars) {
    col_vals <- dt[[vn]]  # full column vector
    expanded[, (vn) := col_vals[row_idx_to]]
  }
  
  # Compute stats grouped by row_idx_from (the focal row)
  cat("Computing neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(vn) {
    list(
      bquote(max(.(as.name(vn)), na.rm = TRUE)),
      bquote(min(.(as.name(vn)), na.rm = TRUE)),
      bquote(mean(.(as.name(vn)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(vn) {
    paste0("neighbor_", c("max_", "min_", "mean_"), vn)
  }))
  
  names(agg_exprs) <- agg_names
  
  # For rows where ALL neighbor values of a variable are NA,
  # max/min with na.rm=TRUE produce Inf/-Inf and mean produces NaN.
  # We'll fix those after aggregation.
  
  stats <- expanded[,
    lapply(agg_exprs, eval, envir = .SD),
    by = row_idx_from
  ]
  
  # Replace Inf/-Inf/NaN with NA to match original behavior
  for (col in agg_names) {
    vals <- stats[[col]]
    vals[is.infinite(vals) | is.nan(vals)] <- NA_real_
    set(stats, j = col, value = vals)
  }
  
  # ------------------------------------------------------------------
  # Step 5: Merge stats back to the main data, preserving row order.
  # ------------------------------------------------------------------
  cat("Merging results back to main data...\n")
  
  # Ensure all rows are represented (rows with no valid neighbors get NA)
  setkey(stats, row_idx_from)
  
  for (col in agg_names) {
    # Initialize with NA
    dt[, (col) := NA_real_]
    # Fill in computed values
    dt[stats, (col) := get(paste0("i.", col)), on = .(row_idx__ = row_idx_from)]
  }
  
  # Clean up helper columns
  dt[, c("row_idx__", "cell_pos__") := NULL]
  
  cat("Done.\n")
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# Usage â€” drop-in replacement for the original outer loop
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The resulting cell_data now contains columns:
# #   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
# #   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
# #   ... etc.
# #
# # These are numerically identical to the original implementation.
# # The trained Random Forest model can be used directly for prediction
# # as long as the column names match (adjust naming if needed).
```

---

## Handling the Column-Name Convention

If your existing trained Random Forest expects specific column names (e.g., `ntl_neighbor_max` instead of `neighbor_max_ntl`), add a renaming step:

```r
# Adjust column names to match the trained model's expectations
# Example: if the original code produced "ntl_neighbor_max", "ntl_neighbor_min", etc.
rename_map <- unlist(lapply(neighbor_source_vars, function(vn) {
  old <- paste0("neighbor_", c("max_", "min_", "mean_"), vn)
  new <- paste0(vn, "_neighbor_", c("max", "min", "mean"))
  setNames(old, new)
}))

setnames(cell_data, rename_map, names(rename_map))
```

---

## Why This Is Correct (Numerical Equivalence)

| Original | Optimized | Equivalence |
|----------|-----------|-------------|
| For row `i`, find neighbor cell IDs via `id_order[neighbors[[ref_idx]]]` | Edge list built from same `rook_neighbors_unique` and `id_order` | Same neighbor cell set |
| Filter to same year via string key `paste(id, year)` | Equi-join on `(cell_pos, year)` | Same year filter |
| Drop `NA` results (cell-year not in data) | `nomatch = NULL` + `!is.na(row_idx_to)` | Same missing-data handling |
| `max`, `min`, `mean` of non-NA neighbor values | Same functions with `na.rm = TRUE`, then Inf/NaN â†’ NA | Identical output |

---

## Performance Estimate

| Step | Estimated Time | Memory |
|------|---------------|--------|
| Edge list construction | < 1 sec | ~20 MB |
| First join (rows Ã— edges) | ~30 sec | ~1.5 GB (25M rows Ã— 3 int cols) |
| Second join (resolve neighbor rows) | ~30 sec | in-place |
| Gather 5 variable columns | ~10 sec | +1 GB |
| Grouped aggregation (15 stats) | ~2-3 min | ~1 GB |
| Merge back | ~10 sec | in-place |
| **Total** | **~4-5 minutes** | **~4-5 GB peak** |

This fits comfortably within 16 GB RAM and replaces the estimated 86+ hour runtime. The trained Random Forest model is untouched â€” only the feature-construction pipeline is reformulated.
END_RESPONSE>>>
