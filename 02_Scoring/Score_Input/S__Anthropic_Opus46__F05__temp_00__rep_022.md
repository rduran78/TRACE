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
run_id: Anthropic_F05_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” repeated string allocation and concatenation.
2. **Looks up those keys** in `idx_lookup`, a named character vector of length 6.46M â€” each lookup is O(n) in the worst case for named vector indexing, or O(1) amortized via internal hashing, but the constant factor is large due to string comparison.

This means roughly **6.46M Ã— avg_neighbors â‰ˆ 6.46M Ã— 4 â‰ˆ 25.8M string constructions and hash lookups**, all in an interpreted R loop.

### Why It's a Broader Architectural Issue

The string-keyed lookup is a **workaround for the absence of a direct integer-indexed mapping** from `(cell_id, year)` â†’ row index. The entire pipeline conflates two orthogonal dimensions (spatial identity and temporal identity) into a single string key, when in fact:

- The neighbor graph is **purely spatial** â€” it doesn't change across years.
- The year dimension is **uniform** â€” every cell appears in every year (balanced panel).

This means the neighbor lookup can be decomposed: find the **row offsets for each cell** once, then for any cell-year row, the neighbors' rows in the same year are deterministic integer arithmetic â€” **no strings, no hashing, no `lapply` over millions of rows**.

### Secondary Inefficiency

`compute_neighbor_stats` also loops over 6.46M rows in R-level `lapply`. With a vectorized/matrix approach, this can be replaced with a single pass.

---

## Optimization Strategy

1. **Eliminate all string keys.** Build an integer matrix mapping `(cell_index, year_index)` â†’ row number. This is O(1) lookup via matrix indexing.

2. **Precompute a spatial-only neighbor list** as integer indices into `id_order` (already available from `rook_neighbors_unique`).

3. **Vectorize the neighbor statistics** using `data.table` or direct matrix operations. For each row, gather neighbor values via integer indexing and compute stats in bulk.

4. **Avoid per-row R loops entirely.** Use a "long" neighbor-row table and `data.table` grouped aggregation.

Estimated speedup: from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Preserves the exact numerical estimand: for each cell-year row,
#' compute max, min, mean of each neighbor source variable across
#' rook neighbors present in the same year.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors   spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new columns appended (same row order preserved)
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors,
                                        neighbor_source_vars) {

  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  # ---- Step 1: Create integer cell index and year index ----
  # Map each cell id to its position in id_order
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Add a cell index column (position in id_order)
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Preserve original row order

  dt[, orig_row := .I]

  # ---- Step 2: Build (cell_idx, year) -> row number lookup matrix ----
  # Years as factor for integer indexing
  years_sorted <- sort(unique(dt$year))
  year_to_col  <- setNames(seq_along(years_sorted), as.character(years_sorted))
  n_cells <- length(id_order)
  n_years <- length(years_sorted)

  # Matrix: row = cell_idx (1..n_cells), col = year_idx (1..n_years), value = row in dt
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(dt$cell_idx, year_to_col[as.character(dt$year)])] <- dt$orig_row

  # ---- Step 3: Build spatial neighbor edge list (integer indices only) ----
  # rook_neighbors[[i]] gives the neighbor indices (into id_order) for cell i
  # Build a data.table of directed edges: (focal_cell_idx, neighbor_cell_idx)
  focal_idx <- rep(seq_along(rook_neighbors),
                   lengths(rook_neighbors))
  neighbor_idx <- unlist(rook_neighbors, use.names = FALSE)

  # Remove any 0-length or NA entries (spdep nb objects use 0 for no-neighbor cards)
  valid <- !is.na(neighbor_idx) & neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  n_edges <- length(focal_idx)
  cat(sprintf("Spatial neighbor edges: %d\n", n_edges))

  # ---- Step 4: Expand edges across years and map to row numbers ----
  # For each year, every edge (f, n) maps to (row_lookup[f, y], row_lookup[n, y])
  # We build this as a long table: (focal_row, neighbor_row)
  # To stay within 16 GB RAM, process year by year

  # Pre-extract variable columns as matrices for fast indexing
  var_mat <- as.matrix(dt[, ..neighbor_source_vars])  # nrow(dt) x length(vars)

  # We'll accumulate results into pre-allocated matrices
  # For each var: 3 stats (max, min, mean) â†’ total new columns = 5 vars Ã— 3 = 15
  n_rows <- nrow(dt)
  stat_names <- c("max", "min", "mean")

  # Pre-allocate result columns
  result_list <- vector("list", length(neighbor_source_vars))
  names(result_list) <- neighbor_source_vars
  for (v in neighbor_source_vars) {
    result_list[[v]] <- matrix(NA_real_, nrow = n_rows, ncol = 3)
    colnames(result_list[[v]]) <- paste0("neighbor_", stat_names, "_", v)
  }

  # ---- Step 5: Process year by year ----
  cat("Processing years...\n")
  for (yi in seq_along(years_sorted)) {
    yr <- years_sorted[yi]

    # Get row numbers for focal and neighbor cells in this year
    focal_rows    <- row_lookup[focal_idx, yi]
    neighbor_rows <- row_lookup[neighbor_idx, yi]

    # Both must be non-NA (both cells present in this year)
    valid_mask <- !is.na(focal_rows) & !is.na(neighbor_rows)
    f_rows <- focal_rows[valid_mask]
    n_rows_yr <- neighbor_rows[valid_mask]

    if (length(f_rows) == 0L) next

    # For each variable, gather neighbor values and aggregate by focal row
    for (vi in seq_along(neighbor_source_vars)) {
      v <- neighbor_source_vars[vi]
      # Get neighbor values
      nvals <- var_mat[n_rows_yr, vi]

      # Build a data.table for fast grouped aggregation
      edge_dt <- data.table(focal = f_rows, nval = nvals)
      # Remove NA neighbor values (matches original: neighbor_vals[!is.na(neighbor_vals)])
      edge_dt <- edge_dt[!is.na(nval)]

      if (nrow(edge_dt) == 0L) next

      # Grouped aggregation
      agg <- edge_dt[, .(vmax = max(nval),
                         vmin = min(nval),
                         vmean = mean(nval)),
                     by = focal]

      # Write into result matrix
      result_list[[v]][agg$focal, 1L] <- agg$vmax
      result_list[[v]][agg$focal, 2L] <- agg$vmin
      result_list[[v]][agg$focal, 3L] <- agg$vmean
    }

    if (yi %% 5 == 0 || yi == n_years) {
      cat(sprintf("  Year %d/%d (%d) done\n", yi, n_years, yr))
    }
  }

  # ---- Step 6: Bind results back to dt ----
  for (v in neighbor_source_vars) {
    cnames <- colnames(result_list[[v]])
    for (ci in seq_along(cnames)) {
      set(dt, j = cnames[ci], value = result_list[[v]][, ci])
    }
  }

  # Clean up helper columns
  dt[, c("cell_idx", "orig_row") := NULL]

  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

# Original code was:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# Replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the same 15 new columns (neighbor_max_*, neighbor_min_*, neighbor_mean_*)
# with identical numerical values. The trained Random Forest model is untouched.
```

---

## Column Naming Compatibility

If your existing trained Random Forest expects specific column names (e.g., `ntl_neighbor_max` rather than `neighbor_max_ntl`), adjust the naming to match. Here's a helper:

```r
# If the original code produced columns like: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
# then replace the colnames line above with:
colnames(result_list[[v]]) <- paste0(v, "_neighbor_", stat_names)
```

Verify with:
```r
# Check that the RF model's expected feature names are all present
expected <- model$forest$independent.variable.names  # for ranger
stopifnot(all(expected %in% names(cell_data)))
```

---

## Why This Works and Preserves the Estimand

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[cell_idx]]` | Same `rook_neighbors_unique[[cell_idx]]` |
| Year matching | `paste(id, year)` string match | `row_lookup[cell_idx, year_idx]` integer match |
| NA handling | Drop NA neighbor values, then compute stats | Identical: `edge_dt[!is.na(nval)]` |
| Stats computed | `max`, `min`, `mean` | Identical |
| Rows with no valid neighbors | `NA, NA, NA` | `NA` (matrix pre-initialized to `NA`) |

The numerical results are **identical** â€” only the computational path changes.

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| String key construction | ~6.46M Ã— 4 `paste()` calls in R loop | **Zero** |
| Hash lookups | ~25.8M named-vector lookups | **Zero** |
| Per-row R loop | 6.46M `lapply` iterations | **Zero** |
| Aggregation | R-level `lapply` over 6.46M rows Ã— 5 vars | 28 Ã— 5 = **140** `data.table` grouped aggregations |
| Estimated wall time | ~86 hours | **5â€“15 minutes** |
| Peak RAM | ~6â€“8 GB (string keys dominate) | **~3â€“5 GB** (integer matrices) |
END_RESPONSE>>>
