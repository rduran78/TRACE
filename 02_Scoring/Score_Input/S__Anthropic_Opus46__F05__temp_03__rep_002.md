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
run_id: Anthropic_F05_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is built once, but then **6.46 million `paste()` + named-vector lookups** are performed inside the `lapply` â€” each call to `paste(neighbor_cell_ids, data$year[i], sep = "_")` constructs temporary character vectors and does hash-table lookups against a 6.46M-entry named character vector. String hashing at this scale is extremely expensive.

### Broader Algorithmic Problem
The real issue is that **the entire neighbor-lookup concept is being solved in string-key space when it should be solved in integer-index space**. The data has a regular panel structure (each cell appears once per year), so the mapping from `(cell_id, year)` â†’ row index can be represented as a **dense integer matrix** (cell Ã— year), turning every neighbor lookup into a direct integer-indexed matrix access â€” O(1) with no hashing.

Furthermore, `compute_neighbor_stats` is called **5 separate times**, each time iterating over 6.46M rows and chasing the same neighbor indices. These passes can be **fused into a single pass** or, better yet, **fully vectorized** using sparse-matrix multiplication or data.table joins.

### Cost Breakdown (Current)
| Step | Operations | Cost Driver |
|---|---|---|
| `paste()` for `idx_lookup` | 6.46M string concatenations | One-time, tolerable |
| `paste()` inside `lapply` | ~6.46M Ã— ~4 neighbors = ~25.8M concatenations | Dominant cost |
| Named-vector lookup | ~25.8M hash lookups against 6.46M-entry table | Dominant cost |
| `compute_neighbor_stats` | 5 vars Ã— 6.46M rows Ã— index chasing | Repeated traversal |

**Estimated total: 86+ hours** â€” almost entirely from the string operations and R-level loop overhead.

---

## Optimization Strategy

1. **Replace string keys with a dense integer lookup matrix** `row_matrix[cell_index, year_index]` â†’ row number. Lookup becomes a single integer matrix access.

2. **Pre-build all neighbor row-indices as a single integer operation** using vectorized construction â€” no `lapply` over 6.46M rows.

3. **Compute all 5 variables' neighbor stats in one vectorized pass** using a sparse adjacency matrix (or a single grouped data.table operation), eliminating per-row R-level loops entirely.

4. **Preserve exact numerical output**: max, min, mean of non-NA neighbor values per cell-year, per variable â€” identical column names and values.

---

## Working R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 1: Build dense integer lookup matrix  (cell_index Ã— year_index) â†’ row

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Create integer mappings
  unique_ids   <- as.character(id_order)
  unique_years <- sort(unique(dt$year))

  id_to_int   <- setNames(seq_along(unique_ids), unique_ids)
  year_to_int <- setNames(seq_along(unique_years), as.character(unique_years))

  n_ids   <- length(unique_ids)
  n_years <- length(unique_years)

  # Dense lookup matrix: row_matrix[cell_int, year_int] = row index in dt
  # Initialize with NA
  row_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)

  cell_ints <- id_to_int[as.character(dt$id)]
  year_ints <- year_to_int[as.character(dt$year)]
  row_matrix[cbind(cell_ints, year_ints)] <- dt$row_idx

  cat("Step 1 complete: dense lookup matrix built (",
      n_ids, "cells x", n_years, "years )\n")

  # ---------------------------------------------------------------
  # STEP 2: Build sparse directed neighbor adjacency in cell-index space

  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: list of length n_ids,
  # each element is an integer vector of neighbor indices into id_order
  # (with 0L meaning no neighbors per spdep convention)

  # Build edge list (from_cell_int, to_cell_int)
  from_list <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  to_list   <- unlist(rook_neighbors_unique)

  # Remove spdep's 0-encoded "no neighbor" entries
  valid <- to_list != 0L
  from_list <- from_list[valid]
  to_list   <- to_list[valid]

  n_edges <- length(from_list)
  cat("Step 2 complete:", n_edges, "directed neighbor edges\n")

  # ---------------------------------------------------------------
  # STEP 3: Expand edges to cell-year level (vectorized)
  # ---------------------------------------------------------------
  # For each year, every edge (iâ†’j) in cell space becomes
  # (row_matrix[i,y] â†’ row_matrix[j,y]) in row space.
  # We vectorize across all years at once.

  # Replicate edges for each year
  from_cell_expanded <- rep(from_list, times = n_years)
  to_cell_expanded   <- rep(to_list,   times = n_years)
  year_int_expanded  <- rep(seq_len(n_years), each = n_edges)

  # Map to row indices
  from_row <- row_matrix[cbind(from_cell_expanded, year_int_expanded)]
  to_row   <- row_matrix[cbind(to_cell_expanded,   year_int_expanded)]

  # Remove pairs where either cell-year doesn't exist in the data
  valid2 <- !is.na(from_row) & !is.na(to_row)
  from_row <- from_row[valid2]
  to_row   <- to_row[valid2]

  cat("Step 3 complete:", sum(valid2),
      "cell-year neighbor pairs constructed\n")

  # Clean up large temporaries
  rm(from_cell_expanded, to_cell_expanded, year_int_expanded, valid2)
  gc()

  # ---------------------------------------------------------------
  # STEP 4: Compute neighbor stats for each variable (vectorized)
  # ---------------------------------------------------------------
  # Strategy: use data.table grouping.
  # Build an edge table: for each "from_row", gather neighbor values from "to_row".
  # Group by from_row, compute max/min/mean.

  edge_dt <- data.table(from_row = from_row, to_row = to_row)
  rm(from_row, to_row)
  gc()

  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    cat("  Computing neighbor stats for:", var_name, "...")
    t0 <- proc.time()

    # Attach neighbor values
    vals <- dt[[var_name]]
    edge_dt[, nval := vals[to_row]]

    # Compute grouped stats (excluding NAs)
    stats <- edge_dt[!is.na(nval),
                     .(vmax  = max(nval),
                       vmin  = min(nval),
                       vmean = mean(nval)),
                     by = from_row]

    # Initialize result columns with NA
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]

    # Fill in computed values
    dt[stats$from_row, (col_max)  := stats$vmax]
    dt[stats$from_row, (col_min)  := stats$vmin]
    dt[stats$from_row, (col_mean) := stats$vmean]

    elapsed <- (proc.time() - t0)[3]
    cat(" done in", round(elapsed, 1), "sec\n")
  }

  # Clean up the temporary column
  edge_dt[, nval := NULL]

  # ---------------------------------------------------------------
  # STEP 5: Return as data.frame (preserving compatibility)
  # ---------------------------------------------------------------
  dt[, row_idx := NULL]
  result <- as.data.frame(dt)

  cat("All neighbor features computed.\n")
  return(result)
}

# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- optimize_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is unchanged â€” just call predict():
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per cell-year. The grouping key `from_row` is a 1:1 mapping to the original row. |
| **Column names** | Output columns are named `neighbor_max_{var}`, `neighbor_min_{var}`, `neighbor_mean_{var}` â€” adjust the prefix in the code if your original names differ. |
| **NA handling** | Rows with zero valid neighbors get `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |
| **Trained RF model** | No model retraining. The feature matrix is numerically identical; just pass it to `predict()`. |

---

## Expected Performance

| Step | Estimated Time | Estimated RAM |
|---|---|---|
| Dense lookup matrix | ~2 sec | ~40 MB |
| Edge expansion (vectorized) | ~10 sec | ~600 MB peak |
| Per-variable grouped stats (Ã—5) | ~15â€“30 sec each | ~500 MB working |
| **Total** | **~3â€“5 minutes** | **< 8 GB peak** |

This is a **~1,000Ã— speedup** over the original 86+ hour estimate, achieved by eliminating all string operations and R-level row loops in favor of vectorized integer indexing and data.table grouped aggregation.
END_RESPONSE>>>
