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
run_id: Anthropic_F05_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts an ID to character and looks it up in a named vector â€” O(1) amortized but with string allocation overhead.
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current year to form string keys â€” allocates new strings per row.
4. **Looks up** those keys in `idx_lookup` (a named vector of 6.46M entries) â€” named vector lookup is hash-based but still involves repeated string hashing.

This means roughly **6.46M Ã— avg_neighbors â‰ˆ 25â€“50 million `paste` + hash-lookup operations**, all in an interpreted R `lapply` loop. The string allocation and hashing dominates.

Then `compute_neighbor_stats` is called 5 times (once per variable), each iterating over the 6.46M-element `neighbor_lookup` list â€” but this is comparatively cheap since the index lists are already built.

### The Deeper Structural Insight

The neighbor relationship is **year-invariant**: cell A's rook neighbors are the same cells every year. The only reason the code builds string keys with year is to find the **row index** of (neighbor_id, year) in the stacked panel. This means:

- The neighbor **topology** is fixed across years (344,208 cells Ã— ~4 neighbors each).
- The panel is simply the topology **replicated** across 28 years.
- We don't need string keys at all. We need an **integer matrix** mapping `(cell_index, year_index) â†’ row_index`, then neighbor row indices for row `i` are simply the row indices of `(neighbors_of_cell[i], year_of_row[i])`.

### Summary

| Layer | Problem | Impact |
|-------|---------|--------|
| **String keys** | `paste()` + named-vector hash lookup inside 6.46M-iteration loop | ~50M string allocations |
| **Redundant topology expansion** | Neighbor topology is year-invariant but re-derived per row | 28Ã— redundant work |
| **R-level loop** | `lapply` over 6.46M rows in interpreted R | No vectorization |
| **`compute_neighbor_stats`** | 5 separate passes over 6.46M-element list, each extracting scalar stats | Could be vectorized |

## Optimization Strategy

1. **Build an integer lookup matrix** `row_matrix[cell_index, year_index] â†’ row_in_data` once. This is a 344,208 Ã— 28 integer matrix (~38 MB). No strings.

2. **Convert the `nb` object to a flat adjacency representation** once (two integer vectors: `adj_start`, `adj_target`), so neighbor retrieval is a slice of an integer vector.

3. **Vectorize the neighbor-stats computation** using `data.table` or direct vectorized R: explode each row into its neighbor rows, join, and compute grouped `max/min/mean` â€” all in vectorized C-level operations.

4. **Process all 5 variables simultaneously** in a single pass over the exploded edge table rather than 5 separate passes.

This reduces the estimated runtime from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Ensure data is a data.table with correct ordering
# ==============================================================
# cell_data must have columns: id, year, and the 5 neighbor source vars.
# id_order is the vector of unique cell IDs matching rook_neighbors_unique.
# rook_neighbors_unique is the spdep nb object.

build_and_apply_neighbor_features <- function(cell_data, id_order,
                                              rook_neighbors_unique,
                                              neighbor_source_vars) {

  # Convert to data.table if needed (by reference if already one)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # ----------------------------------------------------------
  # STEP 1: Build integer cell-index mapping

  # ----------------------------------------------------------
  # Map each cell id to a sequential integer index (1..N_cells)
  n_cells <- length(id_order)
  id_to_cidx <- setNames(seq_len(n_cells), as.character(id_order))

  # Add cell index to data
  cell_data[, .cidx := id_to_cidx[as.character(id)]]

  # ----------------------------------------------------------
  # STEP 2: Build year-index mapping
  # ----------------------------------------------------------
  years_sorted <- sort(unique(cell_data$year))
  n_years <- length(years_sorted)
  year_to_yidx <- setNames(seq_len(n_years), as.character(years_sorted))

  cell_data[, .yidx := year_to_yidx[as.character(year)]]

  # ----------------------------------------------------------
  # STEP 3: Build row-lookup matrix (cell_index, year_index) -> row
  # ----------------------------------------------------------
  # This is a 344,208 x 28 integer matrix (~38 MB)
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

  # Populate: for each row in cell_data, store its row number
  row_matrix[cbind(cell_data$.cidx, cell_data$.yidx)] <- seq_len(nrow(cell_data))

  # ----------------------------------------------------------
  # STEP 4: Flatten the nb object into an edge list
  # ----------------------------------------------------------
  # rook_neighbors_unique[[k]] gives the neighbor indices (into id_order)
  # for cell id_order[k].
  # Build a data.table of directed edges: (from_cidx, to_cidx)

  from_cidx <- rep(
    seq_len(n_cells),
    times = lengths(rook_neighbors_unique)
  )
  to_cidx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-length entries (cells with no neighbors, if any)
  valid <- !is.na(to_cidx) & to_cidx > 0L
  edges <- data.table(from_cidx = from_cidx[valid],
                      to_cidx   = to_cidx[valid])

  cat(sprintf("Edge list: %d directed neighbor relationships\n", nrow(edges)))

  # ----------------------------------------------------------
  # STEP 5: For each year, look up neighbor row indices
  # ----------------------------------------------------------
  # We need to build a long table:
  #   (focal_row, neighbor_row)
  # Then join to get neighbor values and aggregate.
  #
  # Strategy: iterate over years (only 28), vectorize within each year.

  # Pre-extract the columns we need for speed
  var_cols <- neighbor_source_vars
  n_vars <- length(var_cols)

  # Pre-allocate result columns (max, min, mean for each var)
  for (v in var_cols) {
    cell_data[, paste0("n_max_", v) := NA_real_]
    cell_data[, paste0("n_min_", v) := NA_real_]
    cell_data[, paste0("n_mean_", v) := NA_real_]
  }

  # Extract variable data as a matrix for fast column access
  var_mat <- as.matrix(cell_data[, ..var_cols])

  cat("Processing neighbor stats by year...\n")

  for (yi in seq_len(n_years)) {
    if (yi %% 5 == 1) cat(sprintf("  Year %d/%d (%d)\n", yi, n_years, years_sorted[yi]))

    # Row indices of focal cells in this year
    focal_rows_this_year <- row_matrix[, yi]  # length = n_cells, NA if cell absent

    # For each edge (from_cidx -> to_cidx), the focal row is
    # row_matrix[from_cidx, yi] and the neighbor row is row_matrix[to_cidx, yi]
    focal_row    <- focal_rows_this_year[edges$from_cidx]
    neighbor_row <- focal_rows_this_year[edges$to_cidx]

    # Drop edges where either focal or neighbor is missing this year
    valid_mask <- !is.na(focal_row) & !is.na(neighbor_row)
    f_rows <- focal_row[valid_mask]
    n_rows <- neighbor_row[valid_mask]

    if (length(f_rows) == 0L) next

    # Extract neighbor values for all variables at once
    # n_vals is a matrix: (n_valid_edges x n_vars)
    n_vals <- var_mat[n_rows, , drop = FALSE]

    # Build a data.table for grouped aggregation
    # Using data.table for fast grouped max/min/mean
    agg_dt <- data.table(
      focal_row = f_rows
    )

    # Add each variable's neighbor values as columns
    for (j in seq_len(n_vars)) {
      set(agg_dt, j = var_cols[j], value = n_vals[, j])
    }

    # Aggregate: for each focal_row, compute max/min/mean of each variable
    # Build the aggregation expression dynamically
    agg_exprs <- list()
    agg_names <- character(0)
    for (v in var_cols) {
      agg_exprs[[paste0("n_max_", v)]]  <- parse(text = sprintf("max(%s, na.rm = TRUE)", v))[[1]]
      agg_exprs[[paste0("n_min_", v)]]  <- parse(text = sprintf("min(%s, na.rm = TRUE)", v))[[1]]
      agg_exprs[[paste0("n_mean_", v)]] <- parse(text = sprintf("mean(%s, na.rm = TRUE)", v))[[1]]
    }

    # Construct the j-expression for data.table
    j_expr <- as.call(c(
      as.name("list"),
      agg_exprs
    ))

    agg_result <- agg_dt[, eval(j_expr), by = focal_row]

    # Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen after
    # filtering, but be safe)
    for (col_name in names(agg_result)[-1]) {
      vals <- agg_result[[col_name]]
      vals[is.infinite(vals)] <- NA_real_
      set(agg_result, j = col_name, value = vals)
    }

    # Write results back into cell_data
    result_cols <- names(agg_result)[-1]  # exclude 'focal_row'
    target_rows <- agg_result$focal_row

    for (col_name in result_cols) {
      set(cell_data, i = target_rows, j = col_name, value = agg_result[[col_name]])
    }
  }

  # ----------------------------------------------------------
  # STEP 6: Clean up temporary columns
  # ----------------------------------------------------------
  cell_data[, .cidx := NULL]
  cell_data[, .yidx := NULL]

  cat("Done. Neighbor features added.\n")
  return(cell_data)
}

# ==============================================================
# USAGE (replaces the original outer loop)
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_apply_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The output columns (n_max_*, n_min_*, n_mean_*) contain
# identical numerical values to the original implementation.
```

## Why This Works and Complexity Analysis

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Outer loop** | 6.46M R-level iterations | 28 iterations (one per year) |
| **Key construction** | ~50M `paste()` calls | Zero string operations |
| **Lookup method** | Named-vector hash on strings | Integer matrix indexing `O(1)` |
| **Neighbor stats** | 5 separate `lapply` passes Ã— 6.46M | Single `data.table` grouped aggregation per year |
| **Memory** | 6.46M-element list of integer vectors | ~38 MB integer matrix + edge table (~22 MB) |
| **Estimated time** | 86+ hours | **5â€“15 minutes** |

### Key Correctness Guarantees

1. **Numerical equivalence**: The `max`, `min`, and `mean` operations are identical â€” same neighbor sets, same `na.rm = TRUE` semantics, same values. Rows with no valid neighbors get `NA` (matching the original `c(NA, NA, NA)` return).

2. **Random Forest compatibility**: No model columns are renamed or removed. The output column names follow the same `n_max_*`, `n_min_*`, `n_mean_*` pattern. Adjust the prefix if your original `compute_and_add_neighbor_features` used different names.

3. **Year-invariant topology**: The optimization exploits the fact that rook contiguity doesn't change across years â€” this is mathematically guaranteed for a fixed spatial grid, so the reformulation is exact.
END_RESPONSE>>>
