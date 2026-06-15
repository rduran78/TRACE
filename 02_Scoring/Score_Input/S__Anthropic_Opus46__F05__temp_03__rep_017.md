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
run_id: Anthropic_F05_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_017.md

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
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup on a 6.46M-element character vector, repeated 6.46M times**.

The named-vector `idx_lookup` has 6.46M entries. Each lookup into it is O(n) in the worst case for partial matching, or O(1) amortized with hashing â€” but R's `[.character` on named vectors is **not a hash table**; it uses linear search or internal hashing that still has significant per-call overhead. Doing this 6.46M times is catastrophic.

### The Deeper Structural Insight

The neighbor relationships are **spatial** (cell-to-cell) and **time-invariant**. The year dimension is only used to find "the same neighbor in the same year." Since the panel is balanced (every cell appears in every year), the neighbor lookup can be reformulated as a **purely positional/integer operation** that never touches strings at all.

**Key insight:** If the data is sorted by `(id, year)` or `(year, id)` in a known order, then for any row `i` belonging to cell `c` in year `t`, its neighbor rows can be computed by **integer arithmetic** on row indices â€” no string keys, no hash lookups, no `paste`.

Furthermore, `compute_neighbor_stats` is called 5 times, each time iterating over the full 6.46M-element `neighbor_lookup` list. This list-of-integer-vectors structure forces R into slow per-element `lapply` iteration. A **vectorized matrix-based approach** can replace this entirely.

---

## Optimization Strategy

1. **Eliminate all string operations.** Sort data by `(id, year)`. With `N_cells = 344,208` and `N_years = 28`, row index for cell `c` (0-indexed among cells) in year `t` (0-indexed among years) is `c * N_years + t + 1`. Neighbor row indices are pure integer arithmetic.

2. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a single vectorized construction of a sparse neighbor matrix or a flat integer-index structure.

3. **Replace per-variable `lapply` in `compute_neighbor_stats`** with sparse matrix multiplication. If `W` is the row-adjacency matrix (6.46M Ã— 6.46M sparse), then `W %*% x` gives neighbor sums, `W %*% ones` gives neighbor counts, and neighbor means = sums / counts. For max and min, use grouped operations on a long-form edge list.

4. **Compute all 5 variables' stats in one pass** over the edge list for max/min, and via sparse matrix multiply for mean.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement preserving the original numerical estimand.
# =============================================================================

library(Matrix)   # for sparse matrices
library(data.table)

build_and_apply_neighbor_features <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # STEP 0: Convert to data.table for fast manipulation; record original order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, orig_row_idx__ := .I]

  N_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  N_years <- length(years)
  stopifnot(nrow(dt) == N_cells * N_years)  # balanced panel check

  # -------------------------------------------------------------------------
  # STEP 1: Create integer mappings (no strings anywhere)
  # -------------------------------------------------------------------------
  # Map cell id -> 1-based cell index (in id_order ordering)
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))

  # Map year -> 1-based year index
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  # Sort data by (cell_index, year) so row position is deterministic
  dt[, cell_idx__ := id_to_cidx[as.character(id)]]
  dt[, year_idx__ := year_to_yidx[as.character(year)]]
  setorder(dt, cell_idx__, year_idx__)

  # Now row index for cell c (1-based) in year t (1-based) is:
  #   row = (c - 1) * N_years + t
  # Verify:
  dt[, expected_row__ := (cell_idx__ - 1L) * N_years + year_idx__]
  stopifnot(all(dt$expected_row__ == seq_len(nrow(dt))))
  dt[, expected_row__ := NULL]

  # -------------------------------------------------------------------------
  # STEP 2: Build cell-level directed edge list from rook_neighbors_unique
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: list of length N_cells,
  # where element i contains integer indices of neighbors of cell i
  # (in id_order indexing, matching our cell_idx__).

  # Build edge list: from_cell -> to_cell (1-based cell indices)
  n_neighbors <- vapply(rook_neighbors_unique, length, integer(1))
  from_cell   <- rep(seq_len(N_cells), times = n_neighbors)
  to_cell     <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the nb "0" sentinel for cells with no neighbors (if any)
  valid <- to_cell > 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  N_edges <- length(from_cell)
  cat(sprintf("Cell-level directed edges: %d\n", N_edges))

  # -------------------------------------------------------------------------
  # STEP 3: Expand cell-level edges to row-level edges (across all years)
  #
  # For each year t, edge (c1 -> c2) at cell level becomes
  #   row_from = (c1-1)*N_years + t  ->  row_to = (c2-1)*N_years + t
  #
  # Total row-level edges = N_edges * N_years
  # ~1.37M * 28 â‰ˆ 38.5M edges â€” manageable in memory as integer vectors.
  # -------------------------------------------------------------------------

  # Pre-compute cell base offsets: (cell_idx - 1) * N_years
  from_base <- (from_cell - 1L) * N_years
  to_base   <- (to_cell   - 1L) * N_years

  # Expand across years
  year_offsets <- seq_len(N_years)  # 1..28

  # Use outer-sum approach: each column is one year
  # row_from[e, t] = from_base[e] + t
  # Flatten in column-major order (all edges for year 1, then year 2, ...)
  row_from <- rep(from_base, times = N_years) +
              rep(year_offsets, each = N_edges)
  row_to   <- rep(to_base,   times = N_years) +
              rep(year_offsets, each = N_edges)

  N_row_edges <- length(row_from)
  N_rows      <- nrow(dt)
  cat(sprintf("Row-level directed edges: %d (rows: %d)\n", N_row_edges, N_rows))

  # -------------------------------------------------------------------------
  # STEP 4: Build sparse adjacency matrix W (N_rows x N_rows)
  #         W[i,j] = 1 means j is a neighbor of i.
  #         So W %*% x gives sum of neighbor values for each row.
  # -------------------------------------------------------------------------
  W <- sparseMatrix(
    i    = row_from,
    j    = row_to,
    x    = 1,
    dims = c(N_rows, N_rows)
  )

  # Neighbor count per row (for computing means)
  neighbor_count <- as.numeric(W %*% rep(1, N_rows))

  # -------------------------------------------------------------------------
  # STEP 5: For each variable, compute max, min, mean of neighbors
  # -------------------------------------------------------------------------
  # Mean: use sparse matrix multiply.
  # Max, Min: use data.table grouped operations on the edge list.

  # Pre-build the edge data.table for grouped max/min
  # We only need row_from and row_to; we'll join variable values on row_to.
  edge_dt <- data.table(from = row_from, to = row_to)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))

    vals <- dt[[var_name]]

    # --- MEAN via sparse matrix multiply ---
    # Handle NAs: replace with 0 for sum, and track non-NA count
    not_na   <- as.numeric(!is.na(vals))
    vals_0   <- ifelse(is.na(vals), 0, vals)

    neighbor_sum    <- as.numeric(W %*% vals_0)
    neighbor_nonna  <- as.numeric(W %*% not_na)
    neighbor_mean   <- ifelse(neighbor_nonna > 0,
                              neighbor_sum / neighbor_nonna,
                              NA_real_)
    # Rows with no neighbors at all -> NA
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # --- MAX and MIN via grouped edge-list operations ---
    edge_dt[, val := vals[to]]

    # Remove edges where neighbor value is NA
    valid_edges <- edge_dt[!is.na(val)]

    if (nrow(valid_edges) > 0) {
      agg <- valid_edges[, .(nmax = max(val), nmin = min(val)), by = from]

      # Initialize with NA
      neighbor_max <- rep(NA_real_, N_rows)
      neighbor_min <- rep(NA_real_, N_rows)

      neighbor_max[agg$from] <- agg$nmax
      neighbor_min[agg$from] <- agg$nmin
    } else {
      neighbor_max <- rep(NA_real_, N_rows)
      neighbor_min <- rep(NA_real_, N_rows)
    }

    # --- Assign columns (same naming convention as original) ---
    max_col  <- paste0("n_max_",  var_name)
    min_col  <- paste0("n_min_",  var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := neighbor_max]
    dt[, (min_col)  := neighbor_min]
    dt[, (mean_col) := neighbor_mean]
  }

  # Clean up the temporary edge val column
  edge_dt[, val := NULL]

  # -------------------------------------------------------------------------
  # STEP 6: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, orig_row_idx__)
  dt[, c("orig_row_idx__", "cell_idx__", "year_idx__") := NULL]

  return(as.data.frame(dt))
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_apply_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original.
```

---

## Why This Works and Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| **String constructions** | ~51.6M (`paste` calls inside 6.46M-iteration `lapply`) | **0** |
| **Named-vector lookups on 6.46M-element vector** | ~25.8M | **0** |
| **R-level loop iterations** | 6.46M (in `build_neighbor_lookup`) + 5 Ã— 6.46M (in `compute_neighbor_stats`) | **0** (fully vectorized) |
| **Core operation for mean** | R `lapply` + per-element subsetting | **Sparse matrix multiply** (`Matrix::crossprod`-level C code) |
| **Core operation for max/min** | R `lapply` + per-element subsetting | **`data.table` grouped aggregation** (radix-sort + single pass, in C) |
| **Memory for edge list** | 6.46M-element list of integer vectors (high overhead) | 2 integer vectors of ~38.5M each (~308 MB) + sparse matrix |
| **Estimated wall time** | 86+ hours | **~5â€“15 minutes** |

### Why numerical equivalence is preserved

- The sparse matrix `W` encodes exactly the same directed neighbor relationships as the original `build_neighbor_lookup`.
- The NA-handling logic (exclude NAs before computing max/min/mean) is replicated exactly: for mean, we sum non-NA values and divide by the non-NA count; for max/min, we filter out NA edges before grouping.
- The column names follow the same `n_max_`, `n_min_`, `n_mean_` convention, so the trained Random Forest model's feature references remain valid.
- The original row order is restored via `orig_row_idx__` before returning.
END_RESPONSE>>>
