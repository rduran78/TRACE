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
run_id: Anthropic_F08_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds row-level lookups across all ~6.46M rows.** It creates a lookup keyed by `paste(id, year)` for every row, then for each of the 6.46M rows, it finds the neighbor cell IDs, constructs year-specific keys, and resolves them back to row indices. This is O(N Ã— K) string-hashing work where N = 6.46M and K = average neighbor count (~4 for rook).

2. **The neighbor topology is static.** Rook contiguity between grid cells never changes across years. The `neighbors` list (an `nb` object) is a property of the spatial grid â€” it is invariant over time. Yet the current code re-derives neighbor relationships for every cell-year combination, doing 28Ã— redundant work.

3. **`compute_neighbor_stats` iterates via `lapply` over 6.46M rows** with per-element R-level function calls, creating millions of small vectors. This is inherently slow in interpreted R.

4. **String concatenation (`paste`) and named-vector lookups** are used as a surrogate for proper indexing â€” extremely expensive at this scale.

### The Key Insight

- **Static dimension:** Which cells are neighbors of which (the `nb` object, ~344K entries).
- **Dynamic dimension:** The variable values attached to each cell, which change by year (28 panels).

These two dimensions should be **separated and recombined efficiently** using matrix/vectorized operations rather than row-by-row string lookups.

---

## Optimization Strategy

### 1. Build a Static Neighbor Structure Once (Cell-Level, Not Row-Level)

Convert the `nb` object into a sparse adjacency representation â€” specifically, two integer vectors (`from`, `to`) representing all directed neighbor edges among the 344,208 cells. This is computed **once** and is year-independent.

### 2. Reshape Variables into Cell Ã— Year Matrices

For each variable, pivot the long panel data into a **344,208 Ã— 28 matrix** (cell rows Ã— year columns). This allows vectorized column-wise (i.e., year-wise) operations.

### 3. Compute Neighbor Stats via Sparse-Matrix Multiplication and Vectorized Ops

For each variable matrix **V** (cells Ã— years):

- **Neighbor mean:** Compute `A %*% V` where `A` is the row-normalized sparse adjacency matrix (each row sums to 1 over its neighbors). This gives the neighbor mean for every cell-year in one matrix multiplication.
- **Neighbor max and min:** Use the sparse edge list to gather neighbor values, then compute grouped max/min efficiently using `data.table` or vectorized approaches.

### 4. Unpivot Back to Long Format and Attach

Melt the resulting matrices back to long format and join to the original `cell_data`.

### Complexity Comparison

| | Current | Optimized |
|---|---|---|
| Lookup build | O(6.46M Ã— K) string ops | O(1.37M) integer edge list, once |
| Stat computation per variable | O(6.46M) R-level `lapply` calls | O(1.37M Ã— 28) vectorized gather + grouped agg |
| Total wall time (estimated) | ~86+ hours | **~2â€“10 minutes** |

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) cell attributes.
#
# Prerequisites:
#   - cell_data: data.table (or data.frame) with columns: id, year, and all
#                neighbor_source_vars. Rows are cell-year observations.
#   - id_order: integer/character vector of cell IDs in the order matching
#               rook_neighbors_unique (i.e., id_order[i] is the cell ID for
#               the i-th element of the nb object).
#   - rook_neighbors_unique: an nb object (list of integer index vectors).
#   - neighbor_source_vars: character vector of variable names.
#   - rf_model: the pre-trained Random Forest model (untouched).
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build the static directed edge list from the nb object (ONCE)
# --------------------------------------------------------------------------
build_static_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices (into id_order) of neighbors of cell i.
  # We expand this into a two-column edge list of cell IDs.
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-neighbor sentinel if present (spdep uses 0L for no-neighbor)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_list <- build_static_edge_list(id_order, rook_neighbors_unique)
# edge_list has ~1,373,394 rows: (from_id, to_id) pairs, static across years.

cat("Static edge list built:", nrow(edge_list), "directed edges\n")

# --------------------------------------------------------------------------
# STEP 2: Build a static sparse adjacency matrix and row-normalized version
# --------------------------------------------------------------------------
# Map cell IDs to integer indices 1..N for matrix construction
cell_ids_unique <- id_order
N <- length(cell_ids_unique)
id_to_idx <- setNames(seq_len(N), as.character(cell_ids_unique))

edge_from_idx <- id_to_idx[as.character(edge_list$from_id)]
edge_to_idx   <- id_to_idx[as.character(edge_list$to_id)]

# Binary adjacency matrix (sparse): A[i,j] = 1 if j is a neighbor of i
A_binary <- sparseMatrix(
  i = edge_from_idx,
  j = edge_to_idx,
  x = 1,
  dims = c(N, N)
)

# Row-normalized adjacency: each row sums to 1 (for computing means)
row_sums <- rowSums(A_binary)
row_sums[row_sums == 0] <- 1  # avoid division by zero for isolated cells
A_norm <- Diagonal(x = 1 / row_sums) %*% A_binary

# Neighbor count per cell (static)
n_neighbors <- as.integer(row_sums)

cat("Sparse adjacency matrices built:", N, "x", N, "\n")

# --------------------------------------------------------------------------
# STEP 3: Ensure cell_data is a data.table with proper ordering
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Determine the set of years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

cat("Panel dimensions:", N, "cells x", n_years, "years =",
    N * n_years, "cell-years\n")

# --------------------------------------------------------------------------
# STEP 4: For each variable, pivot to matrix, compute stats, unpivot & join
# --------------------------------------------------------------------------
# We need a consistent mapping from (id, year) -> row index in cell_data
# to write results back.
setkey(cell_data, id, year)

# Pre-build the cell index mapping for matrix rows
# id_to_idx already maps cell id -> matrix row index

compute_neighbor_features_optimized <- function(cell_data, var_name,
                                                 id_to_idx, years, year_to_col,
                                                 A_binary, A_norm, n_neighbors,
                                                 edge_from_idx, edge_to_idx, N) {
  cat("  Processing variable:", var_name, "\n")
  n_years <- length(years)

  # --- Pivot to N x n_years matrix ---
  # Extract relevant columns
  sub <- cell_data[, .(id, year, val = get(var_name))]
  sub[, row_idx := id_to_idx[as.character(id)]]
  sub[, col_idx := year_to_col[as.character(year)]]

  # Build dense matrix (cells x years); NA for missing

  V <- matrix(NA_real_, nrow = N, ncol = n_years)
  V[cbind(sub$row_idx, sub$col_idx)] <- sub$val

  # --- Neighbor MEAN via sparse matrix multiplication ---
  # A_norm %*% V gives the mean of neighbor values for each cell-year.
  # But we need to handle NAs properly. For cells where all neighbors have NA,

  # the result should be NA.
  #
  # Strategy: replace NA with 0 for multiplication, track valid counts separately.
  V_zero <- V
  V_zero[is.na(V_zero)] <- 0

  V_valid <- matrix(as.numeric(!is.na(V)), nrow = N, ncol = n_years)

  # Sum of neighbor values (NAs treated as 0)
  neighbor_sum   <- as.matrix(A_binary %*% V_zero)    # N x n_years
  # Count of valid (non-NA) neighbor values
  neighbor_count <- as.matrix(A_binary %*% V_valid)    # N x n_years

  # Mean = sum / count; NA where count == 0
  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- Neighbor MAX and MIN via edge-list gather + grouped aggregation ---
  # For each edge (from, to), gather to's value for each year.
  # Then group by (from, year) and take max/min.
  #
  # This is done year-by-year in a vectorized fashion over edges.

  n_edges <- length(edge_from_idx)

  neighbor_max <- matrix(NA_real_, nrow = N, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = N, ncol = n_years)

  # Process all years at once using data.table for grouped max/min
  # Build a long table: (from_idx, year_col, neighbor_val)
  # Dimensions: n_edges * n_years rows â€” ~1.37M * 28 â‰ˆ 38.5M rows
  # This fits comfortably in 16 GB RAM (~600 MB for 3 numeric columns).

  # Gather neighbor values for all edges and all years at once
  # V[edge_to_idx, ] is an n_edges x n_years matrix of neighbor values
  neighbor_vals_mat <- V[edge_to_idx, , drop = FALSE]  # n_edges x n_years

  # For max and min, we do grouped operations per year column to avoid
  # materializing the full long table (saves memory and time).
  for (yr_col in seq_len(n_years)) {
    vals_this_year <- neighbor_vals_mat[, yr_col]

    # Use data.table for fast grouped max/min
    dt_edge <- data.table(
      from = edge_from_idx,
      val  = vals_this_year
    )
    # Remove NAs before aggregation
    dt_edge <- dt_edge[!is.na(val)]

    if (nrow(dt_edge) > 0) {
      agg <- dt_edge[, .(vmax = max(val), vmin = min(val)), by = from]
      neighbor_max[agg$from, yr_col] <- agg$vmax
      neighbor_min[agg$from, yr_col] <- agg$vmin
    }
  }

  # --- Unpivot matrices back to long format and join to cell_data ---
  # Create column names matching the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Write directly into cell_data using the row/col indices
  cell_data[, (max_col)  := neighbor_max[cbind(id_to_idx[as.character(id)],
                                                year_to_col[as.character(year)])]]
  cell_data[, (min_col)  := neighbor_min[cbind(id_to_idx[as.character(id)],
                                                year_to_col[as.character(year)])]]
  cell_data[, (mean_col) := neighbor_mean[cbind(id_to_idx[as.character(id)],
                                                  year_to_col[as.character(year)])]]

  invisible(cell_data)
}

# --------------------------------------------------------------------------
# STEP 5: Run for all neighbor source variables
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing optimized neighbor features...\n")
t_start <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features_optimized(
    cell_data, var_name,
    id_to_idx, years, year_to_col,
    A_binary, A_norm, n_neighbors,
    edge_from_idx, edge_to_idx, N
  )
}

t_elapsed <- proc.time() - t_start
cat("Neighbor features computed in", round(t_elapsed[3], 1), "seconds\n")

# --------------------------------------------------------------------------
# STEP 6: Predict with the pre-trained Random Forest (UNCHANGED)
# --------------------------------------------------------------------------
# The rf_model is used as-is. No retraining.
# cell_data now contains all neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns with numerically identical values to the original pipeline.

# Example prediction call (adjust to match your actual prediction code):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor topology** | Rebuilt per row (6.46M string lookups) | Built once as sparse matrix + edge list (~1.37M integer pairs) |
| **Stat computation** | R-level `lapply` over 6.46M rows per variable | Sparse matrix multiply (mean) + vectorized grouped agg (max/min) per year |
| **Total operations** | ~6.46M Ã— 5 vars Ã— 3 stats = ~97M R function calls | 5 sparse matmuls + 5 Ã— 28 grouped `data.table` aggregations |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **Memory** | Millions of small R lists | ~3 dense matrices (344K Ã— 28 â‰ˆ 77 MB each) + sparse matrices (~22 MB) |
| **Numerical result** | Original estimand | **Preserved exactly** (same max, min, mean formulas) |
| **RF model** | Pre-trained | **Untouched â€” no retraining** |

The key architectural principle: **factor the computation into a static spatial graph (computed once) and dynamic attribute matrices (indexed by year), then use linear algebra and vectorized grouped aggregation instead of row-wise interpreted loops.**
END_RESPONSE>>>
