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
run_id: Anthropic_F07_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` call that performs character pasting, hash lookups, and NA filtering row-by-row.** This is ~6.46M iterations of interpreted R code with repeated string allocation (`paste`, `setNames`, named-vector lookups). Then `compute_neighbor_stats` iterates over the same 6.46M list entries again for each of the 5 variables (another ~32.3M interpreted iterations total).

Specific problems:

1. **O(N) `lapply` with string-key lookups inside `build_neighbor_lookup`**: Each of the 6.46M rows does `paste()`, named-vector indexing (`idx_lookup[neighbor_keys]`), and NA removal. Named-vector lookup in R is linear-scan or hash-based but with per-call overhead that dominates at this scale.

2. **Redundant recomputation across years**: The neighbor *topology* is time-invariant (cell A's rook neighbors are the same in every year), but the lookup is rebuilt as if it's unique per cell-year. The 344,208 cell neighbor relationships are replicated 28 times.

3. **`lapply` + `do.call(rbind, ...)` anti-pattern in `compute_neighbor_stats`**: Allocates 6.46M small vectors, then row-binds them. This is extremely slow.

4. **No vectorization or use of sparse-matrix algebra**: The neighbor relationship is a sparse adjacency matrix. Max, min, and mean over neighbors can be computed via sparse matrix operations in seconds.

## Optimization Strategy

**Replace the entire list-based lookup with a sparse adjacency matrix approach using the `Matrix` package.** 

Key ideas:

- Represent the rook-neighbor topology as a **sparse adjacency matrix W** (344,208 Ã— 344,208).
- For each year, extract the variable vector, then compute neighbor **mean** via sparse matrix-vector multiply: `W %*% x / rowSums(W)`.
- For neighbor **max** and **min**, use a grouped-operation approach: expand the sparse matrix into a long (i, j) edge list, look up values for j, then aggregate by i using `data.table`.
- This replaces ~6.46M R-level iterations with vectorized C-level operations. Expected runtime: **minutes, not days**.

The trained Random Forest model is untouched. The numerical outputs (neighbor max, min, mean) are identical because we preserve the exact same rook-neighbor graph and the same arithmetic.

## Working R Code

```r
library(Matrix)
library(data.table)

# ============================================================
# STEP 1: Build sparse adjacency matrix from spdep::nb object
# ============================================================
# rook_neighbors_unique: an nb object of length = length(id_order) = 344,208
# id_order: vector of cell IDs in the order matching the nb object

n_cells <- length(id_order)

# Build COO (coordinate) representation
from_idx <- rep(seq_along(rook_neighbors_unique),
                lengths(rook_neighbors_unique))
to_idx   <- unlist(rook_neighbors_unique)

# Remove any 0-neighbor placeholders (spdep uses integer(0) for islands,
# but sometimes stores 0L)
valid <- to_idx > 0L
from_idx <- from_idx[valid]
to_idx   <- to_idx[valid]

# Sparse binary adjacency matrix (n_cells x n_cells)
W <- sparseMatrix(i = from_idx, j = to_idx, x = 1,
                  dims = c(n_cells, n_cells))

# Number of neighbors per cell (for computing mean)
n_neighbors <- as.integer(rowSums(W))  # integer vector, length n_cells

# ============================================================
# STEP 2: Convert cell_data to data.table for fast operations
# ============================================================
cell_dt <- as.data.table(cell_data)

# Create a mapping: cell ID -> matrix row index (1..n_cells)
id_to_matrow <- setNames(seq_along(id_order), as.character(id_order))

# Add matrix row index to data
cell_dt[, mat_row := id_to_matrow[as.character(id)]]

# Ensure data is keyed for fast year-group operations
setkey(cell_dt, year)

# ============================================================
# STEP 3: Precompute edge list for max/min operations
# ============================================================
# Edge list: data.table with columns (from, to)
edge_dt <- data.table(from = from_idx, to = to_idx)

# ============================================================
# STEP 4: Function to compute neighbor stats for one variable
# ============================================================
compute_neighbor_features_fast <- function(dt, var_name, W, n_neighbors,
                                            edge_dt, n_cells, id_order) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Pre-allocate output columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]

  years <- sort(unique(dt$year))

  for (yr in years) {
    # Row indices in dt for this year
    yr_rows <- which(dt$year == yr)

    # Build a full-length vector for the variable (indexed by mat_row)
    # Cells not present in this year get NA
    x_full <- rep(NA_real_, n_cells)
    mat_rows_yr <- dt$mat_row[yr_rows]
    x_full[mat_rows_yr] <- dt[[var_name]][yr_rows]

    # ------ NEIGHBOR MEAN via sparse matrix-vector multiply ------
    # Replace NA with 0 for multiplication, track valid counts
    x_nona <- x_full
    x_valid <- rep(1, n_cells)
    na_mask <- is.na(x_full)
    x_nona[na_mask] <- 0
    x_valid[na_mask] <- 0

    neighbor_sum   <- as.numeric(W %*% x_nona)
    neighbor_count <- as.integer(W %*% x_valid)

    n_mean <- ifelse(neighbor_count > 0L,
                     neighbor_sum / neighbor_count,
                     NA_real_)

    # ------ NEIGHBOR MAX and MIN via edge-list aggregation ------
    # Look up neighbor values for every edge
    neighbor_vals <- x_full[edge_dt$to]

    # Build a temporary DT for grouped aggregation
    # Only keep edges where the neighbor value is not NA
    valid_edges <- !is.na(neighbor_vals)
    agg_dt <- data.table(
      from = edge_dt$from[valid_edges],
      val  = neighbor_vals[valid_edges]
    )

    if (nrow(agg_dt) > 0) {
      agg <- agg_dt[, .(nmax = max(val), nmin = min(val)), by = from]

      # Map aggregated results back to full vector
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
      n_max_full[agg$from] <- agg$nmax
      n_min_full[agg$from] <- agg$nmin
    } else {
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
    }

    # ------ Write results back to the data.table ------
    set(dt, i = yr_rows, j = max_col,  value = n_max_full[mat_rows_yr])
    set(dt, i = yr_rows, j = min_col,  value = n_min_full[mat_rows_yr])
    set(dt, i = yr_rows, j = mean_col, value = n_mean[mat_rows_yr])
  }

  return(dt)
}

# ============================================================
# STEP 5: Run for all 5 neighbor source variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_fast(
    dt          = cell_dt,
    var_name    = var_name,
    W           = W,
    n_neighbors = n_neighbors,
    edge_dt     = edge_dt,
    n_cells     = n_cells,
    id_order    = id_order
  )
}

# ============================================================
# STEP 6: Convert back to data.frame if needed downstream
# ============================================================
cell_data <- as.data.frame(cell_dt)
cell_data$mat_row <- NULL  # remove helper column

# The trained Random Forest model is used as-is with predict().
# No retraining is needed; the new columns are numerically identical
# to the original implementation's output.
```

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops) | Eliminated (sparse matrix built once, ~seconds) |
| `compute_neighbor_stats` Ã— 5 vars | ~80+ hours (32.3M list iterations) | ~5 Ã— 28 sparse mat-vec + grouped agg â‰ˆ **2â€“5 minutes** |
| Memory | 6.46M-element list of integer vectors | Sparse matrix (~5.5M nonzeros) + edge DT â‰ˆ **~200 MB** |

**Expected total runtime: under 10 minutes on a 16 GB laptop**, versus the original 86+ hour estimate â€” a speedup of roughly **500â€“1000Ã—**.

## Why Numerical Equivalence Is Preserved

- The sparse matrix `W` encodes exactly the same directed rook-neighbor pairs as `rook_neighbors_unique`.
- `neighbor_mean` uses the same definition: sum of non-NA neighbor values divided by count of non-NA neighbor values.
- `neighbor_max` and `neighbor_min` use the same `max()` and `min()` over non-NA neighbor values.
- Cells with zero valid neighbors produce `NA`, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` logic.
- The trained Random Forest model is never modified or retrained.
END_RESPONSE>>>
