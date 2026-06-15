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
run_id: Anthropic_F07_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by string-pasting keys and looking them up in a named character vector. Specifically:

1. **`paste()` + named-vector lookup per row (~6.46M iterations in an R `lapply`)**: For every cell-year row, the code pastes each neighbor's ID with the current year to form a key, then does a named-vector lookup (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) hash probing per call, and doing this 6.46 million times with variable-length neighbor sets is extremely slow.

2. **`compute_neighbor_stats` is also list-based but less severe**: Once the lookup is built, subsetting a numeric vector by integer indices is fast, but wrapping 6.46M calls in `lapply` with per-element `max/min/mean` still has significant R-loop overhead.

3. **Memory pressure**: A list of 6.46M integer vectors, plus intermediate character vectors, can push well past comfortable limits on 16 GB RAM.

**Root cause**: The algorithm is O(N_rows Ã— avg_neighbors) executed in interpreted R loops with expensive string operations. The 86+ hour estimate comes almost entirely from this.

---

## Optimization Strategy

### Key Insight
The neighbor topology is **time-invariant** â€” the same rook-neighbor graph applies to every year. We should:

1. **Build a sparse adjacency matrix once** (344,208 Ã— 344,208) from the `nb` object â€” this is a standard, fast operation via `spdep::nb2listw` or direct construction.

2. **Reshape each variable into a matrix** of dimension (344,208 cells Ã— 28 years), so column `t` holds all cell values for year `t`.

3. **Use sparse matrixâ€“dense matrix multiplication** (via the `Matrix` package) to compute neighbor sums and neighbor counts in one shot. Neighbor mean = sum / count. For max and min, use a grouped operation over the sparse adjacency structure (vectorized in C via `data.table`).

This replaces ~6.46M R-level iterations with a handful of vectorized matrix operations that complete in **seconds to minutes**.

### Preserving the Estimand
The sparse-matrix approach computes **exactly** the same neighbor max, neighbor min, and neighbor mean as the original code (same rook topology, same NA handling). The trained Random Forest model is untouched â€” we only reproduce the same feature columns faster.

---

## Working R Code

```r
# ==============================================================================
# FAST NEIGHBOR FEATURE COMPUTATION
# Replaces build_neighbor_lookup + compute_neighbor_stats
# Preserves exact numerical equivalence with the original implementation.
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# Step 1: Build sparse binary adjacency matrix from the nb object (once)
# --------------------------------------------------------------------------
build_sparse_adjacency <- function(nb_obj, n) {

  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial units (length of nb_obj)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove any 0-length entries (islands) â€” they simply won't appear
  valid <- to > 0L
  sparseMatrix(
    i = from[valid], j = to[valid],
    x = 1, dims = c(n, n), giveCsparse = TRUE
  )
}

n_cells <- length(rook_neighbors_unique)   # 344,208
W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# --------------------------------------------------------------------------
# Step 2: Convert cell_data to data.table for fast manipulation
# --------------------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure consistent cell ordering: map cell id -> row index in id_order
id_to_row <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_idx := id_to_row[as.character(id)]]

# Ensure years are integer and define year mapping
years <- sort(unique(cell_dt$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
cell_dt[, year_idx := year_to_col[as.character(year)]]

# --------------------------------------------------------------------------
# Step 3: Function to compute neighbor max, min, mean for one variable
# --------------------------------------------------------------------------
compute_neighbor_features_fast <- function(dt, var_name, W, n_cells, n_years,
                                           years, year_to_col) {
  # Build cell Ã— year matrix (NA where cell-year is missing)
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  V[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]

  # --- Neighbor MEAN via sparse matrix multiplication ---
  # For mean, we need sum of non-NA neighbor values / count of non-NA neighbors
  # Replace NA with 0 for sum; build indicator for non-NA for count
  V_zero <- V
  V_zero[is.na(V_zero)] <- 0
  V_ind <- (!is.na(V)) * 1.0   # indicator matrix


  # W %*% V_zero  gives, for each cell, the sum of neighbor values (per year)
  # W %*% V_ind   gives, for each cell, the count of non-NA neighbors (per year)
  neighbor_sum   <- as.matrix(W %*% V_zero)   # n_cells Ã— n_years
  neighbor_count <- as.matrix(W %*% V_ind)

  neighbor_mean_mat <- neighbor_sum / neighbor_count
  # Where count == 0, result is NaN from 0/0; convert to NA

  neighbor_mean_mat[neighbor_count == 0] <- NA_real_

  # --- Neighbor MAX and MIN via direct sparse iteration ---
  # We iterate over cells (in C-level vectorized fashion via data.table)
  # Extract the adjacency list from the sparse matrix (CSC format)
  # For max/min we must handle NAs properly: only non-NA neighbor values count.

  # Build an edge-level data.table: (from_cell, to_cell)
  Wt <- summary(W)  # gives i, j, x triplets
  edges <- data.table(from = Wt$i, to = Wt$j)

  # For each year, look up the value of the neighbor cell, then group by (from, year)
  # To avoid a massive long table (edges Ã— years), process year by year

  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (t in seq_len(n_years)) {
    # Get values for this year for all cells
    vals_t <- V[, t]  # length n_cells, may contain NA

    # Look up neighbor values
    edges[, nval := vals_t[to]]

    # Group by 'from' cell, compute max and min ignoring NA
    agg <- edges[!is.na(nval), .(nmax = max(nval), nmin = min(nval)), by = from]

    neighbor_max_mat[agg$from, t] <- agg$nmax
    neighbor_min_mat[agg$from, t] <- agg$nmin
  }

  # --- Map results back to the rows of dt ---
  idx <- cbind(dt$cell_idx, dt$year_idx)

  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := neighbor_max_mat[idx]]
  dt[, (min_col)  := neighbor_min_mat[idx]]
  dt[, (mean_col) := neighbor_mean_mat[idx]]

  invisible(dt)
}

# --------------------------------------------------------------------------
# Step 4: Run for all neighbor source variables
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_neighbor_features_fast(
    cell_dt, var_name, W, n_cells, n_years, years, year_to_col
  )
}

# --------------------------------------------------------------------------
# Step 5: Remove helper columns and convert back if needed
# --------------------------------------------------------------------------
cell_dt[, c("cell_idx", "year_idx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is unchanged â€” use cell_data for prediction
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Summary

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~60â€“80 hrs (6.46M string ops) | ~2 sec (sparse matrix construction) |
| Neighbor mean (per var) | Included above | ~3 sec (sparse mat-mul, 344K Ã— 28) |
| Neighbor max/min (per var) | Included above | ~30 sec (data.table grouped agg Ã— 28 years) |
| **Total for 5 variables** | **~86+ hours** | **~5â€“10 minutes** |
| Peak RAM | ~10â€“14 GB (list of 6.46M vectors) | ~3â€“5 GB (sparse matrix + dense matrices) |

The numerical results are identical: same rook topology, same NA handling, same max/min/mean definitions. The trained Random Forest model is never touched.
END_RESPONSE>>>
