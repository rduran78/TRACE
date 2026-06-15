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
run_id: Anthropic_F08_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is in `build_neighbor_lookup`. It constructs a lookup **per row** (â‰ˆ6.46 million rows), but the neighbor *topology* is purely spatial and **never changes across years**. The function redundantly recomputes the same neighbor cell-ID sets 28 times (once per year) for each of the 344,208 cells. Additionally, the string-key-based index lookup (`paste` + named vector matching) over 6.46 million keys is extremely slow â€” O(n) hash lookups repeated ~6.46 million times, each scanning a neighbor set.

`compute_neighbor_stats` then iterates over the 6.46 million-element list with `lapply`, which is also slow compared to vectorized or matrix-based approaches.

**Summary of waste:**
1. **Redundant topology expansion:** The same neighbor graph is re-resolved 28Ã— (once per year). Topology is static; only variable values change.
2. **String-key indexing:** `paste(id, year)` keys and named-vector lookups are far slower than integer-indexed matrix operations.
3. **Row-level R loops:** `lapply` over 6.46M rows in pure R is inherently slow for what is essentially a sparse-matrixâ€“vector product (mean) or parallel-min/max.

## Optimization Strategy

**Separate the static topology from the dynamic variable values:**

1. **Build the neighbor topology once** as a sparse adjacency structure over the 344,208 cells (not over 6.46M cell-years). Use a sparse matrix (`Matrix::sparseMatrix`) â€” this is built once and reused for all years and all variables.

2. **For each variable, operate year-by-year using sparse matrix algebra:**
   - Reshape the variable into a 344,208 Ã— 28 matrix (cells Ã— years).
   - Neighbor **mean** = sparse matrix multiply (`W %*% values`) divided by neighbor count â€” fully vectorized.
   - Neighbor **max** and **min**: iterate over years (28 iterations, not 6.46M), and for each year use the sparse structure with vectorized grouped operations.

3. **Avoid any string keys or per-row R loops.**

This reduces the work from ~6.46M R-level iterations to 28 vectorized sparse operations per variable, yielding orders-of-magnitude speedup.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the sparse adjacency matrix ONCE (static topology)
# ==============================================================================
# Inputs:
#   id_order             â€” vector of 344,208 cell IDs (defines positional index)
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)

build_sparse_adjacency <- function(id_order, neighbors) {
  n <- length(id_order)
  # Build COO (coordinate) representation
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0L) {
      from <- c(from, rep.int(i, length(nb_i)))
      to   <- c(to, nb_i)
    }
  }
  # Sparse logical/binary adjacency matrix (row i has 1s in columns that are i's neighbors)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  # Also compute the number of neighbors per cell (for mean calculation)
  neighbor_count <- diff(W@p)  # for dgCMatrix, column counts; but we need row counts
  # Actually for row counts:
  neighbor_count <- rowSums(W)
  list(W = W, neighbor_count = neighbor_count)
}

# ==============================================================================
# STEP 2: Compute neighbor stats for one variable using sparse ops
# ==============================================================================
# This function takes the data as a data.table, the adjacency info, and a
# variable name, and returns the data.table with neighbor_max, neighbor_min,
# neighbor_mean columns added.

compute_neighbor_features_sparse <- function(dt, var_name, W, neighbor_count,
                                             id_order, years) {
  n_cells <- length(id_order)
  n_years <- length(years)

  # Create a mapping from cell ID to positional index
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Map each row to its cell position
  cell_pos <- id_to_pos[as.character(dt$id)]

  # --- Build a cells x years matrix of the variable values ---
  # We need a consistent year ordering
  year_to_col <- setNames(seq_along(years), as.character(years))
  year_col <- year_to_col[as.character(dt$year)]

  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(cell_pos, year_col)] <- dt[[var_name]]

  # --- Neighbor MEAN (sparse matrix multiply) ---
  # W %*% val_mat gives sum of neighbor values for each cell and year
  neighbor_sum <- as.matrix(W %*% val_mat)   # n_cells x n_years
  # Divide by neighbor count (avoid /0)
  safe_count <- ifelse(neighbor_count == 0, NA_real_, neighbor_count)
  neighbor_mean_mat <- neighbor_sum / safe_count  # recycled column-wise

  # --- Neighbor MAX and MIN ---
  # Strategy: use the sparse structure of W. For each cell i, we need
  # max/min of val_mat[neighbors_of_i, year]. We iterate over years (28 times)
  # and use the CSR-like structure.

  # Convert W to dgRMatrix (row-oriented) for efficient row-wise neighbor access
  # Or use the dgCMatrix and work with its transpose for column slicing.
  # Actually, we can extract the neighbor list from W once:

  # Extract neighbor indices from sparse matrix (CSC format)
  Wt <- t(W)  # now column j of Wt = row j of W = neighbors of j
  # Wt is dgCMatrix: Wt@p, Wt@i give column pointers and row indices

  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  p <- Wt@p
  idx <- Wt@i + 1L  # 0-based to 1-based

  for (yr_col in seq_len(n_years)) {
    vals_this_year <- val_mat[, yr_col]

    # Vectorized grouped max/min using the CSC structure
    # For each cell j, neighbors are idx[(p[j]+1):p[j+1]]
    # We process all cells at once using C-style pointer arithmetic in R

    # Pre-extract all neighbor values in one shot
    all_nb_vals <- vals_this_year[idx]  # length = number of directed edges

    # Now compute grouped max/min over the ragged structure defined by p
    # Use a fast approach: rep cell index, then tapply or data.table
    # But even faster: direct C-pointer loop in R vectorized form

    # Build group vector: cell j owns entries (p[j]+1):(p[j+1])
    # group_id <- rep(seq_len(n_cells), times = diff(p))
    # This is vectorized and fast

    grp_sizes <- diff(p)
    has_neighbors <- grp_sizes > 0L
    grp_id <- rep(which(has_neighbors), times = grp_sizes[has_neighbors])

    # Subset to non-NA neighbor values
    valid <- !is.na(all_nb_vals)
    if (any(valid)) {
      grp_id_valid <- grp_id[valid]
      vals_valid    <- all_nb_vals[valid]

      # Use data.table for fast grouped min/max
      tmp <- data.table(g = grp_id_valid, v = vals_valid)
      agg <- tmp[, .(mx = max(v), mn = min(v)), by = g]

      neighbor_max_mat[agg$g, yr_col] <- agg$mx
      neighbor_min_mat[agg$g, yr_col] <- agg$mn
    }
  }

  # --- Map results back to the original row order of dt ---
  row_idx <- cbind(cell_pos, year_col)

  max_col_name  <- paste0("neighbor_max_", var_name)
  min_col_name  <- paste0("neighbor_min_", var_name)
  mean_col_name <- paste0("neighbor_mean_", var_name)

  dt[[max_col_name]]  <- neighbor_max_mat[row_idx]
  dt[[min_col_name]]  <- neighbor_min_mat[row_idx]
  dt[[mean_col_name]] <- neighbor_mean_mat[row_idx]

  dt
}

# ==============================================================================
# STEP 3: Full pipeline
# ==============================================================================

# Convert to data.table for speed (if not already)
cell_data <- as.data.table(cell_data)

# Unique sorted years
years <- sort(unique(cell_data$year))

# Build sparse adjacency ONCE
message("Building sparse adjacency matrix (one-time cost)...")
adj <- build_sparse_adjacency(id_order, rook_neighbors_unique)
W              <- adj$W
neighbor_count <- adj$neighbor_count

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  cell_data <- compute_neighbor_features_sparse(
    dt             = cell_data,
    var_name       = var_name,
    W              = W,
    neighbor_count = neighbor_count,
    id_order       = id_order,
    years          = years
  )
}

# --- Prediction with the pre-trained Random Forest (unchanged) ---
# The model object and predict() call remain exactly as before.
# e.g.:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | Neighbor mean = `(W %*% vals) / count` is identical to averaging neighbor values. Max and min are computed over exactly the same neighbor sets defined by the same `rook_neighbors_unique` topology. |
| **Trained RF model untouched** | No retraining occurs. The code only prepares the same input features with the same column names (`neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>`), then calls `predict()` on the existing model object. |
| **Static vs. dynamic separation** | The sparse matrix `W` and `neighbor_count` are built **once** from the time-invariant topology. The variable matrices are sliced per year from the changing panel data. |

## Expected Speedup

| Component | Before | After |
|---|---|---|
| Topology resolution | 6.46M string-key lookups | 1 sparse matrix construction (~1.37M entries) |
| Neighbor mean | 6.46M `lapply` iterations | 5 sparse matrix multiplies (each ~seconds) |
| Neighbor max/min | 6.46M `lapply` iterations | 5 vars Ã— 28 years = 140 vectorized grouped ops |
| **Estimated total** | **86+ hours** | **~5â€“15 minutes** |
END_RESPONSE>>>
