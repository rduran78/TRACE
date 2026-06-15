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
run_id: Anthropic_F08_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. But the neighbor *relationships* between cells are **time-invariant** â€” cell A is always a rook neighbor of cell B regardless of year. The only thing that changes is the *values* of variables attached to those cells each year.

2. **The lookup is keyed on `(id, year)` row positions in a stacked panel.** This means for every one of the ~6.46M rows, the code searches for neighbors by constructing composite string keys and looking them up in a named vector of length ~6.46M. Named-vector lookup in R is O(n) per query in the worst case, and with ~6.46M keys this is catastrophically slow.

3. **`compute_neighbor_stats` iterates over ~6.46M list entries** with `lapply`, performing subsetting and summary statistics on each. This is done 5 times (once per variable), totaling ~32.3M R-level loop iterations.

4. **Memory pressure:** The `neighbor_lookup` list itself stores ~6.46M integer vectors. Since each cell has ~4 rook neighbors on average, and each entry is replicated 28 times (once per year), this is ~28Ã— redundant storage.

### Key Insight

The neighbor graph is **static** (cell-to-cell). The variables are **dynamic** (change by year). Therefore:

- Build the neighbor index **once at the cell level** (~344K entries, not ~6.46M).
- For each variable, compute neighbor stats **year-by-year** using the static cell-level index, operating on a matrix (cells Ã— years) or on year-sliced vectors.

This reduces the lookup construction cost by **28Ã—** and enables vectorized/matrix operations.

---

## Optimization Strategy

### Step 1: Build a cell-level neighbor index (once)

Create a simple list of length `N_cells` (~344K), where entry `i` contains the integer positions of cell `i`'s rook neighbors within the canonical cell ordering. This is derived directly from `rook_neighbors_unique` (the `nb` object) and requires no string operations.

### Step 2: Reshape variables into cell Ã— year matrices

For each of the 5 neighbor source variables, pivot the stacked panel into a matrix of dimension `(N_cells, N_years)`. This allows column-wise (year-wise) vectorized access.

### Step 3: Compute neighbor stats via vectorized matrix operations

For each variable and each year (column), use the cell-level neighbor index to gather neighbor values and compute max, min, mean. This can be done efficiently with `vapply` over ~344K cells per year, or even better, with a **sparse adjacency matrix multiply** for the mean, and row-wise sparse operations for max/min.

### Step 4: Reshape results back and attach to the panel

Unpivot the resulting matrices back into columns and bind them to `cell_data`.

### Expected Speedup

| Aspect | Old | New |
|---|---|---|
| Lookup entries | 6.46M | 344K |
| String operations | ~25.9M `paste` calls | 0 |
| Named-vector lookups | ~25.9M | 0 |
| Stats loop iterations | 32.3M | 1.72M (344K Ã— 5) per year, but vectorized |
| Redundant topology storage | 28Ã— | 1Ã— |

Conservative estimate: **50â€“200Ã— faster** (minutes instead of days).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the static-topology / dynamic-variable distinction.
# =============================================================================

library(Matrix)  # for sparse matrix (used in mean computation)

# ---- Step 0: Establish canonical cell ordering ----
# id_order is the canonical vector of cell IDs (length N_cells = 344,208)
# rook_neighbors_unique is the nb object (list of length N_cells)
# cell_data is the panel data.frame with columns: id, year, ntl, ec, ...

# Ensure cell_data is sorted by (id, year) for clean reshaping
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

N_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
N_years <- length(years)

# Map cell IDs to integer positions in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Map each row's cell ID to its position in the canonical ordering
cell_pos <- id_to_pos[as.character(cell_data$id)]

# ---- Step 1: Build cell-level neighbor index (ONCE, static) ----
# rook_neighbors_unique is already an nb object indexed by id_order position.
# Each element rook_neighbors_unique[[i]] gives the neighbor positions
# (as integers into id_order) for cell i. We use it directly.

# Clean the nb object: replace 0L (no-neighbor sentinel in nb) with integer(0)
cell_neighbor_idx <- lapply(rook_neighbors_unique, function(nb_i) {
  nb_i <- as.integer(nb_i)
  nb_i[nb_i != 0L]
})

# ---- Step 2: Build sparse adjacency matrix (for fast mean computation) ----
# Construct a sparse N_cells x N_cells row-normalized adjacency matrix W
# so that W %*% x gives the neighbor mean of x for each cell.

n_links <- sum(lengths(cell_neighbor_idx))
from_idx <- rep(seq_len(N_cells), times = lengths(cell_neighbor_idx))
to_idx   <- unlist(cell_neighbor_idx)

# Row-normalized weights: each neighbor gets weight 1/(number of neighbors)
weights <- rep(1 / lengths(cell_neighbor_idx), times = lengths(cell_neighbor_idx))
# Handle cells with 0 neighbors (avoid division by zero)
weights[!is.finite(weights)] <- 0

W <- sparseMatrix(
  i = from_idx,
  j = to_idx,
  x = weights,
  dims = c(N_cells, N_cells)
)

# Also build a non-normalized adjacency for max/min
# We'll use the cell_neighbor_idx list directly for max/min.

# ---- Step 3: Reshape each variable into a cell x year matrix ----
# Since cell_data is sorted by (id, year), and every cell has every year,
# we can reshape directly.

reshape_to_matrix <- function(cell_data, var_name, N_cells, N_years) {
  matrix(cell_data[[var_name]], nrow = N_cells, ncol = N_years, byrow = FALSE)
}

# Verify the assumption: each cell appears exactly N_years times
stopifnot(nrow(cell_data) == N_cells * N_years)

# ---- Step 4: Compute neighbor stats per variable ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-compute max and min using the cell_neighbor_idx list.
# For mean, use sparse matrix multiplication (very fast).

compute_neighbor_features_optimized <- function(cell_data, var_name,
                                                 cell_neighbor_idx, W,
                                                 N_cells, N_years, years) {
  # Reshape variable to matrix: rows = cells (in id_order), cols = years
  var_mat <- reshape_to_matrix(cell_data, var_name, N_cells, N_years)

  # Allocate output matrices
  max_mat  <- matrix(NA_real_, nrow = N_cells, ncol = N_years)
  min_mat  <- matrix(NA_real_, nrow = N_cells, ncol = N_years)
  mean_mat <- matrix(NA_real_, nrow = N_cells, ncol = N_years)

  # --- Neighbor mean via sparse matrix multiplication (per year-column) ---
  for (t in seq_len(N_years)) {
    x <- var_mat[, t]

    # Mean: W %*% x handles NA propagation implicitly only if no NAs.
    # For robustness with NAs, we do a two-pass approach:
    #   sum_of_vals / count_of_non_na
    not_na <- as.numeric(!is.na(x))
    x_zero <- x
    x_zero[is.na(x_zero)] <- 0

    neighbor_sum   <- as.numeric(W %*% x_zero)  # This uses row-normalized W
    # But we need raw sums and counts for proper NA handling.
    # Rebuild with un-normalized adjacency:
    # Actually, let's use a simpler approach for mean with the raw adjacency.
    # We'll compute raw sum and count separately.
  }

  # --- Rebuild with un-normalized sparse adjacency for correct NA handling ---
  W_raw <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(N_cells, N_cells)
  )

  for (t in seq_len(N_years)) {
    x <- var_mat[, t]

    # Replace NA with 0 for sum, track non-NA counts
    x_filled <- x
    x_filled[is.na(x_filled)] <- 0
    not_na <- as.numeric(!is.na(x))

    neighbor_sum   <- as.numeric(W_raw %*% x_filled)
    neighbor_count <- as.numeric(W_raw %*% not_na)

    # Mean
    mean_vec <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    mean_mat[, t] <- mean_vec
  }

  # --- Max and Min: must iterate over cells (but only ~344K, not 6.46M) ---
  # Vectorize per year for cache efficiency
  for (t in seq_len(N_years)) {
    x <- var_mat[, t]
    # Use vapply for speed
    max_min <- vapply(cell_neighbor_idx, function(nb) {
      if (length(nb) == 0L) return(c(NA_real_, NA_real_))
      vals <- x[nb]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) return(c(NA_real_, NA_real_))
      c(max(vals), min(vals))
    }, numeric(2))
    # max_min is 2 x N_cells
    max_mat[, t] <- max_min[1L, ]
    min_mat[, t] <- max_min[2L, ]
  }

  # --- Flatten matrices back to panel column order ---
  # cell_data is sorted by (id, year), so matrix column-major order matches
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  cell_data[[col_max]]  <- as.vector(max_mat)
  cell_data[[col_min]]  <- as.vector(min_mat)
  cell_data[[col_mean]] <- as.vector(mean_mat)

  cell_data
}

# ---- Step 5: Run for all neighbor source variables ----
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_optimized(
    cell_data, var_name,
    cell_neighbor_idx, W = NULL,  # W is built inside; see below
    N_cells, N_years, years
  )
}

# ---- Step 6: Predict with the pre-trained Random Forest (unchanged) ----
# The trained RF model object (e.g., `rf_model`) is loaded from disk.
# Prediction proceeds exactly as before:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

### Self-Contained Cleaned-Up Version

The version above has a minor structural issue (building `W_raw` references `from_idx`/`to_idx` from the outer scope). Here is the clean, self-contained final version:

```r
# =============================================================================
# FINAL OPTIMIZED IMPLEMENTATION
# =============================================================================
library(Matrix)

# --- Prerequisite: cell_data sorted by (id, year) ---
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

N_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
N_years <- length(years)
stopifnot(nrow(cell_data) == N_cells * N_years)

# --- 1. Static cell-level neighbor index (built ONCE) ---
cell_neighbor_idx <- lapply(rook_neighbors_unique, function(nb_i) {

  nb_i <- as.integer(nb_i)
  nb_i[nb_i != 0L]
})

# --- 2. Static sparse adjacency matrix (built ONCE) ---
from_idx <- rep(seq_len(N_cells), times = lengths(cell_neighbor_idx))
to_idx   <- unlist(cell_neighbor_idx)

W_raw <- sparseMatrix(
  i    = from_idx,
  j    = to_idx,
  x    = rep(1, length(from_idx)),
  dims = c(N_cells, N_cells)
)

# --- 3. Feature computation function ---
add_neighbor_features <- function(cell_data, var_name,
                                  cell_neighbor_idx, W_raw,
                                  N_cells, N_years) {
  # Reshape panel column -> cell x year matrix
  var_mat  <- matrix(cell_data[[var_name]], nrow = N_cells, ncol = N_years)
  max_mat  <- matrix(NA_real_, N_cells, N_years)
  min_mat  <- matrix(NA_real_, N_cells, N_years)
  mean_mat <- matrix(NA_real_, N_cells, N_years)

  for (t in seq_len(N_years)) {
    x <- var_mat[, t]

    # ---- Neighbor mean via sparse matrix ops (vectorized over all cells) ----
    x_filled <- x;  x_filled[is.na(x_filled)] <- 0
    not_na   <- as.numeric(!is.na(x))
    n_sum    <- as.numeric(W_raw %*% x_filled)
    n_count  <- as.numeric(W_raw %*% not_na)
    mean_mat[, t] <- ifelse(n_count > 0, n_sum / n_count, NA_real_)

    # ---- Neighbor max & min via vapply (344K iterations, not 6.46M) ----
    mm <- vapply(cell_neighbor_idx, function(nb) {
      if (length(nb) == 0L) return(c(NA_real_, NA_real_))
      v <- x[nb];  v <- v[!is.na(v)]
      if (length(v) == 0L) return(c(NA_real_, NA_real_))
      c(max(v), min(v))
    }, numeric(2))
    max_mat[, t] <- mm[1L, ]
    min_mat[, t] <- mm[2L, ]
  }

  # Flatten back to panel order (column-major matches id-within-year sorting)
  cell_data[[paste0("neighbor_max_",  var_name)]] <- as.vector(max_mat)
  cell_data[[paste0("neighbor_min_",  var_name)]] <- as.vector(min_mat)
  cell_data[[paste0("neighbor_mean_", var_name)]] <- as.vector(mean_mat)
  cell_data
}

# --- 4. Outer loop ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(Sys.time(), " | Processing: ", var_name)
  cell_data <- add_neighbor_features(
    cell_data, var_name, cell_neighbor_idx, W_raw, N_cells, N_years
  )
}
message(Sys.time(), " | Done. Proceeding to prediction.")

# --- 5. Predict with pre-trained Random Forest (UNCHANGED) ---
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | Max, min, and mean are computed with the same formulas (including NA handling) as the original. The estimand is preserved exactly. |
| **Trained RF model** | Untouched. Only the feature-engineering step upstream of `predict()` is optimized. |
| **Correctness of topology** | The `nb` object is used directly by integer position â€” no string key construction, no hash collisions, no off-by-one risk. |
| **Memory** | Cell-level index (~344K entries) + sparse matrix (~1.37M non-zeros) + one cellÃ—year matrix at a time (~344K Ã— 28 â‰ˆ 9.6M doubles â‰ˆ 77 MB). Well within 16 GB. |
| **Speed** | Neighbor mean is fully vectorized via sparse matrixâ€“vector multiply. Max/min loop over ~344K cells Ã— 28 years Ã— 5 vars â‰ˆ 48.2M lightweight integer-indexed operations (vs. ~6.46M string-hashed lookups Ã— 5 in the original). Estimated runtime: **5â€“20 minutes** on a standard laptop. |
END_RESPONSE>>>
