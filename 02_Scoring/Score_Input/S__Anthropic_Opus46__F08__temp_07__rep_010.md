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
run_id: Anthropic_F08_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds row-index mappings per cell-year (~6.46M rows):** For each of the 6.46 million rows, it performs string-paste key construction (`paste(id, year, sep="_")`) and named-vector lookups to resolve neighbor *row indices*. These row indices change every time the data is reordered or subsetted, but the underlying **neighbor relationships between cells never change** â€” they are purely spatial.

2. **The neighbor lookup is a list of 6.46M elements**, each containing row indices into the full panel. This is ~6.46M list allocations with string-hashing operations â€” enormously wasteful.

3. **`compute_neighbor_stats` iterates over 6.46M list elements** with R-level `lapply`, extracting and summarizing neighbor values one row at a time. This is called 5 times (once per variable), totaling ~32.3 million R-level iterations.

4. **The key insight being missed:** The neighbor graph is a property of the 344,208 *cells*, not of the 6.46M *cell-years*. For any given year, the neighbor structure is identical. The current code redundantly re-resolves the same cell-to-cell topology 28 times (once per year) embedded inside a single monolithic row-index lookup.

### Bottleneck Summary

| Component | Current Scale | Optimal Scale |
|---|---|---|
| Neighbor lookup construction | 6.46M row-level | 344,208 cell-level (once) |
| Neighbor stat computation per variable | 6.46M R-level iterations | 344,208 cells Ã— 28 years, vectorized |
| Total R-level iterations (5 vars) | ~32.3M | Vectorized matrix operations |

---

## Optimization Strategy

**Separate the static topology from the dynamic variable values:**

1. **Build a cell-level neighbor index once (344K entries, not 6.46M).** Store for each cell the indices of its neighbors in cell-space (not row-space). This is a sparse adjacency structure over 344K cells.

2. **Organize variable data as a cell Ã— year matrix.** If the data is sorted by `(id, year)` and every cell has all 28 years, each variable becomes a 344,208 Ã— 28 matrix. Neighbor statistics become matrix-row operations.

3. **Use sparse matrix multiplication for neighbor aggregation.** Construct a sparse adjacency matrix `W` (344,208 Ã— 344,208) from `rook_neighbors_unique`. Then:
   - Neighbor **sum** = `W %*% X` (where `X` is a 344K Ã— 28 matrix of variable values)
   - Neighbor **count** = `W %*% (!is.na(X))` (for proper mean with NA handling)
   - Neighbor **mean** = sum / count
   - Neighbor **max** and **min** require a grouped-row approach (sparse matrix multiplication only handles sums), but can be computed efficiently with vectorized cell-level operations.

4. **Write results back into the panel data frame** in the correct row positions.

This reduces the problem from billions of R-level operations to a handful of sparse matrix multiplications and vectorized group operations over 344K cells.

**Expected speedup:** From ~86 hours to **minutes** (roughly 2â€“10 minutes depending on max/min strategy).

**Preservation guarantees:**
- The Random Forest model is never touched â€” we only recompute the same 15 input features (5 vars Ã— 3 stats) with identical numerical values.
- The original numerical estimand is preserved: max, min, and mean of each variable across rook neighbors, per cell-year.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static cell topology from dynamic year-varying variables
# =============================================================================

library(Matrix)  # for sparse matrix operations

# ---- Step 1: Build the static sparse adjacency matrix (once) ----------------
# This encodes the neighbor topology over 344,208 cells.
# rook_neighbors_unique: an nb object (list of length n_cells),
#   where each element contains integer indices of neighbors in id_order.
# id_order: vector of cell IDs defining the canonical cell ordering.

build_sparse_adjacency <- function(nb_object, n_cells) {
  # nb_object[[i]] contains the neighbor indices (in 1:n_cells) for cell i.
  # We build a sparse logical/binary adjacency matrix W of size n_cells x n_cells.
  
  from <- rep(seq_len(n_cells), times = lengths(nb_object))
  to   <- unlist(nb_object)
  
  # Remove the "0" entries that spdep uses to denote no-neighbor cells
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format, efficient for column operations
  )
  
  return(W)
}

# ---- Step 2: Prepare cell-year data as matrices -----------------------------
# Assumption: cell_data is a data.frame/data.table with columns id, year, and
#   the variable columns. We sort by (id, year) and reshape each variable
#   into a cell x year matrix.

prepare_cell_year_matrices <- function(cell_data, id_order, years, var_names) {
  # Ensure data is sorted by id (in id_order sequence) then year
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Map cell IDs to their position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Compute the row position in a cell x year matrix for each row of cell_data
  cell_pos <- id_to_pos[as.character(cell_data$id)]
  year_pos <- match(cell_data$year, years)
  
  # Linear index into a (n_cells x n_years) matrix stored column-major
  lin_idx <- cell_pos + (year_pos - 1L) * n_cells
  
  matrices <- list()
  for (vn in var_names) {
    mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat[lin_idx] <- cell_data[[vn]]
    matrices[[vn]] <- mat
  }
  
  list(
    matrices = matrices,
    cell_pos = cell_pos,
    year_pos = year_pos,
    lin_idx  = lin_idx,
    n_cells  = n_cells,
    n_years  = n_years
  )
}

# ---- Step 3: Compute neighbor stats using the static adjacency matrix -------
# For mean: use sparse matrix multiply (sum) and count.
# For max/min: iterate over cells using the nb list (vectorized over years).

compute_neighbor_features_optimized <- function(W, nb_object, var_matrix, 
                                                 n_cells, n_years) {
  # var_matrix: n_cells x n_years matrix of the variable values
  
  # --- Neighbor Mean (via sparse matrix multiplication) ---
  # Handle NAs: replace NA with 0 for sum, track non-NA counts
  not_na   <- !is.na(var_matrix)
  var_zero <- var_matrix
  var_zero[is.na(var_zero)] <- 0
  
  # W %*% var_zero gives sum of neighbor values (treating NA as 0)
  # W %*% not_na gives count of non-NA neighbors
  neighbor_sum   <- as.matrix(W %*% var_zero)    # n_cells x n_years
  neighbor_count <- as.matrix(W %*% not_na)      # n_cells x n_years
  
  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  # --- Neighbor Max and Min (cell-level vectorized over years) ---
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- nb_object[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) next
    
    # Extract the sub-matrix of neighbor values: (n_neighbors x n_years)
    nb_vals <- var_matrix[nb_idx, , drop = FALSE]
    
    # Compute column-wise (year-wise) max and min, respecting NAs
    # suppressWarnings for all-NA columns (returns -Inf/Inf, we fix below)
    col_max <- suppressWarnings(apply(nb_vals, 2, max, na.rm = TRUE))
    col_min <- suppressWarnings(apply(nb_vals, 2, min, na.rm = TRUE))
    
    # Fix all-NA columns
    all_na <- apply(is.na(nb_vals), 2, all)
    col_max[all_na] <- NA_real_
    col_min[all_na] <- NA_real_
    
    neighbor_max[i, ] <- col_max
    neighbor_min[i, ] <- col_min
  }
  
  list(
    n_max  = neighbor_max,
    n_min  = neighbor_min,
    n_mean = neighbor_mean
  )
}

# ---- Step 3b: Faster max/min using Rcpp-style vectorization in pure R -------
# The cell-level loop above (344K iterations with apply) may still be slow.
# We can speed up max/min by pre-building an edge list and using vectorized ops.

compute_neighbor_max_min_fast <- function(nb_object, var_matrix, n_cells, n_years) {
  # Build edge list: for each cell, which neighbors does it have?
  from_cell <- rep(seq_len(n_cells), times = lengths(nb_object))
  to_cell   <- unlist(nb_object)
  valid     <- to_cell > 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]
  
  # For each directed edge (from -> to), the "to" cell is a neighbor of "from".
  # We need: for each "from" cell, the max and min of var_matrix[to, year].
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Process year by year to keep memory manageable
  for (y in seq_len(n_years)) {
    vals_this_year <- var_matrix[, y]
    
    # Get neighbor values for all edges
    nb_vals <- vals_this_year[to_cell]  # length = n_edges
    
    # Use tapply or data.table for grouped max/min
    # tapply approach:
    from_factor <- from_cell  # already integer indices 1:n_cells
    
    # For max: need to handle NAs
    # Split and compute â€” but split on 344K groups with ~1.37M edges is fast
    max_vals <- rep(NA_real_, n_cells)
    min_vals <- rep(NA_real_, n_cells)
    
    # Remove edges where neighbor value is NA
    not_na_edge <- !is.na(nb_vals)
    if (any(not_na_edge)) {
      from_valid <- from_cell[not_na_edge]
      nb_valid   <- nb_vals[not_na_edge]
      
      # Use base R rowmax via tapply
      max_result <- tapply(nb_valid, from_valid, max)
      min_result <- tapply(nb_valid, from_valid, min)
      
      idx_max <- as.integer(names(max_result))
      idx_min <- as.integer(names(min_result))
      
      max_vals[idx_max] <- as.numeric(max_result)
      min_vals[idx_min] <- as.numeric(min_result)
    }
    
    neighbor_max[, y] <- max_vals
    neighbor_min[, y] <- min_vals
  }
  
  list(n_max = neighbor_max, n_min = neighbor_min)
}


# =============================================================================
# MAIN PIPELINE: Drop-in replacement for the outer loop
# =============================================================================

run_optimized_neighbor_pipeline <- function(cell_data, id_order, 
                                             rook_neighbors_unique,
                                             neighbor_source_vars) {
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat("Building static sparse adjacency matrix (", n_cells, " cells)...\n")
  W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)
  
  cat("Reshaping panel data into cell x year matrices...\n")
  prep <- prepare_cell_year_matrices(cell_data, id_order, years, 
                                      neighbor_source_vars)
  
  # We'll write results back using prep$lin_idx
  lin_idx <- prep$lin_idx
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    
    var_matrix <- prep$matrices[[var_name]]
    
    # Compute mean via sparse matrix multiplication
    not_na   <- !is.na(var_matrix)
    var_zero <- var_matrix
    var_zero[is.na(var_zero)] <- 0
    
    neighbor_sum   <- as.matrix(W %*% var_zero)
    neighbor_count <- as.matrix(W %*% not_na)
    neighbor_mean  <- neighbor_sum / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA_real_
    
    # Compute max and min via edge-list vectorized approach
    max_min <- compute_neighbor_max_min_fast(
      rook_neighbors_unique, var_matrix, n_cells, n_years
    )
    
    # Map results back from cell x year matrices to panel rows
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)
    
    cell_data[[col_max]]  <- max_min$n_max[lin_idx]
    cell_data[[col_min]]  <- max_min$n_min[lin_idx]
    cell_data[[col_mean]] <- neighbor_mean[lin_idx]
    
    cat("  Done:", var_name, "\n")
  }
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}

# =============================================================================
# EXECUTION
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_optimized_neighbor_pipeline(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# Now proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-hash lookups | 1 sparse matrix build (344K cells) |
| **Mean computation** | 6.46M R-level iterations Ã— 5 vars | 5 sparse matrix multiplications |
| **Max/Min computation** | 6.46M R-level iterations Ã— 5 vars | 28 years Ã— 5 vars vectorized `tapply` over 1.37M edges |
| **Memory for lookup** | 6.46M-element list of integer vectors | 344K Ã— 28 matrices + sparse matrix (~20 MB) |
| **Estimated time** | ~86+ hours | **~2â€“10 minutes** |

### Why Numerical Results Are Identical

- **Mean:** `W %*% X` computes exactly `sum(neighbor_values)` per cell; dividing by `W %*% (!is.na(X))` gives the identical `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min:** `tapply(nb_valid, from_valid, max)` computes the same grouped maximum as iterating per row and calling `max(neighbor_vals)`.
- **NA handling:** Edges with NA neighbor values are excluded before aggregation, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` logic.
- **The Random Forest model object is never modified** â€” only the input features are recomputed (identically) faster.
END_RESPONSE>>>
