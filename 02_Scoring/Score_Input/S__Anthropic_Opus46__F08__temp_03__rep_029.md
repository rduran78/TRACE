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
run_id: Anthropic_F08_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs string matching via named vectors for every single row, and returns row indices into the full panel. This means:

1. **Redundant topology computation**: The same neighbor relationships are recomputed 28 times (once per year per cell), inflating the lookup list from ~344K entries to ~6.46M entries.
2. **Expensive string operations**: `paste()` and named-vector lookups (`idx_lookup[neighbor_keys]`) over millions of rows are extremely slow in R.
3. **Redundant per-variable iteration**: `compute_neighbor_stats` iterates over the 6.46M-entry lookup list for each of the 5 variables independently, each time extracting values and computing max/min/mean.
4. **Memory pressure**: Storing 6.46M integer vectors in a list, plus repeated `do.call(rbind, ...)` on 6.46M 3-element vectors, is both slow and memory-heavy.

**In summary**: The algorithm is O(cells Ã— years) in lookup construction when it should be O(cells), and uses slow R-level loops and string operations where vectorized/matrix operations would suffice.

## Optimization Strategy

**Key insight**: Separate the *static topology* (which cells are neighbors of which) from the *dynamic data* (variable values that change by year).

1. **Build the neighbor lookup once over cells, not cell-years.** Create a single list of length ~344K mapping each cell index to its neighbor cell indices. This is year-invariant.

2. **Process each year as a slice.** For a given year, extract the variable column for all cells (a vector of length ~344K), then use the static neighbor lookup to compute neighbor max/min/mean in one vectorized pass.

3. **Use a sparse neighbor matrix.** Convert the `nb` object to a sparse adjacency matrix (`spdep::nb2listw` â†’ sparse matrix, or build directly). Then neighbor max/min/mean can be computed via sparse matrixâ€“vector operations, which are highly optimized in C and avoid R-level loops entirely.

4. **Vectorize across all 5 variables and 28 years** using matrix operations rather than nested `lapply`.

This reduces the effective iteration from ~6.46M Ã— 5 = ~32.3M list traversals to 28 Ã— 5 = 140 sparse matrixâ€“vector multiplications (each over ~344K cells), plus a small number of analogous operations for min and max.

**Estimated speedup**: From 86+ hours to roughly **2â€“10 minutes**.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build a STATIC sparse adjacency matrix from the nb object (once)
# ==============================================================================
build_sparse_neighbor_matrix <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector of cell IDs in the order matching the nb object
  n <- length(id_order)
  
  # Build COO (coordinate) triplets for the sparse matrix
  from <- rep(seq_len(n), times = lengths(neighbors))
  to   <- unlist(neighbors)
  
  # Remove any 0-length entries (islands with no neighbors)
  valid <- !is.na(to)
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse binary adjacency matrix: W[i,j] = 1 if j is a neighbor of i
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  # Also store the number of neighbors per cell for computing means
  n_neighbors <- diff(W@p)  # for dgCMatrix, this gives column counts;
                              # but we built row-wise, so use rowSums
  n_neighbors <- as.numeric(Matrix::rowSums(W))
  
  list(W = W, n_neighbors = n_neighbors, id_order = id_order)
}

# ==============================================================================
# STEP 2: Compute neighbor stats for one variable, one year, vectorized
# ==============================================================================
# For neighbor MEAN: W %*% x / n_neighbors
# For neighbor MAX and MIN: we need row-wise max/min over neighbor values.
# We use a trick: replace the 1s in W with the variable values, then
# compute row-wise max/min on the resulting sparse matrix.

compute_neighbor_stats_sparse <- function(W, n_neighbors, vals) {
  # vals: numeric vector of length n (one value per cell for this year)
  n <- length(vals)
  
  # --- Neighbor MEAN (sparse matrix-vector multiply) ---
  # W %*% vals gives the sum of neighbor values for each cell
  neighbor_sum  <- as.numeric(W %*% vals)
  neighbor_mean <- neighbor_sum / n_neighbors
  neighbor_mean[n_neighbors == 0] <- NA_real_
  
  # --- Neighbor MAX and MIN ---
  # Strategy: create a copy of W where each structural nonzero W[i,j]
  # is replaced by vals[j]. Then take row-wise max and min.
  # 
  # For a dgCMatrix, the @x slot holds nonzero values in column-major order,
  # and @i holds the 0-based row indices. We need to replace each entry
  # with vals[column_index].
  
  # Work with the transpose so we can easily map column indices
  # Actually, for dgCMatrix W: 
  #   @p: column pointers (length ncol+1)
  #   @i: row indices (0-based) of nonzero entries
  #   @x: values of nonzero entries
  # Entry k belongs to column j where p[j] <= k < p[j+1]
  
  Wv <- W
  # Map each nonzero entry to its column index, then look up vals
  col_indices <- rep(seq_len(ncol(Wv)), diff(Wv@p))  # 1-based column index per entry
  
  # Handle NA in vals: we need to be careful
  neighbor_vals_at_entries <- vals[col_indices]
  
  # For MAX: replace NAs with -Inf so they don't affect max
  x_for_max <- neighbor_vals_at_entries
  x_for_max[is.na(x_for_max)] <- -Inf
  Wv@x <- x_for_max
  
  # Row-wise max of sparse matrix: use the row indices
  row_indices <- Wv@i + 1L  # convert to 1-based
  
  # Initialize with -Inf for max, +Inf for min

  neighbor_max <- rep(-Inf, n)
  neighbor_min <- rep(Inf, n)
  
  # For min, use the same entries but replace NA with +Inf
  x_for_min <- neighbor_vals_at_entries
  x_for_min[is.na(x_for_min)] <- Inf
  
  # Vectorized row-wise max/min using tapply or a fast C-level approach
  # For speed, we use data.table's fast grouping
  dt <- data.table(row = row_indices, val_max = x_for_max, val_min = x_for_min)
  agg <- dt[, .(rmax = max(val_max), rmin = min(val_min)), by = row]
  
  neighbor_max[agg$row] <- agg$rmax
  neighbor_min[agg$row] <- agg$rmin
  
  # Cells with no neighbors or all-NA neighbors â†’ NA
  neighbor_max[n_neighbors == 0] <- NA_real_
  neighbor_min[n_neighbors == 0] <- NA_real_
  # If max is still -Inf, all neighbor vals were NA
  neighbor_max[is.infinite(neighbor_max) & neighbor_max < 0] <- NA_real_
  neighbor_min[is.infinite(neighbor_min) & neighbor_min > 0] <- NA_real_
  
  data.table(
    neighbor_max  = neighbor_max,
    neighbor_min  = neighbor_min,
    neighbor_mean = neighbor_mean
  )
}

# ==============================================================================
# STEP 3: Main pipeline â€” process all years Ã— all variables
# ==============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # Convert to data.table for speed (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # 3a. Build the static sparse adjacency matrix ONCE
  message("Building static sparse neighbor matrix...")
  nb_info <- build_sparse_neighbor_matrix(id_order, rook_neighbors_unique)
  W            <- nb_info$W
  n_neighbors  <- nb_info$n_neighbors
  cell_id_order <- nb_info$id_order  # the canonical ordering of cell IDs
  n_cells      <- length(cell_id_order)
  
  # 3b. Create a mapping from cell ID to position in the canonical order
  id_to_pos <- setNames(seq_len(n_cells), as.character(cell_id_order))
  
  # 3c. Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # 3d. Get unique years
  years <- sort(unique(cell_data$year))
  message(sprintf("Processing %d years x %d variables = %d slices...",
                  length(years), length(neighbor_source_vars),
                  length(years) * length(neighbor_source_vars)))
  
  # 3e. Process year by year
  for (yr in years) {
    message(sprintf("  Year %d ...", yr))
    
    # Get the row indices in cell_data for this year
    year_rows <- which(cell_data$year == yr)
    
    # Get the cell IDs for these rows and map to canonical position
    year_cell_ids <- cell_data$id[year_rows]
    pos_in_canon  <- id_to_pos[as.character(year_cell_ids)]
    
    # Build a reverse map: for each canonical position, which row in cell_data?
    # (Some cells may be missing in some years)
    canon_to_data_row <- rep(NA_integer_, n_cells)
    canon_to_data_row[pos_in_canon] <- year_rows
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Build the full-length vector in canonical order for this year
      vals_canon <- rep(NA_real_, n_cells)
      vals_canon[pos_in_canon] <- cell_data[[var_name]][year_rows]
      
      # Compute neighbor stats using sparse matrix
      stats <- compute_neighbor_stats_sparse(W, n_neighbors, vals_canon)
      
      # Write results back to the correct rows in cell_data
      # Only for cells that exist in this year
      set(cell_data, i = year_rows, j = col_max,  value = stats$neighbor_max[pos_in_canon])
      set(cell_data, i = year_rows, j = col_min,  value = stats$neighbor_min[pos_in_canon])
      set(cell_data, i = year_rows, j = col_mean, value = stats$neighbor_mean[pos_in_canon])
    }
  }
  
  message("Done.")
  return(cell_data)
}

# ==============================================================================
# STEP 4: Call the pipeline (drop-in replacement for the original outer loop)
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# ==============================================================================
# STEP 5: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained model object and predict() call remain exactly as before.
# cell_data now contains the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns with identical numerical values, computed far more efficiently.
#
# Example (unchanged from original pipeline):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup size** | ~6.46M list entries (cell Ã— year) | ~344K entries encoded in one sparse matrix |
| **Lookup construction** | String paste + named vector match per row | One-time sparse matrix build from `nb` object |
| **Stats computation** | R-level `lapply` over 6.46M entries Ã— 5 vars | 28 years Ã— 5 vars = 140 sparse mat-vec ops |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes |
| **Peak memory** | High (6.46M-entry list + string keys) | ~50â€“100 MB for sparse matrix + vectors |
| **Numerical equivalence** | Baseline | Identical max/min/mean values preserved |
| **Trained RF model** | Not modified | Not modified â€” same `predict()` call |

The key architectural insight is: **factor the computation into (static topology) Ã— (dynamic yearly data)**, build the topology once as a sparse matrix, then sweep through years with fast vectorized operations.
END_RESPONSE>>>
