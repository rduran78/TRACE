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
run_id: Anthropic_F05_T00_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_004.md

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
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup via string matching on a 6.46M-element named vector**, which is O(N) per probe in the worst case (hash collisions aside, the sheer memory traffic is enormous).

This means the string-key construction and lookup is **O(N Ã— K)** where N = 6.46M rows and K = average neighbor count (~4), yielding ~25.8 billion character comparisons in the worst case. Even with R's internal hashing of named vectors, the constant factors are brutal: each `paste` allocates, each lookup probes a 6.46M-entry hash table.

### The Deeper Structural Issue

The lookup conflates **two independent dimensions** (cell ID and year) into a single string key. But the neighbor structure is **time-invariant** â€” cell 42's neighbors are the same in 1992 and 2019. The year dimension is only needed to align rows. This means:

- The neighbor topology needs to be resolved **once per cell** (344K cells), not once per cell-year (6.46M rows).
- The year-to-row mapping is a simple structured index, not something requiring string hashing.

### `compute_neighbor_stats` Is Also Suboptimal

It loops over 6.46M entries in `neighbor_lookup`, each time subsetting a numeric vector by integer indices. This is acceptable but can be replaced with a single vectorized matrix operation.

---

## Optimization Strategy

**Principle: Separate the spatial dimension from the temporal dimension.**

1. **Build a cell-index â†’ row-indices mapping** (344K cells Ã— 28 years). Since the panel is balanced (or near-balanced), create a matrix where `row_matrix[cell_pos, year_pos]` gives the row number in `data`. This is O(N) to build, no strings.

2. **Build a neighbor-row-index matrix** by expanding the `nb` object once. For each cell, look up its neighbors' cell positions, then use the row matrix to get all (neighbor, year) row indices. This produces a **pre-expanded integer index list** â€” one per cell-year row â€” using only integer arithmetic.

3. **Vectorize `compute_neighbor_stats`** using the pre-built integer index list, or better yet, use a sparse-matrix multiplication approach: construct a sparse neighbor-weight matrix W (6.46M Ã— 6.46M) where entry (i, j) = 1 if row j is a spatial neighbor of row i in the same year. Then neighbor means = `(W %*% x) / (W %*% 1)`, neighbor max/min via row-wise operations.

4. **For max/min**, sparse matrix multiplication doesn't directly help, but we can use `data.table` grouped operations or a chunked approach.

The **most practical approach** for a 16 GB laptop: use `data.table` to avoid the string-key pattern entirely, and compute neighbor stats via vectorized joins.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# =============================================================================
# STEP 1: Build the neighbor lookup using integer arithmetic only
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table if not already
  dt <- as.data.table(data)
  
  # Create integer mappings: cell_id -> cell_position (1-based in id_order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Create year -> year_position mapping
  years_sorted <- sort(unique(dt$year))
  year_to_pos <- setNames(seq_along(years_sorted), as.character(years_sorted))
  
  # Add cell_pos and year_pos columns to data
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_pos := year_to_pos[as.character(year)]]
  
  # Build a matrix: row_matrix[cell_pos, year_pos] = row index in dt
  # This replaces the entire string-key lookup
  n_cells <- length(id_order)
  n_years <- length(years_sorted)
  
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$cell_pos, dt$year_pos)] <- seq_len(nrow(dt))
  
  list(
    dt = dt,
    row_matrix = row_matrix,
    id_to_pos = id_to_pos,
    year_to_pos = year_to_pos,
    n_cells = n_cells,
    n_years = n_years,
    years_sorted = years_sorted
  )
}

# =============================================================================
# STEP 2: Build sparse neighbor matrix (same-year neighbors only)
#          W is (N x N) where N = nrow(data), W[i,j] = 1 iff row j is a
#          spatial neighbor of row i AND they share the same year.
# =============================================================================

build_sparse_neighbor_matrix <- function(lookup, neighbors) {
  row_matrix <- lookup$row_matrix
  n_cells    <- lookup$n_cells
  n_years    <- lookup$n_years
  dt         <- lookup$dt
  N          <- nrow(dt)
  
  # Pre-calculate total number of non-zero entries for memory pre-allocation
  # For each cell, count neighbors; multiply by number of years it appears
  neighbor_counts <- vapply(neighbors, length, integer(1))  # length 344K
  
  # For each cell_pos, count how many years have non-NA rows
  years_per_cell <- rowSums(!is.na(row_matrix))  # length 344K
  
  # Total directed neighbor-year pairs (upper bound for nnz)
  total_nnz <- sum(as.numeric(neighbor_counts) * as.numeric(years_per_cell))
  cat("Estimated nnz in sparse matrix:", total_nnz, "\n")
  
  # Pre-allocate vectors for sparse matrix triplets
  row_i <- integer(total_nnz)
  col_j <- integer(total_nnz)
  ptr <- 0L
  
  # Iterate over cells (344K iterations, not 6.46M)
  for (c_pos in seq_len(n_cells)) {
    nb_positions <- neighbors[[c_pos]]
    if (length(nb_positions) == 0L) next
    
    # Get neighbor cell positions in id_order
    # nb_positions already indexes into id_order (spdep::nb convention)
    
    for (y_pos in seq_len(n_years)) {
      focal_row <- row_matrix[c_pos, y_pos]
      if (is.na(focal_row)) next
      
      # Get neighbor rows for the same year
      nb_rows <- row_matrix[nb_positions, y_pos]
      nb_rows <- nb_rows[!is.na(nb_rows)]
      if (length(nb_rows) == 0L) next
      
      idx_range <- (ptr + 1L):(ptr + length(nb_rows))
      row_i[idx_range] <- focal_row
      col_j[idx_range] <- nb_rows
      ptr <- ptr + length(nb_rows)
    }
  }
  
  # Trim to actual size
  row_i <- row_i[1:ptr]
  col_j <- col_j[1:ptr]
  
  W <- sparseMatrix(
    i = row_i, j = col_j, x = rep(1, ptr),
    dims = c(N, N)
  )
  
  return(W)
}

# =============================================================================
# STEP 3: Compute neighbor stats vectorized using sparse matrix
# =============================================================================

compute_neighbor_stats_sparse <- function(dt, W, var_name) {
  x <- dt[[var_name]]
  
  # Replace NA with 0 for matrix multiplication, but track validity
  not_na <- as.numeric(!is.na(x))
  x_clean <- ifelse(is.na(x), 0, x)
  
  # Number of non-NA neighbors per row
  n_valid <- as.vector(W %*% not_na)
  
  # Sum of neighbor values (only non-NA contribute)
  neighbor_sum <- as.vector(W %*% x_clean)
  
  # Mean
  neighbor_mean <- ifelse(n_valid > 0, neighbor_sum / n_valid, NA_real_)
  
  # For max and min, we need a different approach since sparse matmul

  # doesn't give us max/min directly. We use a chunked row-wise approach.
  # 
  # Key insight: W is sparse, so we iterate over its row structure.
  # With dgCMatrix (column-sparse), we transpose to get row access via columns.
  
  Wt <- t(W)  # Now columns of Wt correspond to rows of W
  
  neighbor_max <- rep(NA_real_, nrow(dt))
  neighbor_min <- rep(NA_real_, nrow(dt))
  
  # Process in chunks to manage memory
  chunk_size <- 50000L
  N <- nrow(dt)
  n_chunks <- ceiling(N / chunk_size)
  
  cat("Computing max/min for", var_name, "in", n_chunks, "chunks\n")
  
  for (ch in seq_len(n_chunks)) {
    start_row <- (ch - 1L) * chunk_size + 1L
    end_row   <- min(ch * chunk_size, N)
    rows      <- start_row:end_row
    
    # Extract the submatrix: columns of Wt for these rows
    Wt_sub <- Wt[, rows, drop = FALSE]
    
    # For each column (= each focal row), find non-zero entries
    # dgCMatrix: @p gives column pointers, @i gives row indices (0-based)
    p <- Wt_sub@p
    idx_all <- Wt_sub@i + 1L  # 1-based row indices
    
    for (k in seq_along(rows)) {
      col_start <- p[k] + 1L
      col_end   <- p[k + 1L]
      if (col_end < col_start) next  # no neighbors
      
      nb_idx <- idx_all[col_start:col_end]
      nb_vals <- x[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      
      if (length(nb_vals) == 0L) next
      
      neighbor_max[rows[k]] <- max(nb_vals)
      neighbor_min[rows[k]] <- min(nb_vals)
    }
  }
  
  list(
    max  = neighbor_max,
    min  = neighbor_min,
    mean = neighbor_mean
  )
}

# =============================================================================
# STEP 4: Alternative â€” fully vectorized max/min using data.table edge list
#          This avoids the inner R loop entirely.
# =============================================================================

compute_neighbor_stats_dt <- function(dt, W, var_name) {
  x <- dt[[var_name]]
  
  # --- Mean via sparse matrix (fast) ---
  not_na  <- as.numeric(!is.na(x))
  x_clean <- ifelse(is.na(x), 0, x)
  n_valid <- as.vector(W %*% not_na)
  nb_sum  <- as.vector(W %*% x_clean)
  nb_mean <- ifelse(n_valid > 0, nb_sum / n_valid, NA_real_)
  
  # --- Max/Min via data.table grouped operations ---
  # Extract edge list from sparse matrix
  W_coo <- summary(W)  # gives i, j, x columns (COO format)
  
  edges <- data.table(
    focal    = W_coo$i,
    neighbor = W_coo$j
  )
  
  # Attach neighbor values
  edges[, nb_val := x[neighbor]]
  
  # Remove edges where neighbor value is NA
  edges <- edges[!is.na(nb_val)]
  
  # Grouped aggregation â€” fully vectorized in data.table
  agg <- edges[, .(
    nb_max = max(nb_val),
    nb_min = min(nb_val)
  ), by = focal]
  
  # Map back to full row set
  nb_max <- rep(NA_real_, length(x))
  nb_min <- rep(NA_real_, length(x))
  nb_max[agg$focal] <- agg$nb_max
  nb_min[agg$focal] <- agg$nb_min
  
  list(max = nb_max, min = nb_min, mean = nb_mean)
}

# =============================================================================
# STEP 5: Main pipeline â€” drop-in replacement
# =============================================================================

run_optimized_neighbor_features <- function(cell_data, id_order, 
                                             rook_neighbors_unique) {
  cat("=== Building integer lookup structures ===\n")
  t0 <- proc.time()
  
  lookup <- build_neighbor_lookup_fast(cell_data, id_order, 
                                        rook_neighbors_unique)
  
  cat("  Lookup built in", (proc.time() - t0)[3], "sec\n")
  
  cat("=== Building sparse neighbor matrix ===\n")
  t1 <- proc.time()
  
  W <- build_sparse_neighbor_matrix(lookup, rook_neighbors_unique)
  
  cat("  Sparse matrix built in", (proc.time() - t1)[3], "sec\n")
  cat("  Dimensions:", dim(W)[1], "x", dim(W)[2], 
      " nnz:", nnzero(W), "\n")
  
  # Extract edge list once (reused across all variables)
  cat("=== Extracting edge list ===\n")
  W_coo <- summary(W)
  edges_template <- data.table(
    focal    = W_coo$i,
    neighbor = W_coo$j
  )
  
  dt <- as.data.table(lookup$dt)
  
  # Precompute not_na-count vector (for mean denominator) â€” 
  # reuse W for each variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  cat("=== Computing neighbor features for", 
      length(neighbor_source_vars), "variables ===\n")
  
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    tv <- proc.time()
    
    x <- dt[[var_name]]
    
    # --- Mean via sparse matrix multiplication ---
    not_na  <- as.numeric(!is.na(x))
    x_clean <- ifelse(is.na(x), 0, x)
    n_valid <- as.vector(W %*% not_na)
    nb_sum  <- as.vector(W %*% x_clean)
    nb_mean <- ifelse(n_valid > 0, nb_sum / n_valid, NA_real_)
    
    # --- Max/Min via data.table grouped aggregation ---
    edges <- copy(edges_template)
    edges[, nb_val := x[neighbor]]
    edges <- edges[!is.na(nb_val)]
    
    agg <- edges[, .(
      nb_max = max(nb_val),
      nb_min = min(nb_val)
    ), by = focal]
    
    nb_max <- rep(NA_real_, nrow(dt))
    nb_min <- rep(NA_real_, nrow(dt))
    nb_max[agg$focal] <- agg$nb_max
    nb_min[agg$focal] <- agg$nb_min
    
    # --- Assign to data.table (same column names as original) ---
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := nb_max]
    dt[, (min_col)  := nb_min]
    dt[, (mean_col) := nb_mean]
    
    cat("    Done in", (proc.time() - tv)[3], "sec\n")
  }
  
  # Remove helper columns
  dt[, c("cell_pos", "year_pos") := NULL]
  
  cat("=== All neighbor features complete ===\n")
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# cell_data <- run_optimized_neighbor_features(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The trained Random Forest model is unchanged â€” just predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| **String key construction** | 6.46M Ã— `paste()` calls for lookup + 6.46M Ã— K Ã— `paste()` inside `lapply` | **Zero** string operations | âˆž |
| **Lookup structure** | 6.46M-entry named character vector (hash table with string keys) | Integer matrix `[344K Ã— 28]` â€” direct indexing | ~100Ã— |
| **Neighbor resolution** | 6.46M R-level `lapply` iterations, each doing string paste + hash probe | 344K Ã— 28 iterations building sparse matrix (integer arithmetic only) | ~10-50Ã— |
| **Neighbor mean** | R-level loop over 6.46M entries, subsetting + `mean()` each | Single sparse matrix multiplication `W %*% x` | ~500-1000Ã— |
| **Neighbor max/min** | R-level loop over 6.46M entries | `data.table` grouped aggregation on edge list | ~50-100Ã— |
| **Per-variable cost** | Rebuilds nothing but still loops 6.46M times | Sparse matmul + one `data.table` group-by | ~100Ã— |
| **Total 5 variables** | ~86+ hours estimated | **~5-15 minutes expected** | ~350-1000Ã— |

### Why This Works on 16 GB RAM

- **Sparse matrix W**: ~25.8M non-zeros (6.46M rows Ã— ~4 neighbors) Ã— 12 bytes â‰ˆ **310 MB** in CSC format.
- **Edge list**: ~25.8M rows Ã— 3 columns Ã— 8 bytes â‰ˆ **620 MB** (temporary, freed per variable).
- **Row matrix**: 344K Ã— 28 Ã— 4 bytes â‰ˆ **39 MB**.
- **Data itself**: 6.46M Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB**.
- **Total peak**: ~7-8 GB â€” fits in 16 GB with headroom.

### Numerical Equivalence

The optimized code computes **exactly the same `max`, `min`, and `mean`** over the same set of non-NA neighbor values for each row. The sparse matrix `W` encodes the identical neighbor-in-same-year relationship that the original string-key lookup resolved. The `data.table` aggregation and sparse matrix multiplication produce the same arithmetic results. The trained Random Forest model is never touched.
END_RESPONSE>>>
