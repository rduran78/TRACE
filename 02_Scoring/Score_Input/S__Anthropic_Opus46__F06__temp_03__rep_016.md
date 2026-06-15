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
run_id: Anthropic_F06_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation Ã— ~4 neighbors Ã— 6.46M rows â‰ˆ 25.8M string constructions).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) â€” named vector lookup in R is **O(n)** per query in the worst case because R's named vectors use linear hashing with potential collisions, and the vector has 6.46M entries.

**Net cost:** ~6.46M iterations Ã— ~4 neighbors Ã— expensive string ops + hash lookups = extremely slow. This alone could take hours.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows, called 5 times

Each call iterates over all 6.46M rows, subsetting a numeric vector by index and computing `max`, `min`, `mean`. The R-level `lapply` loop with per-element function calls is slow due to interpreter overhead. Called 5 times (once per variable), this is ~32.3M R function invocations.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics in C and are extremely fast â€” but they assume a **regular rectangular grid** where every cell has the same neighborhood structure. Here, the panel has an `spdep::nb` object (irregular neighbor structure, potentially with boundary cells having fewer neighbors, and the neighbor list is precomputed). However, the **key insight** from the focal analogy is: **we should vectorize the neighbor aggregation using sparse matrix multiplication or data.table joins rather than row-by-row R loops.**

### Memory estimate

6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ 5.7 GB. With neighbor features (5 vars Ã— 3 stats = 15 new columns), we add ~0.77 GB. Total ~6.5 GB fits in 16 GB RAM, but we must avoid unnecessary copies.

---

## 2. Optimization Strategy

### Strategy: Sparse adjacency matrix + vectorized matrix operations

1. **Replace `build_neighbor_lookup`** with a sparse **row-adjacency matrix** `W` of dimension `(n_rows Ã— n_rows)` where `n_rows = 6.46M`. Entry `W[i,j] = 1` if row `j` is a rook neighbor of row `i` **in the same year**. This matrix is constructed once using the spatial neighbor list and year matching.

2. **Replace `compute_neighbor_stats`** for `mean` with a single sparse matrix-vector multiplication: `W %*% x / row_counts`. For `max` and `min`, use a grouped operation via `data.table`.

3. **Key realization:** Since the spatial neighbor structure is **identical across all 28 years**, we can:
   - Build a small spatial adjacency matrix `W_spatial` (344,208 Ã— 344,208) once.
   - For each variable, reshape to a wide matrix (344,208 rows Ã— 28 columns), then compute neighbor stats using sparse matrix ops on each year-column simultaneously.

This avoids the 6.46M-row loop entirely.

### Expected speedup

- `build_neighbor_lookup`: eliminated entirely (replaced by sparse matrix construction, ~seconds).
- `compute_neighbor_stats`: replaced by sparse matrix multiplication (~seconds per variable).
- **Total estimated time: 2â€“10 minutes** instead of 86+ hours.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# 
# Prerequisites:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the
#     neighbor_source_vars. Rows are ordered consistently.
#   - id_order: vector of unique spatial cell IDs (same order as rook_neighbors_unique)
#   - rook_neighbors_unique: spdep::nb object (list of integer index vectors)
#   - The trained Random Forest model object (untouched)
#
# This code preserves the exact same numerical results as the original
# implementation: for each cell-year row, it computes max, min, and mean
# of each source variable across that cell's rook neighbors in the same year.
# =============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build the spatial sparse adjacency matrix (344,208 x 344,208)
# --------------------------------------------------------------------------
build_spatial_adjacency <- function(id_order, neighbors_nb) {
  n <- length(id_order)
  # Build COO (coordinate) format triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0; skip those
    if (length(nb_i) == 1L && nb_i[1L] == 0L) next
    from_list[[i]] <- rep.int(i, length(nb_i))
    to_list[[i]]   <- nb_i
  }
  
  row_idx <- unlist(from_list, use.names = FALSE)
  col_idx <- unlist(to_list,   use.names = FALSE)
  
  W <- sparseMatrix(
    i = row_idx,
    j = col_idx,
    x = rep.int(1, length(row_idx)),
    dims = c(n, n)
  )
  return(W)
}

cat("Building spatial adjacency matrix...\n")
W_spatial <- build_spatial_adjacency(id_order, rook_neighbors_unique)
cat("  Dimensions:", dim(W_spatial), "\n")
cat("  Non-zeros: ", nnzero(W_spatial), "\n")

# --------------------------------------------------------------------------
# STEP 2: Convert cell_data to data.table and create mapping
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure we know the mapping from cell id -> spatial index
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)
n_cells <- length(id_order)

cat("Panel: ", n_cells, "cells x", n_years, "years =", 
    n_cells * n_years, "potential rows\n")

# --------------------------------------------------------------------------
# STEP 3: For each variable, reshape to matrix, compute stats, reshape back
# --------------------------------------------------------------------------
# We reshape each variable into a (n_cells x n_years) matrix where
# row i corresponds to id_order[i] and column j corresponds to years[j].
# Then:
#   neighbor_mean = (W_spatial %*% X) / (W_spatial %*% valid_mask)
#   neighbor_max and neighbor_min require a different approach since
#   sparse matrix algebra doesn't directly support max/min.
#
# For max/min, we use an efficient grouped approach:
#   - Expand the neighbor pairs, join values, and aggregate.
#
# However, since we have the matrix form, we can iterate over cells
# in a VECTORIZED way per year using the sparse structure.
# --------------------------------------------------------------------------

# Pre-compute the neighbor list from the sparse matrix (CSC format) for max/min
# This is just the nb object re-indexed â€” we already have it.
# We'll use it for max/min via data.table.

# Create a long-form neighbor edge table (spatial only, ~1.37M rows)
cat("Building spatial edge table for max/min...\n")
edge_from <- vector("list", n_cells)
edge_to   <- vector("list", n_cells)
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 1L && nb_i[1L] == 0L) next
  edge_from[[i]] <- rep.int(i, length(nb_i))
  edge_to[[i]]   <- nb_i
}
edges_dt <- data.table(
  from_spatial = unlist(edge_from, use.names = FALSE),
  to_spatial   = unlist(edge_to,   use.names = FALSE)
)
cat("  Edge table rows:", nrow(edges_dt), "\n")

# Add spatial index to cell_data
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Create a key for fast joining: (spatial_idx, year)
setkey(cell_data, spatial_idx, year)

# --------------------------------------------------------------------------
# STEP 4: Compute neighbor features for each variable
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edges_dt, W_spatial,
                                          id_order, years, var_name) {
  cat("Processing variable:", var_name, "\n")
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # --- MEAN via sparse matrix multiplication (fastest) ---
  cat("  Computing neighbor means via sparse matmul...\n")
  
  # Build (n_cells x n_years) matrix
  # Map each row of cell_data to (spatial_idx, year_idx)
  year_to_idx <- setNames(seq_along(years), as.character(years))
  
  sp_idx  <- cell_data$spatial_idx
  yr_idx  <- year_to_idx[as.character(cell_data$year)]
  vals    <- cell_data[[var_name]]
  
  # Sparse matrix of values: rows = spatial cells, cols = years
  # For cells with NA, we need to handle them carefully
  valid_mask <- as.numeric(!is.na(vals))
  vals_clean <- ifelse(is.na(vals), 0, vals)
  
  X_val <- sparseMatrix(
    i = sp_idx, j = yr_idx, x = vals_clean,
    dims = c(n_cells, n_years)
  )
  X_mask <- sparseMatrix(
    i = sp_idx, j = yr_idx, x = valid_mask,
    dims = c(n_cells, n_years)
  )
  
  # Neighbor sums and counts
  neighbor_sum   <- W_spatial %*% X_val    # (n_cells x n_years)
  neighbor_count <- W_spatial %*% X_mask   # (n_cells x n_years)
  
  # Mean = sum / count (NA where count == 0)
  # Extract back to long form
  mean_vals <- numeric(nrow(cell_data))
  count_vals <- numeric(nrow(cell_data))
  
  # Extract efficiently
  for (j in seq_len(n_years)) {
    rows_j <- which(yr_idx == j)
    sp_j   <- sp_idx[rows_j]
    s_vec  <- neighbor_sum[, j]
    c_vec  <- neighbor_count[, j]
    mean_vals[rows_j]  <- s_vec[sp_j]
    count_vals[rows_j] <- c_vec[sp_j]
  }
  
  mean_result <- ifelse(count_vals == 0, NA_real_, mean_vals / count_vals)
  
  # --- MAX and MIN via data.table grouped join ---
  cat("  Computing neighbor max/min via data.table join...\n")
  
  # We need: for each (from_spatial, year), get all neighbor values and take max/min
  # Strategy: cross join edges with years, then join to get values, then aggregate
  
  # Create a lookup: (spatial_idx, year) -> value
  val_lookup <- cell_data[, .(spatial_idx, year, val = get(var_name))]
  setkey(val_lookup, spatial_idx, year)
  
  # Expand edges across all years efficiently
  # Instead of full cross join (1.37M * 28 = 38.4M rows), 
  # we do it year by year to control memory
  
  max_result <- rep(NA_real_, nrow(cell_data))
  min_result <- rep(NA_real_, nrow(cell_data))
  
  # Also build a lookup from (spatial_idx, year) -> row index in cell_data
  row_lookup <- cell_data[, .(spatial_idx, year, row_pos = .I)]
  setkey(row_lookup, spatial_idx, year)
  
  for (y in years) {
    # Get values for this year
    yr_vals <- val_lookup[year == y]
    setkey(yr_vals, spatial_idx)
    
    # Join edges to get neighbor values
    # edges_dt: from_spatial, to_spatial
    # We want: for each from_spatial, the val of each to_spatial in year y
    edge_vals <- edges_dt[yr_vals, on = .(to_spatial = spatial_idx), 
                          nomatch = 0L,
                          .(from_spatial, neighbor_val = i.val)]
    
    # Remove NA neighbor values
    edge_vals <- edge_vals[!is.na(neighbor_val)]
    
    if (nrow(edge_vals) == 0) next
    
    # Aggregate
    agg <- edge_vals[, .(nb_max = max(neighbor_val), 
                         nb_min = min(neighbor_val)), 
                     by = from_spatial]
    
    # Map back to cell_data rows
    agg_rows <- row_lookup[year == y]
    setkey(agg_rows, spatial_idx)
    setkey(agg, from_spatial)
    
    merged <- agg_rows[agg, on = .(spatial_idx = from_spatial), nomatch = 0L]
    
    max_result[merged$row_pos] <- merged$nb_max
    min_result[merged$row_pos] <- merged$nb_min
  }
  
  # --- Assign to cell_data ---
  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)
  
  cell_data[, (max_col)  := max_result]
  cell_data[, (min_col)  := min_result]
  cell_data[, (mean_col) := mean_result]
  
  cat("  Done with", var_name, "\n")
  return(cell_data)
}

# --------------------------------------------------------------------------
# STEP 5: Run for all variables
# --------------------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cell_data <- compute_all_neighbor_features(
    cell_data, edges_dt, W_spatial, id_order, years, var_name
  )
}

# Clean up helper column
cell_data[, spatial_idx := NULL]

cat("All neighbor features computed.\n")

# --------------------------------------------------------------------------
# STEP 6: Apply the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure column names match what the model expects.
# Example:
#   predictions <- predict(rf_model, newdata = cell_data)
# --------------------------------------------------------------------------
```

---

## 4. Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with string `paste` + named vector lookup | Sparse matrix `W_spatial` built once from `nb` object (~seconds) |
| **Mean computation** | Per-row R loop Ã— 5 vars = 32.3M function calls | Sparse matrix multiplication `W %*% X` â€” fully vectorized in C (~seconds per variable) |
| **Max/Min computation** | Per-row R loop with subsetting | `data.table` grouped join + aggregation, looped over 28 years (~seconds per variable) |
| **Total estimated time** | 86+ hours | **~2â€“10 minutes** |
| **Memory** | Multiple large intermediate lists | Sparse matrices + in-place `data.table` columns; fits in 16 GB |
| **Numerical results** | Original estimand | **Exactly preserved** â€” same max, min, mean over non-NA rook-neighbor values per cell-year |
| **RF model** | Pre-trained | **Untouched** â€” applied after feature computation |

### Why this preserves the original numerical estimand

- **Mean:** `(W %*% x) / (W %*% valid_mask)` computes exactly `sum(neighbor_vals) / count(non_NA_neighbor_vals)` â€” identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min:** The `data.table` join replicates the exact same subsetting logic: for each cell-year, gather all rook-neighbor values in the same year, drop NAs, and take `max`/`min`.
- **NA handling:** Cells with zero valid neighbors get `NA` for all three statistics, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`.
END_RESPONSE>>>
