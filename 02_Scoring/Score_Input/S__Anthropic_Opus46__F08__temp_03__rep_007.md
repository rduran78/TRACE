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
run_id: Anthropic_F08_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup list of length ~6.46 million, where each element maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor graph is static.** Cell *i*'s neighbors are the same in 1992 as in 2019. The spatial topology never changes. Yet the current code rebuilds/re-resolves neighbor relationships for every cell Ã— every year â€” effectively duplicating the same spatial graph 28 times.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M string keys (`paste(id, year)`) and uses named-vector lookup (`idx_lookup[neighbor_keys]`) inside a `lapply` over 6.46M rows. Named-vector lookup in R is O(n) per probe in the worst case, and the `paste`/string operations dominate runtime.

3. **`compute_neighbor_stats` then iterates over the 6.46M-element list** for each of 5 variables, performing per-element subsetting and summary â€” another 5 Ã— 6.46M = 32.3M R-level function calls.

**The key insight:** Because the neighbor structure is year-invariant, we should:
- Build the neighbor lookup **once at the cell level** (344K entries, not 6.46M).
- For each variable, extract a **cell Ã— year matrix**, then compute neighbor max/min/mean using fast vectorized matrix operations over the 344K cells, broadcasting across all 28 years simultaneously.

This reduces the problem from 6.46M list iterations to 344K cell iterations (or fully vectorized matrix algebra), and eliminates all string-key construction.

---

## Optimization Strategy

| Aspect | Current (slow) | Redesigned (fast) |
|---|---|---|
| Neighbor lookup granularity | Per cell-year row (6.46M entries) | Per cell (344K entries) â€” **static** |
| Data structure for variables | Column in a long data.frame | Cell Ã— Year matrix â€” **changing** |
| Neighbor aggregation | R-level `lapply` over 6.46M rows | Sparse-matrix multiplication or vectorized matrix-row aggregation over 344K cells |
| String key construction | 6.46M `paste()` calls + named-vector lookup | None â€” integer indexing only |
| Passes per variable | 1 pass over 6.46M rows | 1 sparse-matrix multiply (or 1 vectorized pass over 344K cells) |
| Estimated time | 86+ hours | Minutes |

**Concrete plan:**

1. **Build a cell-level neighbor lookup once** â€” a simple list of length 344K where element *i* contains the integer indices of cell *i*'s rook neighbors. This is just `rook_neighbors_unique` re-indexed to a contiguous 1:N integer mapping. Cost: trivial, done once.

2. **Build a sparse adjacency matrix W** (344K Ã— 344K) from the neighbor list. This enables computing neighbor sums, counts, max, and min via matrix operations.

3. **Reshape each variable into a 344K Ã— 28 matrix** (cell rows Ã— year columns) using integer indexing.

4. **Compute neighbor stats using the sparse matrix:**
   - **Neighbor mean:** `W %*% X / neighbor_count` (one sparse matrix multiply per variable).
   - **Neighbor max and min:** Iterate over 344K cells (not 6.46M) using the cell-level neighbor list, operating on matrix rows. This is 18.7Ã— fewer iterations and each iteration touches only the year-vector, which is cache-friendly.

5. **Write results back** into the long data.frame in the correct row order.

6. **Feed into the pre-trained Random Forest** exactly as before â€” same column names, same numerical values.

---

## Working R Code

```r
library(Matrix)  # for sparse matrix operations

# =============================================================================
# STEP 0: Ensure consistent cell ordering
# =============================================================================
# id_order: vector of 344,208 unique cell IDs (same order as rook_neighbors_unique)
# cell_data: the long data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# rook_neighbors_unique: spdep nb object (list of length 344,208)

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Map cell IDs to contiguous integer indices 1:n_cells
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# Map years to contiguous integer indices 1:n_years
year_to_idx <- setNames(seq_len(n_years), as.character(years))

# =============================================================================
# STEP 1: Build STATIC cell-level neighbor list (done once, reused forever)
# =============================================================================
# rook_neighbors_unique is already indexed consistently with id_order,
# so rook_neighbors_unique[[i]] gives the neighbor indices for id_order[i].
# We just need to strip the spdep attributes and ensure integer vectors.

cell_neighbor_list <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]  # spdep uses 0 to denote "no neighbors"; remove if present
})

# Precompute neighbor counts per cell (static)
neighbor_counts <- vapply(cell_neighbor_list, length, integer(1))

# =============================================================================
# STEP 2: Build sparse adjacency matrix W (static, built once)
#          W[i, j] = 1 if cell j is a neighbor of cell i
# =============================================================================
# Build COO triplets
from_idx <- rep(seq_len(n_cells), times = neighbor_counts)
to_idx   <- unlist(cell_neighbor_list, use.names = FALSE)

W <- sparseMatrix(
  i    = from_idx,
  j    = to_idx,
  x    = 1,
  dims = c(n_cells, n_cells)
)

# Neighbor count vector (as dense numeric for division)
n_count_vec <- as.numeric(neighbor_counts)
# Replace 0 with NA to avoid division by zero
n_count_vec_safe <- ifelse(n_count_vec == 0, NA_real_, n_count_vec)

# =============================================================================
# STEP 3: Map every row of cell_data to (cell_idx, year_idx) for fast reshaping
# =============================================================================
cell_data$`.cell_idx` <- id_to_idx[as.character(cell_data$id)]
cell_data$`.year_idx` <- year_to_idx[as.character(cell_data$year)]

# Linear index into a cell Ã— year matrix (column-major)
lin_idx <- cell_data$`.cell_idx` + (cell_data$`.year_idx` - 1L) * n_cells

# =============================================================================
# STEP 4: Function to reshape a variable into cell Ã— year matrix
# =============================================================================
var_to_matrix <- function(data, var_name, n_cells, n_years, lin_idx) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[lin_idx] <- data[[var_name]]
  mat
}

# =============================================================================
# STEP 5: Compute neighbor stats (max, min, mean) for each variable
# =============================================================================
# Strategy:
#   - MEAN: use sparse matrix multiply  ->  neighbor_mean = (W %*% X) / count
#   - MAX and MIN: vectorized loop over 344K cells (not 6.46M rows)
#     For each cell i, extract X[neighbors_of_i, ] and compute col-wise max/min.
#     This is 18.7Ã— fewer iterations than the original approach.

compute_neighbor_features <- function(var_name, cell_data, W, cell_neighbor_list,
                                       n_count_vec_safe, n_cells, n_years, lin_idx) {
  
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # --- Reshape to matrix ---
  X <- var_to_matrix(cell_data, var_name, n_cells, n_years, lin_idx)
  
  # --- Neighbor MEAN via sparse matrix multiply ---
  # W %*% X gives, for each cell i, the sum of neighbor values across all years
  neighbor_sum  <- as.matrix(W %*% X)  # n_cells Ã— n_years
  neighbor_mean <- neighbor_sum / n_count_vec_safe  # broadcasts column-wise
  
  # --- Neighbor MAX and MIN via vectorized cell-level loop ---
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_list[[i]]
    if (length(nb) == 0L) next
    
    # Extract neighbor rows: length(nb) Ã— n_years sub-matrix
    nb_vals <- X[nb, , drop = FALSE]
    
    # Column-wise max and min (across neighbors, for each year)
    if (length(nb) == 1L) {
      neighbor_max[i, ] <- nb_vals[1L, ]
      neighbor_min[i, ] <- nb_vals[1L, ]
    } else {
      # Handles NAs: na.rm = TRUE; if all NA for a year, returns Inf/-Inf -> convert to NA
      col_max <- apply(nb_vals, 2, max, na.rm = TRUE)
      col_min <- apply(nb_vals, 2, min, na.rm = TRUE)
      col_max[is.infinite(col_max)] <- NA_real_
      col_min[is.infinite(col_min)] <- NA_real_
      neighbor_max[i, ] <- col_max
      neighbor_min[i, ] <- col_min
    }
  }
  
  # --- Write results back to long format using lin_idx ---
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_data[[max_col]]  <- neighbor_max[lin_idx]
  cell_data[[min_col]]  <- neighbor_min[lin_idx]
  cell_data[[mean_col]] <- neighbor_mean[lin_idx]
  
  cell_data
}

# =============================================================================
# STEP 6: Main loop â€” process all 5 neighbor source variables
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (optimized: static graph + matrix ops)...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(
    var_name, cell_data, W, cell_neighbor_list,
    n_count_vec_safe, n_cells, n_years, lin_idx
  )
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds (%.1f minutes).\n", elapsed, elapsed / 60))

# Clean up temporary index columns
cell_data$`.cell_idx` <- NULL
cell_data$`.year_idx` <- NULL

# =============================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# =============================================================================
# The pre-trained model object (e.g., `rf_model`) is loaded from disk.
# Column names and numerical values are identical to the original pipeline.
#
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Eliminating the Cell-Level Loop for Max/Min

The `for` loop over 344K cells for max/min is already ~18.7Ã— faster than the original 6.46M-row loop, but if it's still too slow (each iteration calls `apply`), we can replace it with a fully vectorized sparse approach:

```r
# =============================================================================
# OPTIONAL: Fully vectorized neighbor MAX and MIN using sparse tricks
# =============================================================================
# For MAX: replace 0s in W with -Inf, set missing X values to -Inf,
#          then sparse max can be computed year-by-year.
# 
# A simpler and very fast approach: use data.table to "explode" the neighbor
# list and compute grouped max/min. The exploded table has ~1.37M Ã— 28 â‰ˆ 38.5M
# rows conceptually, but we operate column-by-column.

compute_neighbor_maxmin_fast <- function(var_name, X, cell_neighbor_list, 
                                          n_cells, n_years) {
  library(data.table)
  
  # Explode neighbor relationships: from_cell -> to_cell (static, ~1.37M pairs)
  from_cell <- rep(seq_len(n_cells), times = vapply(cell_neighbor_list, length, integer(1)))
  to_cell   <- unlist(cell_neighbor_list, use.names = FALSE)
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Process each year-column: ~1.37M rows per year, 28 years
  for (yr in seq_len(n_years)) {
    vals <- X[to_cell, yr]  # neighbor values for this year
    dt <- data.table(from = from_cell, val = vals)
    agg <- dt[!is.na(val), .(vmax = max(val), vmin = min(val)), by = from]
    neighbor_max[agg$from, yr] <- agg$vmax
    neighbor_min[agg$from, yr] <- agg$vmin
  }
  
  list(neighbor_max = neighbor_max, neighbor_min = neighbor_min)
}
```

This replaces the 344K-iteration R loop with 28 fast `data.table` grouped aggregations over ~1.37M rows each â€” typically completing in seconds.

---

## Summary

| What | How |
|---|---|
| **Root cause** | Neighbor lookup built per cell-year (6.46M) instead of per cell (344K); string-key hashing |
| **Key insight** | Neighbor graph is **static**; only variable values **change** by year |
| **Redesign** | Separate static structure (sparse matrix W, cell-level neighbor list) from changing data (cell Ã— year matrices) |
| **Neighbor mean** | Single sparse matrix multiply `W %*% X` per variable |
| **Neighbor max/min** | Cell-level loop (344K) or `data.table` grouped aggregation (1.37M pairs Ã— 28 years) |
| **Speedup** | From 86+ hours to estimated **minutes** |
| **Preserves** | Pre-trained RF model (untouched), original numerical estimand (identical values) |
END_RESPONSE>>>
