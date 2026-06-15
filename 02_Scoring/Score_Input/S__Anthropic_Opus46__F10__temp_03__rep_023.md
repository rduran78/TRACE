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
run_id: Anthropic_F10_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates a per-row (6.46M) list of neighbor row indices.** It does this by pasting string keys (`id_year`) and performing named-vector lookups for every single row. String concatenation and named-vector lookup in R are O(n) per lookup in the worst case, and doing this ~6.46M Ã— avg_neighbors times is catastrophically slow. The resulting list object itself (~6.46M entries) also consumes enormous memory.

2. **`compute_neighbor_stats` iterates over 6.46M list entries in an `lapply` loop**, extracting subsets of a numeric vector and computing `max/min/mean`. While each individual operation is trivial, the R-level loop overhead across 6.46M iterations, repeated for 5 variables (= 32.3M iterations total), dominates runtime.

3. **The topology is year-invariant but the lookup is built as if it's year-specific.** Rook neighbors are a spatial relationship â€” they don't change across years. Yet the current code embeds year into every key, inflating the lookup from ~344K spatial entries to ~6.46M spatiotemporal entries.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~30â€“40% of 86 hours (string ops, named vector lookups)
- `compute_neighbor_stats` Ã— 5 vars: ~50â€“60% (R-level loop over 6.46M rows Ã— 5)
- Memory pressure / GC: ~10%

---

## Optimization Strategy

### Core Insight: Sparse Matrix Aggregation

The neighbor aggregation is a **sparse matrixâ€“vector product** (and analogous operations for max/min). We can:

1. **Build the adjacency structure once** as a sparse matrix (344,208 Ã— 344,208) from the `nb` object â€” this is the graph topology, year-invariant.
2. **For each variable and each year**, extract the column vector of values for all cells in that year, then use the sparse matrix to compute neighbor sums (for mean), neighbor counts, neighbor max, and neighbor min â€” all vectorized.
3. **Sparse matrix Ã— dense vector** for sum/mean is a single `%*%` call via the `Matrix` package â€” highly optimized C code, no R-level loops.
4. **For max and min**, we use a grouped operation via the sparse matrix's row/column indices, leveraging `data.table` grouping or a custom C-level aggregation.

### Complexity Reduction

| Step | Original | Optimized |
|---|---|---|
| Build topology | O(6.46M Ã— k) string ops | O(344K Ã— k) integer sparse matrix, once |
| Mean per var-year | O(6.46M) R loop | O(nnz) sparse mat-vec multiply, 28 batches |
| Max/Min per var-year | O(6.46M) R loop | O(nnz) grouped agg via data.table, 28 batches |
| Total iterations | ~32.3M R-level | ~0 R-level loops (all vectorized/C) |

**Expected runtime: ~2â€“5 minutes** (vs. 86+ hours).

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build sparse adjacency matrix ONCE from nb object ----
# rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
# id_order: vector of cell IDs in the order matching the nb object

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj[[i]] contains the indices of neighbors of node i

  # Build a sparse matrix A where A[i,j] = 1 means j is a neighbor of i
  # (i.e., row i aggregates over its neighbors in columns)
  
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # spdep nb objects use 0L to indicate no neighbors; remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Precompute neighbor counts per cell (constant across years)
neighbor_counts <- as.numeric(A %*% rep(1, n_cells))  # = rowSums(A)

cat("Adjacency matrix:", n_cells, "x", n_cells,
    "with", nnzero(A), "nonzeros\n")

# ---- Step 2: Convert cell_data to data.table for fast indexing ----
dt <- as.data.table(cell_data)

# Create a mapping from cell ID to spatial index (position in id_order / nb object)
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Ensure sorted by year and spatial_idx for consistent vectorized access
setkey(dt, year, spatial_idx)

# Verify all cells present in every year (panel is balanced)
years <- sort(unique(dt$year))
n_years <- length(years)
stopifnot(nrow(dt) == n_cells * n_years)

# ---- Step 3: Extract sparse matrix structure for max/min ----
# We need row indices, column indices from A for grouped max/min
A_coo <- summary(A)  # returns data.frame with i, j, x columns
adj_i <- A_coo$i     # row (target node)
adj_j <- A_coo$j     # col (source neighbor)
n_edges <- length(adj_i)

# Pre-create a data.table template for grouped aggregation
edge_dt <- data.table(target = adj_i, source = adj_j)

# ---- Step 4: Neighbor aggregation function (vectorized per year) ----

compute_neighbor_features_fast <- function(dt, A, neighbor_counts,
                                           edge_dt, adj_j,
                                           var_name, years, n_cells) {
  max_col <- paste0("max_", var_name)
  min_col <- paste0("min_", var_name)
  mean_col <- paste0("mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  for (yr in years) {
    # Extract the value vector for this year, ordered by spatial_idx
    # Because we keyed by (year, spatial_idx), rows for this year are contiguous
    # and ordered by spatial_idx
    year_rows <- which(dt$year == yr)
    vals <- dt[[var_name]][year_rows]  # length = n_cells, ordered by spatial_idx
    
    # --- MEAN via sparse matrix-vector product ---
    # Replace NA with 0 for sum, and track non-NA for correct mean
    not_na <- as.numeric(!is.na(vals))
    vals_zero <- ifelse(is.na(vals), 0, vals)
    
    neighbor_sum     <- as.numeric(A %*% vals_zero)
    neighbor_non_na  <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_non_na > 0,
                            neighbor_sum / neighbor_non_na,
                            NA_real_)
    
    # --- MAX and MIN via grouped aggregation on edges ---
    # Get neighbor values for all edges
    neighbor_vals_edge <- vals[adj_j]  # length = n_edges
    
    # Grouped max and min using data.table
    agg <- edge_dt[, .(
      nmax = if (all(is.na(neighbor_vals_edge[.I])))
                NA_real_
             else
                max(neighbor_vals_edge[.I], na.rm = TRUE),
      nmin = if (all(is.na(neighbor_vals_edge[.I])))
                NA_real_
             else
                min(neighbor_vals_edge[.I], na.rm = TRUE)
    ), by = target]
    
    # Initialize with NA (for cells with 0 neighbors)
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[agg$target] <- agg$nmax
    neighbor_min[agg$target] <- agg$nmin
    
    # Also set to NA where all neighbors had NA values
    no_valid <- neighbor_non_na == 0
    neighbor_max[no_valid] <- NA_real_
    neighbor_min[no_valid] <- NA_real_
    neighbor_mean[no_valid] <- NA_real_
    
    # Write back
    set(dt, i = year_rows, j = max_col,  value = neighbor_max)
    set(dt, i = year_rows, j = min_col,  value = neighbor_min)
    set(dt, i = year_rows, j = mean_col, value = neighbor_mean)
  }
  
  dt
}

# ---- Step 5: Run for all neighbor source variables ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# However, the grouped data.table aggregation above uses .I which references
# the edge_dt rows. We need a slightly different approach to avoid the .I issue.
# Let's use a cleaner vectorized grouped aggregation:

compute_neighbor_features_v2 <- function(dt, A, neighbor_counts,
                                         adj_i, adj_j,
                                         var_name, years, n_cells) {
  max_col  <- paste0("max_", var_name)
  min_col  <- paste0("min_", var_name)
  mean_col <- paste0("mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  n_edges <- length(adj_i)
  
  for (yr in years) {
    year_rows <- which(dt$year == yr)
    vals <- dt[[var_name]][year_rows]  # length n_cells, by spatial_idx
    
    # ---- MEAN via sparse mat-vec ----
    not_na     <- as.numeric(!is.na(vals))
    vals_zero  <- ifelse(is.na(vals), 0, vals)
    
    neighbor_sum    <- as.numeric(A %*% vals_zero)
    neighbor_nvalid <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_nvalid > 0,
                            neighbor_sum / neighbor_nvalid,
                            NA_real_)
    
    # ---- MAX / MIN via data.table grouped agg on edge list ----
    nv <- vals[adj_j]  # neighbor values for each edge
    
    agg_dt <- data.table(target = adj_i, nv = nv)
    agg <- agg_dt[!is.na(nv),
                  .(nmax = max(nv), nmin = min(nv)),
                  by = target]
    
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[agg$target] <- agg$nmax
    neighbor_min[agg$target] <- agg$nmin
    
    # Write results
    set(dt, i = year_rows, j = max_col,  value = neighbor_max)
    set(dt, i = year_rows, j = min_col,  value = neighbor_min)
    set(dt, i = year_rows, j = mean_col, value = neighbor_mean)
  }
  
  dt
}

# ---- Execute ----
cat("Starting optimized neighbor feature computation...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")
  dt <- compute_neighbor_features_v2(
    dt, A, neighbor_counts,
    adj_i, adj_j,
    var_name, years, n_cells
  )
}

elapsed <- (proc.time() - t0)[3]
cat(sprintf("Neighbor features computed in %.1f seconds\n", elapsed))

# ---- Step 6: Convert back and predict with trained Random Forest ----
cell_data <- as.data.frame(dt)

# The pre-trained RF model object (do NOT retrain)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **mean** | `mean(vals[neighbors])` ignoring NA | `(A %*% vals_zero) / (A %*% not_na)` | Identical: sum of non-NA values Ã· count of non-NA values |
| **max** | `max(vals[neighbors], na.rm=TRUE)` | `data.table [!is.na(nv), max(nv), by=target]` | Identical: max over same non-NA neighbor set |
| **min** | `min(vals[neighbors], na.rm=TRUE)` | `data.table [!is.na(nv), min(nv), by=target]` | Identical: min over same non-NA neighbor set |
| **all-NA / no neighbors** | Returns `c(NA, NA, NA)` | All three set to `NA_real_` | Identical |

The sparse matrix `A` encodes exactly the same directed edges as `rook_neighbors_unique`. The per-year slicing ensures temporal alignment. No approximation is introduced.

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Topology build | ~6.46M string keys, named lookups | One sparse matrix build (~1.37M entries) |
| Mean computation | R loop Ã— 6.46M Ã— 5 | 28 Ã— 5 = 140 sparse mat-vec products (C-level) |
| Max/Min computation | R loop Ã— 6.46M Ã— 5 | 140 `data.table` grouped aggregations (C-level) |
| Memory for topology | ~6.46M-entry list of integer vectors | One 344KÃ—344K sparse matrix (~22 MB) |
| **Estimated runtime** | **86+ hours** | **2â€“5 minutes** |
| RF model | Unchanged | Unchanged |
END_RESPONSE>>>
