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
run_id: Anthropic_F10_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector â€” this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` removal dominates.

3. **The topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt monolithically across all cell-years, entangling spatial structure with temporal indexing. This prevents vectorized, year-parallel computation.

**Core insight:** The rook-neighbor adjacency is a **sparse spatial graph with 344,208 nodes and ~1.37M directed edges**. The neighbor aggregation (max, min, mean per node) is a **sparse matrixâ€“vector operation** that can be executed independently per year, reusing the same adjacency structure. This is equivalent to a single-hop Graph Neural Network neighborhood aggregation pass.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** (`Matrix::sparseMatrix`, 344K Ã— 344K, ~1.37M nonzeros). This is the graph topology.

2. **For each year, extract the column vector of node attributes**, then compute:
   - **Mean:** `A %*% x / degree` (sparse matrixâ€“vector multiply, ~1.37M flops).
   - **Max/Min:** Use grouped operations via the sparse matrix structure (CSC column pointers), implemented in C++ via `Rcpp` or via `data.table` grouped aggregation on the edge list.

3. **Avoid any per-row R-level iteration.** The 6.46M-row loop is replaced by 28 sparse mat-vec multiplies (for mean) and 28 grouped aggregations (for max/min) per variable.

4. **Memory:** The sparse matrix is ~33 MB. Per-year vectors are ~2.6 MB. Total memory stays well under 16 GB.

5. **Expected speedup:** From 86+ hours to **minutes** (sparse mat-vec on 344K nodes with 1.37M edges is sub-second; 28 years Ã— 5 variables Ã— 3 stats = 420 operations).

## Working R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation via Sparse Graph Operations
# Preserves numerical equivalence with original max/min/mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is a data.table keyed by (id, year) ----------
cell_dt <- as.data.table(cell_data)

# Canonical ordering of spatial cell IDs (must match rook_neighbors_unique index)
# id_order is the vector of cell IDs in the order used by the nb object.
stopifnot(length(id_order) == 344208L)
n_cells <- length(id_order)

# Map cell IDs to integer indices 1..n_cells
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# Map cell_dt rows to their spatial index
cell_dt[, sp_idx := id_to_idx[as.character(id)]]

# ---- Step 1: Build sparse adjacency matrix ONCE ----------------------------
# rook_neighbors_unique is an nb object: a list of length n_cells,
# each element is an integer vector of neighbor indices (into id_order).

build_sparse_adjacency <- function(nb_obj, n) {
  # Build COO (coordinate) representation of the directed edge list

  # Edge from j -> i means "j is a neighbor of i" so that A %*% x
  # gives the sum of neighbor values for each node.
  from_list <- lapply(seq_len(n), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1L] == 0L) return(integer(0))
    nbrs
  })
  
  # Row indices (the node whose neighbors we aggregate INTO)
  row_idx <- rep(seq_len(n), lengths(from_list))
  # Column indices (the neighbor node)
  col_idx <- unlist(from_list, use.names = FALSE)
  
  sparseMatrix(
    i = row_idx,
    j = col_idx,
    x = rep(1, length(row_idx)),
    dims = c(n, n),
    repr = "C"   # CSC format for efficient column operations
  )
}

A <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# Degree vector (number of neighbors per node) â€” used for mean
degree <- as.numeric(A %*% rep(1, n_cells))

# ---- Step 2: Build edge list for max/min (grouped aggregation) --------------
# Extract COO from sparse matrix
A_T <- as(A, "TsparseMatrix")  # triplet form
edge_dt <- data.table(
  target = A_T@i + 1L,   # 1-based row index (node receiving aggregation)
  source = A_T@j + 1L    # 1-based col index (neighbor node)
)
setkey(edge_dt, target)

# ---- Step 3: Aggregate per year, per variable --------------------------------
# We process each year independently, reusing the same graph topology.

years <- sort(unique(cell_dt$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_dt is keyed for fast subsetting
setkey(cell_dt, year, sp_idx)

# Pre-allocate result columns in cell_dt
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("n_max_", var_name)
  min_col  <- paste0("n_min_", var_name)
  mean_col <- paste0("n_mean_", var_name)
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
}

cat("Processing", length(years), "years x", length(neighbor_source_vars), "variables...\n")

for (yr in years) {
  # Extract the rows for this year, ordered by sp_idx
  yr_rows <- cell_dt[.(yr)]  # keyed lookup on year
  
  # Build a full-length vector for each variable (indexed by sp_idx)
  # Some cells may be missing in a given year; those stay NA.
  # We need a dense vector of length n_cells.
  
  # Get the sp_idx values present this year
  present_idx <- yr_rows$sp_idx
  
  # Row indices in cell_dt for this year (for writing results back)
  # We need the actual row positions in cell_dt
  yr_row_positions <- which(cell_dt$year == yr)
  # Create a map from sp_idx -> position in yr_row_positions
  sp_to_yrpos <- integer(n_cells)
  sp_to_yrpos[cell_dt$sp_idx[yr_row_positions]] <- yr_row_positions
  
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)
    
    # Dense vector of attribute values, length n_cells, NA for missing
    x <- rep(NA_real_, n_cells)
    x[present_idx] <- yr_rows[[var_name]]
    
    # ---- MEAN via sparse matrix-vector multiply ----------------------------
    # A %*% x gives sum of neighbor values (NAs propagate, so handle them)
    # To handle NAs correctly (matching original: mean of non-NA neighbors):
    #   sum_vals = A %*% x_nona   (where x_nona has 0 for NA)
    #   count    = A %*% notna    (count of non-NA neighbors)
    #   mean     = sum_vals / count
    
    not_na <- as.numeric(!is.na(x))
    x_nona <- x
    x_nona[is.na(x_nona)] <- 0
    
    sum_vals <- as.numeric(A %*% x_nona)
    count    <- as.numeric(A %*% not_na)
    
    n_mean <- ifelse(count > 0, sum_vals / count, NA_real_)
    # Nodes with degree == 0 (no neighbors at all) -> NA
    n_mean[degree == 0] <- NA_real_
    
    # ---- MAX and MIN via grouped edge-list aggregation ---------------------
    # Look up source values for all edges
    edge_vals <- x[edge_dt$source]
    
    # Grouped aggregation (data.table is highly optimized for this)
    agg_dt <- data.table(
      target = edge_dt$target,
      val    = edge_vals
    )
    # Remove edges where source value is NA
    agg_dt <- agg_dt[!is.na(val)]
    
    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(nmax = max(val), nmin = min(val)), by = target]
      
      n_max <- rep(NA_real_, n_cells)
      n_min <- rep(NA_real_, n_cells)
      n_max[stats$target] <- stats$nmax
      n_min[stats$target] <- stats$nmin
    } else {
      n_max <- rep(NA_real_, n_cells)
      n_min <- rep(NA_real_, n_cells)
    }
    
    # ---- Write results back to cell_dt at correct row positions ------------
    # Only write for cells present this year
    write_idx <- sp_to_yrpos[present_idx]
    set(cell_dt, i = write_idx, j = max_col,  value = n_max[present_idx])
    set(cell_dt, i = write_idx, j = min_col,  value = n_min[present_idx])
    set(cell_dt, i = write_idx, j = mean_col, value = n_mean[present_idx])
  }
  
  cat("  Year", yr, "done.\n")
}

# ---- Step 4: Convert back to data.frame if needed for predict() -------------
cell_data <- as.data.frame(cell_dt[, sp_idx := NULL])

# ---- Step 5: Apply the pre-trained Random Forest (unchanged) ----------------
# predictions <- predict(rf_model, newdata = cell_data)

cat("Done. Neighbor features computed.\n")
```

## Complexity Comparison

| Operation | Original | Optimized |
|---|---|---|
| Build lookup | O(NÂ·kÂ·M) string ops, M=6.46M | O(E) sparse matrix build, E=1.37M, **once** |
| Mean per var-year | O(M) R-level list iteration | O(E) sparse mat-vec multiply (BLAS) |
| Max/Min per var-year | O(M) R-level list iteration | O(E) `data.table` grouped agg (radix) |
| Total operations | ~6.46M Ã— 5 Ã— R-loop overhead | 28 Ã— 5 Ã— 3 sparse ops (~420 sub-second ops) |
| **Estimated wall time** | **86+ hours** | **5â€“15 minutes** |

## Why Numerical Equivalence Is Preserved

- **Mean:** `sum(non-NA neighbor values) / count(non-NA neighbors)` â€” identical to the original `mean(neighbor_vals)` after `NA` removal, computed via `(A %*% x_nona) / (A %*% not_na)`.
- **Max/Min:** Grouped aggregation on the exact same edge list with `NA` removal â€” identical to the original `max(neighbor_vals)` and `min(neighbor_vals)` after `NA` removal.
- **The Random Forest model is never retrained** â€” only `predict()` is called on the augmented dataset with the same feature columns.
END_RESPONSE>>>
