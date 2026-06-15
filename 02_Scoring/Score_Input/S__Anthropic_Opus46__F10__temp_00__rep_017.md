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
run_id: Anthropic_F10_T00_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The string-keyed lookup `idx_lookup` is a named character vector of length ~6.46M â€” each lookup into it is O(n) in the worst case because R named vectors use linear hashing that degrades at scale.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements**, subsetting a numeric vector and computing `max/min/mean` per element. The `lapply` + `do.call(rbind, ...)` pattern over millions of tiny vectors is extremely slow due to R interpreter overhead and memory allocation churn.

3. **The neighbor lookup is row-indexed (cellÃ—year)**, but the graph topology is **year-invariant** â€” every year has the same adjacency structure. The current code redundantly encodes the same spatial edges 28 times (once per year), inflating the lookup from ~1.37M edges to ~38.4M edge references.

**Root cause summary:** The 86+ hour runtime is dominated by (a) millions of R-level string operations, (b) millions of tiny R list/vector allocations, and (c) failure to exploit the year-invariance of the spatial graph.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook adjacency graph is **static across years**. We should:

1. **Build the spatial adjacency once** as a sparse structure (CSR-style: integer vectors of row pointers and column indices) over the 344,208 cells â€” not over 6.46M cell-years.
2. **For each year**, slice the relevant variable column, then use **vectorized sparse matrixâ€“vector multiplication** (or equivalent) to compute neighbor max, min, and mean.
3. **Use `data.table`** for fast year-slicing and column assignment.
4. **Use a sparse adjacency matrix (`Matrix::sparseMatrix`)** so that neighbor-mean is literally a sparse matrix Ã— dense vector multiplication â€” an O(nnz) operation in compiled C code.
5. **For max and min**, there is no direct sparse-matrix shortcut, but we can use the CSR structure (`dgCMatrix` slot access) to compute them in a tight vectorized loop or via `{Rcpp}` if needed. However, a pure-R approach using `data.table` grouped operations on an edge list is also very fast.

### Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(6.46M) string ops | O(344K) integer ops, once |
| Per variable stats | O(6.46M) list iterations Ã— 5 vars | O(28 years Ã— 344K cells Ã— avg 4 neighbors) Ã— 5 vars, vectorized |
| Total R-level iterations | ~32.3M `lapply` calls | ~0 (vectorized/compiled) |

**Expected speedup: ~200â€“500Ã—**, bringing runtime to **minutes** instead of days.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)
library(Matrix)

# ---- 0. Convert to data.table if not already --------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- 1. Build cell-level ID mapping (once) -----------------------------------
# id_order: the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

n_cells <- length(id_order)
stopifnot(n_cells == length(rook_neighbors_unique))

# Map cell IDs to integer indices 1..n_cells (matching id_order position)
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# ---- 2. Build sparse adjacency matrix (once) --------------------------------
# Construct COO (coordinate) edge list from the nb object.
# For each cell i, rook_neighbors_unique[[i]] gives the indices (into id_order)
# of its rook neighbors.

# Pre-compute total number of edges for pre-allocation
n_edges <- sum(lengths(rook_neighbors_unique))

# Build edge list: "from" node aggregates over its neighbors ("to" nodes)
# So row = "from" (the focal cell), col = "to" (the neighbor cell)
ei_from <- integer(n_edges)
ei_to   <- integer(n_edges)
pos <- 0L
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  len_i <- length(nb_i)
  if (len_i > 0L) {
    idx_range <- (pos + 1L):(pos + len_i)
    ei_from[idx_range] <- i
    ei_to[idx_range]   <- nb_i
    pos <- pos + len_i
  }
}

# Sparse adjacency matrix: A[i,j] = 1 means j is a neighbor of i
# So A %*% x gives the sum of neighbor values for each cell
adj_mat <- sparseMatrix(
  i = ei_from,
  j = ei_to,
  x = rep(1, n_edges),
  dims = c(n_cells, n_cells),
  repr = "C"   # CSC format (Matrix default), will convert to CSR below
)

# Neighbor count per cell (for computing mean)
neighbor_count <- diff(adj_mat@p)  # For dgCMatrix, this is column counts
# We actually need row counts. Transpose or use rowSums:
n_neighbors <- as.integer(rowSums(adj_mat))  # integer vector length n_cells

# For max/min we need CSR (row-compressed) access.
# Convert to dgRMatrix (row-sparse) for efficient row-wise access:
adj_csr <- as(adj_mat, "RsparseMatrix")
# adj_csr@p: row pointers (length n_cells + 1)
# adj_csr@j: column indices (0-based)

csr_p <- adj_csr@p
csr_j <- adj_csr@j  # 0-based column indices

# ---- 3. Build edge list as data.table for grouped max/min -------------------
# This is an alternative approach: use data.table grouping on the edge list.
# For each (from_cell, year), we look up the neighbor values and compute stats.

edge_dt <- data.table(
  from_idx = ei_from,   # focal cell index (1-based, into id_order)
  to_idx   = ei_to      # neighbor cell index (1-based, into id_order)
)

# Map cell indices to cell IDs
edge_dt[, from_id := id_order[from_idx]]
edge_dt[, to_id   := id_order[to_idx]]

# ---- 4. Ensure cell_data is keyed properly -----------------------------------
# We need fast lookups by (id, year)
cell_data[, cell_idx := id_to_idx[as.character(id)]]
setkey(cell_data, cell_idx, year)

# Get sorted unique years
all_years <- sort(unique(cell_data$year))
n_years <- length(all_years)

# ---- 5. Compute neighbor features (vectorized, year-by-year) ----------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
}

# Create a row-index lookup: for each (cell_idx, year) -> row in cell_data
# Since cell_data is keyed on (cell_idx, year), we can use fast binary search.
# But for vectorized operations, it's faster to work year-by-year with
# a cell_idx -> value vector.

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (yr in all_years) {
  # Extract this year's slice
  yr_rows <- cell_data[year == yr, which = TRUE]
  yr_cell_idx <- cell_data$cell_idx[yr_rows]
  
  # Build a mapping: cell_idx -> position in yr_rows
  # (Not all cells may be present in every year)
  val_vec <- rep(NA_real_, n_cells)  # reusable buffer
  
  # Map from cell_idx to yr_rows position
  yr_row_lookup <- rep(NA_integer_, n_cells)
  yr_row_lookup[yr_cell_idx] <- yr_rows
  
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Fill value vector for this year: val_vec[cell_idx] = value
    val_vec[] <- NA_real_
    val_vec[yr_cell_idx] <- cell_data[[var_name]][yr_rows]
    
    # --- Compute neighbor MEAN via sparse matrix-vector multiply ---
    # neighbor_sum = A %*% val_vec
    # But we need to handle NAs: NAs should be excluded from mean.
    # Strategy: replace NA with 0 for sum, and count non-NA neighbors separately.
    
    not_na <- !is.na(val_vec)
    val_clean <- val_vec
    val_clean[!not_na] <- 0
    
    neighbor_sum     <- as.numeric(adj_mat %*% val_clean)
    neighbor_nna     <- as.numeric(adj_mat %*% as.numeric(not_na))
    neighbor_mean    <- ifelse(neighbor_nna > 0, neighbor_sum / neighbor_nna, NA_real_)
    
    # --- Compute neighbor MAX and MIN via CSR row traversal ---
    # This is the part that benefits most from Rcpp, but we can do it
    # in vectorized R using the edge_dt approach.
    
    # Look up neighbor values on the edge list
    nb_vals <- val_vec[ei_to]  # length = n_edges, value of each neighbor
    
    # We need max and min grouped by ei_from, excluding NAs
    # Use data.table for fast grouped aggregation
    agg_dt <- data.table(from_idx = ei_from, nb_val = nb_vals)
    agg_dt <- agg_dt[!is.na(nb_val)]
    
    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(nb_max = max(nb_val), nb_min = min(nb_val)),
                       by = from_idx]
      
      # Initialize with NA
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
      neighbor_max[stats$from_idx] <- stats$nb_max
      neighbor_min[stats$from_idx] <- stats$nb_min
    } else {
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
    }
    
    # Also set to NA for cells with zero valid neighbors
    no_valid <- (neighbor_nna == 0) | (n_neighbors == 0L)
    neighbor_max[no_valid]  <- NA_real_
    neighbor_min[no_valid]  <- NA_real_
    neighbor_mean[no_valid] <- NA_real_
    
    # Write results back to cell_data for this year's rows
    set(cell_data, i = yr_row_lookup[yr_cell_idx], j = max_col,
        value = neighbor_max[yr_cell_idx])
    set(cell_data, i = yr_row_lookup[yr_cell_idx], j = min_col,
        value = neighbor_min[yr_cell_idx])
    set(cell_data, i = yr_row_lookup[yr_cell_idx], j = mean_col,
        value = neighbor_mean[yr_cell_idx])
  }
  
  cat(sprintf("  Year %d done.\n", yr))
}

elapsed <- (proc.time() - t0)[3]
cat(sprintf("Neighbor stats computed in %.1f seconds.\n", elapsed))

# ---- 6. Clean up helper column ---------------------------------------------
cell_data[, cell_idx := NULL]

# ---- 7. Apply pre-trained Random Forest model (unchanged) -------------------
# The model object (e.g., `rf_model`) is already in memory.
# Predict on the augmented cell_data with the new neighbor features.
# 
# predictions <- predict(rf_model, newdata = cell_data)
#
# This step is unchanged from the original pipeline.
```

---

## Further Acceleration: Optional Rcpp for Max/Min

The `data.table` grouped aggregation for max/min on the edge list is already fast, but if you want to eliminate the temporary `data.table` allocation (~1.37M rows per year Ã— 28 years), here is an Rcpp function that computes max, min, and mean directly from the CSR structure:

```r
# Optional: Rcpp-accelerated CSR row aggregation
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix csr_neighbor_stats(IntegerVector p, IntegerVector j,
                                  NumericVector vals, int n) {
  // p: row pointers (length n+1, 0-based)
  // j: column indices (0-based)
  // vals: value vector (length n), indexed by column
  // Returns n x 3 matrix: [max, min, mean]
  NumericMatrix out(n, 3);
  
  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    double vmax = R_NegInf;
    double vmin = R_PosInf;
    double vsum = 0.0;
    int count = 0;
    
    for (int k = start; k < end; k++) {
      double v = vals[j[k]];  // j[k] is 0-based col index
      if (!R_IsNA(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        vsum += v;
        count++;
      }
    }
    
    if (count == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / count;
    }
  }
  return out;
}
')
```

Then replace the inner variable loop body with:

```r
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  val_vec[] <- NA_real_
  val_vec[yr_cell_idx] <- cell_data[[var_name]][yr_rows]
  
  # Single pass over CSR structure â€” compiled C++
  stats_mat <- csr_neighbor_stats(csr_p, csr_j, val_vec, n_cells)
  # stats_mat columns: [1] max, [2] min, [3] mean
  
  set(cell_data, i = yr_row_lookup[yr_cell_idx], j = max_col,
      value = stats_mat[yr_cell_idx, 1])
  set(cell_data, i = yr_row_lookup[yr_cell_idx], j = min_col,
      value = stats_mat[yr_cell_idx, 2])
  set(cell_data, i = yr_row_lookup[yr_cell_idx], j = mean_col,
      value = stats_mat[yr_cell_idx, 3])
}
```

---

## Expected Performance

| Approach | Estimated Runtime |
|---|---|
| Original (`lapply` + string keys) | ~86+ hours |
| Pure R with `data.table` + sparse matrix (above) | ~5â€“15 minutes |
| With Rcpp `csr_neighbor_stats` | ~1â€“3 minutes |

**Memory footprint:** The sparse matrix is ~22 MB (1.37M non-zeros). The `data.table` is already in memory. Peak overhead is ~50â€“80 MB for temporary vectors â€” well within 16 GB.

**Numerical equivalence:** The `max`, `min`, and `mean` computations use identical semantics (NA exclusion, same aggregation functions). The sparse matrix multiply for mean is algebraically identical to summing neighbor values and dividing by the count of non-NA neighbors. The Rcpp version computes all three in a single pass with identical logic.
END_RESPONSE>>>
