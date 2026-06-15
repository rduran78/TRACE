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
run_id: Anthropic_F02_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and named-vector lookups, which are extremely slow in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which again loops over 6.46 million entries with per-element subsetting and summary computation. Together, these two stages dominate the 86+ hour runtime.

**Specific problems:**

1. **String-key hashing per row in `build_neighbor_lookup`:** For each of 6.46M rows, the code pastes neighbor IDs with the current year, then looks them up in a named character vector (`idx_lookup`). This is O(n Ã— k) string operations where k is the average neighbor count (~4 for rook contiguity), yielding ~26M paste + lookup operations in slow interpreted R.

2. **`lapply` over millions of rows:** Both functions use `lapply` over 6.46M elements, producing millions of tiny R objects (vectors of length 3 or small integer vectors), causing massive GC pressure and memory fragmentation.

3. **`do.call(rbind, result)` on millions of rows:** Binding ~6.46M small vectors into a matrix is itself slow and memory-intensive.

4. **Redundant recomputation:** The neighbor lookup is the same structural relationship replicated across years. The code rebuilds string keys per row instead of exploiting the panel structure (same neighbor graph repeated for each year).

---

## Optimization Strategy

### Key Insight: Exploit the Panel Structure

The neighbor graph is **fixed across years**. Cell `i`'s rook neighbors are the same in 1992 as in 2019. So instead of building a 6.46M-row lookup with string keys, we can:

1. **Sort data by (year, id)** so that within each year, cells appear in a consistent order.
2. **Build the neighbor lookup once at the cell level** (~344K cells), then apply it identically within each year using integer-offset arithmetic.
3. **Vectorize `compute_neighbor_stats`** using `data.table` or sparse-matrix multiplication instead of per-row `lapply`.

### Concrete Plan

- **Replace string-keyed lookup with integer-indexed lookup.** Sort data by `(year, id)`, assign a within-year positional index. Since the sort order is identical each year, the neighbor indices for cell position `j` are the same positions in every year-block. The global row index is simply `(year_block_offset + neighbor_position)`.
- **Vectorize neighbor stats using sparse matrix multiplication.** Construct a sparse adjacency matrix `W` (344K Ã— 344K) once. For each year, extract the variable column as a vector, then compute `W %*% x` (neighbor sum), use the row-sums of `W` for counts, and analogous operations for max/min. This replaces millions of R-level loops with optimized C-level sparse BLAS.
- **For max and min**, sparse matrix multiplication doesn't directly help, so we use a `data.table` join-and-aggregate approach or a targeted C++ snippet via `Rcpp`.

### Expected Improvement

| Component | Before | After |
|---|---|---|
| Neighbor lookup build | ~hours (string ops on 6.46M rows) | ~seconds (integer arithmetic) |
| Neighbor stats (mean) | ~hours (lapply over 6.46M) | ~seconds (sparse matrix Ã— vector, 28 times) |
| Neighbor stats (max/min) | ~hours | ~minutes (vectorized data.table or Rcpp) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |

Memory stays well within 16 GB: the sparse matrix is ~1.4M non-zero entries (trivial), and we process one variable at a time.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# =============================================================================
# Prerequisites: data.table, Matrix, (optionally Rcpp)
# Preserves: trained Random Forest model (untouched)
# Preserves: original numerical estimand (identical max, min, mean computation)
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 0: Convert to data.table and establish consistent ordering --------

cell_dt <- as.data.table(cell_data)

# Ensure id_order is the canonical ordering of cell IDs (from the nb object).
# id_order: vector of length 344,208 â€” the cell IDs in the order matching
# rook_neighbors_unique (i.e., rook_neighbors_unique[[k]] gives neighbors
# of id_order[k]).

# Create a map from cell ID to its positional index in id_order (1-based).
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add positional index to data
cell_dt[, cell_pos := id_to_pos[as.character(id)]]

# Sort by year then by cell_pos â€” this is the key structural trick.
setkey(cell_dt, year, cell_pos)

# Verify all years have the same set of cells in the same order.
# (If some cell-years are missing, we handle that below.)
years <- sort(unique(cell_dt$year))
n_cells <- length(id_order)
n_years <- length(years)

# Check for complete panel
is_complete_panel <- (nrow(cell_dt) == n_cells * n_years)

if (is_complete_panel) {
  message("Complete panel detected. Using fast offset method.")
} else {
  message("Incomplete panel detected. Using join-based method.")
}

# ---- Step 1: Build sparse adjacency matrix from nb object (once) ------------

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Returns: sparse logical/numeric matrix W of dimension n x n
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove zero-length entries (islands with no neighbors, coded as 0 in spdep)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)
# W is 344208 x 344208 with ~1.37M non-zero entries â€” very small in memory.

# Precompute neighbor counts per cell (used for mean calculation).
neighbor_counts <- as.numeric(rowSums(W))  # length 344208

# ---- Step 2: Compute neighbor stats â€” vectorized ---------------------------

# For a COMPLETE panel, rows are arranged in blocks of n_cells, one block per
# year, with cells in the same positional order within each block.
# Row i in year-block t corresponds to cell_pos = i.
# So we can simply extract each year-block as a vector and multiply by W.

compute_neighbor_features_fast <- function(dt, W, neighbor_counts,
                                           var_name, years, n_cells,
                                           is_complete) {
  
  mean_col <- paste0("neighbor_mean_", var_name)
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  
  # Pre-allocate output columns
  dt[, (mean_col) := NA_real_]
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  
  if (is_complete) {
    # ---- FAST PATH: complete panel, offset arithmetic ----
    
    # Data is sorted by (year, cell_pos). Each year-block is exactly n_cells rows.
    # Block for year t starts at row (t_index - 1) * n_cells + 1.
    
    for (t_idx in seq_along(years)) {
      row_start <- (t_idx - 1L) * n_cells + 1L
      row_end   <- t_idx * n_cells
      
      x <- dt[[var_name]][row_start:row_end]  # length n_cells, ordered by cell_pos
      
      # --- Neighbor mean via sparse matrix multiplication ---
      # Replace NA with 0 for multiplication, track valid counts
      x_nona <- x
      x_nona[is.na(x_nona)] <- 0
      valid_indicator <- as.numeric(!is.na(x))
      
      neighbor_sum   <- as.numeric(W %*% x_nona)
      neighbor_valid <- as.numeric(W %*% valid_indicator)
      
      n_mean <- ifelse(neighbor_valid > 0, neighbor_sum / neighbor_valid, NA_real_)
      
      # --- Neighbor max and min via sparse row operations ---
      # We iterate over the sparse structure efficiently.
      # For max/min, we use a direct C-level approach via the sparse matrix.
      
      # Extract the sparse structure of W (CSC format from Matrix package)
      # We convert to dgRMatrix (row-sparse) for row-wise iteration, or
      # use the column pointers cleverly.
      # 
      # More practical: use the (i, j) triplet representation, then aggregate
      # with data.table â€” this is very fast.
      
      # Build a data.table of (row_cell, neighbor_value)
      # We only need to do the triplet extraction once (outside the year loop).
      # But neighbor values change per year, so we fill them in per year.
      
      # We'll use a pre-extracted edge list (see below).
      # For now, store x and compute after the loop.
      
      set(dt, i = row_start:row_end, j = mean_col, value = n_mean)
    }
    
    # --- Max and Min: edge-list approach (vectorized via data.table) ---
    
    # Extract edge list from W once
    W_triplet <- summary(W)  # gives (i, j, x) triplet â€” i and j are cell positions
    edges <- data.table(
      from_pos = W_triplet$i,
      to_pos   = W_triplet$j
    )
    # For each row cell (from_pos), its neighbors are to_pos.
    # But actually in our adjacency: W[i,j] = 1 means j is a neighbor of i.
    # So for cell i, neighbor values are x[j] for all j where W[i,j]=1.
    # In the triplet, each (i, j) with x=1 means j is neighbor of i.
    
    for (t_idx in seq_along(years)) {
      row_start <- (t_idx - 1L) * n_cells + 1L
      row_end   <- t_idx * n_cells
      
      x <- dt[[var_name]][row_start:row_end]
      
      # Map neighbor values
      edges[, nval := x[to_pos]]
      
      # Aggregate by from_pos
      agg <- edges[!is.na(nval), .(
        nmax = max(nval),
        nmin = min(nval)
      ), by = from_pos]
      
      # Fill results â€” cells not in agg have all-NA neighbors â†’ stay NA
      max_vec <- rep(NA_real_, n_cells)
      min_vec <- rep(NA_real_, n_cells)
      max_vec[agg$from_pos] <- agg$nmax
      min_vec[agg$from_pos] <- agg$nmin
      
      set(dt, i = row_start:row_end, j = max_col, value = max_vec)
      set(dt, i = row_start:row_end, j = min_col, value = min_vec)
    }
    
  } else {
    # ---- SAFE PATH: incomplete panel, join-based ----
    
    W_triplet <- summary(W)
    edges <- data.table(from_pos = W_triplet$i, to_pos = W_triplet$j)
    
    for (yr in years) {
      yr_rows <- which(dt$year == yr)
      
      # Map from cell_pos to row index within this year
      pos_in_year <- dt$cell_pos[yr_rows]
      pos_to_row  <- setNames(yr_rows, as.character(pos_in_year))
      
      x_full <- rep(NA_real_, n_cells)
      x_full[pos_in_year] <- dt[[var_name]][yr_rows]
      
      # Mean
      x_nona <- x_full; x_nona[is.na(x_nona)] <- 0
      valid_ind <- as.numeric(!is.na(x_full))
      n_sum   <- as.numeric(W %*% x_nona)
      n_valid <- as.numeric(W %*% valid_ind)
      n_mean  <- ifelse(n_valid > 0, n_sum / n_valid, NA_real_)
      
      set(dt, i = yr_rows, j = mean_col, value = n_mean[pos_in_year])
      
      # Max / Min
      edges[, nval := x_full[to_pos]]
      agg <- edges[!is.na(nval), .(nmax = max(nval), nmin = min(nval)), by = from_pos]
      
      max_full <- rep(NA_real_, n_cells)
      min_full <- rep(NA_real_, n_cells)
      max_full[agg$from_pos] <- agg$nmax
      min_full[agg$from_pos] <- agg$nmin
      
      set(dt, i = yr_rows, j = max_col, value = max_full[pos_in_year])
      set(dt, i = yr_rows, j = min_col, value = min_full[pos_in_year])
    }
  }
  
  return(dt)
}

# ---- Step 3: Run for all neighbor source variables --------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  cell_dt <- compute_neighbor_features_fast(
    dt               = cell_dt,
    W                = W,
    neighbor_counts  = neighbor_counts,
    var_name         = var_name,
    years            = years,
    n_cells          = n_cells,
    is_complete      = is_complete_panel
  )
}

# ---- Step 4: Convert back to data.frame if needed for predict() -------------

# Remove the helper column
cell_dt[, cell_pos := NULL]

# Restore original row order if needed (the RF predict call may expect it)
setkey(cell_dt, id, year)  # or whatever order the trained model expects

cell_data <- as.data.frame(cell_dt)

# ---- Step 5: Predict with the existing trained Random Forest ----------------
# The trained model object is untouched. Column names and semantics are preserved.
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Why They Work

| Original Problem | Optimization Applied | Speedup Factor |
|---|---|---|
| 6.46M string `paste` + named-vector lookups in `build_neighbor_lookup` | **Eliminated entirely** â€” replaced with integer positional arithmetic on a sorted panel | ~1000Ã— |
| 6.46M `lapply` iterations for `mean` in `compute_neighbor_stats` | **Sparse matrixâ€“vector multiply** (`W %*% x`), executed 28 times (once per year) on 344K-length vectors | ~500Ã— |
| 6.46M `lapply` iterations for `max`/`min` | **Vectorized `data.table` aggregation** on a pre-extracted edge list (~1.37M edges), 28 times | ~200Ã— |
| `do.call(rbind, ...)` on 6.46M 3-element vectors | **Eliminated** â€” results written directly via `data.table::set()` | ~100Ã— |
| Peak memory from millions of small R objects | **Eliminated** â€” only dense vectors of length 344K and one sparse matrix in memory | ~5Ã— reduction |

**Estimated new runtime:** 5â€“15 minutes on a 16 GB laptop.  
**Numerical equivalence:** The mean is computed as `sum / valid_count` (identical to `mean` on non-NA values). Max and min use the same `max()` and `min()` on the same neighbor value sets. The trained Random Forest model is never modified or retrained.
END_RESPONSE>>>
