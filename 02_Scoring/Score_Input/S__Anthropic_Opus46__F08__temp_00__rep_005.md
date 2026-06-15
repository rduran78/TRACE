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
run_id: Anthropic_F08_T00_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination â€” effectively duplicating the same spatial relationship 28 times.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M `paste(id, year)` keys and performs millions of character-based lookups into a named vector (`idx_lookup`), which is O(n) or O(n log n) in R's named-vector implementation.

3. **`lapply` over 6.46M rows** with per-element anonymous functions and repeated subsetting is inherently slow in R, even before considering the string operations inside each iteration.

4. **`compute_neighbor_stats` then iterates again** over the 6.46M-element lookup list, performing per-row subsetting of a numeric vector. This is called 5 times (once per variable), adding another 5 Ã— 6.46M iterations.

**Net effect:** ~6.46M list elements created, each containing a variable-length integer vector, followed by 5 Ã— 6.46M `lapply` calls for stats. The combined memory allocation, string hashing, and interpreted-loop overhead easily explains the 86+ hour estimate.

## Optimization Strategy

**Key insight:** Separate the *static spatial topology* (which cells are neighbors) from the *dynamic yearly values* (what values those neighbors hold).

### Step 1: Build a cell-level neighbor index once (344K entries, not 6.46M)

Convert `rook_neighbors_unique` (an `nb` object, already a list of integer vectors indexed by cell position) into a sparse adjacency representation. This is done **once** and reused for all years and all variables.

### Step 2: Compute neighbor stats per year using matrix operations

For each year, extract the column of values for a given variable as a vector aligned to `id_order`. Then use the static neighbor index to compute max, min, and mean via vectorized operations â€” ideally by converting the neighbor list to a sparse matrix and using matrix-vector multiplication (for mean) and row-wise operations (for max/min).

### Step 3: Map results back to the full cell-year data frame

Join the per-cell-per-year results back to the panel data frame.

This reduces the problem from 6.46M list lookups to **28 Ã— 5 = 140 vectorized operations** over 344K cells, plus a single merge step.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build static sparse adjacency matrix ONCE (344,208 Ã— 344,208)
# ==============================================================================
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element is an integer vector of neighbor positions (into id_order).
# id_order is the vector of cell IDs in the order matching the nb object.

build_sparse_adjacency <- function(nb_obj) {
  n <- length(nb_obj)
  # Build COO (coordinate) representation
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove the 0-neighbor sentinel that spdep uses (nb objects use 0L for no neighbors)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse logical/numeric matrix: W[i,j] = 1 if j is a neighbor of i
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  # Also store the number of neighbors per cell for computing means
  n_neighbors <- diff(W@p)  # for dgCMatrix, column counts; but we want row counts
  # Actually, let's just use rowSums which is efficient for sparse matrices
  
  return(W)
}

W <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- length(id_order)

# Precompute neighbor counts per cell (static)
neighbor_counts <- as.numeric(rowSums(W))  # length = n_cells

# ==============================================================================
# STEP 2: Convert panel data to data.table for fast operations
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# Create a mapping from cell ID to position index in id_order (1-based)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add the position index to the data.table
cell_dt[, pos := id_to_pos[as.character(id)]]

# Verify alignment
stopifnot(all(!is.na(cell_dt$pos)))

# Sort by year and pos for efficient grouped operations
setkey(cell_dt, year, pos)

# ==============================================================================
# STEP 3: Vectorized neighbor stats computation
# ==============================================================================
# For each variable and each year:
#   - Extract the value vector v (length n_cells, ordered by pos)
#   - neighbor_sum  = W %*% v   (sparse matrix-vector multiply)
#   - neighbor_mean = neighbor_sum / neighbor_counts
#   - For max and min: iterate over the nb list (but only 344K cells, not 6.46M)

# Pre-extract the neighbor list as a plain list of integer vectors for max/min
# (The sparse matrix handles mean efficiently, but max/min need explicit iteration)
nb_list <- lapply(rook_neighbors_unique, function(x) {
  x <- x[x != 0L]
  as.integer(x)
})

compute_neighbor_stats_fast <- function(dt, var_name, W, nb_list, 
                                         neighbor_counts, n_cells, id_order) {
  years <- sort(unique(dt$year))
  
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns in the data.table
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  for (yr in years) {
    # Extract the value vector for this year, ordered by pos (1..n_cells)
    # Because we keyed by (year, pos), the subset for a given year is already
    # sorted by pos. But some cells might be missing for some years, so we
    # need to handle that.
    
    yr_rows <- dt[.(yr)]  # keyed lookup: all rows for this year
    
    # Build a full-length vector (n_cells) with NAs for missing cells
    vals <- rep(NA_real_, n_cells)
    vals[yr_rows$pos] <- yr_rows[[var_name]]
    
    # --- MEAN via sparse matrix-vector multiply ---
    # Replace NAs with 0 for the multiply, but track valid counts
    vals_zero <- vals
    vals_zero[is.na(vals_zero)] <- 0
    valid_flag <- as.numeric(!is.na(vals))
    
    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_valid <- as.numeric(W %*% valid_flag)
    
    n_mean <- ifelse(neighbor_valid > 0, neighbor_sum / neighbor_valid, NA_real_)
    
    # --- MAX and MIN via vectorized C-level operations on the nb list ---
    # Use vapply over 344K cells (not 6.46M)
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
    
    # Vectorized approach: build a matrix of neighbor values using the nb list
    # For cells with neighbors, compute max and min
    for (ci in seq_len(n_cells)) {
      nb_idx <- nb_list[[ci]]
      if (length(nb_idx) == 0L) next
      nv <- vals[nb_idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) next
      n_max[ci] <- max(nv)
      n_min[ci] <- min(nv)
    }
    
    # Map results back to the data.table rows for this year
    row_indices <- which(dt$year == yr)
    positions   <- dt$pos[row_indices]
    
    set(dt, i = row_indices, j = max_col,  value = n_max[positions])
    set(dt, i = row_indices, j = min_col,  value = n_min[positions])
    set(dt, i = row_indices, j = mean_col, value = n_mean[positions])
  }
  
  return(dt)
}
```

The inner loop over 344K cells for max/min is still interpreted R. We can accelerate it dramatically with `Rcpp` or by restructuring into a padded matrix approach:

```r
# ==============================================================================
# FASTER max/min: Rcpp inline (avoids 344K R-level loop iterations per year)
# ==============================================================================
library(Rcpp)

cppFunction('
NumericMatrix neighbor_max_min_cpp(NumericVector vals, List nb_list) {
  int n = nb_list.size();
  NumericMatrix out(n, 2);  // col 0 = max, col 1 = min
  
  for (int i = 0; i < n; i++) {
    IntegerVector idx = nb_list[i];
    int m = idx.size();
    if (m == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      continue;
    }
    double vmax = NA_REAL;
    double vmin = NA_REAL;
    bool found = false;
    for (int j = 0; j < m; j++) {
      double v = vals[idx[j] - 1];  // R is 1-indexed
      if (ISNA(v) || ISNAN(v)) continue;
      if (!found) {
        vmax = v;
        vmin = v;
        found = true;
      } else {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
      }
    }
    if (!found) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
    }
  }
  return out;
}
')

# ==============================================================================
# FINAL OPTIMIZED PIPELINE
# ==============================================================================

compute_all_neighbor_features <- function(dt, var_names, W, nb_list,
                                           neighbor_counts, n_cells) {
  years <- sort(unique(dt$year))
  
  # Pre-allocate all result columns
  for (var_name in var_names) {
    dt[, paste0("neighbor_max_",  var_name) := NA_real_]
    dt[, paste0("neighbor_min_",  var_name) := NA_real_]
    dt[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  for (yr in years) {
    cat("Processing year:", yr, "\n")
    
    # Identify rows for this year
    row_indices <- which(dt$year == yr)
    positions   <- dt$pos[row_indices]
    
    for (var_name in var_names) {
      # Build full-length value vector for this year
      vals <- rep(NA_real_, n_cells)
      vals[positions] <- dt[[var_name]][row_indices]
      
      # --- MEAN via sparse mat-vec ---
      vals_zero <- vals
      vals_zero[is.na(vals_zero)] <- 0
      valid_flag <- as.numeric(!is.na(vals))
      
      neighbor_sum   <- as.numeric(W %*% vals_zero)
      neighbor_valid <- as.numeric(W %*% valid_flag)
      
      n_mean <- ifelse(neighbor_valid > 0, neighbor_sum / neighbor_valid, NA_real_)
      
      # --- MAX / MIN via Rcpp ---
      mm <- neighbor_max_min_cpp(vals, nb_list)
      n_max <- mm[, 1]
      n_min <- mm[, 2]
      
      # Write back to data.table
      max_col  <- paste0("neighbor_max_",  var_name)
      min_col  <- paste0("neighbor_min_",  var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      set(dt, i = row_indices, j = max_col,  value = n_max[positions])
      set(dt, i = row_indices, j = min_col,  value = n_min[positions])
      set(dt, i = row_indices, j = mean_col, value = n_mean[positions])
    }
  }
  
  return(dt)
}

# ==============================================================================
# EXECUTION
# ==============================================================================

# 1. Build static structures ONCE
W <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- length(id_order)

nb_list <- lapply(rook_neighbors_unique, function(x) {
  x <- x[x != 0L]
  as.integer(x)
})

# 2. Convert to data.table and add position index
cell_dt <- as.data.table(cell_data)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, pos := id_to_pos[as.character(id)]]

# 3. Compute all neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_dt <- compute_all_neighbor_features(
  dt = cell_dt,
  var_names = neighbor_source_vars,
  W = W,
  nb_list = nb_list,
  neighbor_counts = NULL,  # not needed; we use neighbor_valid per year
  n_cells = n_cells
)

# 4. Remove helper column, convert back if needed
cell_dt[, pos := NULL]
cell_data <- as.data.frame(cell_dt)

# 5. Predict with the pre-trained Random Forest (UNCHANGED)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | 6.46M-element list of character-keyed row indices | 344K-element nb list + sparse matrix (built once) |
| String operations | ~13M `paste()` calls + named-vector lookups | **Zero** |
| Iterations for stats | 5 vars Ã— 6.46M rows = 32.3M R-level iterations | 5 vars Ã— 28 years Ã— 344K cells = 48.2M **C-level** iterations (Rcpp) + 280 sparse mat-vec multiplies |
| Mean computation | R-level loop with subsetting | Sparse matrixâ€“vector multiply (optimized BLAS) |
| Estimated time | 86+ hours | **~5â€“15 minutes** |
| Numerical result | max, min, mean of non-NA neighbor values | **Identical** (same estimand preserved) |
| RF model | Untouched | **Untouched** |
| RAM peak | ~6.46M list + copies | ~sparse matrix (few MB) + 3 vectors of 344K per year-var |

The redesign exploits the static-versus-changing distinction: the spatial topology is encoded **once** in a sparse matrix and neighbor list, and only the year-varying value vectors are swapped through on each iteration.
END_RESPONSE>>>
