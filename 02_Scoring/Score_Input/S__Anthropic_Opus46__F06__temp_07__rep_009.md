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
run_id: Anthropic_F06_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) effective cost
The function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) per lookup).
2. Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) for every single row.

Character-based named-vector lookups in R are hash-table lookups, but doing ~6.46M iterations Ã— ~4 neighbors Ã— string construction + hash lookup is extremely slow in an interpreted `lapply` loop. The lookup table `idx_lookup` itself has ~6.46M entries, so each probe is non-trivial.

**More fundamentally, the spatial neighbor structure is invariant across years.** The function redundantly recomputes the same spatial neighbor mapping 28 times (once per year for each cell). There are only 344,208 unique cells; the neighbor graph doesn't change from year to year.

### Bottleneck 2: `compute_neighbor_stats` â€” Repeated R-level loops
For each of the 5 variables, the function runs an `lapply` over all 6.46M rows, extracting neighbor values, removing NAs, and computing max/min/mean. This is 5 Ã— 6.46M = ~32.3M R-level function calls with vector subsetting.

### Summary
| Component | Calls | Cost driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | String construction, hash lookup |
| `compute_neighbor_stats` | 5 Ã— 6.46M iterations | Repeated subsetting, R-level loop |
| **Total** | ~38.7M R-level iterations | Interpreted loops on large data |

The 86+ hour estimate is consistent with these costs on a laptop.

---

## Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Since the rook-neighbor graph is **purely spatial** and **time-invariant**, we can:

1. **Build the neighbor lookup once at the cell level** (344K cells, not 6.46M cell-years).
2. **Compute neighbor stats year-by-year** using vectorized matrix operations, not row-by-row `lapply`.

### Specific techniques:

1. **Sparse adjacency matrix (Matrix package):** Encode the rook-neighbor graph as a sparse logical/binary matrix `W` of dimension 344,208 Ã— 344,208. Then for a given year, the neighbor-max, neighbor-min, and neighbor-mean of a variable can be computed via sparse matrix operations or efficient grouped operations â€” no R-level row loop needed.

2. **Year-sliced vectorized computation:** For each year, extract the variable vector (length 344,208), then use the sparse matrix to gather neighbor values. For **mean**, `W %*% x / rowSums(W)` gives the exact neighbor mean in one sparse matrix-vector multiply. For **max** and **min**, we use an efficient row-wise grouped operation over the sparse structure.

3. **data.table for fast joins and column assignment:** Replace data.frame operations with `data.table` for zero-copy column additions and fast keyed joins.

4. **Why not raster focal?** The grid cells come from an irregular (or at least ID-indexed) spatial panel, not a regular raster. The neighbor structure is precomputed as an `nb` object, which may encode irregular boundaries, islands, etc. A raster focal operation assumes a regular grid kernel and could **silently produce wrong results** at boundaries or for cells with fewer than 4 neighbors. We must preserve the exact `nb`-defined neighbor structure to preserve the original numerical estimand for the pre-trained Random Forest.

### Expected speedup:
- `build_neighbor_lookup`: eliminated entirely (replaced by one-time sparse matrix construction, ~seconds).
- `compute_neighbor_stats`: from ~32M R-level iterations to 28 years Ã— 5 vars Ã— vectorized sparse operations = ~140 sparse-matrix operations, each taking ~1-3 seconds â†’ **~5-10 minutes total**.
- Overall: from **86+ hours â†’ ~5-15 minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the exact numerical results of the original implementation.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)
library(Matrix)

#' Build a sparse binary adjacency matrix from an spdep nb object.
#' 
#' @param nb_obj   An nb object (e.g., rook_neighbors_unique from spdep).
#' @param id_order Character or integer vector of cell IDs in the order
#'                 corresponding to the nb object indices.
#' @return A list with:
#'   - W: a sparse dgCMatrix (n_cells x n_cells) binary adjacency matrix
#'   - id_order: the cell ID vector (defines row/col ordering)
build_sparse_adjacency <- function(nb_obj, id_order) {
  n <- length(nb_obj)
  stopifnot(n == length(id_order))
  
  # Build COO (coordinate) triplets
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_len(n)) {
    neighs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(neighs) == 1L && neighs[1] == 0L) next
    neighs <- neighs[neighs != 0L]
    if (length(neighs) == 0L) next
    from_idx <- c(from_idx, rep(i, length(neighs)))
    to_idx   <- c(to_idx, neighs)
  }
  
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n, n),
    dimnames = list(as.character(id_order), as.character(id_order))
  )
  
  list(W = W, id_order = id_order)
}

#' Compute neighbor max, min, mean for one variable across all cell-years,
#' using sparse matrix operations (mean) and efficient grouped ops (max, min).
#'
#' @param dt         A data.table with columns: id, year, and the variable.
#' @param var_name   Name of the source variable.
#' @param adj        Output of build_sparse_adjacency().
#' @return The data.table dt with three new columns added in place.
compute_neighbor_features_sparse <- function(dt, var_name, adj) {
  W        <- adj$W
  id_order <- adj$id_order
  n_cells  <- length(id_order)
  
  # Column names for output (matching original naming convention)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns with NA
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Create a mapping from cell id to sparse-matrix row index
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add the matrix row index to dt for fast alignment
  dt[, .sp_row := id_to_row[as.character(id)]]
  
  # Get the neighbor count per cell (for mean denominator, excluding NAs later)
  # W_row_nnz <- diff(W@p)  # for dgCMatrix, number of nonzeros per column
  # Actually for row-wise ops, convert to dgRMatrix or use rowSums
  # We'll work column-by-column in year slices for clarity.
  
  # Pre-extract the CSC structure for efficient row-gather of neighbor values

  # For max and min, we need actual neighbor values â€” sparse matrix multiply
  # only gives sum. We use the adjacency list extracted from the sparse matrix.
  # Extract adjacency list once (from sparse matrix, very fast).
  W_t <- t(W)  # transpose so columns of W_t = neighbors of each cell
  
  # Process each year independently
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Extract the variable values for this year, aligned to id_order
    # Use a keyed join for speed
    year_slice <- dt[year == yr, .(id, .sp_row, val = get(var_name))]
    
    # Build a full-length vector aligned to sparse matrix rows
    x <- rep(NA_real_, n_cells)
    x[year_slice$.sp_row] <- year_slice$val
    
    # --- NEIGHBOR MEAN (via sparse matrix multiply) ---
    # Replace NA with 0 for the multiply, but track valid counts
    x_nona <- x
    x_nona[is.na(x_nona)] <- 0
    valid <- as.numeric(!is.na(x))
    
    neighbor_sum   <- as.numeric(W %*% x_nona)       # sum of neighbor values (NAâ†’0)
    neighbor_count <- as.numeric(W %*% valid)         # count of non-NA neighbors
    
    neighbor_mean_vec <- ifelse(neighbor_count > 0,
                                neighbor_sum / neighbor_count,
                                NA_real_)
    
    # --- NEIGHBOR MAX and MIN (row-wise over sparse structure) ---
    # We iterate over the sparse matrix structure, but in C-level vectorized
    # fashion using the column pointers of W (CSC format).
    # 
    # For moderate-size problems, the fastest pure-R approach:
    # expand the neighbor pairs and do grouped max/min via data.table.
    
    # Extract (row, col) pairs from W where W[row, col] = 1
    # row = focal cell index, col = neighbor cell index
    # In dgCMatrix: W@i = row indices (0-based), W@p = column pointers
    # But we want rowâ†’neighbors, so use W's structure directly.
    
    # Actually, we already have W as dgCMatrix.
    # Rows of W = focal cells, columns = neighbors.
    # For row-wise operations, it's more efficient to work with W as dgRMatrix
    # or to use the transpose trick.
    
    # W_t (transposed) is dgCMatrix where column j = neighbors of cell j.
    # W_t@i[  (W_t@p[j]+1) : W_t@p[j+1]  ] gives 0-based row indices = 
    # neighbor indices of cell j.
    
    # Vectorized extraction using data.table:
    # Build a neighbor-value table for this year
    
    # Extract all (focal, neighbor) pairs from sparse matrix
    # Do this once outside the year loop if memory allows â€” but the pairs
    # are the same every year. Let's extract once before the loop.
    # (We'll restructure below.)
    
    # For now, use the pre-extracted edge list approach:
    neighbor_max_vec <- rep(NA_real_, n_cells)
    neighbor_min_vec <- rep(NA_real_, n_cells)
    
    # We'll compute max/min via the edge data.table (see restructured code below)
    # For this version, assign mean now, and handle max/min via edge table.
    
    # Assign mean results back to dt
    rows_this_year <- which(dt$year == yr)
    sp_rows <- dt$.sp_row[rows_this_year]
    
    dt[rows_this_year, (col_mean) := neighbor_mean_vec[sp_rows]]
  }
  
  # --- MAX and MIN via edge-list + data.table grouped ops (all years at once) ---
  
  # Step 1: Extract the edge list from sparse matrix (once)
  # W is dgCMatrix: columns are indexed by @p, rows by @i (0-based)
  # W[i,j] = 1 means cell i has neighbor j
  wt <- summary(W)  # returns (i, j, x) triplets with 1-based indices
  edges <- data.table(focal = wt$i, neighbor = wt$j)
  
  # Step 2: For each year, join neighbor values and compute grouped max/min
  # Build a lookup: (sp_row, year) â†’ value
  val_lookup <- dt[, .(sp_row = .sp_row, year, val = get(var_name))]
  setkey(val_lookup, sp_row, year)
  
  # Expand edges Ã— years
  # To avoid a massive cross join (edges Ã— years), process year by year
  
  for (yr in years) {
    # Get values for this year
    yr_vals <- val_lookup[year == yr, .(sp_row, val)]
    setkey(yr_vals, sp_row)
    
    # Join neighbor values onto edge list
    edge_yr <- copy(edges)
    edge_yr[yr_vals, neighbor_val := i.val, on = .(neighbor = sp_row)]
    
    # Remove edges where neighbor value is NA
    edge_yr <- edge_yr[!is.na(neighbor_val)]
    
    # Grouped max and min by focal cell
    stats <- edge_yr[, .(nmax = max(neighbor_val),
                         nmin = min(neighbor_val)),
                     by = focal]
    
    # Map back to dt rows for this year
    rows_this_year <- which(dt$year == yr)
    sp_rows <- dt$.sp_row[rows_this_year]
    
    # Build a vector indexed by sp_row
    max_vec <- rep(NA_real_, n_cells)
    min_vec <- rep(NA_real_, n_cells)
    max_vec[stats$focal] <- stats$nmax
    min_vec[stats$focal] <- stats$nmin
    
    set(dt, i = rows_this_year, j = col_max, value = max_vec[sp_rows])
    set(dt, i = rows_this_year, j = col_min, value = min_vec[sp_rows])
  }
  
  # Clean up temporary column
  dt[, .sp_row := NULL]
  
  invisible(dt)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build sparse adjacency matrix (once, ~seconds)
adj <- build_sparse_adjacency(rook_neighbors_unique, id_order)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  compute_neighbor_features_sparse(cell_data, var_name, adj)
}

message("Done. Neighbor features added. Ready for Random Forest prediction.")
```

---

## Cleaner Refactored Version (Recommended)

The above interleaves the mean computation inside a year loop separately from max/min. Below is a cleaner, unified version:

```r
library(data.table)
library(Matrix)

# â”€â”€ 1. Build sparse adjacency matrix (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_sparse_adjacency <- function(nb_obj, id_order) {
  n <- length(nb_obj)
  stopifnot(n == length(id_order))
  
  from <- integer(); to <- integer()
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    nb <- nb[nb != 0L]
    if (length(nb)) {
      from <- c(from, rep.int(i, length(nb)))
      to   <- c(to, nb)
    }
  }
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  list(W = W, id_order = id_order, 
       edges = data.table(focal = from, neighbor = to))
}

# â”€â”€ 2. Compute all three stats for one variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

add_neighbor_features <- function(dt, var_name, adj) {
  W        <- adj$W
  id_order <- adj$id_order
  edges    <- adj$edges
  n_cells  <- length(id_order)
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Map cell id â†’ matrix row index
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  dt[, .sp_row := id_map[as.character(id)]]
  
  # Pre-allocate
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    idx_yr <- which(dt$year == yr)
    sp     <- dt$.sp_row[idx_yr]
    
    # Full-length vector aligned to matrix rows
    x <- rep(NA_real_, n_cells)
    x[sp] <- dt[[var_name]][idx_yr]
    
    # â”€â”€ Mean via sparse matrix multiply â”€â”€
    x0 <- x; x0[is.na(x0)] <- 0
    v  <- as.numeric(!is.na(x))
    
    s <- as.numeric(W %*% x0)
    cnt <- as.numeric(W %*% v)
    mn <- ifelse(cnt > 0, s / cnt, NA_real_)
    
    # â”€â”€ Max & Min via data.table grouped ops on edge list â”€â”€
    edge_yr <- edges[, .(focal, nval = x[neighbor])]
    edge_yr <- edge_yr[!is.na(nval)]
    
    mm <- edge_yr[, .(nmax = max(nval), nmin = min(nval)), by = focal]
    
    mx <- rep(NA_real_, n_cells); mx[mm$focal] <- mm$nmax
    mi <- rep(NA_real_, n_cells); mi[mm$focal] <- mm$nmin
    
    # â”€â”€ Write back â”€â”€
    set(dt, i = idx_yr, j = col_max,  value = mx[sp])
    set(dt, i = idx_yr, j = col_min,  value = mi[sp])
    set(dt, i = idx_yr, j = col_mean, value = mn[sp])
  }
  
  dt[, .sp_row := NULL]
  invisible(dt)
}

# â”€â”€ 3. Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data <- as.data.table(cell_data)

adj <- build_sparse_adjacency(rook_neighbors_unique, id_order)

for (v in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  message("Processing: ", v)
  add_neighbor_features(cell_data, v, adj)
}

# cell_data now has 15 new columns (3 stats Ã— 5 vars), numerically identical
# to the original implementation. The pre-trained Random Forest model is
# unchanged and can be applied directly.
```

---

## Why This Works and What It Preserves

| Concern | Resolution |
|---|---|
| **Numerical equivalence** | Mean = `sum / count` (excluding NAs) via sparse multiply is algebraically identical to the original `mean(neighbor_vals)`. Max and min use the same grouped values. No floating-point reordering issues for max/min (exact). Mean may differ by â‰¤ machine epsilon due to summation order â€” negligible for RF prediction. |
| **Boundary / irregular cells** | The sparse matrix `W` is built directly from the same `spdep::nb` object (`rook_neighbors_unique`), so cells with 1, 2, or 3 neighbors (edges, corners, islands) are handled identically. |
| **NA handling** | Matches original: NAs among neighbors are excluded; if all neighbors are NA (or no neighbors exist), the result is `NA`. |
| **Pre-trained RF model** | Untouched. The code only computes predictor columns; no model retraining occurs. |
| **Memory (16 GB laptop)** | The sparse matrix is ~1.37M nonzeros Ã— 16 bytes â‰ˆ 22 MB. The edge table is ~1.37M rows Ã— 2 cols â‰ˆ 22 MB. Year slices are 344K-length vectors â‰ˆ 2.7 MB each. Total overhead is well under 1 GB. |
| **Why not raster focal?** | The `nb` object may encode irregular boundaries, missing cells, or non-rectangular domains. Raster focal operations assume a regular kernel and would silently produce incorrect neighbor sets at boundaries, **changing the numerical estimand** that the trained RF expects. |

### Expected runtime: **~5â€“15 minutes** (vs. 86+ hours originally).
END_RESPONSE>>>
