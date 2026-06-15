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
run_id: Anthropic_F02_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements using `lapply` with per-row string pasting, named-vector lookups, and NA filtering.** This is an O(n) loop in pure R over ~6.46M rows, each doing character coercion, `paste`, and named-vector indexing â€” all of which are slow in interpreted R. Then `compute_neighbor_stats` loops over the same 6.46M-element list again, once per variable (Ã—5 variables). The combination yields an estimated 86+ hours.

**Specific problems:**

1. **`build_neighbor_lookup`:** The `lapply` over 6.46M rows performs millions of `paste()` and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector indexing in R is hash-table lookup but the overhead of creating millions of small character vectors and calling `paste` row-by-row is enormous.

2. **`compute_neighbor_stats`:** The `lapply` over 6.46M entries, each extracting a small integer vector of neighbor indices, computing `max/min/mean`, and returning a length-3 vector, then `do.call(rbind, ...)` on 6.46M 3-element vectors â€” this is slow and memory-wasteful.

3. **Memory:** The neighbor lookup list alone (6.46M entries, each a small integer vector) consumes several GB. Combined with the 6.46M Ã— 110 data frame and intermediate copies, 16 GB RAM is tight.

---

## Optimization Strategy

The key insight is to **replace the row-level R loops with vectorized operations on a sparse adjacency matrix and use matrix algebra for neighbor statistics.**

### Step-by-step plan:

1. **Build a sparse cell-year adjacency matrix (once).** Convert the spatial `nb` object + year panel structure into a single sparse matrix `W` of dimension `n_rows Ã— n_rows` (6.46M Ã— 6.46M) where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` in the same year. Because there are ~1.37M directed neighbor pairs per year Ã— 28 years â‰ˆ 38.5M non-zero entries, this is extremely sparse and fits in memory (~600 MB).

2. **Compute neighbor stats via sparse matrix multiplication.** For each variable `x`:
   - `neighbor_sum = W %*% x`
   - `neighbor_count = W %*% (!is.na(x))` (to handle NAs)
   - `neighbor_mean = neighbor_sum / neighbor_count`
   - For `max` and `min`: use a grouped approach on the sparse matrix's explicit triplet structure.

3. **This eliminates all `lapply` loops and `paste` operations.** The sparse matrix approach is vectorized in C (via the `Matrix` package) and runs in minutes, not days.

4. **The trained Random Forest model and original numerical estimand are fully preserved** â€” we only change how features are computed, not their values.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ============================================================
# 1. Build a sparse row-adjacency matrix (replaces build_neighbor_lookup)
# ============================================================
build_sparse_neighbor_matrix <- function(cell_data, id_order, neighbors) {
  # Convert cell_data to data.table for fast keyed joins
  dt <- as.data.table(cell_data)
  n <- nrow(dt)
  
  # Create a fast lookup: (id, year) -> row index
  dt[, row_idx := .I]
  setkey(dt, id, year)
  
  # Map each cell id to its position in id_order (spatial index)
  id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build the edge list: for every row i, find its neighbor rows j (same year)
  # We do this by expanding the nb object into an edge list of (cell_id, neighbor_cell_id),
  # then joining with the panel to get (row_i, row_j).
  
  # Step A: Build spatial edge list from nb object
  from_spatial <- rep(seq_along(neighbors), lengths(neighbors))
  to_spatial   <- unlist(neighbors)
  
  # Remove zero-neighbor entries (spdep::nb uses integer(0) for islands)
  valid <- to_spatial > 0L
  from_spatial <- from_spatial[valid]
  to_spatial   <- to_spatial[valid]
  
  from_cell_id <- id_order[from_spatial]
  to_cell_id   <- id_order[to_spatial]
  
  spatial_edges <- data.table(from_id = from_cell_id, to_id = to_cell_id)
  
  cat(sprintf("Spatial edge list: %d directed edges\n", nrow(spatial_edges)))
  
  # Step B: For each year, join spatial edges with row indices
  years <- sort(unique(dt$year))
  
  # Prepare a lookup table: (id) -> list of (year, row_idx)
  id_year_lookup <- dt[, .(id, year, row_idx)]
  
  # We'll build triplet vectors incrementally
  all_i <- vector("list", length(years))
  all_j <- vector("list", length(years))
  
  for (k in seq_along(years)) {
    yr <- years[k]
    # Row indices for this year
    yr_rows <- id_year_lookup[year == yr, .(id, row_idx)]
    setkey(yr_rows, id)
    
    # Join: from_id -> from_row_idx
    edges_yr <- spatial_edges[yr_rows, on = .(from_id = id), nomatch = 0L,
                              .(from_row = i.row_idx, to_id)]
    # Join: to_id -> to_row_idx
    edges_yr <- edges_yr[yr_rows, on = .(to_id = id), nomatch = 0L,
                          .(from_row, to_row = i.row_idx)]
    
    all_i[[k]] <- edges_yr$from_row
    all_j[[k]] <- edges_yr$to_row
    
    if (k %% 5 == 0 || k == length(years)) {
      cat(sprintf("  Year %d (%d/%d): %d edges\n", yr, k, length(years), nrow(edges_yr)))
    }
  }
  
  # Concatenate and build sparse matrix
  all_i <- unlist(all_i)
  all_j <- unlist(all_j)
  
  cat(sprintf("Total cell-year edges: %d\n", length(all_i)))
  
  W <- sparseMatrix(
    i = all_i,
    j = all_j,
    x = rep(1, length(all_i)),
    dims = c(n, n)
  )
  
  return(W)
}

# ============================================================
# 2. Compute neighbor stats via sparse matrix ops
#    (replaces compute_neighbor_stats + loop)
# ============================================================
compute_neighbor_features_sparse <- function(cell_data, W, neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  n  <- nrow(dt)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for: %s\n", var_name))
    
    x <- dt[[var_name]]
    x_num <- as.numeric(x)
    
    # --- Neighbor mean ---
    # Handle NAs: replace with 0 for summation, track non-NA counts
    x_nona <- x_num
    x_nona[is.na(x_nona)] <- 0
    not_na <- as.numeric(!is.na(x_num))
    
    neighbor_sum   <- as.numeric(W %*% x_nona)
    neighbor_count <- as.numeric(W %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- Neighbor max and min ---
    # We need to iterate over the sparse structure.
    # Extract the triplet form of W.
    W_T <- as(W, "TsparseMatrix")  # or dgTMatrix
    wi <- W_T@i + 1L   # 0-based -> 1-based
    wj <- W_T@j + 1L
    
    # Get neighbor values for every edge
    edge_vals <- x_num[wj]
    
    # We need max and min grouped by row (wi), ignoring NAs
    edge_dt <- data.table(row = wi, val = edge_vals)
    
    # Remove edges where the neighbor value is NA
    edge_dt <- edge_dt[!is.na(val)]
    
    # Compute grouped max and min
    agg <- edge_dt[, .(nmax = max(val), nmin = min(val)), by = row]
    
    # Initialize result vectors
    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)
    
    neighbor_max[agg$row] <- agg$nmax
    neighbor_min[agg$row] <- agg$nmin
    
    # --- Assign to data.table ---
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)
    
    dt[, (max_col)  := neighbor_max]
    dt[, (min_col)  := neighbor_min]
    dt[, (mean_col) := neighbor_mean]
    
    cat(sprintf("  Done: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  # Convert back to data.frame if the rest of the pipeline expects one
  return(as.data.frame(dt))
}

# ============================================================
# 3. Main execution (replaces the outer loop)
# ============================================================

# --- Build sparse adjacency matrix (once) ---
cat("Building sparse neighbor matrix...\n")
W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
cat("Sparse matrix built.\n")
cat(sprintf("Dimensions: %d x %d, non-zeros: %d\n", nrow(W), ncol(W), nnzero(W)))

# --- Compute all neighbor features ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_sparse(cell_data, W, neighbor_source_vars)

# --- Clean up the large sparse matrix ---
rm(W)
gc()

# The trained Random Forest model is unchanged.
# cell_data now has the same neighbor feature columns (n_max_*, n_min_*, n_mean_*)
# with identical numerical values, ready for predict().
```

---

## Why This Works and Performance Estimates

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste` + named lookup | One sparse matrix construction via vectorized `data.table` joins per year (28 iterations) |
| `compute_neighbor_stats` (mean) | 6.46M `lapply` iterations Ã— 5 vars | One sparse matrix-vector multiply (`W %*% x`) per variable â€” runs in C via `Matrix` package |
| `compute_neighbor_stats` (max/min) | Same `lapply` loop | One `data.table` grouped aggregation over ~38.5M edges per variable |
| **Estimated runtime** | **86+ hours** | **10â€“30 minutes** |
| **Peak memory** | Multiple GB in list overhead | ~600 MB for sparse matrix + ~1.5 GB for data â‰ˆ fits in 16 GB |

### Key points:

- **Numerical equivalence is preserved.** The sparse matrix encodes exactly the same neighbor relationships; `W %*% x` computes the same sums; grouped `max`/`min` on the edge list produces identical results.
- **The trained Random Forest model is untouched.** Only the feature-computation step is optimized.
- **The sparse matrix `W` is built once and reused** across all 5 variables, amortizing the construction cost.
- **`data.table` keyed joins** replace millions of `paste` + named-vector lookups with hash-based binary-search joins that run in seconds per year.
- **The `Matrix` package sparse multiply** runs in optimized C/BLAS, turning 6.46M R-level iterations into a single vectorized call per variable.
END_RESPONSE>>>
