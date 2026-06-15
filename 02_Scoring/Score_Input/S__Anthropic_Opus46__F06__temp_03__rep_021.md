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
run_id: Anthropic_F06_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(id, year)`) for each neighbor in that year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) against a 6.46M-element named character vector.

Named vector lookup in R is **hash-based but slow at scale** â€” doing ~6.46M lookups, each touching ~4 neighbors on average, against a 6.46M-length named vector is extremely expensive. The `paste`-based key construction also allocates millions of small character vectors.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows Ã— 5 variables

Even after the lookup is built, `compute_neighbor_stats` runs an `lapply` over 6.46M rows for **each** of the 5 variables, performing subsetting and `max/min/mean` in pure R loops. That's ~32.3M R-level function calls total.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a **complete regular grid** with uniform rectangular neighborhoods. Here:
- The panel is irregular in the sense that not all cells exist in all years (potential NA/missing cells).
- The neighbor structure is precomputed as an `nb` object (not necessarily a perfect rectangular grid after subsetting).
- We must **preserve the exact numerical results** of the rook-neighbor computation.

However, the **conceptual analogy is valid**: we are computing focal statistics (max, min, mean) over a spatial neighborhood. The optimization should use **vectorized sparse-matrix operations** that mimic focal computations without requiring a complete raster grid.

---

## 2. Optimization Strategy

### Strategy: Sparse Adjacency Matrix + Vectorized Column Operations

1. **Replace `build_neighbor_lookup`** with a sparse matrix `W` of dimension `(N_rows Ã— N_rows)` where `N_rows = nrow(cell_data)` (~6.46M). Entry `W[i,j] = 1` if row `j` is a rook neighbor of row `i` in the same year. This matrix is extremely sparse (~4 nonzero entries per row on average).

2. **Replace `compute_neighbor_stats`** with vectorized operations:
   - **Mean**: `W %*% x / W %*% ones` (sparse matrix-vector multiply â€” blazing fast via `Matrix` package).
   - **Max and Min**: Use a grouped operation via `data.table` keyed joins, or iterate over the sparse matrix in C++ via `Rcpp`, or use the sparse structure directly.

3. **Build the sparse matrix efficiently** using `data.table` for the join (cell ID + year), avoiding `paste` keys entirely.

### Expected speedup

| Step | Current | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~hours (paste + named vector) | ~1-2 min (data.table join + sparse matrix) | ~60-100Ã— |
| Neighbor stats (mean) | ~hours (lapply Ã— 5 vars) | ~seconds (sparse mat-vec) | ~1000Ã— |
| Neighbor stats (max/min) | ~hours (lapply Ã— 5 vars) | ~2-5 min (Rcpp or grouped ops) | ~50-100Ã— |
| **Total** | **86+ hours** | **~5-15 minutes** | **~350-1000Ã—** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Dependencies: data.table, Matrix, Rcpp (optional but recommended for max/min)
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build sparse adjacency matrix (cell-year level) ----------------

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year

  # id_order: vector of cell IDs in the order matching rook_neighbors (nb object)
  # rook_neighbors: nb object (list of integer index vectors into id_order)
  
  n_cells <- length(id_order)
  
  # --- 1a. Build spatial edge list (cell-level) ---
  # Each entry in rook_neighbors[[i]] gives indices into id_order
  from_cell <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  to_cell   <- unlist(rook_neighbors, use.names = FALSE)
  
  # Remove zero-neighbor entries (spdep uses integer(0) or 0 for no neighbors)
  valid <- to_cell > 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]
  
  # Convert to actual cell IDs
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]
  
  # Spatial edge table
  edges_spatial <- data.table(from_id = from_id, to_id = to_id)
  
  # --- 1b. Expand to cell-year level via join ---
  # Create a data.table with row indices
  dt <- data.table(
    row_idx = seq_len(nrow(cell_data)),
    id      = cell_data$id,
    year    = cell_data$year
  )
  
  # Join: for each spatial edge (from_id, to_id), find all years where BOTH exist
  # First, join edges with "from" rows
  setkey(dt, id, year)
  
  # from side
  edges_from <- edges_spatial[dt, on = .(from_id = id), 
                               .(to_id, year, from_row = row_idx),
                               nomatch = 0L, allow.cartesian = TRUE]
  
  # to side: join to get the row index of the neighbor in the same year
  edges_full <- edges_from[dt, on = .(to_id = id, year = year),
                            .(from_row, to_row = i.row_idx),
                            nomatch = 0L]
  
  # --- 1c. Build sparse matrix ---
  n <- nrow(cell_data)
  W <- sparseMatrix(
    i = edges_full$from_row,
    j = edges_full$to_row,
    x = 1,
    dims = c(n, n)
  )
  
  return(W)
}

# ---- Step 2: Compute neighbor stats using sparse matrix ---------------------

compute_neighbor_features_sparse <- function(cell_data, W, var_name) {
  # Extracts max, min, mean of var_name across rook neighbors
  # W: sparse adjacency matrix (row i has 1s in columns that are i's neighbors)
  
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- Handle NAs: we need to exclude NA neighbors ---
  not_na <- as.numeric(!is.na(x))
  x_safe <- x
  x_safe[is.na(x_safe)] <- 0  # zero out NAs for summation
  
  # Number of non-NA neighbors per row
  neighbor_count <- as.numeric(W %*% not_na)
  
  # Sum of non-NA neighbor values
  neighbor_sum <- as.numeric(W %*% x_safe)
  
  # Mean
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # --- Max and Min: requires iterating over sparse structure ---
  # We use the sparse matrix column pointers directly for efficiency
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  # Convert W to dgTMatrix (triplet form) for easy iteration by row
  # But iterating 6.46M rows in R is slow. Better: use dgCMatrix and Rcpp,
  # or use a data.table grouped approach.
  
  # data.table approach (fast grouped max/min):
  Wt <- as(W, "TsparseMatrix")  # triplet: i, j, x (0-indexed)
  
  edge_dt <- data.table(
    from_row = Wt@i + 1L,   # convert to 1-indexed
    to_row   = Wt@j + 1L
  )
  
  # Attach the variable values of the neighbor (to_row)
  edge_dt[, val := x[to_row]]
  
  # Remove edges where neighbor value is NA
  edge_dt <- edge_dt[!is.na(val)]
  
  # Grouped max and min
  stats <- edge_dt[, .(nb_max = max(val), nb_min = min(val)), by = from_row]
  
  neighbor_max[stats$from_row] <- stats$nb_max
  neighbor_min[stats$from_row] <- stats$nb_min
  
  # --- Assemble output names ---
  prefix <- paste0("nb_", var_name, "_")
  cell_data[[paste0(prefix, "max")]]  <- neighbor_max
  cell_data[[paste0(prefix, "min")]]  <- neighbor_min
  cell_data[[paste0(prefix, "mean")]] <- neighbor_mean
  
  return(cell_data)
}

# ---- Step 3: Main pipeline -------------------------------------------------

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cat("Building sparse neighbor matrix...\n")
  t0 <- Sys.time()
  W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
  cat("  Matrix dimensions:", nrow(W), "x", ncol(W), "\n")
  cat("  Non-zero entries:", nnzero(W), "\n")
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "...\n")
    t1 <- Sys.time()
    cell_data <- compute_neighbor_features_sparse(cell_data, W, var_name)
    cat("  Done in", round(difftime(Sys.time(), t1, units = "mins"), 2), "min\n")
  }
  
  return(cell_data)
}

# ---- Execute ----------------------------------------------------------------
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# Now proceed with prediction using the pre-trained Random Forest model
# (model is NOT retrained â€” only predict)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Property | Original | Optimized | Match? |
|---|---|---|---|
| Neighbor definition | Rook contiguity from `nb` object | Same `nb` object, same edges | âœ… |
| Year matching | Neighbors matched within same year | Same join logic (`on = year`) | âœ… |
| NA handling | Excluded before `max/min/mean` | Excluded identically | âœ… |
| `max` computation | `max(neighbor_vals)` | `data.table` grouped `max(val)` | âœ… Identical |
| `min` computation | `min(neighbor_vals)` | `data.table` grouped `min(val)` | âœ… Identical |
| `mean` computation | `mean(neighbor_vals)` | `sum / count` (NA-excluded) | âœ… Identical |
| No-neighbor rows | Returns `c(NA, NA, NA)` | Defaults to `NA` | âœ… |
| RF model | Not retrained | Not retrained | âœ… |

The sparse matrix `W` encodes **exactly** the same directed neighbor relationships as the original `neighbor_lookup` list. The `data.table` grouped operations and sparse matrix-vector products compute **algebraically identical** results to the original `max`, `min`, and `mean` calls â€” just vectorized instead of row-by-row in R.

---

## 5. Memory Estimate

| Object | Size |
|---|---|
| Sparse matrix W (6.46M Ã— 6.46M, ~38.5M nonzeros) | ~600 MB |
| Triplet edge `data.table` (~38.5M rows Ã— 3 cols) | ~900 MB |
| `cell_data` (~6.46M Ã— 125 cols) | ~6-8 GB |
| **Total peak** | **~10-12 GB** (fits in 16 GB) |

If memory is tight, process variables one at a time and drop the triplet `edge_dt` between iterations (it's recreated each call but could be cached). Alternatively, cache `edge_dt` once and reuse:

```r
# Memory-optimized variant: build edge_dt once, reuse for all variables
Wt <- as(W, "TsparseMatrix")
edge_dt_base <- data.table(
  from_row = Wt@i + 1L,
  to_row   = Wt@j + 1L
)
rm(Wt); gc()

for (var_name in neighbor_source_vars) {
  x <- cell_data[[var_name]]
  not_na <- !is.na(x)
  x_safe <- replace(x, !not_na, 0)
  
  # Mean via sparse mat-vec
  nb_count <- as.numeric(W %*% as.numeric(not_na))
  nb_sum   <- as.numeric(W %*% x_safe)
  cell_data[[paste0("nb_", var_name, "_mean")]] <- 
    ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
  
  # Max/Min via data.table
  ed <- copy(edge_dt_base)
  ed[, val := x[to_row]]
  ed <- ed[!is.na(val)]
  stats <- ed[, .(nb_max = max(val), nb_min = min(val)), by = from_row]
  
  nb_max <- rep(NA_real_, length(x))
  nb_min <- rep(NA_real_, length(x))
  nb_max[stats$from_row] <- stats$nb_max
  nb_min[stats$from_row] <- stats$nb_min
  
  cell_data[[paste0("nb_", var_name, "_max")]] <- nb_max
  cell_data[[paste0("nb_", var_name, "_min")]] <- nb_min
  
  rm(ed, stats); gc()
}
```

This should complete the entire neighbor feature computation in **approximately 5â€“15 minutes** on a standard 16 GB laptop, down from the estimated 86+ hours.
END_RESPONSE>>>
