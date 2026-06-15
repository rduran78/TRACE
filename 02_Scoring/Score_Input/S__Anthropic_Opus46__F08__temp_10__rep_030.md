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
run_id: Anthropic_F08_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per cell-year row (~6.46M entries).** Each row's neighbor indices are recomputed by pasting `id` and `year` into string keys and performing hash lookups. This produces a list of ~6.46 million integer vectors, when the underlying neighbor *structure* is identical across all 28 years. The topology (which cell neighbors which cell) never changes â€” only the *values* attached to cells change year by year.

2. **String key construction is O(N) where N = 6.46M.** The `paste(id, year, sep="_")` and named-vector lookup pattern is extremely expensive in R at this scale. For every row `i`, it reconstructs neighbor keys by pasting cell IDs with `data$year[i]`, then indexes into a 6.46M-length named vector.

3. **`compute_neighbor_stats` iterates over 6.46M list elements with `lapply`.** For each of 5 variables, it walks through 6.46M entries â€” that's ~32.3M R-level loop iterations total, each involving subsetting, `is.na` filtering, and three summary statistics.

4. **The lookup list itself consumes enormous memory.** ~6.46M list elements, each containing ~4 integer neighbor indices (rook), creates massive overhead from R's list/vector object headers (~6.46M SEXP headers).

### The Key Insight

The neighbor relationship is **cell-to-cell**, not **row-to-row**. There are only 344,208 cells with ~1.37M directed neighbor relationships. This topology is static. The current code re-expresses this 28 times (once per year), inflating a 344K-element problem into a 6.46M-element problem.

---

## Optimization Strategy

**Separate the static topology from the dynamic computation:**

| Aspect | Current | Redesigned |
|---|---|---|
| Neighbor lookup granularity | Per cell-year row (6.46M) | Per cell (344K) â€” built once |
| Lookup key mechanism | String paste + named vector | Integer index into cell-ordered matrix |
| Stats computation | R-level lapply over 6.46M | Vectorized sparse-matrix multiplication over 344K cells Ã— 28 years |
| Time complexity | O(6.46M Ã— 5 vars Ã— string ops) | O(344K Ã— avg_neighbors Ã— 28 Ã— 5) via matrix ops |
| Expected speedup | Baseline (~86h) | **~minutes** |

### Specific steps:

1. **Build a sparse adjacency matrix `W` once** from `rook_neighbors_unique` (344,208 Ã— 344,208). This encodes topology as a sparse matrix (~1.37M nonzero entries).

2. **Reshape each variable into a cell Ã— year matrix** (344,208 rows Ã— 28 columns), ordered by a canonical cell-ID ordering.

3. **Compute neighbor stats via sparse matrix operations:**
   - **Neighbor mean:** `W %*% X / degree` (where `degree` = number of non-NA neighbors per cell).
   - **Neighbor max/min:** Use row-wise sparse operations with an efficient grouped-max/min strategy.

4. **Reshape results back** into the long cell-year format and attach to `cell_data`.

5. **Feed into the pre-trained Random Forest** â€” column names and numerical values are preserved exactly.

---

## Working R Code

```r
library(Matrix)   # for sparse matrices
library(data.table)

#' Redesigned pipeline: separate static topology from dynamic values
#' Preserves original numerical estimand and trained RF model.

# ==============================================================
# STEP 1: Build sparse adjacency matrix ONCE from nb object
# ==============================================================

build_adjacency_matrix <- function(nb_obj) {
  # nb_obj is a list of length n_cells; nb_obj[[i]] gives integer

  # indices of neighbors of cell i (1-based, 0 means no neighbors).
  n <- length(nb_obj)
  
  # Pre-allocate vectors for triplet construction
  from <- vector("integer", 0)
  to   <- vector("integer", 0)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # spdep uses 0 for no-neighbor sentinel
    if (length(nbrs) > 0L) {
      from <- c(from, rep.int(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }
  
  W <- sparseMatrix(
    i = from, j = to, x = 1,
    dims = c(n, n), giveCsparse = TRUE
  )
  return(W)
}

# ==============================================================
# STEP 2: Fast neighbor max / min / mean via sparse operations
# ==============================================================

compute_neighbor_stats_sparse <- function(W, x_vec, n_cells) {
  # x_vec: numeric vector of length n_cells (one year's values for one var)
  # W:     n_cells x n_cells sparse adjacency (binary)
  # Returns: data.frame with columns neighbor_max, neighbor_min, neighbor_mean
  
  # --- Neighbor mean (and sum) via sparse matrix-vector multiply ---
  # Replace NA with 0 for summation; track valid counts separately
  x_valid   <- !is.na(x_vec)
  x_zero_na <- ifelse(x_valid, x_vec, 0)
  
  neighbor_sum   <- as.numeric(W %*% x_zero_na)
  neighbor_count <- as.numeric(W %*% as.numeric(x_valid))
  
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # --- Neighbor max and min via row-wise sparse iteration ---
  # Use the sparse structure of W to index into x_vec efficiently
  # W is dgCMatrix (compressed sparse column). Convert to dgRMatrix 

  # for efficient row access, or iterate via column-format.
  
  # Efficient approach: use the triplet representation
  Wt <- as(W, "TsparseMatrix")  # gives @i (0-based row), @j (0-based col)
  
  # Build a data.table for grouped max/min
  dt_edges <- data.table(
    row_idx = Wt@i + 1L,    # 1-based
    val     = x_vec[Wt@j + 1L]  # neighbor's value
  )
  
  # Remove edges where neighbor value is NA
  dt_edges <- dt_edges[!is.na(val)]
  
  # Grouped max and min
  if (nrow(dt_edges) > 0) {
    stats <- dt_edges[, .(nmax = max(val), nmin = min(val)), by = row_idx]
    
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[stats$row_idx] <- stats$nmax
    neighbor_min[stats$row_idx] <- stats$nmin
  } else {
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
  }
  
  data.table(
    neighbor_max  = neighbor_max,
    neighbor_min  = neighbor_min,
    neighbor_mean = neighbor_mean
  )
}

# ==============================================================
# STEP 3: Main pipeline â€” process all vars Ã— all years
# ==============================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # cell_data: data.frame/data.table with columns id, year, and the source vars
  # id_order: vector of cell IDs in the same order as rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object
  # neighbor_source_vars: character vector of variable names
  
  cat("Converting cell_data to data.table...\n")
  cell_data <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))
  
  # --- STEP 1: Build adjacency matrix ONCE (static topology) ---
  cat("Building sparse adjacency matrix (once)...\n")
  t0 <- Sys.time()
  W <- build_adjacency_matrix(rook_neighbors_unique)
  cat(sprintf("  Adjacency matrix: %d x %d, %d nonzeros. Time: %.1fs\n",
              nrow(W), ncol(W), nnzero(W), as.numeric(Sys.time() - t0, units = "secs")))
  
  # Pre-compute triplet form once for max/min operations
  cat("Converting to triplet form for max/min...\n")
  Wt <- as(W, "TsparseMatrix")
  Wt_row <- Wt@i + 1L
  Wt_col <- Wt@j + 1L
  rm(Wt)
  
  # --- STEP 2: Create canonical ordering: map cell id -> position 1..n_cells ---
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # --- STEP 3: Create row index for fast (cell_pos, year) -> data row mapping ---
  cat("Building cell-position and row index...\n")
  cell_data[, cell_pos := id_to_pos[as.character(id)]]
  
  # Sort to ensure consistent ordering
  setkey(cell_data, year, cell_pos)
  
  # --- STEP 4: Loop over variables (5) and years (28) ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    t_var <- Sys.time()
    
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Pre-allocate result columns
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
    
    for (yr in years) {
      # Extract this year's slice in canonical cell order
      yr_rows <- cell_data[year == yr]
      
      # Build a vector of length n_cells in canonical position order
      x_vec <- rep(NA_real_, n_cells)
      x_vec[yr_rows$cell_pos] <- yr_rows[[var_name]]
      
      # --- Neighbor MEAN via sparse mat-vec (vectorized) ---
      x_valid    <- !is.na(x_vec)
      x_zero_na  <- ifelse(x_valid, x_vec, 0)
      n_sum      <- as.numeric(W %*% x_zero_na)
      n_count    <- as.numeric(W %*% as.numeric(x_valid))
      n_mean     <- ifelse(n_count > 0, n_sum / n_count, NA_real_)
      
      # --- Neighbor MAX and MIN via data.table grouped ops ---
      neighbor_vals <- x_vec[Wt_col]  # values at neighbor positions
      valid_mask    <- !is.na(neighbor_vals)
      
      if (any(valid_mask)) {
        dt_e <- data.table(
          row_idx = Wt_row[valid_mask],
          val     = neighbor_vals[valid_mask]
        )
        stats <- dt_e[, .(nmax = max(val), nmin = min(val)), by = row_idx]
        
        n_max <- rep(NA_real_, n_cells)
        n_min <- rep(NA_real_, n_cells)
        n_max[stats$row_idx] <- stats$nmax
        n_min[stats$row_idx] <- stats$nmin
      } else {
        n_max <- rep(NA_real_, n_cells)
        n_min <- rep(NA_real_, n_cells)
      }
      
      # Write results back: match on year and cell_pos
      idx <- cell_data[year == yr, which = TRUE]
      pos <- cell_data$cell_pos[idx]
      
      set(cell_data, i = idx, j = col_max,  value = n_max[pos])
      set(cell_data, i = idx, j = col_min,  value = n_min[pos])
      set(cell_data, i = idx, j = col_mean, value = n_mean[pos])
    }
    
    cat(sprintf("  %s done in %.1fs\n", var_name,
                as.numeric(Sys.time() - t_var, units = "secs")))
  }
  
  # Clean up helper column
  cell_data[, cell_pos := NULL]
  
  return(cell_data)
}

# ==============================================================
# STEP 4: Execute and predict with existing RF model
# ==============================================================

# --- Run the optimized pipeline ---
cell_data <- run_neighbor_feature_pipeline(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# --- Predict with the pre-trained Random Forest (unchanged) ---
# The column names and numerical values are identical to the original
# implementation, so the trained model is fully compatible.
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Neighbor mean** | Computed as `sum(neighbor_values) / count(non_NA_neighbors)` â€” identical to `mean(neighbor_vals)` after NA removal in the original code. |
| **Neighbor max / min** | Computed via `max(val)` / `min(val)` grouped by row â€” identical to the original `max(neighbor_vals)` / `min(neighbor_vals)` after NA removal. |
| **Column naming** | `neighbor_max_{var}`, `neighbor_min_{var}`, `neighbor_mean_{var}` â€” matches whatever `compute_and_add_neighbor_features` produced (adjust prefix/suffix if your RF model expects a different naming convention). |
| **NA handling** | Cells with zero valid neighbors get `NA` for all three stats â€” identical to the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`. |
| **RF model** | Never retrained. Only `predict()` is called on identically-structured features. |

## Expected Performance

| Stage | Operations | Estimated Time |
|---|---|---|
| Adjacency matrix build | 344K cells, ~1.37M edges | ~2â€“5 seconds |
| Triplet extraction | One-time conversion | ~1 second |
| Per variable Ã— year (sparse mat-vec + grouped max/min) | 5 Ã— 28 = 140 iterations, each ~344K cells | ~3â€“8 seconds each |
| **Total** | | **~10â€“20 minutes** |

This represents a **~250â€“500Ã— speedup** over the original 86+ hour estimate, achieved entirely by honoring the static-versus-changing distinction: build the topology once, apply it per-year via vectorized sparse operations.
END_RESPONSE>>>
