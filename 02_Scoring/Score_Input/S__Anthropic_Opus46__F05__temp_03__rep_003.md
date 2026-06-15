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
run_id: Anthropic_F05_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is built once, but then `paste()` is called **per row** inside `lapply` over 6.46M rows. Each call constructs character keys and performs named-vector lookups (which are hash-table lookups on character strings). That's ~6.46M Ã— avg_neighbors string constructions and hash probes â€” billions of character operations.

### Broader Structural Problem
The deeper issue is that the entire approach is **row-wise and character-keyed** when the data has a natural **integer-indexed structure**. Because every cell appears in every year (a balanced panel), the neighbor relationships are **time-invariant**: cell `i`'s neighbors in year `t` are the same cells as in year `t+1`. The string-key lookup is re-discovering, for every single row, a spatial relationship that is constant across all 28 years. Furthermore, `compute_neighbor_stats` is also row-wise via `lapply` over 6.46M rows, repeated 5 times (once per variable).

### Root Cause Summary
| Layer | Problem |
|---|---|
| Key construction | `paste()` on 6.46M rows Ã— avg_neighbors per row |
| Lookup mechanism | Character hash lookup instead of integer indexing |
| Redundant recomputation | Spatial neighbor mapping recomputed for every year when it's time-invariant |
| Stats computation | Row-wise `lapply` over 6.46M rows Ã— 5 variables |
| Overall | An O(R Ã— N_avg) character-operation algorithm where an O(C Ã— N_avg Ã— Y) integer-matrix algorithm suffices, and the latter is vectorizable |

## Optimization Strategy

1. **Exploit the balanced panel**: Sort data by `(id, year)`. Since every cell appears for all 28 years, cell `c`'s rows are a contiguous block of 28 rows at positions `((c-1)*28 + 1)` through `(c*28)`. Neighbor lookup becomes pure integer arithmetic â€” no strings, no hashing.

2. **Build a sparse neighbor matrix once** (cell-level, not row-level): Convert the `nb` object to a sparse adjacency matrix or a simple integer list indexed by cell position.

3. **Vectorized stats via matrix operations**: For each variable, reshape the 6.46M-length vector into a 344,208 Ã— 28 matrix (cells Ã— years). Then use the sparse adjacency structure to compute neighbor max/min/mean as matrix operations across cells, broadcasting across all years simultaneously.

4. **Memory-efficient**: A 344,208 Ã— 28 numeric matrix is ~77 MB. The sparse neighbor structure is small. Total peak overhead is well under 1 GB.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure data is a data.table sorted by (id, year)
# ==============================================================================
cell_data <- as.data.table(cell_data)
setorder(cell_data, id, year)

# Verify balanced panel
years <- sort(unique(cell_data$year))
n_years <- length(years)  # 28
n_cells <- length(unique(cell_data$id))  # 344,208
stopifnot(nrow(cell_data) == n_cells * n_years)

# ==============================================================================
# STEP 1: Build integer cell-index mapping and neighbor structure
# ==============================================================================
# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# (the same id_order used in the original code)

# Map each cell ID to its positional index in id_order (1-based)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Map each cell ID to its block-start row in the sorted cell_data
# Because cell_data is sorted by (id, year), cell at position p in id_order
# may not be at block p â€” we need to map via the actual sorted order.
sorted_ids <- unique(cell_data$id)  # unique preserves first-appearance order in sorted DT
cell_id_to_block <- setNames(seq_along(sorted_ids), as.character(sorted_ids))

# Build neighbor list in terms of "block index" in sorted cell_data
# rook_neighbors_unique[[k]] gives neighbor indices into id_order for cell id_order[k]
n_nb <- length(rook_neighbors_unique)
neighbor_block_list <- vector("list", n_cells)

# For each cell in sorted_ids, find its neighbors' block indices
for (b in seq_len(n_cells)) {

  cid <- sorted_ids[b]
  pos_in_id_order <- id_to_pos[as.character(cid)]
  if (is.na(pos_in_id_order)) {
    neighbor_block_list[[b]] <- integer(0)
    next
  }
  nb_positions <- rook_neighbors_unique[[pos_in_id_order]]
  if (length(nb_positions) == 0) {
    neighbor_block_list[[b]] <- integer(0)
    next
  }
  nb_cell_ids <- id_order[nb_positions]
  nb_blocks <- cell_id_to_block[as.character(nb_cell_ids)]
  neighbor_block_list[[b]] <- as.integer(nb_blocks[!is.na(nb_blocks)])
}

# ==============================================================================
# STEP 2: Build sparse adjacency matrix (n_cells x n_cells)
# ==============================================================================
# Row i has 1s in columns corresponding to neighbors of cell i
i_idx <- rep(seq_len(n_cells), lengths(neighbor_block_list))
j_idx <- unlist(neighbor_block_list)
adj <- sparseMatrix(
  i = i_idx, j = j_idx,
  x = 1, dims = c(n_cells, n_cells)
)
# Number of neighbors per cell (for computing means)
n_neighbors <- as.numeric(adj %*% rep(1, n_cells))  # length n_cells

# ==============================================================================
# STEP 3: Compute neighbor stats for each variable â€” fully vectorized
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  vals <- cell_data[[var_name]]
  
  # Reshape to n_cells x n_years matrix (row = cell block, col = year)
  val_mat <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = TRUE)
  
  # --- NEIGHBOR MEAN (vectorized via sparse matrix multiply) ---
  # For each year column, sum neighbor values then divide by neighbor count
  # adj %*% val_mat gives (n_cells x n_years) where entry [i,t] = sum of
  # neighbor values for cell i in year t
  
  # Handle NAs: replace NA with 0 for sum, track counts of non-NA neighbors
  val_mat_nona <- val_mat
  val_mat_nona[is.na(val_mat_nona)] <- 0
  is_valid <- !is.na(val_mat)  # logical matrix
  valid_num <- matrix(as.numeric(is_valid), nrow = n_cells, ncol = n_years)
  
  neighbor_sum  <- as.matrix(adj %*% val_mat_nona)   # n_cells x n_years
  neighbor_cnt  <- as.matrix(adj %*% valid_num)       # n_cells x n_years
  
  neighbor_mean <- neighbor_sum / neighbor_cnt
  neighbor_mean[neighbor_cnt == 0] <- NA
  
  # --- NEIGHBOR MAX and MIN (loop over cells, vectorized over years) ---
  # For max/min we cannot use simple matrix multiply. We use the neighbor list
  # but operate on whole year-vectors at once (28 elements), not per row.
  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (b in seq_len(n_cells)) {
    nb <- neighbor_block_list[[b]]
    if (length(nb) == 0L) next
    # sub_mat: length(nb) x n_years
    sub_mat <- val_mat[nb, , drop = FALSE]
    if (length(nb) == 1L) {
      neighbor_max_mat[b, ] <- sub_mat[1, ]
      neighbor_min_mat[b, ] <- sub_mat[1, ]
    } else {
      # Column-wise max/min ignoring NAs
      neighbor_max_mat[b, ] <- apply(sub_mat, 2, max, na.rm = TRUE)
      neighbor_min_mat[b, ] <- apply(sub_mat, 2, min, na.rm = TRUE)
      # apply with na.rm=TRUE on all-NA columns gives Inf/-Inf; fix:
    }
    # Fix all-NA columns
    all_na_cols <- (neighbor_cnt[b, ] == 0)
    if (any(all_na_cols)) {
      neighbor_max_mat[b, all_na_cols] <- NA
      neighbor_min_mat[b, all_na_cols] <- NA
    }
  }
  # Fix Inf/-Inf from max/min on all-NA
  neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA
  neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA
  
  # --- Flatten back to column vectors (row-major to match cell_data order) ---
  max_col_name  <- paste0(var_name, "_neighbor_max")
  min_col_name  <- paste0(var_name, "_neighbor_min")
  mean_col_name <- paste0(var_name, "_neighbor_mean")
  
  cell_data[, (max_col_name)  := as.vector(t(neighbor_max_mat))]
  cell_data[, (min_col_name)  := as.vector(t(neighbor_min_mat))]
  cell_data[, (mean_col_name) := as.vector(t(neighbor_mean))]
}
```

### Further Optimization: Eliminate the Cell-Loop for Max/Min

The cell-loop for max/min (344K iterations) is the remaining bottleneck. We can replace it with a **grouped operation on the sparse edge list**:

```r
# ==============================================================================
# STEP 3 (ALTERNATIVE): Fully vectorized max/min/mean using edge-list approach
# ==============================================================================
# Build edge list once
edges <- summary(adj)  # gives i, j, x columns for non-zero entries
# edges$i = focal cell block index, edges$j = neighbor cell block index
# ~1.37M edges

edge_i <- edges$i
edge_j <- edges$j
n_edges <- length(edge_i)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  vals <- cell_data[[var_name]]
  val_mat <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = TRUE)
  
  # For each edge, pull the neighbor's value for all 28 years
  # Result: n_edges x n_years matrix
  neighbor_val_mat <- val_mat[edge_j, , drop = FALSE]  # n_edges x 28
  
  # Group by focal cell (edge_i) to compute max, min, mean per year
  # Use data.table for fast grouped aggregation
  
  # Melt to long form: edge_index x year
  edge_dt <- data.table(
    focal = rep(edge_i, n_years),
    year_idx = rep(seq_len(n_years), each = n_edges),
    val = as.vector(neighbor_val_mat)  # column-major: all edges for year 1, then year 2, ...
  )
  
  # Remove NA values before aggregation
  edge_dt <- edge_dt[!is.na(val)]
  
  # Aggregate
  agg <- edge_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(focal, year_idx)]
  
  # Create full grid to handle cells/years with no valid neighbors (-> NA)
  full_grid <- CJ(focal = seq_len(n_cells), year_idx = seq_len(n_years))
  agg <- agg[full_grid, on = .(focal, year_idx)]
  setorder(agg, focal, year_idx)
  
  # Assign to cell_data (which is sorted by id, year â€” same order)
  max_col_name  <- paste0(var_name, "_neighbor_max")
  min_col_name  <- paste0(var_name, "_neighbor_min")
  mean_col_name <- paste0(var_name, "_neighbor_mean")
  
  cell_data[, (max_col_name)  := agg$nb_max]
  cell_data[, (min_col_name)  := agg$nb_min]
  cell_data[, (mean_col_name) := agg$nb_mean]
  
  # Free memory
  rm(edge_dt, agg, neighbor_val_mat, full_grid)
  gc()
}
```

## Performance Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `paste()` + char hash lookups | One-time integer index build |
| Stats: mean | 6.46M `lapply` iterations Ã— 5 vars | Sparse matrix multiply (seconds) |
| Stats: max/min | 6.46M `lapply` iterations Ã— 5 vars | data.table grouped agg on 1.37M edges Ã— 28 years |
| String operations | ~billions | **Zero** at runtime |
| Estimated time | 86+ hours | **~5â€“15 minutes** |
| Peak RAM | Moderate (but slow) | ~2â€“3 GB (well within 16 GB) |
| Numerical output | max, min, mean per cell-year | **Identical** max, min, mean per cell-year |

## Why the Trained Random Forest Is Preserved

The code above only reconstructs the **input features** (neighbor max/min/mean columns) using the same mathematical definitions. The values are numerically identical to the originals. The trained Random Forest model object is never touched â€” it can be applied to the resulting `cell_data` with `predict(rf_model, cell_data)` exactly as before.
END_RESPONSE>>>
