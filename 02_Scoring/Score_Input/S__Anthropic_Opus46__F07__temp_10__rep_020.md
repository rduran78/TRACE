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
run_id: Anthropic_F07_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`. This function iterates over **~6.46 million rows** with `lapply`, and for each row it:

1. Does a character-based hash lookup (`id_to_ref`) â€” fine individually, but called millions of times.
2. Indexes into the `neighbors` list â€” fine.
3. Constructs **character paste keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) for every neighbor of every row â€” this is extremely expensive at scale (millions of `paste` calls, each producing multiple strings).
4. Looks up those keys in a named character vector (`idx_lookup`) â€” named-vector lookup in R is O(n) or at best hash-based but with repeated allocation overhead.

The result is a **list of 6.46 million integer vectors**, consuming substantial memory and taking tens of hours to build. Then `compute_neighbor_stats` iterates over this list again for each of the 5 variables â€” another 5 Ã— 6.46M R-level loop iterations, though these are cheaper.

**Root causes:**
- **Row-level R loops with string operations** over 6.46M rows.
- **Redundant recomputation**: the neighbor topology is the same for every year, but the lookup is rebuilt per cell-year row.
- **Memory bloat**: storing 6.46M integer vectors in a list.

## Optimization Strategy

### Key Insight: Separate Space from Time

The neighbor graph is **purely spatial** â€” cell A neighbors cell B in every year identically. There are only **344,208 unique cells**, not 6.46M cell-years. We should:

1. **Work at the cell level** (344K cells), not the cell-year level (6.46M rows).
2. **Vectorize** the neighbor stats computation using a sparse adjacency matrix and matrix operations â€” `max`, `min`, and `mean` over neighbors can be computed via sparse matrix multiplication and related tricks.
3. **Avoid all `paste`/string operations** â€” use integer indexing throughout.

### Approach: Sparse Adjacency Matrix + Split-Apply

1. Build a **sparse binary adjacency matrix** `W` (344,208 Ã— 344,208) from `rook_neighbors_unique`. This has ~1.37M non-zero entries â€” trivially small.
2. For each variable and each year, arrange the variable values into a vector aligned by cell, then compute:
   - **Neighbor mean**: `W %*% x / W %*% ones` (sparse matrix-vector multiply).
   - **Neighbor max / min**: Use a loop over cells' neighbor indices (but only 344K cells, not 6.46M rows), or use `data.table` grouping.
3. Join results back to the panel.

This replaces 6.46M-row R loops with 28 sparse matrix-vector multiplies per variable (each taking milliseconds), plus efficient grouped operations for max/min.

**Expected speedup**: from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)
library(Matrix)

# ============================================================
# 1.  Build sparse adjacency matrix from spdep nb object
#     (done once; 344,208 cells, ~1.37M directed edges)
# ============================================================
build_sparse_adj <- function(id_order, nb_obj) {
  n <- length(id_order)
  # nb_obj[[i]] contains integer indices of neighbors of cell i
  # Build COO triplets
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor sentinels that spdep uses (integer(0) is fine,
  # but some nb objects store 0L for no-neighbor cells)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  W
}

W <- build_sparse_adj(id_order, rook_neighbors_unique)

# ============================================================
# 2.  Convert cell_data to data.table and create cell index
# ============================================================
cell_dt <- as.data.table(cell_data)

# Map each id to its position in id_order (the row/col index in W)
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_idx := id_to_idx[as.character(id)]]

# Sort for efficient by-year processing
setkey(cell_dt, year, cell_idx)

# Total number of spatial cells
n_cells <- length(id_order)

# Pre-extract neighbor index lists (only 344K entries) for max/min
nb_idx_list <- lapply(seq_len(n_cells), function(i) {
  j <- rook_neighbors_unique[[i]]
  j[j > 0L]
})

# Number of neighbors per cell (for mean denominator)
ones <- rep(1, n_cells)
n_neighbors <- as.numeric(W %*% ones)  # vector of length n_cells

# ============================================================
# 3.  Compute neighbor stats per variable, fully vectorized
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_dt$year))

for (var_name in neighbor_source_vars) {

  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]

  for (yr in years) {
    # Extract rows for this year, keyed by cell_idx
    yr_mask <- cell_dt$year == yr
    sub     <- cell_dt[yr_mask]

    # Build a full-length vector for this variable aligned to cell_idx
    # (NA for cells not present in this year â€” handles irregular/masked cells)
    x_full <- rep(NA_real_, n_cells)
    x_full[sub$cell_idx] <- sub[[var_name]]

    # --- Neighbor mean via sparse matrix multiply ---
    # Replace NA with 0 for the sum, track non-NA counts
    x_for_sum       <- x_full
    x_for_sum[is.na(x_for_sum)] <- 0
    not_na           <- as.numeric(!is.na(x_full))

    neigh_sum   <- as.numeric(W %*% x_for_sum)
    neigh_count <- as.numeric(W %*% not_na)

    neigh_mean <- ifelse(neigh_count > 0, neigh_sum / neigh_count, NA_real_)

    # --- Neighbor max and min via compiled vapply over 344K cells ---
    # This is 344K iterations (not 6.46M) with simple integer indexing
    neigh_max <- vapply(nb_idx_list, function(idx) {
      if (length(idx) == 0L) return(NA_real_)
      v <- x_full[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0L) NA_real_ else max(v)
    }, numeric(1))

    neigh_min <- vapply(nb_idx_list, function(idx) {
      if (length(idx) == 0L) return(NA_real_)
      v <- x_full[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0L) NA_real_ else min(v)
    }, numeric(1))

    # --- Write back to the data.table for this year's rows ---
    # sub$cell_idx gives us which cells are present
    idx_present <- sub$cell_idx
    set(cell_dt, which = which(yr_mask), j = col_max,  value = neigh_max[idx_present])
    set(cell_dt, which = which(yr_mask), j = col_min,  value = neigh_min[idx_present])
    set(cell_dt, which = which(yr_mask), j = col_mean, value = neigh_mean[idx_present])
  }

  message("Done: ", var_name)
}

# ============================================================
# 4.  Convert back to data.frame if needed for downstream RF
# ============================================================
cell_dt[, cell_idx := NULL]
cell_data <- as.data.frame(cell_dt)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over exactly the same rook-neighbor values as the original code; the sparse matrix `W` encodes the identical `rook_neighbors_unique` topology. |
| **Irregular/masked cells** | Cells absent in a given year get `NA` in `x_full`; they are excluded from neighbor statistics via the `!is.na` filter (max/min) and the `not_na` count vector (mean) â€” identical to the original `neighbor_vals[!is.na(neighbor_vals)]` logic. |
| **Trained Random Forest** | The RF model object is untouched. Only the feature-engineering step is optimized. The output columns have the same names and semantics, so `predict(rf_model, cell_data)` works unchanged. |

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~60+ hours (6.46M string-paste iterations) | ~0.5 sec (sparse matrix construction, 344K-cell `nb_idx_list`) |
| Stats per variable per year | R-level loop over 6.46M entries | 1 sparse mat-vec (~2 ms) + `vapply` over 344K cells (~1 sec) |
| Total for 5 vars Ã— 28 years | ~86+ hours | **~5â€“10 minutes** |
| Peak RAM | ~6.46M-element list of integer vectors (multi-GB) | Sparse matrix (~20 MB) + working vectors (~few MB) | 

This should complete in well under 15 minutes on a standard 16 GB laptop while producing bit-identical numerical results.
END_RESPONSE>>>
