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
run_id: Anthropic_F05_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **O(N) string-key construction inside `build_neighbor_lookup`**: `idx_lookup` is built once (good), but inside the `lapply` over ~6.46M rows, `paste()` is called per-row to construct `neighbor_keys`. With an average of ~8 rook neighbors per cell (1.37M directed relationships / ~344K cells â‰ˆ 4 per cell, but shared across years), this generates tens of millions of small `paste()` calls plus hash lookups into a 6.46M-entry named vector. The named-vector lookup `idx_lookup[neighbor_keys]` is itself O(k) per call but with enormous constant overhead due to string hashing on a 6.46M-name vector.

2. **The entire string-keying approach is unnecessary**: The data is a balanced panel (344,208 cells Ã— 28 years). The neighbor structure is **time-invariant** â€” cell A's rook neighbors are the same in every year. Therefore, the neighbor relationships can be expressed as **row-index offsets** in a data.table sorted by (id, year) or (year, id). There is no need to do any per-row string lookup at all.

3. **`lapply` over 6.46M rows returns a list of integer vectors** â€” this is inherently slow in R and memory-wasteful. The neighbor lookup can be vectorized entirely using a **sparse adjacency approach** or a **data.table join**.

4. **`compute_neighbor_stats` re-traverses the same list structure 5 times** (once per variable), each time pulling values by index. This could be done in a single pass or via matrix operations.

### Summary of the cost hierarchy

| Layer | Operation | Calls | Bottleneck |
|-------|-----------|-------|------------|
| String key build | `paste()` + named-vector construction | 1Ã— for 6.46M keys | Moderate |
| Per-row neighbor key lookup | `paste()` + `idx_lookup[keys]` | 6.46M Ã— ~4 neighbors | **Dominant** |
| Per-variable stats | List traversal Ã— 5 vars | 5 Ã— 6.46M | Significant |
| `do.call(rbind, ...)` on 6.46M-element list | Memory allocation | 5Ã— | Significant |

## Optimization Strategy

**Core insight**: Since the panel is balanced and the neighbor structure is time-invariant, we can:

1. **Sort data by (year, id)** so that within each year-block, cells appear in the same order.
2. **Express the neighbor graph as a sparse matrix** (or equivalently, a two-column edge list of cell-position indices).
3. **For each year-block**, the row positions are a simple offset from the cell's position index. Neighbor row indices become `offset + neighbor_cell_positions` â€” pure integer arithmetic, no strings.
4. **Compute all 5 variables' stats in a single vectorized pass** using sparse matrixâ€“vector multiplication (for mean/sum) and grouped operations for min/max.

The most efficient approach uses `Matrix::sparseMatrix` to represent the adjacency, then computes neighbor means via matrix multiplication and neighbor min/max via grouped operations on an edge list â€” all fully vectorized with no R-level loops over 6.46M rows.

## Working R Code

```r
library(data.table)
library(Matrix)

#' Optimized neighbor feature construction for a balanced cell-year panel.
#'
#' @param cell_data        data.frame/data.table with columns: id, year, and the source vars
#' @param id_order         integer vector of cell IDs in the order used by rook_neighbors_unique
#' @param rook_nb          spdep::nb object (rook_neighbors_unique)
#' @param neighbor_source_vars character vector of variable names
#' @return data.table with original columns plus neighbor features appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_nb,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # ---------------------------------------------------------------
  # 1. Build a mapping from cell id -> position index (1..N_cells)
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  # Assign each row its cell-position index
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Sort by (year, cell_pos) so within each year the cells are in
  # canonical position order. This is the key enabler.
  setorder(dt, year, cell_pos)

  # Verify balanced panel
  years <- sort(unique(dt$year))
  n_years <- length(years)
  stopifnot(nrow(dt) == n_cells * n_years)

  # ---------------------------------------------------------------
  # 2. Build directed edge list from the nb object (cell-position space)
  #    from_pos -> to_pos  (time-invariant)
  # ---------------------------------------------------------------
  edge_from <- integer(0)
  edge_to   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_nb[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
      edge_from <- c(edge_from, rep.int(i, length(nb_i)))
      edge_to   <- c(edge_to,   as.integer(nb_i))
    }
  }
  n_edges <- length(edge_from)
  cat(sprintf("Neighbor edge list: %d directed edges\n", n_edges))

  # ---------------------------------------------------------------
  # 3. Build row-level edge list for ALL year-blocks at once.
  #
  #    Because dt is sorted by (year, cell_pos), the row for

  #    cell_pos=p in year_index=t (0-based) is:  row = t * n_cells + p
  #
  #    So we replicate the edge list across all years via offset.
  # ---------------------------------------------------------------
  year_offsets <- (seq_len(n_years) - 1L) * n_cells  # length n_years

  # Pre-allocate full edge list: n_edges * n_years entries
  total_edges <- as.double(n_edges) * n_years
  cat(sprintf("Total row-level edges: %.0f\n", total_edges))

  row_from <- integer(total_edges)
  row_to   <- integer(total_edges)

  for (t_idx in seq_len(n_years)) {
    off <- year_offsets[t_idx]
    start <- (t_idx - 1L) * n_edges + 1L
    end   <- t_idx * n_edges
    row_from[start:end] <- edge_from + off
    row_to[start:end]   <- edge_to   + off
  }

  # ---------------------------------------------------------------
  # 4. Build sparse adjacency matrix (n_rows x n_rows) â€” but we only
  #    need it for matrix-vector products (for mean).
  #    Also compute the number of neighbors per row (degree) for mean.
  # ---------------------------------------------------------------
  n_rows <- nrow(dt)

  # Sparse matrix: adj[i, j] = 1 means j is a neighbor of i
  # So adj %*% vals = sum of neighbor values for each row
  adj <- sparseMatrix(
    i = row_from,
    j = row_to,
    x = rep.int(1, length(row_from)),
    dims = c(n_rows, n_rows)
  )

  # Degree (number of non-NA neighbors will be adjusted per variable)
  degree <- as.integer(rowSums(adj))  # number of neighbors per row

  # ---------------------------------------------------------------
  # 5. For each variable, compute neighbor max, min, mean
  #    - mean: use sparse mat-vec product, divide by count of non-NA neighbors
  #    - min/max: use edge list with data.table grouped operations
  # ---------------------------------------------------------------

  # Pre-build the edge data.table (reusable across variables)
  edge_dt <- data.table(from_row = row_from, to_row = row_to)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))

    vals <- dt[[var_name]]

    # --- Neighbor MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for summation, track non-NA count separately
    vals_nona <- vals
    is_na_val <- is.na(vals_nona)
    vals_nona[is_na_val] <- 0

    neighbor_sum   <- as.numeric(adj %*% vals_nona)
    # Count of non-NA neighbors: use indicator vector
    not_na_ind <- as.numeric(!is_na_val)
    neighbor_count <- as.numeric(adj %*% not_na_ind)

    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- Neighbor MAX and MIN via edge-list grouped operations ---
    # Attach the neighbor's value to each edge
    edge_dt[, nb_val := vals[to_row]]

    # Remove edges where neighbor value is NA
    edge_valid <- edge_dt[!is.na(nb_val)]

    if (nrow(edge_valid) > 0) {
      stats_dt <- edge_valid[, .(
        nb_max = max(nb_val),
        nb_min = min(nb_val)
      ), by = from_row]

      # Initialize with NA
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)

      neighbor_max[stats_dt$from_row] <- stats_dt$nb_max
      neighbor_min[stats_dt$from_row] <- stats_dt$nb_min
    } else {
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)
    }

    # --- Assign columns using the ORIGINAL naming convention ---
    # Original code produces: {var}_max, {var}_min, {var}_mean
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")

    set(dt, j = max_col,  value = neighbor_max)
    set(dt, j = min_col,  value = neighbor_min)
    set(dt, j = mean_col, value = neighbor_mean)
  }

  # ---------------------------------------------------------------
  # 6. Restore original row order and return
  # ---------------------------------------------------------------
  # Remove helper column
  dt[, cell_pos := NULL]

  # Restore original order if needed â€” the caller may expect id,year order.
  # The original data's order should be preserved. We store original row index:
  # Actually, let's be safe: sort back to (id, year) which is a common panel order.
  setorder(dt, id, year)

  return(dt)
}

# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  rook_nb          = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is used downstream unchanged â€”
# the columns it expects ({var}_max, {var}_min, {var}_mean) are
# now present with identical numerical values.
```

## Complexity and Runtime Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | 6.46M `paste()` + string hash lookups | Integer addition (offset + position) |
| **Per-variable stats** | R-level `lapply` over 6.46M list elements | Sparse mat-vec (`Matrix`) + `data.table` grouped agg |
| **String allocations** | ~50M+ transient strings | Zero |
| **Total passes over edges** | 5 vars Ã— 6.46M rows Ã— ~4 neighbors = ~129M R-level iterations | 5 vars Ã— 1 sparse mat-vec + 1 grouped agg |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **Peak RAM** | Lists of 6.46M integer vectors (~2â€“4 GB) | Sparse matrix (~600 MB) + edge DT (~300 MB) |
| **Numerical output** | max, min, mean per variable | **Identical** max, min, mean per variable |

### Why correctness is preserved

- The sparse matrix `adj` encodes exactly the same neighbor relationships as the original `build_neighbor_lookup`. Each `adj[i,j] = 1` means "row j is a rook neighbor of row i in the same year" â€” the year constraint is enforced structurally by the block-diagonal construction (edges only connect rows within the same year-offset block).
- `adj %*% vals` computes the exact sum of neighbor values; dividing by the count of non-NA neighbors yields the identical `mean()`.
- `data.table` grouped `max()`/`min()` over the edge list yields the identical `max()`/`min()` per row.
- The Random Forest model sees the same column names with the same numerical values, so no retraining is needed.
END_RESPONSE>>>
