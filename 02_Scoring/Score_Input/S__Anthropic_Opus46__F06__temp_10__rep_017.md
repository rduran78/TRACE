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
run_id: Anthropic_F06_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors via character key matching.
2. Constructs `paste(..., sep="_")` keys for every neighbor Ã— that row's year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) â€” which in R is **hash-table lookup on character keys repeated 6.46M times**.

This produces a **list of length 6.46M**, each element being an integer vector of row indices. The `paste` and named-vector lookup inside the inner function are extremely expensive at this scale.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows, called 5 times

For each of the 5 source variables, `compute_neighbor_stats` iterates over 6.46M list elements, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M tiny vectors is inherently slow due to R's per-call overhead and memory allocation churn.

### Why raster focal/kernel operations are **not** appropriate here

Focal operations assume a regular grid with a fixed rectangular kernel. Here, the grid cells use an **irregular (spdep::nb) neighbor structure** â€” cells on boundaries, coastlines, or masked regions have variable numbers of neighbors. Focal operations would silently alter the neighbor set, changing the numerical results. We must preserve the exact rook-neighbor topology.

### Estimated time breakdown

| Step | Approximate share |
|---|---|
| `build_neighbor_lookup` (paste + named lookup Ã— 6.46M) | ~35â€“45% |
| `compute_neighbor_stats` (lapply Ã— 6.46M Ã— 5 vars) | ~50â€“60% |
| Overhead / GC | ~5% |

---

## 2. Optimization Strategy

### Strategy 2a: Replace the row-level list lookup with a **sparse adjacency matrix in cell-year space**

Instead of building a list of 6.46M integer vectors, we construct a single sparse matrix `W` of dimension `(n_rows Ã— n_rows)` where `W[i,j] = 1` iff row `j` is a rook neighbor of row `i` **in the same year**. Then:

- `neighbor_max = row-wise max of W âŠ™ vals` (element-wise product, then row max)
- `neighbor_min = row-wise min` (same idea)
- `neighbor_mean = (W %*% vals) / (W %*% 1)` (sparse matrix-vector multiply)

**Sparse matrix-vector multiplication** for mean is trivially fast via `Matrix::` package (compiled C). Max and min require iterating over the sparse structure, but only once.

### Strategy 2b: Vectorize the lookup construction using `data.table` merge

Instead of 6.46M `paste` + hash lookups, we:
1. Build an **edge list** `(cell_i, cell_j)` from the `nb` object (only ~1.37M edges, spatial).
2. Join this edge list to the data on `(cell_j, year)` to get the row index of each neighbor-year pair â€” a single `data.table` merge.
3. Group by the focal row index and compute `max/min/mean` in one pass per variable â€” fully vectorized via `data.table`.

This reduces the time from hours to **minutes**.

### Strategy 2c: Compute all 5 variables in one grouped pass

Instead of 5 separate loops, compute all 15 summary statistics (3 stats Ã— 5 vars) in a single `data.table` grouped aggregation.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup construction | ~30â€“40 hours | ~30 seconds |
| Neighbor stats (5 vars Ã— 6.46M) | ~45â€“55 hours | ~2â€“5 minutes |
| **Total** | **86+ hours** | **~3â€“8 minutes** |

---

## 3. Working R Code

```r
# ==============================================================================
# Optimized neighbor feature computation
# Preserves exact rook-neighbor topology and numerical results.
# Preserves the trained Random Forest model (no retraining).
# ==============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ------------------------------------------------------------------
  # STEP 1: Build the spatial edge list from the nb object (once)
  # ------------------------------------------------------------------
  # rook_neighbors_unique is a list of length length(id_order),

  # where element [[i]] contains integer indices into id_order of
  # the rook neighbors of id_order[i].
  
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edges now has ~1,373,394 rows: (focal_id, neighbor_id)
  
  # ------------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table; add row index
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # Create lookup: for each (id, year) -> row index
  # We will join edges against this to find the row indices of neighbors
  setkey(dt, id, year)
  
  # Focal row lookup: (id, year) -> .row_idx
  focal_lookup <- dt[, .(focal_id = id, year, focal_row = .row_idx)]
  
  # Neighbor row lookup: (id, year) -> .row_idx and variable values
  neighbor_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  neighbor_lookup <- dt[, ..neighbor_cols]
  setnames(neighbor_lookup, c("id", ".row_idx"),
           c("neighbor_id", "neighbor_row"))
  
  # ------------------------------------------------------------------
  # STEP 3: Expand edges Ã— years via merge
  # ------------------------------------------------------------------
  # Join focal_lookup to edges on focal_id to get (focal_id, year, focal_row, neighbor_id)
  setkey(edges, focal_id)
  setkey(focal_lookup, focal_id)
  expanded <- edges[focal_lookup, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # Result columns: focal_id, neighbor_id, year, focal_row
  
  # Now join to neighbor_lookup on (neighbor_id, year) to get neighbor row + values
  setkey(expanded, neighbor_id, year)
  setkey(neighbor_lookup, neighbor_id, year)
  expanded <- neighbor_lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Now expanded has: neighbor_id, year, neighbor_row, ntl, ec, ..., focal_id, focal_row
  
  # ------------------------------------------------------------------
  # STEP 4: Grouped aggregation â€” compute max, min, mean per focal_row
  # ------------------------------------------------------------------
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Perform the grouped aggregation in one pass
  stats <- expanded[, lapply(agg_exprs, eval), by = focal_row]
  
  # Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen if

  # nomatch = NA filters correctly, but be safe)
  for (col_name in agg_names) {
    vals <- stats[[col_name]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats, j = col_name, value = vals)
  }
  
  # ------------------------------------------------------------------
  # STEP 5: Left-join stats back to the original data
  # ------------------------------------------------------------------
  setkey(stats, focal_row)
  dt[, .row_idx := .I]
  
  for (col_name in agg_names) {
    # Pre-allocate NA column, then fill matched rows
    dt[, (col_name) := NA_real_]
    dt[stats$focal_row, (col_name) := stats[[col_name]]]
  }
  
  dt[, .row_idx := NULL]
  
  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ==============================================================================
# Usage â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The 15 new columns are now in cell_data:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec, neighbor_min_ec, neighbor_mean_ec,
#   ... etc.
#
# The trained Random Forest model can be applied as before:
#   predictions <- predict(rf_model, newdata = cell_data)
```

### Alternative: Sparse-matrix approach for `mean` (even faster for mean only)

If memory is tight or you want maximum speed for the mean calculation specifically, you can use a sparse matrix multiply. This is compatible with the `data.table` approach above for max/min:

```r
library(Matrix)

build_sparse_W <- function(dt, id_order, rook_neighbors_unique) {
  # Map cell id -> position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (cell_pos, year) -> row in dt
  dt[, .cell_pos := id_to_pos[as.character(id)]]
  years <- sort(unique(dt$year))
  year_to_idx <- setNames(seq_along(years), as.character(years))
  dt[, .year_idx := year_to_idx[as.character(year)]]
  
  n_cells <- length(id_order)
  n_years <- length(years)
  N <- nrow(dt)
  
  # Build spatial edge list
  from_pos <- integer(0)
  to_pos   <- integer(0)
  for (i in seq_along(rook_neighbors_unique)) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) > 0L) {
      from_pos <- c(from_pos, rep(i, length(nb)))
      to_pos   <- c(to_pos, nb)
    }
  }
  
  # Expand to row-level: for each year, map (from_pos, yr) and (to_pos, yr) to row indices
  # Use a (cell_pos, year_idx) -> row_idx matrix for fast lookup
  row_lookup <- integer(n_cells * n_years)
  row_lookup[] <- NA_integer_
  row_lookup[(dt$.year_idx - 1L) * n_cells + dt$.cell_pos] <- seq_len(N)
  
  # Vectorized expansion across years
  ii <- integer(0); jj <- integer(0)
  for (y in seq_along(years)) {
    fi <- row_lookup[(y - 1L) * n_cells + from_pos]
    ti <- row_lookup[(y - 1L) * n_cells + to_pos]
    valid <- !is.na(fi) & !is.na(ti)
    ii <- c(ii, fi[valid])
    jj <- c(jj, ti[valid])
  }
  
  W <- sparseMatrix(i = ii, j = jj, x = 1, dims = c(N, N))
  
  dt[, c(".cell_pos", ".year_idx") := NULL]
  return(W)
}

# Usage for mean:
# W <- build_sparse_W(dt, id_order, rook_neighbors_unique)
# row_counts <- as.numeric(W %*% rep(1, nrow(dt)))
# for (var_name in neighbor_source_vars) {
#   vals <- dt[[var_name]]
#   vals[is.na(vals)] <- 0  # handle NA carefully
#   not_na <- as.numeric(!is.na(dt[[var_name]]))
#   neighbor_sum   <- as.numeric(W %*% vals)
#   neighbor_count <- as.numeric(W %*% not_na)
#   dt[[paste0("neighbor_mean_", var_name)]] <- ifelse(
#     neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_
#   )
# }
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste` + named vector lookup | Single `data.table` merge on edge list (~1.37M edges Ã— 28 years) |
| **Stats computation** | `lapply` over 6.46M list elements Ã— 5 variables | One grouped `data.table` aggregation across all 5 variables |
| **Time estimate** | 86+ hours | ~3â€“8 minutes |
| **RAM peak** | Moderate (6.46M-element list of integer vectors) | Moderate (~40M-row expanded edge table, then freed) |
| **Numerical results** | Exact rook-neighbor max/min/mean | **Identical** â€” same neighbor topology, same aggregation functions |
| **Trained RF model** | Unchanged | **Unchanged â€” no retraining** |
END_RESPONSE>>>
