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
run_id: Anthropic_F07_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**: it creates a list of ~6.46 million elements, each produced by an anonymous function inside `lapply` that performs character key construction, hash-table lookups, and NA filtering **per row**. This is an O(n) R-level loop over 6.46 million rows, with heavy per-iteration overhead from `paste`, named-vector indexing, and memory allocation. The subsequent `compute_neighbor_stats` is a second O(n) loop that indexes into a numeric vector â€” cheaper per iteration but still pure-R over 6.46 million rows, repeated 5 times.

**Root causes of the 86+ hour estimate:**

1. **Character-key hashing in a hot loop.** `paste(id, year)` and named-vector lookup (`idx_lookup[neighbor_keys]`) for every row is extremely slow in R.
2. **Per-row `lapply` with allocation.** Each of 6.46M iterations allocates small vectors and returns them; the garbage collector is under constant pressure.
3. **Redundant work across variables.** The neighbor lookup structure is rebuilt conceptually for every variable call (though the lookup itself is reused, the stats loop runs 5 times over 6.46M rows in pure R).
4. **The problem is fundamentally a sparse-matrixâ€“vector operation** that can be expressed as a single matrix multiplication / grouped aggregation, avoiding any R-level row loop.

---

## Optimization Strategy

**Replace the row-level R loop with a sparse adjacency matrix and vectorized matrix operations.**

Key insight: if we construct a sparse **row-adjacency matrix `A`** of dimension `(n_rows Ã— n_rows)` where `A[i, j] = 1` iff row `j` is a rook neighbor of row `i` *in the same year*, then:

| Statistic | Vectorized form |
|---|---|
| **neighbor_mean** | `(A %*% x) / (A %*% ones)` â€” i.e., sparse matrix-vector multiply |
| **neighbor_max** | Replace 0s in `A` with `-Inf`, compute row-max of `A * x` (via `Matrix` utilities or one `data.table` group-by) |
| **neighbor_min** | Analogous with `+Inf` |

Because `Matrix::%*%` is implemented in C (CHOLMOD/CSC), the mean computation for all 6.46M rows takes **seconds**, not hours. Max and min require a grouped operation on the sparse triplet entries, which `data.table` handles in seconds as well.

**Estimated speedup: from 86+ hours â†’ ~2â€“5 minutes.**

Memory: the sparse matrix has ~6.46M Ã— 4 (avg neighbors) â‰ˆ 26M non-zero entries, stored as two integer vectors + one double vector â‰ˆ ~600 MB, well within 16 GB.

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Prerequisites
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(Matrix)
library(data.table)

# cell_data        : data.frame / data.table with columns id, year, ntl, ec, â€¦
# id_order         : integer vector of cell IDs in the order used by spdep::nb
# rook_neighbors_unique : spdep nb object (list of integer index vectors)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a directed edge list of spatial neighbors (cell-ID level)
#     This is done ONCE, independent of year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_spatial_edge_list <- function(id_order, neighbors) {
  # neighbors[[k]] contains the indices (into id_order) of the rook neighbors
  # of cell id_order[k].
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(from_id = id_order[from_idx],
             to_id   = id_order[to_idx])
}

spatial_edges <- build_spatial_edge_list(id_order, rook_neighbors_unique)
cat("Spatial directed edges:", nrow(spatial_edges), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Expand to row-level adjacency (same year) and build sparse matrix
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Create a compact row index (1-based) for every row in cell_data
cell_data[, .row_idx := .I]

# Keyed lookup:  (id, year) â†’ row_idx
row_map <- cell_data[, .(id, year, .row_idx)]
setkey(row_map, id, year)

# Expand spatial edges Ã— years via keyed join
#   For each spatial edge (from_id, to_id) we need every year in which
#   BOTH from_id and to_id appear.
# Join "from" side
edges_from <- spatial_edges[row_map, on = .(from_id = id),
                            nomatch = 0L,
                            allow.cartesian = TRUE]
# edges_from now has columns: from_id, to_id, year, .row_idx  (= from_row)
setnames(edges_from, ".row_idx", "from_row")

# Join "to" side
setkey(edges_from, to_id, year)
edges_full <- edges_from[row_map, on = .(to_id = id, year),
                         nomatch = 0L,
                         allow.cartesian = FALSE]
setnames(edges_full, ".row_idx", "to_row")

# edges_full now has: from_row, to_row  (and other cols we can drop)
cat("Row-level directed edges:", nrow(edges_full), "\n")

# Build sparse adjacency matrix  (from_row  â†’  to_row means
# "to_row is a neighbor of from_row")
n <- nrow(cell_data)
A <- sparseMatrix(
  i    = edges_full$from_row,
  j    = edges_full$to_row,
  x    = 1,
  dims = c(n, n)
)

# Precompute the number of non-NA neighbors per row (updated per variable)
# and a ones vector
ones <- rep(1, n)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute neighbor stats (max, min, mean) â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_fast <- function(cell_data, A, edges_full, var_name) {
  x <- cell_data[[var_name]]

  # --- Handle NAs: zero-out contributions from NA neighbors ----------
  not_na   <- as.double(!is.na(x))
  x_safe   <- ifelse(is.na(x), 0, x)

  # neighbor count (excluding NAs)
  n_neigh  <- as.numeric(A %*% not_na)

  # neighbor sum  (excluding NAs because NA positions contribute 0)
  n_sum    <- as.numeric(A %*% x_safe)

  # MEAN
  nb_mean  <- ifelse(n_neigh > 0, n_sum / n_neigh, NA_real_)

  # --- MAX and MIN via data.table on the sparse edge list ------------
  #     We only need the "to" values for each "from" row.
  dt <- edges_full[, .(from_row, to_row)]
  dt[, val := x[to_row]]
  dt <- dt[!is.na(val)]

  agg <- dt[, .(nb_max = max(val), nb_min = min(val)), by = from_row]

  # Scatter back to full-length vectors
  nb_max <- rep(NA_real_, nrow(cell_data))
  nb_min <- rep(NA_real_, nrow(cell_data))
  nb_max[agg$from_row] <- agg$nb_max
  nb_min[agg$from_row] <- agg$nb_min

  data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Outer loop â€” attach features with original naming convention
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  stats <- compute_neighbor_stats_fast(cell_data, A, edges_full, var_name)

  # Preserve the column-naming convention used by the trained RF model
  set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = stats$nb_max)
  set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = stats$nb_min)
  set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = stats$nb_mean)
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done. Neighbor features added for", length(neighbor_source_vars), "variables.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The sparse matrix encodes *exactly* the same `(from_row, to_row)` pairs as the original `build_neighbor_lookup`. Mean = sum/count, max, and min are computed over the identical neighbor sets. No approximation is introduced. |
| **Trained Random Forest** | No model code is touched. Only the feature-engineering pipeline is accelerated. The output columns have the same names and identical values, so `predict(rf_model, cell_data)` produces the same predictions. |
| **Irregular / masked topology** | The sparse matrix is built directly from `rook_neighbors_unique`, so cells with 0â€“4 neighbors, missing years, or masked cells are handled exactly as before â€” no rectangular-raster assumption is made. |
| **NA handling** | NAs in the source variable are excluded from max, min, and mean, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` logic. Rows with zero valid neighbors receive `NA`. |

**Expected runtime:** ~2â€“5 minutes total (dominated by the `data.table` grouped max/min, run 5 times) on a 16 GB laptop, versus the original 86+ hour estimate â€” roughly a **1,000Ã—** speedup.
END_RESPONSE>>>
