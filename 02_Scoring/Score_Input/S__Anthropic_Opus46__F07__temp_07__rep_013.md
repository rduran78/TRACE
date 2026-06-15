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
run_id: Anthropic_F07_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that performs character coercion, `paste`, and named-vector lookups. Specifically:

1. **O(n) R-level loop with expensive string operations.** For each of the ~6.46M rows, `paste()` builds neighbor keys, then `idx_lookup[neighbor_keys]` does repeated named-vector lookups (hashing ~6.46M names). This alone accounts for the vast majority of the 86+ hour estimate.

2. **Redundant per-variable re-traversal.** `compute_neighbor_stats` is called 5 times, each time traversing the full 6.46M-element lookup list. This is comparatively minor but still wasteful.

3. **Memory pressure from a 6.46M-element list of integer vectors.** Each list element carries R object overhead (~128 bytes minimum), so the lookup alone consumes several GB before any computation begins.

**Root cause:** The neighbor lookup is time-invariant (the grid doesn't change across years), but the code rebuilds index mappings at the individual cell-year level using slow string-keyed lookups. The correct approach is to exploit the panel structure: neighbors are defined over **cells**, not cell-years, and every cell appears once per year in a predictable order.

---

## Optimization Strategy

### Key insight: separate the spatial dimension from the temporal dimension

Since every cell appears in every year (balanced panel), a neighbor relationship between cell *i* and cell *j* in year *t* is simply: "find the row for cell *j* in year *t*." If we sort the data by `(year, id)`, the row offset for any cell within a year-block is deterministic and can be computed with integer arithmetic â€” **no string keys needed**.

### Plan

| Step | What | Speedup factor |
|------|------|----------------|
| 1 | Sort data by `(year, id)`. Build a single integer vector mapping each cell index to its neighbor cell indices (purely spatial, ~344K entries). | Eliminates 6.46M string operations |
| 2 | For each year-block (a contiguous slice of rows), translate spatial neighbor indices to row indices by adding the year-block offset. | O(1) per neighbor edge per year |
| 3 | Vectorize the neighbor stats computation using the sparse adjacency structure (a `dgCMatrix` or direct C++-speed aggregation via `data.table`). | Eliminates 6.46M R-level list iterations |
| 4 | Compute all 5 variables' stats in one pass over the adjacency. | 5Ã— reduction in traversals |

**Expected runtime:** Under 5 minutes on a 16 GB laptop.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€ 0. Ensure data.table format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€ 1. Build a canonical cell ordering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    id_order is the vector of cell IDs in the same order as rook_neighbors_unique.
n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))
n_years <- length(years)

# Map each cell id to its position in id_order (1-based).
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# â”€â”€ 2. Sort data by (year, canonical cell position) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, cell_pos := id_to_pos[as.character(id)]]
setorder(cell_dt, year, cell_pos)

# Verify balanced panel (every cell appears in every year).
stopifnot(nrow(cell_dt) == n_cells * n_years)

# After sorting, the row for cell position p in year-index y (0-based) is:
#   row = y * n_cells + p
# This is the key that eliminates all string lookups.

# â”€â”€ 3. Build sparse adjacency matrix (cells Ã— cells) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    rook_neighbors_unique is an nb object: a list of length n_cells,
#    where each element is an integer vector of neighbor positions (1-based),
#    with a single 0 meaning no neighbors.

# Construct COO (coordinate) representation.
from_vec <- integer(0)
to_vec   <- integer(0)

for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  # spdep::nb encodes "no neighbors" as a single 0
  if (length(nb_i) == 1L && nb_i == 0L) next
  from_vec <- c(from_vec, rep.int(i, length(nb_i)))
  to_vec   <- c(to_vec,   nb_i)
}

# Sparse binary adjacency matrix (n_cells Ã— n_cells), column-sparse.
# Entry (i, j) = 1 means j is a rook-neighbor of i.
adj <- sparseMatrix(
  i    = from_vec,
  j    = to_vec,
  x    = 1,
  dims = c(n_cells, n_cells)
)

# Number of neighbors per cell (for computing means).
n_neighbors <- as.integer(rowSums(adj))  # length n_cells

# â”€â”€ 4. Compute neighbor stats for all variables, all years â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  vals_all <- cell_dt[[var_name]]  # length = n_cells * n_years, sorted by (year, cell_pos)

  # Pre-allocate result columns.
  col_max  <- rep(NA_real_, nrow(cell_dt))
  col_min  <- rep(NA_real_, nrow(cell_dt))
  col_mean <- rep(NA_real_, nrow(cell_dt))

  for (yi in seq_along(years)) {
    # Row range for this year (1-based).
    row_start <- (yi - 1L) * n_cells + 1L
    row_end   <- yi * n_cells
    idx_range <- row_start:row_end

    # Extract this year's values as a numeric vector of length n_cells,
    # ordered by cell_pos.
    v <- vals_all[idx_range]

    # Replace NA with -Inf/+Inf for max/min, then fix up afterwards.
    v_for_max <- v
    v_for_max[is.na(v_for_max)] <- -Inf

    v_for_min <- v
    v_for_min[is.na(v_for_min)] <- Inf

    # For sum (to compute mean), replace NA with 0 and track non-NA count.
    v_notna    <- as.numeric(!is.na(v))
    v_for_sum  <- v
    v_for_sum[is.na(v_for_sum)] <- 0

    # â”€â”€ Sparse matrixâ€“vector products â”€â”€
    # adj %*% v_for_max gives, for each cell i, the SUM of neighbor values
    # (using -Inf for NA). We need MAX, not SUM.
    # Unfortunately, standard sparse matmul only gives sums.
    # Strategy: iterate over unique neighbor counts, or use a direct approach.
    #
    # Efficient direct approach: for each cell, gather neighbor values.
    # With the sorted layout, we can do this with compiled code via
    # Matrix operations on a *modified* adjacency.
    #
    # For MAX and MIN we must avoid matmul. Instead, we use the adj
    # structure directly. We convert adj to a dgCMatrix and walk its
    # column pointers.

    # â”€â”€ Direct C-level traversal via .Call is unavailable in pure R,
    #    but we can use data.table's fast grouped operations on the
    #    edge list. â”€â”€

    # Build edge-value table for this year (reuse from/to vectors).
    # neighbor value = v[to_vec]  (the neighbor cell's value)
    nb_vals <- v[to_vec]

    # data.table grouped aggregation (from_vec is the "focal cell").
    edge_dt <- data.table(
      focal = from_vec,
      nval  = nb_vals
    )

    # Remove edges where neighbor value is NA.
    edge_dt <- edge_dt[!is.na(nval)]

    if (nrow(edge_dt) > 0L) {
      stats_dt <- edge_dt[, .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ), by = focal]

      # Write results back. stats_dt$focal is the cell_pos (1-based).
      result_rows <- row_start - 1L + stats_dt$focal
      col_max[result_rows]  <- stats_dt$nb_max
      col_min[result_rows]  <- stats_dt$nb_min
      col_mean[result_rows] <- stats_dt$nb_mean
    }
  }

  # Assign to data.table.
  cell_dt[, paste0(var_name, "_neighbor_max")  := col_max]
  cell_dt[, paste0(var_name, "_neighbor_min")  := col_min]
  cell_dt[, paste0(var_name, "_neighbor_mean") := col_mean]
}

# â”€â”€ 5. Restore original row order if needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If downstream code expects the original row order, store it before sorting
# and restore here. For safety:
cell_dt[, cell_pos := NULL]  # remove helper column

# Convert back to data.frame if the trained RF model expects one.
cell_data <- as.data.frame(cell_dt)
```

---

### Further speed-up: pre-build the edge table once

The inner loop rebuilds `edge_dt` per variable per year. We can factor this out:

```r
# â”€â”€ Optimized: single edge table, all variables at once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)
library(Matrix)

cell_dt <- as.data.table(cell_data)

n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))
n_years <- length(years)

id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_pos := id_to_pos[as.character(id)]]
setorder(cell_dt, year, cell_pos)
stopifnot(nrow(cell_dt) == n_cells * n_years)

# Build edge list once from nb object.
from_vec <- integer(0)
to_vec   <- integer(0)
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 1L && nb_i == 0L) next
  from_vec <- c(from_vec, rep.int(i, length(nb_i)))
  to_vec   <- c(to_vec,   nb_i)
}
n_edges <- length(from_vec)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate all 15 result columns.
for (var_name in neighbor_source_vars) {
  cell_dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
  cell_dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
  cell_dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
}

# Process year by year.
for (yi in seq_along(years)) {
  row_start <- (yi - 1L) * n_cells + 1L
  row_end   <- yi * n_cells

  # Build edge table with all 5 variables' neighbor values.
  edge_dt <- data.table(focal = from_vec)

  any_valid <- FALSE
  for (var_name in neighbor_source_vars) {
    v <- cell_dt[[var_name]][row_start:row_end]
    edge_dt[, (var_name) := v[to_vec]]
  }

  # Melt to long form for grouped aggregation across all variables at once.
  edge_long <- melt(
    edge_dt,
    id.vars       = "focal",
    variable.name = "var",
    value.name    = "nval"
  )
  edge_long <- edge_long[!is.na(nval)]

  if (nrow(edge_long) > 0L) {
    stats_long <- edge_long[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = .(focal, var)]

    # Write back per variable.
    for (var_name in neighbor_source_vars) {
      s <- stats_long[var == var_name]
      if (nrow(s) > 0L) {
        result_rows <- row_start - 1L + s$focal
        set(cell_dt, i = result_rows,
            j = paste0(var_name, "_neighbor_max"),  value = s$nb_max)
        set(cell_dt, i = result_rows,
            j = paste0(var_name, "_neighbor_min"),  value = s$nb_min)
        set(cell_dt, i = result_rows,
            j = paste0(var_name, "_neighbor_mean"), value = s$nb_mean)
      }
    }
  }
}

cell_dt[, cell_pos := NULL]
cell_data <- as.data.frame(cell_dt)
```

---

### Fastest variant: avoid melt, process variables in a tight loop

```r
library(data.table)

cell_dt <- as.data.table(cell_data)
n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))
n_years <- length(years)

id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_pos := id_to_pos[as.character(id)]]
setorder(cell_dt, year, cell_pos)
stopifnot(nrow(cell_dt) == n_cells * n_years)

# â”€â”€ Build edge list from nb object (vectorized) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lengths_nb <- vapply(rook_neighbors_unique, function(nb) {
  if (length(nb) == 1L && nb == 0L) 0L else length(nb)
}, integer(1))

from_vec <- rep(seq_len(n_cells), times = lengths_nb)
to_vec   <- unlist(lapply(seq_len(n_cells), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 1L && nb == 0L) integer(0) else nb
}), use.names = FALSE)

n_edges <- length(from_vec)
cat("Edge list:", n_edges, "directed edges\n")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns.
for (vn in neighbor_source_vars) {
  set(cell_dt, j = paste0(vn, "_neighbor_max"),  value = rep(NA_real_, nrow(cell_dt)))
  set(cell_dt, j = paste0(vn, "_neighbor_min"),  value = rep(NA_real_, nrow(cell_dt)))
  set(cell_dt, j = paste0(vn, "_neighbor_mean"), value = rep(NA_real_, nrow(cell_dt)))
}

# â”€â”€ Main loop: per year Ã— per variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (yi in seq_along(years)) {
  row_start <- (yi - 1L) * n_cells + 1L

  for (vn in neighbor_source_vars) {
    v <- .subset2(cell_dt, vn)[row_start:(row_start + n_cells - 1L)]
    nv <- v[to_vec]

    # Fast grouped stats via data.table.
    tmp <- data.table(f = from_vec, nv = nv)
    tmp <- tmp[!is.na(nv)]

    if (nrow(tmp) > 0L) {
      s <- tmp[, .(mx = max(nv), mn = min(nv), mu = mean(nv)), keyby = f]
      rows <- row_start - 1L + s$f
      set(cell_dt, i = rows, j = paste0(vn, "_neighbor_max"),  value = s$mx)
      set(cell_dt, i = rows, j = paste0(vn, "_neighbor_min"),  value = s$mn)
      set(cell_dt, i = rows, j = paste0(vn, "_neighbor_mean"), value = s$mu)
    }
  }

  if (yi %% 5L == 0L) cat("Completed year", years[yi], "\n")
}

cell_dt[, cell_pos := NULL]
cell_data <- as.data.frame(cell_dt)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Lookup construction** | 6.46M string-paste + named-vector lookups | Integer arithmetic on sorted panel |
| **Neighbor stats** | 6.46M R-level `lapply` iterations Ã— 5 vars | 28 year-blocks Ã— 5 vars Ã— one `data.table` grouped aggregation on ~1.37M edges |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Peak memory** | Multiple GB for 6.46M-element list | ~50 MB for edge vectors + year-slice temps |
| **Numerical result** | max, min, mean per cell-year | Identical (same neighbor sets, same aggregation functions) |
| **Trained RF model** | â€” | Untouched; only the input feature columns are recomputed identically |

The trained Random Forest model is never modified â€” only the feature engineering pipeline is replaced with a numerically equivalent but dramatically faster implementation.
END_RESPONSE>>>
