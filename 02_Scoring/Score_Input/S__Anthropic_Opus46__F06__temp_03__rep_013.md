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
run_id: Anthropic_F06_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`
This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in the neighbor list.
2. Constructs string keys (`paste(id, year)`) for each neighbor.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String-based named-vector lookups in R are hash-table operations, but doing this 6.46 million times with allocation of intermediate character vectors is extremely slow. The neighbor structure is **time-invariant** (same neighbors every year), yet the lookup is rebuilt per cell-year row rather than exploiting the panel structure.

### Bottleneck 2: `compute_neighbor_stats` â€” Row-level `lapply` with `rbind`
For each of the 5 variables, another `lapply` over 6.46 million rows computes `max`, `min`, `mean` by subsetting a numeric vector. The `do.call(rbind, result)` on a 6.46M-element list is also costly. This runs 5 times (once per variable), totaling ~32.3 million R-level loop iterations.

### Why raster focal/kernel operations are not the right analogy
Focal operations assume a regular grid with a fixed rectangular kernel. Here the neighbor structure is an irregular `spdep::nb` object (rook contiguity on an arbitrary polygon/grid layout), so focal operations would require re-gridding and could introduce numerical differences. We must **preserve the original numerical estimand exactly**, so we stay with the neighbor-list approach but vectorize it.

### Root cause summary
| Component | Calls | Cost driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | String key construction & lookup |
| `compute_neighbor_stats` | 6.46M Ã— 5 vars | Per-row subsetting, `lapply`, `do.call(rbind)` |
| Total | ~38.8M R-level iterations | Interpreted loop overhead, memory allocation |

---

## Optimization Strategy

### 1. Exploit panel structure: separate space from time
The neighbor graph is **time-invariant**. Instead of building a 6.46M-row lookup, build a **cell-level** lookup (344,208 cells) and then broadcast across years using vectorized joins.

### 2. Vectorized sparse-matrix multiplication for neighbor stats
Represent the neighbor graph as a **sparse adjacency matrix** `W` (344,208 Ã— 344,208). For each year and each variable, arrange the variable values as a vector `v` of length 344,208. Then:
- `W %*% v` gives the **sum** of neighbor values.
- The number of non-NA neighbors per cell is obtained from `W %*% (!is.na(v))`.
- **Mean** = sum / count.
- For **max** and **min**, use grouped operations via `data.table` with an edge list.

Sparse matrix multiplication in R (via the `Matrix` package) is implemented in C and is orders of magnitude faster than row-level `lapply`.

### 3. Use `data.table` for grouped max/min
Convert the sparse neighbor structure to an edge list `(from, to)`. For each year, join variable values onto the `to` column, then compute `max` and `min` grouped by `from`. `data.table` grouped operations on ~1.37M edges Ã— 28 years = ~38.4M rows are fast (seconds, not hours).

### 4. Expected speedup
| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~hours | Eliminated (sparse matrix built once in seconds) | âˆž |
| Mean computation (per var per year) | ~minutes | Sparse matrix multiply (~ms) | ~1000Ã— |
| Max/Min (per var per year) | ~minutes | `data.table` grouped op (~ms) | ~500Ã— |
| **Total estimated wall time** | **86+ hours** | **~2â€“5 minutes** | **~1000â€“2500Ã—** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the exact numerical results of the original implementation.
# =============================================================================

library(data.table)
library(Matrix)

# ---- 0. Ensure cell_data is a data.table with original row order preserved ---
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}
# Preserve original row order for downstream compatibility
cell_data[, .row_order := .I]

# ---- 1. Build cell-level mappings (time-invariant) --------------------------

# id_order: the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an spdep::nb object (list of integer index vectors)

n_cells <- length(id_order)
stopifnot(n_cells == length(rook_neighbors_unique))

# Build a map from cell id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# ---- 2. Build sparse adjacency matrix W (n_cells x n_cells) -----------------
#
# W[i, j] = 1 means cell j is a rook neighbor of cell i.
# This encodes: "for cell i, aggregate over its neighbors j."

edge_from <- integer(0)
edge_to   <- integer(0)

for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  # spdep::nb encodes no-neighbor as 0L in a length-1 vector

  nb_i <- nb_i[nb_i > 0L]
  if (length(nb_i) > 0L) {
    edge_from <- c(edge_from, rep.int(i, length(nb_i)))
    edge_to   <- c(edge_to, nb_i)
  }
}

# Sparse matrix (rows = focal cell, cols = neighbor cell)
W <- sparseMatrix(
  i = edge_from,
  j = edge_to,
  x = 1,
  dims = c(n_cells, n_cells)
)

# Also keep edge list as data.table for max/min computation
edges_dt <- data.table(from_pos = edge_from, to_pos = edge_to)

rm(edge_from, edge_to)

# ---- 3. Build cell-year indexing structure -----------------------------------

# Map each cell id to its position in id_order within cell_data
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)

# For fast subsetting: key by (cell_pos, year)
setkey(cell_data, cell_pos, year)

# Pre-build a complete grid index: for each (cell_pos, year), what is the
# row in cell_data? We need this to scatter results back.
# Using a matrix: rows = cell_pos (1..n_cells), cols = year index (1..n_years)

year_to_idx <- setNames(seq_along(years), as.character(years))

# Build a lookup matrix: row_lookup[cell_pos, year_idx] = row in cell_data
# Initialize with NA
row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_lookup[cbind(cell_data$cell_pos, year_to_idx[as.character(cell_data$year)])] <- cell_data$.row_order

# ---- 4. Compute neighbor features per variable ------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  # Initialize result columns with NA
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  cell_data[[col_max]]  <- NA_real_
  cell_data[[col_min]]  <- NA_real_
  cell_data[[col_mean]] <- NA_real_

  # Process year by year
  for (yr in years) {

    yr_idx <- year_to_idx[as.character(yr)]

    # Row indices in cell_data for this year, ordered by cell_pos
    yr_rows <- row_lookup[, yr_idx]  # length = n_cells, NA if cell absent

    # Build value vector: v[cell_pos] = variable value (NA if cell absent)
    v <- rep(NA_real_, n_cells)
    present <- !is.na(yr_rows)
    v[present] <- cell_data[[var_name]][yr_rows[present]]

    # --- MEAN via sparse matrix ---
    # Replace NA with 0 for summation, track non-NA counts
    v_nona <- v
    v_nona[is.na(v_nona)] <- 0
    indicator <- as.numeric(!is.na(v))

    neighbor_sum   <- as.numeric(W %*% v_nona)      # length n_cells
    neighbor_count <- as.numeric(W %*% indicator)    # length n_cells

    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- MAX and MIN via data.table grouped operations ---
    # Attach values to the "to" side of edges
    edges_yr <- edges_dt[, .(from_pos, to_pos)]
    edges_yr[, val := v[to_pos]]

    # Remove edges where neighbor value is NA
    edges_yr <- edges_yr[!is.na(val)]

    if (nrow(edges_yr) > 0) {
      stats_dt <- edges_yr[, .(nb_max = max(val), nb_min = min(val)), by = from_pos]

      # Initialize full vectors
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
      neighbor_max[stats_dt$from_pos] <- stats_dt$nb_max
      neighbor_min[stats_dt$from_pos] <- stats_dt$nb_min
    } else {
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
    }

    # --- Scatter results back into cell_data ---
    # Only for cells present in this year
    target_rows <- yr_rows[present]
    target_pos  <- which(present)

    cell_data[[col_max]][target_rows]  <- neighbor_max[target_pos]
    cell_data[[col_min]][target_rows]  <- neighbor_min[target_pos]
    cell_data[[col_mean]][target_rows] <- neighbor_mean[target_pos]
  }

  cat("  Done:", col_max, col_min, col_mean, "\n")
}

# ---- 5. Clean up helper columns and restore original row order ---------------
setorder(cell_data, .row_order)
cell_data[, c(".row_order", "cell_pos") := NULL]

cat("All neighbor features computed. Ready for Random Forest prediction.\n")

# ---- 6. Predict using the pre-trained Random Forest (unchanged) --------------
# The trained model object (e.g., `rf_model`) is used as-is:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Statistic | Original method | Optimized method | Equivalence |
|---|---|---|---|
| **Mean** | `mean(vals[neighbors])` after removing NAs | `sum(vals[neighbors]) / count_non_na(neighbors)` via sparse matrix multiply | Algebraically identical. Sparse matrix computes exact sum; division by exact count yields identical IEEE 754 result. |
| **Max** | `max(vals[neighbors])` after removing NAs | `data.table` grouped `max` on the same neighbor values | Identical â€” same values, same `max()` function. |
| **Min** | `min(vals[neighbors])` after removing NAs | `data.table` grouped `min` on the same neighbor values | Identical â€” same values, same `min()` function. |
| **Neighbor structure** | `rook_neighbors_unique[[ref_idx]]` | Same `nb` object â†’ same sparse matrix / edge list | Identical graph. |
| **NA handling** | Drops NAs before aggregation; returns NA if all neighbors NA or no neighbors | Indicator-based count for mean; `edges_yr[!is.na(val)]` for max/min; cells with zero valid neighbors get NA | Identical behavior. |

The **trained Random Forest model is never modified** â€” it is applied to the enriched `cell_data` exactly as before, just with features computed ~1000Ã— faster.
END_RESPONSE>>>
