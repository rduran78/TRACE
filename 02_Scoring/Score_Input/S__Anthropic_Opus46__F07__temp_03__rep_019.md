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
run_id: Anthropic_F07_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The lookup is O(n) in list overhead** â€” `idx_lookup` is a named character vector of length ~6.46M. Named lookup on such a vector is O(n) per probe in the worst case (R uses hashing for named vectors, but the sheer number of calls and allocations dominates).
3. **`compute_neighbor_stats`** then iterates over the 6.46M-element list again, extracting small vectors each time â€” this is fast*er* but still slow due to R-level loop overhead on millions of tiny vectors.

**Net effect**: ~6.46M R-level iterations, each doing string operations and hash lookups, with massive intermediate allocation. This is why the estimate is 86+ hours.

### Why naive raster focal operations are unsafe
The document correctly notes that the cell topology may be irregular/masked. A `terra::focal()` or `raster::focal()` assumes a complete rectangular grid with uniform 4- or 8-connectivity. If cells are masked out (ocean, borders, etc.), focal operations silently include or exclude wrong neighbors. The `spdep::nb` object encodes the *exact* irregular rook-neighbor graph and must be honored.

---

## Optimization Strategy

### Key insight: replace per-row R loops with vectorized joins and sparse-matrix multiplication.

**Step 1 â€” Vectorized lookup via `data.table` merge (replaces `build_neighbor_lookup`)**

Instead of building a 6.46M-element list, create an **edge table** (a two-column data.table of `(focal_row, neighbor_row)`) using a single vectorized merge. This avoids all per-row string operations.

**Step 2 â€” Compute neighbor stats via `data.table` grouped aggregation (replaces `compute_neighbor_stats`)**

Group the edge table by `focal_row` and compute `max`, `min`, `mean` of the neighbor values in one vectorized pass. `data.table` does this in C, not R-level loops.

**Step 3 (alternative for even more speed) â€” Sparse matrix multiplication**

For `mean`, construct a row-normalized sparse adjacency matrix **W** (one-time cost). Then `neighbor_mean = W %*% x` is a single sparse matrix-vector multiply for each variable. Similarly, use the binary (non-normalized) adjacency matrix with custom operations for max/min via the `Matrix` package or `slam`. However, max/min are not linear, so the `data.table` grouped approach is simpler and fast enough.

**Expected speedup**: From ~86 hours to **~2â€“5 minutes** (vectorized C-level operations on ~20M edge-year pairs instead of 6.46M R-level iterations).

**Preservation guarantees**:
- The exact same rook-neighbor graph (`rook_neighbors_unique`) is used.
- The same `max`, `min`, `mean` statistics are computed on the same neighbor sets.
- No retraining of the Random Forest model is needed; the output columns are numerically identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact rook-neighbor topology, numerical results, trained RF model
# =============================================================================

library(data.table)

# ---------- 0. Convert cell_data to data.table if not already ----------------
if (!is.data.table(cell_data)) {
 cell_data <- as.data.table(cell_data)
}

# ---------- 1. Build the directed edge table (one-time, vectorized) ----------
# rook_neighbors_unique: an spdep nb object (list of integer vectors)
#   rook_neighbors_unique[[i]] gives the indices (into id_order) of
#   the rook neighbors of id_order[i].
# id_order: vector of cell IDs in the order matching the nb object.

build_edge_table <- function(id_order, neighbors) {
  # Expand the nb list into a two-column table of (focal_cell_id, neighbor_cell_id)
  n_neighbors <- lengths(neighbors)                 # integer vector, fast
  focal_idx   <- rep.int(seq_along(neighbors), n_neighbors)
  nbr_idx     <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id = id_order[focal_idx],
    nbr_id   = id_order[nbr_idx]
  )
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---------- 2. Create a row key in cell_data --------------------------------
# We need to join edges Ã— years. Strategy: merge edge_dt with cell_data on
# the neighbor side to pull in neighbor values, grouped by (focal_id, year).

# Ensure id and year columns are keyed for fast joins
setkey(cell_data, id, year)

# ---------- 3. Compute neighbor stats for all variables (vectorized) ---------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will cross-join the edge table with years, then merge neighbor values.
# But cross-joining 1.37M edges Ã— 28 years = 38.4M rows is fine in RAM
# (~1â€“2 GB for a few numeric columns).

# More memory-efficient: loop over variables, merge only needed columns.

cat("Computing neighbor features...\n")

# Prepare a slim lookup: for each (id, year), the values of all source vars
# This avoids repeated merges.
lookup_cols <- c("id", "year", neighbor_source_vars)
val_lookup  <- cell_data[, ..lookup_cols]
setnames(val_lookup, "id", "nbr_id")
setkey(val_lookup, nbr_id, year)

# Get unique years
years <- sort(unique(cell_data$year))

# Process in year chunks to control peak memory (each year: ~1.37M edge rows)
# Accumulate results in a list, then rbindlist and merge back.

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  set(cell_data, j = paste0("n_max_", var_name),  value = NA_real_)
  set(cell_data, j = paste0("n_min_", var_name),  value = NA_real_)
  set(cell_data, j = paste0("n_mean_", var_name), value = NA_real_)
}

# Create an index for fast update: (id, year) -> row position in cell_data
cell_data[, .row_idx := .I]
setkey(cell_data, id, year)

for (yr in years) {
  # Subset neighbor values for this year
  yr_vals <- val_lookup[year == yr]  # keyed on (nbr_id, year)
  yr_vals[, year := NULL]            # drop year, keep nbr_id + var columns

  # Merge edges with neighbor values: edge_dt has (focal_id, nbr_id)
  # yr_vals has (nbr_id, ntl, ec, pop_density, def, usd_est_n2)
  merged <- merge(edge_dt, yr_vals, by = "nbr_id", all.x = FALSE)
  # all.x = FALSE: drop edges where neighbor has no data this year (same as
  # the original !is.na filter in build_neighbor_lookup)

  # Group by focal_id, compute stats for all variables at once
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    vn <- as.name(var_name)
    agg_exprs[[paste0("n_max_", var_name)]]  <-
      bquote(max(.(vn), na.rm = TRUE))
    agg_exprs[[paste0("n_min_", var_name)]]  <-
      bquote(min(.(vn), na.rm = TRUE))
    agg_exprs[[paste0("n_mean_", var_name)]] <-
      bquote(mean(.(vn), na.rm = TRUE))
  }

  # Build the aggregation call
  stats <- merged[, lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(x), min(x), mean(x))
  }), by = focal_id, .SDcols = neighbor_source_vars]

  # The above returns 3 rows per focal_id (max, min, mean). Instead, use
  # explicit aggregation for clarity and correctness:
  stats <- merged[, {
    res <- list()
    for (vn in neighbor_source_vars) {
      v <- .SD[[vn]]
      v <- v[!is.na(v)]
      if (length(v) == 0L) {
        res[[paste0("n_max_", vn)]]  <- NA_real_
        res[[paste0("n_min_", vn)]]  <- NA_real_
        res[[paste0("n_mean_", vn)]] <- NA_real_
      } else {
        res[[paste0("n_max_", vn)]]  <- max(v)
        res[[paste0("n_min_", vn)]]  <- min(v)
        res[[paste0("n_mean_", vn)]] <- mean(v)
      }
    }
    res
  }, by = focal_id, .SDcols = neighbor_source_vars]

  # Now update cell_data for this year
  stats[, year := yr]
  setkey(stats, focal_id, year)

  stat_cols <- setdiff(names(stats), c("focal_id", "year"))

  # Join and update in place
  # First get the row indices in cell_data that match
  idx_dt <- cell_data[stats, on = .(id = focal_id, year = year),
                      which = TRUE,  # returns row indices
                      nomatch = 0L]

  # Actually, use a proper update join:
  # Rename focal_id to id for joining
  setnames(stats, "focal_id", "id")
  setkey(stats, id, year)

  cell_data[stats, on = .(id, year),
            (stat_cols) := mget(paste0("i.", stat_cols))]

  if (yr %% 5 == 0 || yr == years[1] || yr == years[length(years)]) {
    cat(sprintf("  Year %d done\n", yr))
  }
}

# Clean up helper column
cell_data[, .row_idx := NULL]

# Handle -Inf/Inf from max/min of empty sets (shouldn't occur due to NA guard,
# but defensive)
for (var_name in neighbor_source_vars) {
  for (prefix in c("n_max_", "n_min_", "n_mean_")) {
    col <- paste0(prefix, var_name)
    vals <- cell_data[[col]]
    vals[is.infinite(vals)] <- NA_real_
    set(cell_data, j = col, value = vals)
  }
}

cat("Neighbor feature computation complete.\n")
```

---

### Even Faster: Sparse-Matrix Approach for `mean` (Optional)

If you want maximum speed (seconds, not minutes), use a sparse adjacency matrix for the mean, and data.table for max/min only:

```r
library(Matrix)

# Build sparse row-normalized adjacency matrix (one-time)
build_sparse_W <- function(id_order, neighbors) {
  n <- length(id_order)
  n_nbrs <- lengths(neighbors)
  i <- rep.int(seq_len(n), n_nbrs)
  j <- unlist(neighbors, use.names = FALSE)
  # Row-normalize: each weight = 1 / (number of neighbors of focal cell)
  x <- rep.int(1 / n_nbrs, sum(n_nbrs))
  # Handle cells with 0 neighbors (avoid 1/0)
  x[is.infinite(x)] <- 0
  sparseMatrix(i = i, j = j, x = x, dims = c(n, n))
}

W <- build_sparse_W(id_order, rook_neighbors_unique)

# For a single year-slice of a variable (vector of length n_cells, ordered by id_order):
# neighbor_mean = as.numeric(W %*% values_vector)
# This is O(nnz) â‰ˆ 1.37M multiplications â€” essentially instant.

# To apply across all years:
# Reshape variable into an n_cells Ã— n_years matrix, then:
# result_matrix <- W %*% values_matrix   # sparse Ã— dense, very fast

# For max and min, you still need the data.table grouped approach above,
# or use the igraph/sparse-matrix row-wise max trick.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level iterations with string ops | One vectorized `rep.int` + `unlist` (~0.5s) |
| **Stat computation** | 6.46M R-level `lapply` iterations | `data.table` grouped aggregation in C (~2â€“5 min) |
| **Peak RAM** | ~6.46M-element list of integer vectors (~4â€“8 GB) | Edge table + merge (~2â€“3 GB) |
| **Total time** | ~86+ hours | **~2â€“5 minutes** |
| **Numerical result** | Exact rook-neighbor max/min/mean | **Identical** |
| **RF model** | Trained | **Untouched â€” no retraining** |
END_RESPONSE>>>
