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
run_id: Anthropic_F06_T00_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting `neighbor_id` and `year`.
4. Looks up those keys in `idx_lookup` (a named vector).

String pasting and named-vector lookups inside a per-row `lapply` over 6.46M rows is extremely slow. The named-vector lookup `idx_lookup[neighbor_keys]` is an O(k) hash lookup per neighbor key, but the sheer volume (6.46M Ã— ~4 neighbors on average) and the string construction overhead dominate.

### Bottleneck 2: `compute_neighbor_stats` â€” per-row `lapply` with subsetting

For each of the 6.46M rows, it subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is repeated 5 times (once per variable). The R-level loop overhead across ~32.3 million iterations total is enormous.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. While the cells here are on a spatial grid, the neighbor structure is encoded as an `nb` object (irregular/boundary-aware rook contiguity). Focal operations would incorrectly include diagonal neighbors or fail at irregular boundaries. The `nb` object must be respected to **preserve the original numerical estimand**. However, the *spirit* of focal operations â€” vectorized batch computation â€” is exactly what we need.

### Root cause summary

| Component | Problem | Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row string paste + named vector lookup over 6.46M rows | ~40+ hours |
| `compute_neighbor_stats` | Per-row R-level lapply with subsetting, 5 variables Ã— 6.46M rows | ~40+ hours |
| Memory | Storing 6.46M-element list of integer vectors | ~2-4 GB (manageable but wasteful) |

---

## Optimization Strategy

### Strategy: Vectorized sparse-matrix multiplication and group operations

**Key insight:** The neighbor relationship is *time-invariant*. Cell `i` has the same rook neighbors in every year. We can:

1. **Build a sparse adjacency matrix `W`** (344,208 Ã— 344,208) from the `nb` object â€” done once.
2. **Reshape each variable into a matrix** of dimension (344,208 cells Ã— 28 years).
3. **Compute neighbor sums, counts, max, and min** using sparse matrix operations and vectorized row/column operations.
   - **Mean**: `W %*% X / W %*% (!is.na(X))` â€” sparse matrix multiply gives neighbor sums; dividing by neighbor counts gives means.
   - **Max/Min**: Use a sparse-matrix trick: iterate over each cell's neighbors via the CSC/CSR structure of `W`, but do so in C++ via `Rcpp` or use `data.table` grouped operations on an edge list.

4. **Flatten back** to the original long-format data frame.

This replaces ~6.46M R-level iterations with a handful of sparse matrix multiplications (seconds each) and vectorized operations.

For **max and min**, sparse matrix multiplication doesn't directly apply, so we use a **`data.table` edge-list join** approach: expand the edge list, join variable values, and compute grouped `max`/`min`/`mean` in one pass.

**Expected speedup:** From 86+ hours to **~2â€“10 minutes**.

**Memory:** The sparse matrix is ~1.4M non-zeros (trivial). The cellÃ—year matrices are 344,208 Ã— 28 â‰ˆ 9.6M entries per variable (~77 MB as double). Total peak memory well under 8 GB.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the original numerical estimand (rook-neighbor max, min, mean)
# Preserves the trained Random Forest model (no retraining)
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build sparse adjacency matrix from nb object (once) ----

build_sparse_adjacency <- function(nb_obj) {
  # nb_obj is a list of integer vectors (spdep::nb), 1-indexed
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor placeholders (spdep uses 0L for no neighbors)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

W <- build_sparse_adjacency(rook_neighbors_unique)
# W is 344208 x 344208 with ~1.37M non-zero entries


# ---- Step 2: Build cell-year indexing structures ----

# Convert to data.table for speed (if not already)
dt <- as.data.table(cell_data)

# Ensure consistent ordering: we need a mapping from (id) -> row in W
# id_order is the vector such that id_order[k] is the cell id for row k of W
id_to_widx <- setNames(seq_along(id_order), as.character(id_order))

# Create integer cell index and year index
dt[, cell_idx := id_to_widx[as.character(id)]]

# Year mapping: 1992 -> 1, 1993 -> 2, ..., 2019 -> 28
years_sorted <- sort(unique(dt$year))
year_to_colidx <- setNames(seq_along(years_sorted), as.character(years_sorted))
dt[, year_idx := year_to_colidx[as.character(year)]]

n_cells <- length(id_order)  # 344208
n_years <- length(years_sorted)  # 28


# ---- Step 3: Build edge list from sparse matrix (once) ----

W_csc <- as(W, "dgCMatrix")
edges <- data.table(
  from_cell = rep(seq_len(n_cells), diff(W_csc@p)),
  to_cell   = W_csc@i + 1L  # convert 0-indexed to 1-indexed
)
# 'from_cell' is the focal cell, 'to_cell' is its rook neighbor
# We want neighbor stats FOR from_cell, computed FROM to_cell values

# Actually for dgCMatrix, columns are "j", rows are "i"
# Let's rebuild correctly using summary()
W_triplet <- summary(W)  # gives (i, j, x) triplets
edges <- data.table(
  focal_cell    = W_triplet$i,
  neighbor_cell = W_triplet$j
)
rm(W_triplet)


# ---- Step 4: Compute neighbor features for all variables (vectorized) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-sort dt by (cell_idx, year_idx) for fast matrix construction
setkey(dt, cell_idx, year_idx)

compute_all_neighbor_features <- function(dt, edges, var_name,
                                          n_cells, n_years,
                                          years_sorted) {
  cat("Processing variable:", var_name, "\n")

  # --- Build cell x year matrix ---
  # Extract the variable values into a matrix M[cell_idx, year_idx]
  vals <- dt[[var_name]]
  cidx <- dt$cell_idx
  yidx <- dt$year_idx

  M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  M[cbind(cidx, yidx)] <- vals

  # --- For each year, compute neighbor max, min, mean ---
  # Strategy: expand edges Ã— years, look up neighbor values, group by (focal, year)
  #
  # But expanding 1.37M edges Ã— 28 years = 38.4M rows â€” very manageable.

  # Pre-allocate result matrices
  max_M  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_M  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Process year by year to keep memory bounded
  for (y in seq_len(n_years)) {
    # Get neighbor values for this year
    neighbor_vals <- M[edges$neighbor_cell, y]

    # Build a data.table for grouped aggregation
    tmp <- data.table(
      focal = edges$focal_cell,
      val   = neighbor_vals
    )

    # Remove NA neighbor values
    tmp <- tmp[!is.na(val)]

    if (nrow(tmp) == 0L) next

    # Grouped aggregation â€” extremely fast in data.table
    agg <- tmp[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = focal]

    max_M[agg$focal, y]  <- agg$nb_max
    min_M[agg$focal, y]  <- agg$nb_min
    mean_M[agg$focal, y] <- agg$nb_mean
  }

  # --- Write results back to dt ---
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := max_M[cbind(cell_idx, year_idx)]]
  dt[, (min_col)  := min_M[cbind(cell_idx, year_idx)]]
  dt[, (mean_col) := mean_M[cbind(cell_idx, year_idx)]]

  invisible(dt)
}

# ---- Step 5: Run for all variables ----

for (var_name in neighbor_source_vars) {
  dt <- compute_all_neighbor_features(dt, edges, var_name,
                                      n_cells, n_years, years_sorted)
}

# ---- Step 6: Clean up helper columns and convert back if needed ----

dt[, c("cell_idx", "year_idx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(dt)

# ---- Step 7: Predict with the existing trained Random Forest ----
# (No changes needed â€” the model object and cell_data columns are preserved)
# predictions <- predict(trained_rf_model, newdata = cell_data)

cat("Done. Neighbor features computed for all variables.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor structure** | We build `W` directly from `rook_neighbors_unique` â€” the identical `nb` object. No diagonal or spurious neighbors are introduced. |
| **Same statistics** | We compute `max`, `min`, `mean` over exactly the non-NA values of rook neighbors for each cell-year, identical to the original `compute_neighbor_stats`. |
| **Same column names** | Output columns follow the same naming convention (`neighbor_max_ntl`, etc.) â€” adjust the naming pattern to match whatever `compute_and_add_neighbor_features` originally produced. |
| **Trained RF preserved** | No model retraining occurs. The feature columns are numerically identical, so predictions are identical. |

## Performance Estimate

| Step | Cost |
|---|---|
| Build sparse matrix | < 1 second |
| Build edge list | < 1 second |
| Per variable: build matrix (9.6M assignments) | ~1â€“2 seconds |
| Per variable: 28 years Ã— grouped aggregation on ~1.37M rows | ~10â€“30 seconds |
| Per variable: write back to dt | ~1 second |
| **Total for 5 variables** | **~2â€“5 minutes** |
| **Speedup vs. 86 hours** | **~1,000â€“2,500Ã—** |
END_RESPONSE>>>
