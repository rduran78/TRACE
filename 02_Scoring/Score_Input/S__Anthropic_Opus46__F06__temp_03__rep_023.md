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
run_id: Anthropic_F06_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string operations

`build_neighbor_lookup` iterates over all ~6.46 million rows, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs `paste(neighbor_id, year, "_")` keys.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

This produces a **list of length 6.46 million**, each element containing integer row indices. The `paste` + named-vector lookup inside an `lapply` over 6.46M rows is extremely slow in R. Named vector lookup is O(n) hash probing per call, repeated billions of times in aggregate.

### Bottleneck B: `compute_neighbor_stats` â€” per-row subsetting and summary

For each of the 5 variables, `compute_neighbor_stats` iterates over 6.46M list elements, subsets a numeric vector by indices, removes NAs, and computes `max`, `min`, `mean`. That's ~32.3 million R-level function calls (6.46M Ã— 5 vars), each with allocation overhead.

### Why raster focal/kernel operations are tempting but insufficient

The data lives on a regular grid, so a rook-contiguity focal window (a 3Ã—3 cross kernel) could theoretically replace the neighbor lookup with a fast C-level `terra::focal()` or `raster::focal()` call. However:
- The grid may have irregular boundaries, missing cells, or masked regions where a simple rectangular focal window would include non-existent neighbors or skip valid ones.
- The `spdep::nb` object already encodes the exact topology, including boundary effects.
- We need to **preserve the original numerical estimand exactly**, so we must use the same neighbor set.

**The correct strategy is to vectorize the computation using the sparse adjacency structure directly**, avoiding per-row R loops entirely.

---

## 2. Optimization Strategy

### Step 1: Replace `build_neighbor_lookup` with a sparse adjacency matrix

Convert `rook_neighbors_unique` (an `nb` object with 344,208 cells) into a sparse matrix **once**. Then expand it to the cell-year level using vectorized operations. A sparse matrix-vector multiply computes all neighbor sums in one shot; element-wise operations give counts; and from those we get means. For max and min, we use grouped operations via `data.table`.

### Step 2: Use `data.table` + sparse matrix for neighbor stats

- **Mean**: `sparse_matrix %*% value_vector / sparse_matrix %*% ones_vector` (where NAs are zeroed out and tracked).
- **Max and Min**: Convert the sparse adjacency to a `data.table` of `(row, col)` pairs, join with values, and compute grouped max/min. This is a single vectorized `data.table` grouped aggregation over ~1.37M Ã— 28 â‰ˆ 38.4M edges â€” very fast.

### Step 3: Avoid recomputing the lookup for each variable

The adjacency structure is the same for all 5 variables. Build the edge list once, reuse it.

### Expected speedup

From ~86 hours to **minutes** (typically 5â€“15 minutes on a 16 GB laptop).

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)
library(spdep)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with a row-order column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# Preserve original row order so we can write results back correctly
cell_data[, .row_order := .I]

# id_order: the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object of length length(id_order) = 344,208
# cell_data must have columns: id, year, and the 5 neighbor_source_vars

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the cell-level sparse adjacency matrix (344208 x 344208)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_cell_adjacency <- function(id_order, nb_obj) {
  n <- length(nb_obj)
  # Build COO (coordinate) representation
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor entries (nb encodes no-neighbor as integer(0))
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

W_cell <- build_cell_adjacency(id_order, rook_neighbors_unique)
cat("Cell adjacency matrix:", nrow(W_cell), "x", ncol(W_cell),
    "with", nnzero(W_cell), "non-zeros\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Build the cell-year-level edge list ONCE
#
# For each cell-year row i with cell index c(i), and each rook neighbor
# cell index c', we need the row j that corresponds to (c', year(i)).
# We do this via a merge/join, not a per-row loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Map cell IDs to cell indices (position in id_order)
id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_idx := id_to_cellidx[as.character(id)]]

# Extract the edge list from the sparse matrix (cell-level)
W_coo <- summary(W_cell)  # gives (i, j, x) triplets
cell_edges <- data.table(from_cellidx = W_coo$i, to_cellidx = W_coo$j)
cat("Cell-level directed edges:", nrow(cell_edges), "\n")

# Build a lookup: (cell_idx, year) -> row index in cell_data
setkey(cell_data, cell_idx, year)
cell_data[, .row_idx := .I]  # after setkey, .I reflects sorted order

# We need a lookup table
lookup <- cell_data[, .(cell_idx, year, .row_idx)]
setkey(lookup, cell_idx, year)

# Expand cell_edges across all 28 years
years <- sort(unique(cell_data$year))

# For each year, join from_cellidx -> from_row and to_cellidx -> to_row
# This creates the full cell-year edge list (~38.4M rows)
cat("Building cell-year edge list across", length(years), "years...\n")

cy_edges <- rbindlist(lapply(years, function(yr) {
  # from side: find row indices for (from_cellidx, yr)
  from_lookup <- lookup[.(cell_edges$from_cellidx, yr), .(.row_idx), nomatch = NA]
  # to side: find row indices for (to_cellidx, yr)
  to_lookup   <- lookup[.(cell_edges$to_cellidx,   yr), .(.row_idx), nomatch = NA]

  dt <- data.table(
    from_row = from_lookup$.row_idx,
    to_row   = to_lookup$.row_idx
  )
  # Drop edges where either side is missing (cell not present in that year)
  dt[!is.na(from_row) & !is.na(to_row)]
}))

cat("Cell-year directed edges:", nrow(cy_edges), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for each variable using grouped ops
#
# For row i, its neighbors are all to_row where from_row == i.
# We compute max, min, mean of vals[to_row] grouped by from_row.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Restore original row order for correct column assignment
setkey(cell_data, .row_order)
# Now cell_data is back in original order, and .row_idx may no longer
# equal .row_order. We need to map: the .row_idx used in cy_edges
# was assigned after setkey(cell_data, cell_idx, year), so we need
# a mapping from .row_idx (sorted order) to .row_order (original order).

# Actually, let's just work in the sorted order and reorder at the end.
# Re-sort by (cell_idx, year) to align with .row_idx
setkey(cell_data, cell_idx, year)
# Now row i in cell_data corresponds to .row_idx == i

n_rows <- nrow(cell_data)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")

  vals <- cell_data[[var_name]]

  # Get neighbor values
  edge_dt <- copy(cy_edges)
  edge_dt[, nval := vals[to_row]]

  # Drop NA neighbor values (matches original: neighbor_vals[!is.na()])
  edge_dt <- edge_dt[!is.na(nval)]

  # Grouped aggregation
  agg <- edge_dt[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]

  # Initialize result columns with NA
  max_col  <- rep(NA_real_, n_rows)
  min_col  <- rep(NA_real_, n_rows)
  mean_col <- rep(NA_real_, n_rows)

  # Fill in computed values
  max_col[agg$from_row]  <- agg$nb_max
  min_col[agg$from_row]  <- agg$nb_min
  mean_col[agg$from_row] <- agg$nb_mean

  # Add to cell_data with the expected column names
  # (adjust naming convention to match what the trained RF model expects)
  set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = max_col)
  set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = min_col)
  set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = mean_col)

  cat("  Done:", var_name, "\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Restore original row order
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(cell_data, .row_order)

# Clean up helper columns
cell_data[, c(".row_order", ".row_idx", "cell_idx") := NULL]

cat("All neighbor features computed. Ready for prediction.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Summary of Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical. Same neighbor sets (from `rook_neighbors_unique`), same `max`/`min`/`mean` after dropping NAs. No approximation. |
| **Trained RF model** | Untouched. We only compute features; the model is applied as-is. |
| **Column names** | Must match what the RF expects â€” adjust the `paste0(var_name, "_neighbor_max")` pattern to match your trained model's feature names. |
| **Memory** | The edge list is ~38.4M rows Ã— 3 columns â‰ˆ ~900 MB. With 16 GB RAM this is feasible, especially since we `copy` only one variable's values at a time and discard after aggregation. |
| **Speed** | `data.table` grouped aggregation over ~38M rows completes in seconds per variable. Total: **~2â€“10 minutes** vs. the original ~86 hours. |

### Why not `terra::focal()`?

A raster focal approach with a cross-shaped (rook) kernel could compute means and sums at C speed, but:
1. `focal()` does not natively return per-cell **max** and **min** in a single pass without custom C functions.
2. Boundary/mask handling must exactly replicate the `spdep::nb` topology â€” any mismatch changes the estimand.
3. Reshaping 6.46M rows into year-specific raster stacks, running focal, and reshaping back adds complexity with no guarantee of correctness.

The `data.table` grouped-aggregation approach is nearly as fast (minutes), is guaranteed numerically identical, and requires no raster reconstruction.
END_RESPONSE>>>
