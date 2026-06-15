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
run_id: Anthropic_F08_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It constructs a lookup of **row indices** for every cell-year combination â€” all ~6.46 million rows â€” even though the **neighbor topology is invariant across years**. The function pastes IDs and years into keys, hashes them, and then does ~6.46 million lookups into a named character vector of length ~6.46 million. This produces a list of ~6.46 million elements, each containing row indices into the full panel. The `compute_neighbor_stats` function then loops over that same list of ~6.46 million elements per variable.

**Root cause:** The code conflates the **static spatial graph** (which cell is neighbor to which cell â€” 344,208 cells, ~1.37M edges, year-invariant) with the **dynamic panel data** (variable values that change by year). By rebuilding this mapping per cell-year row instead of per cell, the cost is multiplied by 28Ã— and all string operations (paste, hash lookup) are applied to millions of rows unnecessarily.

**Specific costs:**
1. `build_neighbor_lookup`: Creates ~6.46M string keys, does ~6.46M hash lookups â†’ very slow.
2. `compute_neighbor_stats`: Iterates an R-level `lapply` over ~6.46M elements per variable â†’ slow.
3. Memory: The `neighbor_lookup` list holds ~6.46M integer vectors â†’ large.

---

## Optimization Strategy

**Separate static topology from dynamic values:**

1. **Build the neighbor graph once at the cell level (344K cells, not 6.46M cell-years).** Create a mapping from each cell's position in `id_order` to its neighbors' positions in `id_order`. This is a simple re-index of `rook_neighbors_unique` â€” essentially free.

2. **Organize data so that values for each year can be extracted as a matrix.** Sort the data by `(id, year)` or `(year, id)` and reshape the variable columns into matrices of dimension `[n_cells Ã— n_years]`. Then neighbor stats become matrix operations on indexed rows.

3. **Vectorize neighbor stat computation.** For each variable, build a sparse matrix or use fast row-indexed operations: for each cell, gather neighbor rows from the matrix, compute max/min/mean across neighbors for all 28 years simultaneously.

4. **Use `data.table` for speed** in reshaping and joining results back.

**Complexity reduction:**
- Current: O(n_cells Ã— n_years) string hashing and list construction â†’ ~6.46M operations in slow R.
- Proposed: O(n_cells) integer list construction + O(n_cells Ã— n_years) vectorized numeric operations â†’ ~344K list elements, vectorized column operations.

**Expected speedup:** From 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure data.table format and sort consistently
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# Ensure id_order is the canonical ordering (positions 1..N match rook_neighbors_unique)
# id_order: integer/character vector of cell IDs in the order matching rook_neighbors_unique
n_cells <- length(id_order)

# Create a map: cell_id -> position in id_order (1-based index into rook_neighbors_unique)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# ==============================================================================
# STEP 1: Build STATIC neighbor index at the cell level (not cell-year level)
#
# cell_neighbor_pos[[i]] = integer vector of positions (in id_order) that are
#                          neighbors of the cell at position i.
# This is just rook_neighbors_unique itself (an nb object is already a list of
# integer position vectors), but we ensure it's clean.
# ==============================================================================
cell_neighbor_pos <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]
})
# cell_neighbor_pos[[i]] gives neighbor positions for cell at position i in id_order


# ==============================================================================
# STEP 2: Reshape each variable into a matrix: n_cells rows Ã— n_years columns
#
# Row i corresponds to id_order[i].
# Column j corresponds to the j-th year in sorted order.
# ==============================================================================
years_sorted <- sort(unique(cell_dt$year))
n_years <- length(years_sorted)

# Add position column to data.table
cell_dt[, cell_pos := id_to_pos[as.character(id)]]

# Sort by cell_pos and year for consistent matrix filling
setkey(cell_dt, cell_pos, year)

# Verify we have complete panel (each cell appears in each year)
# If not complete, the matrix approach still works but needs NA-filling
stopifnot(nrow(cell_dt) == n_cells * n_years)

# Build year-to-column-index map
year_to_col <- setNames(seq_along(years_sorted), as.character(years_sorted))
cell_dt[, year_col := year_to_col[as.character(year)]]

# Function to extract a variable as an [n_cells x n_years] matrix
var_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]
  mat
}

# ==============================================================================
# STEP 3: Compute neighbor stats per variable using the static graph + matrices
#
# For each cell i (row i in the matrix), gather the rows of its neighbors,
# then compute columnwise (i.e., per-year) max, min, mean.
#
# To avoid an R-level loop over 344K cells being too slow, we use a sparse
# approach: build an edge list and use data.table grouping.
# ==============================================================================

# Build edge list: (focal_pos, neighbor_pos) â€” one row per directed edge
# This is static and reused for every variable.
edge_focal <- rep(seq_len(n_cells), times = lengths(cell_neighbor_pos))
edge_neighbor <- unlist(cell_neighbor_pos, use.names = FALSE)
n_edges <- length(edge_focal)

cat(sprintf("Edge list built: %d directed edges\n", n_edges))

# For each variable, we need neighbor max/min/mean per cell per year.
# Strategy: index into the matrix using the edge list, then group by focal cell.
#
# edge_values[e, y] = var_matrix[edge_neighbor[e], y]
# Then group by edge_focal[e] and compute max/min/mean per column (year).
#
# With ~1.37M edges Ã— 28 years this is ~38.4M values â€” fits comfortably in RAM.

compute_neighbor_stats_fast <- function(var_matrix, edge_focal, edge_neighbor,
                                        n_cells, n_years) {
  # Extract neighbor values for all edges: matrix [n_edges x n_years]
  neighbor_vals <- var_matrix[edge_neighbor, , drop = FALSE]  # n_edges x n_years

  # We need to group rows of neighbor_vals by edge_focal and compute
  # max, min, mean per year-column.
  #
  # Use a C-level split via data.table or manual approach.
  # Since n_edges ~ 1.37M and n_years = 28, we can use matrix splitting.

  # Pre-allocate result matrices
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Use data.table for fast grouped operations
  # Melt neighbor_vals into long form: (edge_id, year_col, value)
  # Then group by (focal, year_col)
  #
  # But 1.37M Ã— 28 = 38.4M rows is manageable.

  # Alternative: loop over years (only 28 iterations â€” very fast)
  for (y in seq_len(n_years)) {
    col_vals <- neighbor_vals[, y]  # length n_edges

    # Use data.table for fast grouped max/min/mean
    dt_tmp <- data.table(focal = edge_focal, val = col_vals)
    # Remove NAs before aggregation
    dt_tmp <- dt_tmp[!is.na(val)]

    if (nrow(dt_tmp) > 0) {
      agg <- dt_tmp[, .(
        vmax  = max(val),
        vmin  = min(val),
        vmean = mean(val)
      ), by = focal]

      max_mat[agg$focal, y]  <- agg$vmax
      min_mat[agg$focal, y]  <- agg$vmin
      mean_mat[agg$focal, y] <- agg$vmean
    }
  }

  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# ==============================================================================
# STEP 4: Run for all neighbor source variables and attach results to cell_dt
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Build the [n_cells x n_years] matrix for this variable
  var_matrix <- var_to_matrix(cell_dt, var_name, n_cells, n_years)

  # Compute neighbor stats
  stats <- compute_neighbor_stats_fast(var_matrix, edge_focal, edge_neighbor,
                                       n_cells, n_years)

  # Map results back to cell_dt rows using (cell_pos, year_col)
  idx <- cbind(cell_dt$cell_pos, cell_dt$year_col)

  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (max_col)  := stats$max[idx]]
  cell_dt[, (min_col)  := stats$min[idx]]
  cell_dt[, (mean_col) := stats$mean[idx]]

  cat(sprintf("  Done: %s\n", var_name))
}

# ==============================================================================
# STEP 5: Clean up helper columns and convert back if needed
# ==============================================================================
cell_dt[, c("cell_pos", "year_col") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object and prediction code remain exactly as before.
# e.g.:
# predictions <- predict(rf_model, newdata = cell_data)
#
# The column names (var_neighbor_max, var_neighbor_min, var_neighbor_mean)
# and their numerical values are identical to the original implementation,
# preserving the original numerical estimand.
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup level** | Cell-year (6.46M entries) | Cell (344K entries, static) |
| **String hashing** | ~6.46M paste + hash ops | Zero |
| **Lookup list size** | ~6.46M elements | Edge list: ~1.37M pairs (reused) |
| **Stat computation loop** | R `lapply` over 6.46M per variable | 28 `data.table` grouped aggregations per variable |
| **Total inner iterations** | ~32.3M (5 vars Ã— 6.46M) | 140 (5 vars Ã— 28 years), each vectorized over ~1.37M edges |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **Peak RAM** | Large (6.46M-element list + overhead) | ~1â€“2 GB (matrices + edge list) |
| **Numerical output** | Baseline | **Identical** â€” same max/min/mean over same neighbor sets |
| **RF model** | Pre-trained | **Unchanged, not retrained** |

The key insight: **factoring the invariant spatial topology out of the per-year computation** eliminates millions of redundant string operations, reduces the data structure from 6.46M entries to 344K + a reusable edge list, and enables vectorized year-by-year aggregation via `data.table`.
END_RESPONSE>>>
