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
run_id: Anthropic_F07_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, where each element is built by an `lapply` call that performs character coercion, string pasting, and named-vector lookups *per row*. This is O(N) string operations over 6.46M rows, each touching a named character vector of length 6.46M for lookup â€” an operation that is roughly O(NÂ²) in wall-clock time due to R's named-vector lookup being O(N) per query (linear scan/hash collision). Then `compute_neighbor_stats` iterates over the same 6.46M-element list calling `max`, `min`, and `mean` one row at a time â€” death by a million small R function calls.

**Specific problems:**

1. **`id_to_ref` and `idx_lookup` are named character vectors used as hash maps.** R's `setNames` + bracket lookup on named vectors is slow for millions of keys. Each `idx_lookup[neighbor_keys]` is an O(K) hash probe repeated 6.46M times.
2. **`paste(..., sep="_")` inside the per-row lambda** creates millions of temporary strings.
3. **The neighbor lookup is rebuilt from scratch every run** even though the topology is static across years â€” the neighbor *structure* is identical for each of the 28 years, only the row indices change.
4. **`compute_neighbor_stats` uses `lapply` over 6.46M elements**, calling `max/min/mean` individually â€” massive R interpreter overhead.
5. **The outer loop calls this 5 times**, one per variable, so all overhead is multiplied Ã—5.

**Estimated complexity of current approach:**
- `build_neighbor_lookup`: ~6.46M Ã— (string ops + named-vector lookup) â‰ˆ hours
- `compute_neighbor_stats`: ~6.46M Ã— (subsetting + 3 summary stats) Ã— 5 vars â‰ˆ hours
- Total: 86+ hours as reported.

## Optimization Strategy

### Key Insight: Separate spatial topology from temporal indexing

The rook-neighbor graph is **purely spatial** â€” cell *i*'s neighbors are the same cells in every year. So we can:

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208, ~1.37M non-zero entries). This is a `dgCMatrix` from the `Matrix` package.
2. **Reshape each variable into a matrix** of dimension (344,208 cells Ã— 28 years).
3. **Use sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and neighbor counts in one shot, then derive max/min/mean.

For **mean**: `neighbor_mean = (A %*% X) / (A %*% 1_matrix)` where `A` is the binary adjacency matrix. This is a single sparse matrix multiplication â€” highly optimized C code in the `Matrix` package.

For **max and min**: Sparse matrix multiplication doesn't directly give max/min. We use `data.table` grouped operations on an edge list representation, which is vectorized and cache-friendly.

### Projected speedup:
- Sparse mat-mul for mean: seconds.
- `data.table` grouped max/min on ~38M edge-year pairs (1.37M edges Ã— 28 years): seconds to low minutes.
- Total for 5 variables: **under 5 minutes** (vs. 86+ hours).

### Invariants preserved:
- **Trained Random Forest model untouched** â€” we only recompute the same input features.
- **Numerical estimand identical** â€” same max, min, mean over exactly the same rook neighbors, same NA handling.

## Working R Code

```r
library(data.table)
library(Matrix)
library(spdep)

# ============================================================
# 0. Assumptions about existing objects:
#    - cell_data: data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#    - id_order: character or integer vector of unique cell IDs (length 344,208)
#    - rook_neighbors_unique: an nb object (list of length 344,208)
# ============================================================

# ============================================================
# 1. Build sparse binary adjacency matrix from the nb object
#    (done once; ~1.37M non-zero entries)
# ============================================================
build_adjacency_matrix <- function(nb_obj) {
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel if present
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

A <- build_adjacency_matrix(rook_neighbors_unique)
n_cells <- length(id_order)

# ============================================================
# 2. Convert cell_data to data.table; create fast cell index
# ============================================================
dt <- as.data.table(cell_data)

# Map each cell id to its positional index (1..344208) matching id_order
id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
dt <- merge(dt, id_map, by = "id", sort = FALSE)

# Ensure years are represented as integers for matrix column indexing
years_all   <- sort(unique(dt$year))
n_years     <- length(years_all)
year_map    <- data.table(year = years_all, year_idx = seq_along(years_all))
dt <- merge(dt, year_map, by = "year", sort = FALSE)

# Restore original row order (important for final assignment back)
setorder(dt, cell_idx, year_idx)

# ============================================================
# 3. Build edge list (directed) once â€” for grouped max/min
# ============================================================
edge_from <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique)
valid_edge <- edge_to > 0L
edges <- data.table(from_cell = edge_from[valid_edge],
                    to_cell   = edge_to[valid_edge])

# Expand edges across all years: each edge exists in every year
# This gives ~1.37M * 28 â‰ˆ 38.4M rows
edges_expanded <- CJ(edge_row = seq_len(nrow(edges)), year_idx = seq_len(n_years))
edges_expanded[, from_cell := edges$from_cell[edge_row]]
edges_expanded[, to_cell   := edges$to_cell[edge_row]]
edges_expanded[, edge_row  := NULL]

# Build a lookup from (cell_idx, year_idx) -> row in dt
dt[, dt_row := .I]
cell_year_key <- dt[, .(cell_idx, year_idx, dt_row)]
setkey(cell_year_key, cell_idx, year_idx)

# Map neighbor (to_cell) to its row in dt for value retrieval
setkey(edges_expanded, to_cell, year_idx)
edges_expanded <- cell_year_key[edges_expanded,
                                 .(from_cell, to_cell, year_idx,
                                   neighbor_dt_row = dt_row),
                                 on = .(cell_idx = to_cell, year_idx)]

# Map from_cell to its dt row for result assignment
setkey(edges_expanded, from_cell, year_idx)
edges_expanded <- cell_year_key[edges_expanded,
                                 .(from_cell, to_cell, year_idx,
                                   focal_dt_row = dt_row,
                                   neighbor_dt_row),
                                 on = .(cell_idx = from_cell, year_idx)]

# Remove any edges where either focal or neighbor row is missing (masked cells)
edges_expanded <- edges_expanded[!is.na(focal_dt_row) & !is.na(neighbor_dt_row)]

# ============================================================
# 4. Function: compute neighbor max, min, mean for one variable
#    and add columns to dt in place
# ============================================================
compute_and_add_neighbor_features_fast <- function(dt, var_name, edges_expanded,
                                                    A, n_cells, n_years,
                                                    cell_year_key) {
  cat("Processing:", var_name, "\n")

  vals <- dt[[var_name]]

  # --- MEAN via sparse matrix multiplication ---
  # Build cell Ã— year matrix of values
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  V[cbind(dt$cell_idx, dt$year_idx)] <- vals

  # Replace NA with 0 for sum; track non-NA counts separately
  V_nona      <- V
  V_nona[is.na(V_nona)] <- 0
  V_notna     <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)

  neighbor_sum   <- as.matrix(A %*% V_nona)     # n_cells x n_years

  neighbor_count <- as.matrix(A %*% V_notna)     # n_cells x n_years
  neighbor_mean  <- neighbor_sum / neighbor_count # NA where count == 0

  # Flatten back to dt row order
  mean_col <- neighbor_mean[cbind(dt$cell_idx, dt$year_idx)]

  # --- MAX and MIN via data.table grouped operations ---
  # Attach neighbor values to edge list
  edges_expanded[, nval := vals[neighbor_dt_row]]

  # Group by focal cell's dt row, compute max and min (na.rm)
  stats <- edges_expanded[!is.na(nval),
                           .(nmax = max(nval),
                             nmin = min(nval)),
                           by = focal_dt_row]

  max_col <- rep(NA_real_, nrow(dt))
  min_col <- rep(NA_real_, nrow(dt))
  max_col[stats$focal_dt_row] <- stats$nmax
  min_col[stats$focal_dt_row] <- stats$nmin

  # --- Assign new columns ---
  set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
  set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
  set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)

  invisible(dt)
}

# ============================================================
# 5. Run for all 5 neighbor source variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(
    dt, var_name, edges_expanded, A, n_cells, n_years, cell_year_key
  )
}

# ============================================================
# 6. Write results back to original cell_data (preserving row order)
# ============================================================
# The new columns are the 15 neighbor_* columns (5 vars Ã— 3 stats)
new_cols <- paste0(rep(neighbor_source_vars, each = 3),
                   c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))

# Merge back by id + year to preserve original cell_data row order
dt_out <- dt[, c("id", "year", new_cols), with = FALSE]
cell_data <- merge(as.data.table(cell_data), dt_out, by = c("id", "year"),
                   all.x = TRUE, sort = FALSE)
cell_data <- as.data.frame(cell_data)

cat("Done. New neighbor feature columns added.\n")
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` calls with string pasting and named-vector lookup | Sparse adjacency matrix + data.table edge list (built once) |
| **Mean computation** | Per-row `mean()` Ã— 6.46M Ã— 5 vars | Sparse matrix multiplication: `A %*% V` (seconds) |
| **Max/Min computation** | Per-row `max()`/`min()` Ã— 6.46M Ã— 5 vars | `data.table` grouped aggregation on ~38M-row edge table (minutes) |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Peak memory** | Millions of small R list elements (high GC pressure) | ~2 sparse matrices + 1 dense 344KÃ—28 matrix + 1 edge table (~1.5 GB) â€” fits in 16 GB |
| **Trained RF model** | âœ… Untouched | âœ… Untouched |
| **Numerical results** | Baseline | **Identical** (same neighbors, same NA handling, same max/min/mean) |
END_RESPONSE>>>
