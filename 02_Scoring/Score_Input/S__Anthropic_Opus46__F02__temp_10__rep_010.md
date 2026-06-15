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
run_id: Anthropic_F02_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The performance bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string operations

For every single row, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes each neighbor ID with the current year to create lookup keys (string allocation).
- Looks those keys up in a 6.46-million-entry named character vector (`idx_lookup`).

Named-vector lookup in R is **hash-based but with per-call overhead** that becomes devastating at 6.46M iterations. The `paste()` calls inside the loop generate millions of temporary character vectors. This single function likely accounts for the majority of the estimated 86+ hours.

### 2. `compute_neighbor_stats` â€” repeated R-level loops over ragged neighbor lists

For each of the 5 variables, the function iterates over 6.46M list elements in R, subsets a numeric vector, removes NAs, and computes `max/min/mean`. This is called 5 times (once per variable), but the per-element overhead of R's `lapply` with small anonymous functions is significant at this scale.

### 3. Memory pressure

With ~6.46M rows Ã— 110+ columns, the data frame alone is large. Building a 6.46M-element list of integer vectors (`neighbor_lookup`) adds substantial overhead. The `do.call(rbind, ...)` on 6.46M 3-element vectors is also memory-inefficient (creates a huge temporary list before binding).

---

## Optimization Strategy

| Problem | Solution | Expected Speedup |
|---|---|---|
| Per-row string `paste` + named-vector lookup in `build_neighbor_lookup` | Replace with a **vectorized join** using `data.table`. Pre-build an integer-keyed edge list of (row_index â†’ neighbor_row_index) pairs, then split once. | ~100â€“500Ã— |
| R-level `lapply` in `compute_neighbor_stats` | Use the edge list directly in `data.table` grouped aggregation (`max`, `min`, `mean` by source row), fully vectorized in C. | ~50â€“200Ã— |
| `do.call(rbind, ...)` on millions of small vectors | Eliminated â€” `data.table` returns a single matrix/data.table directly. | Memory + time |
| Repeated iteration for each variable | Compute all 5 variables' neighbor stats in a single grouped operation, or at least keep the edge list and avoid re-traversal. | ~5Ã— |
| 16 GB RAM constraint | `data.table` is memory-efficient; the edge list representation is more compact than a 6.46M-element ragged list. | Fits in RAM |

**Estimated wall-clock time after optimization: 2â€“10 minutes** (down from 86+ hours).

The key insight: instead of building a list of neighbor row indices and then looping over it, we build a **flat edge table** `(from_row, to_row)` and use `data.table` grouped operations which execute in compiled C code.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Preserves the trained Random Forest model and all original numerical outputs.
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# Step 1: Build a flat edge table mapping each row to its neighbor rows.
#
# Inputs:
#   cell_data              â€” data.frame/data.table with columns: id, year, ...
#   id_order               â€” character or integer vector; the cell IDs in the
#                            order used by the nb object
#   rook_neighbors_unique  â€” spdep nb object (list of integer index vectors)
#
# Output:
#   A data.table with two columns:
#     from_rowidx  â€” the row index in cell_data of the focal cell-year
#     to_rowidx    â€” the row index in cell_data of a neighbor cell-year
# --------------------------------------------------------------------------

build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {

  # Convert to data.table if needed (by reference if already one)
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- cell_data
  }

  # --- 1a. Expand the nb object into a flat (cell_id, neighbor_cell_id) table
  #     This is done once and is independent of year.
  n_cells <- length(id_order)
  from_cell <- rep(id_order, times = lengths(neighbors))
  to_cell   <- id_order[unlist(neighbors)]

  edges_cell <- data.table(from_id = from_cell, to_id = to_cell)

  # --- 1b. Build a row-index lookup:  (id, year) -> row index in dt
  dt[, rowidx := .I]
  row_lookup <- dt[, .(id, year, rowidx)]
  setkey(row_lookup, id, year)

  # --- 1c. Get unique years
  years <- unique(dt$year)

  # --- 1d. Cross join edges Ã— years, then map to row indices via keyed join
  #     To avoid a massive cross join all at once (1.37M edges Ã— 28 years
  #     = ~38.5M rows, very manageable), we do it in one shot.
  edge_year <- CJ_dt_edges(edges_cell, years)
  #     edge_year now has columns: from_id, to_id, year

  # Map from_id + year -> from_rowidx
  setnames(row_lookup, c("id", "year", "rowidx"), c("from_id", "year", "from_rowidx"))
  setkey(row_lookup, from_id, year)
  setkey(edge_year, from_id, year)
  edge_year <- row_lookup[edge_year, nomatch = 0L]

  # Map to_id + year -> to_rowidx
  # Rebuild lookup for "to" side
  row_lookup2 <- dt[, .(to_id = id, year, to_rowidx = rowidx)]
  setkey(row_lookup2, to_id, year)
  setkey(edge_year, to_id, year)
  edge_year <- row_lookup2[edge_year, nomatch = 0L]

  # Clean up the temporary column
  dt[, rowidx := NULL]

  edge_year[, .(from_rowidx, to_rowidx)]
}

# Helper: cross join edges with years vector
CJ_dt_edges <- function(edges_cell, years) {
  # edges_cell has from_id, to_id  (~1.37M rows)
  # years is a vector of length 28
  # Result: ~38.5M rows â€” fits easily in memory
  yr_dt <- data.table(year = years)
  res <- edges_cell[, .(from_id, to_id)]
  # Cross join via merge on dummy key
  res[, k := 1L]
  yr_dt[, k := 1L]
  out <- res[yr_dt, on = "k", allow.cartesian = TRUE]
  out[, k := NULL]
  out
}


# --------------------------------------------------------------------------
# Step 2: Compute all neighbor statistics in one vectorized pass per variable
#         (or all variables at once).
#
# For each variable, we need per focal row: max, min, mean of neighbor values.
# --------------------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, edge_table, var_names) {

  dt <- as.data.table(cell_data)
  dt[, rowidx := .I]

  # Bring neighbor values into the edge table
  # edge_table has: from_rowidx, to_rowidx
  et <- copy(edge_table)

  # Pre-extract variable columns as a matrix for fast column access
  var_mat <- as.matrix(dt[, ..var_names])

  # For each variable, attach the neighbor's value, then aggregate
  for (v in seq_along(var_names)) {
    vname <- var_names[v]

    # Attach neighbor value
    et[, nval := var_mat[to_rowidx, v]]

    # Compute grouped stats â€” fully vectorized in data.table's C backend
    stats <- et[!is.na(nval),
                .(nmax  = max(nval),
                  nmin  = min(nval),
                  nmean = mean(nval)),
                by = from_rowidx]

    # Create full-length result columns (NA where no valid neighbors)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))

    max_col[stats$from_rowidx]  <- stats$nmax
    min_col[stats$from_rowidx]  <- stats$nmin
    mean_col[stats$from_rowidx] <- stats$nmean

    # Add to the data.table with the same column naming convention
    set(dt, j = paste0("neighbor_max_",  vname), value = max_col)
    set(dt, j = paste0("neighbor_min_",  vname), value = min_col)
    set(dt, j = paste0("neighbor_mean_", vname), value = mean_col)
  }

  et[, nval := NULL]  # clean up
  dt[, rowidx := NULL]

  dt
}


# --------------------------------------------------------------------------
# Step 3: Main execution â€” drop-in replacement for the original outer loop
# --------------------------------------------------------------------------

# ---- Build edge table (once) ----
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# ---- Compute all neighbor features (vectorized) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features...")
cell_data <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)
message("Done.")

# cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# These are numerically identical to the original implementation's output.
# The trained Random Forest model is untouched â€” only the feature preparation
# code has been optimized.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` aggregations are computed over exactly the same set of neighbor values per focal cell-year. The `nomatch = 0L` join drops edges where a neighbor cell-year doesn't exist in the data, exactly matching the original `!is.na(result)` filter. |
| **Trained RF model** | No model retraining. The code only prepares the feature columns that the model expects. Column names follow the same naming convention (`neighbor_max_<var>`, etc.). Adjust the naming if your original code uses a different convention. |
| **Edge cases** | Rows with zero valid neighbors get `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |

---

## Memory Budget Estimate

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 125 cols, numeric) | ~6.5 GB |
| `edge_table` (38.5M Ã— 2 integer cols) | ~0.3 GB |
| Temporary `nval` column in edge table | ~0.3 GB |
| `stats` intermediate | ~0.15 GB |
| **Total peak** | **~8â€“9 GB** (fits in 16 GB) |

This should complete in approximately **2â€“10 minutes** on a standard laptop, down from the estimated 86+ hours.
END_RESPONSE>>>
