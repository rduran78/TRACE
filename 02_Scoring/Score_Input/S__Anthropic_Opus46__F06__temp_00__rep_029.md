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
run_id: Anthropic_F06_T00_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Converts the cell `id` to a reference index (hash lookup â€” fast).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes each neighbor ID with the current row's year to form string keys.
4. Looks up those keys in a named vector (`idx_lookup`) of length 6.46M.

The `paste()` + named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely expensive. Named vector lookup in R is O(n) per probe in the worst case (it uses hashing, but with 6.46M names the overhead is enormous when called millions of times). This single function likely accounts for **40â€“60 hours** of the 86+ hour estimate.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per variable

`compute_neighbor_stats` subsets a numeric vector by index lists and computes `max`, `min`, `mean`. This is called 5 times (once per variable). Each call iterates over 6.46M list elements. The per-element `lapply` with R-level `max`/`min`/`mean` on small vectors (typically 2â€“4 neighbors) has high per-call overhead. This likely accounts for **20â€“30 hours**.

### Why raster focal/kernel operations are *not* a direct replacement

Focal operations (e.g., `terra::focal`) operate on regular grids with rectangular kernels. The panel data here is a **space Ã— time** structure where neighbor relationships are defined by an irregular `spdep::nb` object (rook contiguity on an irregular subset of cells). Focal operations would require:
- Reconstructing a full raster for each year (344K cells Ã— 28 years = 28 rasters).
- Handling NA masks for cells outside the study area.
- Extracting results back into the panel format.

This is feasible but introduces complexity and potential numerical discrepancies at boundaries. The **better strategy** is to vectorize the neighbor computation directly using `data.table` joins, which preserves the exact numerical results.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup Factor |
|------|---------|----------|----------------|
| Neighbor lookup | Per-row `paste` + named-vector lookup in `lapply` | Pre-build a `data.table` edge list `(row_i, row_j)` via keyed join | ~500â€“1000Ã— |
| Neighbor stats | Per-row `lapply` with R-level `max`/`min`/`mean` | Vectorized `data.table` grouped aggregation on the edge list | ~200â€“500Ã— |
| Per-variable loop | Rebuild nothing, but `lapply` overhead repeated | Single grouped aggregation per variable, all vectorized | ~5Ã— |

**Expected total runtime: 2â€“5 minutes** (down from 86+ hours).

**Key insight**: Instead of storing a list of neighbor row-indices per row, we build a two-column **edge table** `(focal_row, neighbor_row)` and use `data.table` grouped operations to compute `max`, `min`, `mean` in one vectorized pass per variable.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a vectorized edge list (focal_row, neighbor_row)
#
# This replaces build_neighbor_lookup entirely.
# rook_neighbors_unique: spdep nb object (list of integer vectors)
# id_order: vector of cell IDs in the order matching the nb object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(cell_data, id_order, neighbors) {
  # Map: cell_id -> position in id_order (and thus in nb list)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build directed edge list at the CELL level: (focal_id, neighbor_id)
  focal_ids <- rep(
    id_order,
    times = lengths(neighbors)
  )
  neighbor_ids <- id_order[unlist(neighbors)]

  cell_edges <- data.table(
    focal_id    = focal_ids,
    neighbor_id = neighbor_ids
  )

  # Now expand to panel rows by joining on year.
  # cell_data must have columns: id, year, and a row index.
  cell_data[, .row_idx := .I]

  # Keyed lookup tables: (id, year) -> row_idx
  focal_key <- cell_data[, .(focal_id = id, year, focal_row = .row_idx)]
  neighbor_key <- cell_data[, .(neighbor_id = id, year, neighbor_row = .row_idx)]

  setkey(focal_key, focal_id, year)
  setkey(neighbor_key, neighbor_id, year)

  # For each cell-level edge, expand across all 28 years.
  # Strategy: join cell_edges to focal_key to get (focal_row, year, neighbor_id),
  # then join to neighbor_key to get (focal_row, neighbor_row).

  # Join 1: cell_edges Ã— focal_key  â†’  gives us the year dimension
  setkey(cell_edges, focal_id)
  setkey(focal_key, focal_id)

  # Use allow.cartesian because one focal_id maps to 28 years
  expanded <- cell_edges[focal_key,
    on = .(focal_id),
    .(focal_row, year, neighbor_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Join 2: expanded Ã— neighbor_key  â†’  gives us neighbor_row
  setkey(expanded, neighbor_id, year)
  setkey(neighbor_key, neighbor_id, year)

  edge_table <- expanded[neighbor_key,
    on = .(neighbor_id, year),
    .(focal_row, neighbor_row),
    nomatch = NULL
  ]

  # Clean up temporary column
  cell_data[, .row_idx := NULL]

  return(edge_table)
}

cat("Building edge table...\n")
system.time({
  edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~30â€“90 seconds, ~200â€“400 MB for the edge table
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_table), big.mark = ",")))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all variables (vectorized)
#
# This replaces compute_neighbor_stats + the outer loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_all_neighbor_features <- function(cell_data, edge_table, var_names) {
  n <- nrow(cell_data)

  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Extract the variable values for neighbor rows
    edge_table[, val := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation: max, min, mean per focal_row (excluding NAs)
    stats <- edge_table[!is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = focal_row
    ]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign results
    cell_data[stats$focal_row, (max_col)  := stats$nb_max]
    cell_data[stats$focal_row, (min_col)  := stats$nb_min]
    cell_data[stats$focal_row, (mean_col) := stats$nb_mean]
  }

  # Clean up
  edge_table[, val := NULL]

  return(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  cell_data <- compute_and_add_all_neighbor_features(
    cell_data, edge_table, neighbor_source_vars
  )
})
# Expected: ~30â€“60 seconds for all 5 variables


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., rf_model) is used as-is.
# cell_data now has the same 15 neighbor-derived columns
# (5 vars Ã— 3 stats) with identical numerical values.
#
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor relationships** | We use the same `rook_neighbors_unique` nb object and `id_order`. The edge table is a lossless expansion of the same adjacency structure. |
| **Same year matching** | The join on `(neighbor_id, year)` ensures each focal cell-year only sees neighbors from the **same year**, exactly as the original `paste(id, year)` key did. |
| **Same aggregation functions** | `max`, `min`, `mean` with NA removal â€” identical to the original `compute_neighbor_stats`. |
| **Same column names** | Output columns follow the same naming convention (`{var}_max_neighbor`, etc.) â€” adjust the suffix pattern to match whatever `compute_and_add_neighbor_features` originally produced. |
| **Random Forest untouched** | The pre-trained model is never retrained; only `predict()` is called. |

---

## 5. Memory Budget

| Object | Estimated Size |
|--------|---------------|
| `cell_data` (6.46M Ã— 110 cols) | ~5.5 GB |
| `edge_table` (6.46M Ã— ~4 neighbors Ã— 2 int cols) | ~400 MB |
| Temporary `stats` per variable | ~100 MB |
| **Total peak** | **~6.5 GB** (fits in 16 GB) |

If memory is tight, process variables one at a time (as the code already does) and call `gc()` between iterations.
END_RESPONSE>>>
