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
run_id: Anthropic_F02_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The performance bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` conversion and named-vector lookup (`id_to_ref`).
- `paste()` to construct neighbor keys.
- Named-vector lookup into `idx_lookup` (which itself is a 6.46M-element named character vector â€” every lookup is an O(n) hash probe on a very large table).
- Filtering `NA`s.

Named vectors in R use hashed environments under the hood, but building and probing a 6.46M-entry named vector millions of times is extremely slow. The result is a **list of 6.46M integer vectors** â€” itself a large, fragmented memory object.

### 2. `compute_neighbor_stats` â€” Another O(n) `lapply` over 6.46M rows, called 5 times

Each call iterates over every row, subsets a numeric vector by index, removes `NA`s, and computes `max`, `min`, `mean`. The `do.call(rbind, ...)` on a 6.46M-element list of length-3 vectors is also slow (repeated memory allocation).

### Combined cost

- `build_neighbor_lookup`: ~6.46M iterations Ã— expensive string operations â‰ˆ many hours.
- `compute_neighbor_stats`: ~6.46M Ã— 5 variables Ã— subsetting/aggregation â‰ˆ many more hours.
- Memory: the 6.46M-element neighbor lookup list, plus intermediate string vectors, can easily exceed 16 GB.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key construction and named-vector lookup | Replace with **integer join** using `data.table`. Build a `(cell_id, year) â†’ row_index` integer lookup table and join on integer keys â€” orders of magnitude faster. |
| 6.46M-element R list for neighbor_lookup | Flatten to a **two-column `data.table`** (`row_idx`, `neighbor_row_idx`). This is compact, vectorized, and groupable. |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with a **single `data.table` grouped aggregation**: join the edge list to the variable column, then `[, .(max, min, mean), by = row_idx]`. Fully vectorized, no R-level loop. |
| `do.call(rbind, ...)` on millions of small vectors | Eliminated â€” `data.table` returns the result directly as columns. |
| 5 sequential variable passes | Process all 5 variables in **one join + one grouped aggregation** (wide pivot), or loop over variables but each pass is now seconds, not hours. |
| Memory pressure | `data.table` is column-oriented and avoids the overhead of millions of list elements. Peak memory drops dramatically. |

**Estimated speedup**: from 86+ hours to **~5â€“20 minutes** on the same laptop.

**Numerical equivalence**: `max`, `min`, `mean` are computed on exactly the same neighbor sets with the same `NA`-removal logic, so the trained Random Forest model's inputs are preserved identically.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the flattened neighbor edge-list (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors) {
  # cell_dt must be a data.table with columns: id, year  (and row order matters)
  # id_order: vector of cell IDs in the same order as the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  ## ---- Step A: expand the nb object into a cell-id edge list ----
  # For each cell index j in id_order, neighbors[[j]] gives the indices
  # of its rook neighbors in id_order.
  n_cells <- length(id_order)

  from_idx <- rep(seq_len(n_cells),
                  times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid     <- to_idx != 0L
  from_idx  <- from_idx[valid]
  to_idx    <- to_idx[valid]

  # Map back to actual cell IDs
  edge_cells <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )

  ## ---- Step B: build an integer row-index lookup keyed on (id, year) ----
  cell_dt[, .row_idx := .I]
  idx_dt <- cell_dt[, .(.row_idx, id, year)]
  setkey(idx_dt, id, year)

  ## ---- Step C: for every (from_id, year) pair, find the row indices
  ##             of its neighbors in the same year ----
  # Cartesian-style: cross edge_cells with every year that the "from" cell
  # appears in, then look up the "to" cell in the same year.

  # All unique years present in the data
  years <- sort(unique(cell_dt$year))

  # Expand edges Ã— years
  # This produces ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in memory.
  edge_year <- CJ_dt(edge_cells, years)   # see helper below, or:
  edge_year <- edge_cells[, .(year = years), by = .(from_id, to_id)]

  # Look up row index of the SOURCE (from) row
  edge_year[idx_dt, on = .(from_id = id, year), from_row := i..row_idx]

  # Look up row index of the NEIGHBOR (to) row

  edge_year[idx_dt, on = .(to_id = id, year), to_row := i..row_idx]

  # Drop edges where either side is missing (cell not observed that year)
  edgelist <- edge_year[!is.na(from_row) & !is.na(to_row),
                        .(from_row, to_row)]

  setkey(edgelist, from_row)
  return(edgelist)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for one variable (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_dt <- function(cell_dt, edgelist, var_name) {
  # Attach the neighbor's value to each edge
  el <- copy(edgelist)
  el[, nval := cell_dt[[var_name]][to_row]]

  # Drop edges where the neighbor value is NA
  el <- el[!is.na(nval)]

  # Grouped aggregation â€” fully vectorized
  agg <- el[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]

  # Build result columns aligned to cell_dt row order
  n <- nrow(cell_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  col_max[agg$from_row]  <- agg$nb_max
  col_min[agg$from_row]  <- agg$nb_min
  col_mean[agg$from_row] <- agg$nb_mean

  # Name the new columns consistently with the original pipeline
  max_name  <- paste0("neighbor_max_", var_name)
  min_name  <- paste0("neighbor_min_", var_name)
  mean_name <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = max_name,  value = col_max)
  set(cell_dt, j = min_name,  value = col_min)
  set(cell_dt, j = mean_name, value = col_mean)

  invisible(cell_dt)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Main driver (replaces the outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build the edge list once  (~38M rows, ~300 MB, takes < 1 min)
message("Building neighbor edge list...")
edgelist <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  compute_neighbor_stats_dt(cell_data, edgelist, var_name)
}

# Clean up helper column
cell_data[, .row_idx := NULL]

message("Done. ", ncol(cell_data), " columns in cell_data.")
```

---

## Why This Preserves the Original Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | The edge list is built from the identical `rook_neighbors_unique` nb object and filters to the same `(cell_id, year)` pairs that exist in the data â€” exactly matching the original `idx_lookup[neighbor_keys]` logic with `NA` removal. |
| **Same aggregation functions** | `max`, `min`, `mean` are applied after dropping `NA` neighbor values, identical to the original `compute_neighbor_stats`. |
| **Same column names** | Output columns follow the `neighbor_{max,min,mean}_{var}` naming convention (adjust the prefix if your `compute_and_add_neighbor_features` wrapper used a different pattern). |
| **Trained RF model untouched** | No model code is changed; only the feature-engineering step is optimized. The model receives numerically identical input columns. |

---

## Memory Budget Estimate

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edgelist` (38.5M Ã— 2 int cols) | ~0.3 GB |
| Temporary `el` inside each variable pass | ~0.6 GB (freed each iteration) |
| Overhead / R session | ~1â€“2 GB |
| **Peak total** | **~8â€“9 GB** (fits in 16 GB) |

This replaces an 86+ hour pipeline with one that should complete in roughly **5â€“20 minutes** on the same 16 GB laptop, while producing bit-identical numerical results for the downstream Random Forest predictions.
END_RESPONSE>>>
