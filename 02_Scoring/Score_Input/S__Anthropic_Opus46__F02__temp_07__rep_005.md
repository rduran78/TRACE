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
run_id: Anthropic_F02_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` and a named-vector lookup (`id_to_ref`).
- `paste()` to build neighbor keys for every neighbor of that cell in that year.
- A second named-vector lookup (`idx_lookup`) on those keys.

Named-vector lookups in R are hash-based but still carry significant per-call overhead at this scale. Pasting strings millions of times is also expensive. The result is a **list of 6.46M integer vectors**, which is both slow to build and memory-heavy (each list element is a separate R object with its own SEXP header â€” roughly 100+ bytes overhead per element, so the list skeleton alone can consume >600 MB before the actual index data).

### 2. `compute_neighbor_stats` â€” another O(n) `lapply` over 6.46M rows, called 5 times

For every row, it subsets a numeric vector by the neighbor indices, removes NAs, and computes max/min/mean. This is called once per variable (Ã—5), so ~32.3M R-level function calls. Each call allocates small temporary vectors. The `do.call(rbind, ...)` at the end also materializes a 6.46M Ã— 3 intermediate list before binding.

### Combined effect

~6.46M list elements built in step 1, then traversed 5 times in step 2, with heavy per-element R overhead â†’ **estimated 86+ hours** on a laptop.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate the per-row list** | Replace the 6.46M-element neighbor lookup list with a flat **edge table** (a two-column `data.table` of `[row_i, row_j]` pairs). This is a sparse-matrix/CSR-style representation that R and `data.table` can process in bulk. |
| **Vectorize the join** | Use `data.table` keyed joins (binary search, no hashing of 6.46M strings) to map `(neighbor_id, year)` â†’ row index in one vectorized pass. |
| **Vectorize the aggregation** | Use `data.table`'s `by=` grouped aggregation (`max`, `min`, `mean`) over the edge table â€” one pass per variable, fully vectorized in C. No R-level `lapply`. |
| **Minimize memory** | The edge table has ~1.37M Ã— 28 â‰ˆ 38.4M rows of two integer columns â‰ˆ **~307 MB**, far less than 6.46M list elements. Intermediate results are column vectors, not lists of 3-vectors. |
| **Preserve the trained RF model and the numerical estimand** | The output columns are identical (same names, same values: neighbor max, min, mean for each variable). The RF model is not retouched. |

Expected speedup: from 86+ hours to **minutes** (the join is O(n log n); each grouped aggregation is O(E) where E â‰ˆ 38.4M).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a flat edge table (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(cell_dt, id_order, neighbors) {

  # cell_dt must be a data.table with columns: id, year, and a row index
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # --- 1a. Expand the nb object into a directed edge list of cell IDs --------
  #     Each element neighbors[[k]] contains indices into id_order.
  n_neighbors <- lengths(neighbors)
  from_idx    <- rep(seq_along(neighbors), times = n_neighbors)
  to_idx      <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-entries that spdep uses to mark cells with no neighbors
  valid       <- to_idx != 0L
  from_idx    <- from_idx[valid]
  to_idx      <- to_idx[valid]

  edge_ids <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  # edge_ids now has ~1.37M rows (directed rook-neighbor pairs)

  # --- 1b. Cross-join with years to get (from_id, to_id, year) ---------------
  years <- sort(unique(cell_dt$year))
  edge_full <- edge_ids[, .(year = years), by = .(from_id, to_id)]
  # ~1.37M Ã— 28 â‰ˆ 38.4M rows

  # --- 1c. Map from_idÃ—year â†’ row position in cell_dt ("row_i")
  #         Map to_idÃ—year   â†’ row position in cell_dt ("row_j")
  #     We add a row-number column to cell_dt for this purpose.
  cell_dt[, .row_idx := .I]

  # Keyed join: map (from_id, year) â†’ row_i
  setkey(cell_dt, id, year)
  edge_full[cell_dt, row_i := i..row_idx, on = .(from_id = id, year)]
  edge_full[cell_dt, row_j := i..row_idx, on = .(to_id   = id, year)]

  # Drop edges where either side is missing (cell not in data for that year)
  edge_full <- edge_full[!is.na(row_i) & !is.na(row_j)]

  # Clean up helper column
  cell_dt[, .row_idx := NULL]

  edge_full[]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor stats for one variable (vectorized)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_vec <- function(cell_dt, edge_table, var_name) {
  # Attach the neighbor's value to every edge
  vals <- cell_dt[[var_name]]
  edge_table[, nval := vals[row_j]]

  # Grouped aggregation: one group per row_i (the focal cell-year row)
  agg <- edge_table[!is.na(nval),
    .(nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)),
    keyby = .(row_i)
  ]

  # Initialize result columns with NA
  n <- nrow(cell_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  # Fill in the rows that have neighbors
  col_max [agg$row_i] <- agg$nb_max
  col_min [agg$row_i] <- agg$nb_min
  col_mean[agg$row_i] <- agg$nb_mean

  # Clean up temporary column
  edge_table[, nval := NULL]

  # Return a data.table of the three new columns
  setnames(
    data.table(col_max, col_min, col_mean),
    c(paste0(var_name, "_nb_max"),
      paste0(var_name, "_nb_min"),
      paste0(var_name, "_nb_mean"))
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Outer loop â€” drop-in replacement
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

## Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

## Build the edge table once (replaces build_neighbor_lookup)
message("Building edge table â€¦")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

## Compute and attach neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  new_cols <- compute_neighbor_stats_vec(cell_data, edge_table, var_name)
  # Bind new columns to cell_data by reference
  cell_data[, names(new_cols) := new_cols]
}

message("Done. Neighbor features added.")
```

---

## Summary of What Changed and What Is Preserved

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-element R list built via `lapply` + `paste` | Flat `data.table` edge table built via vectorized joins |
| Aggregation | `lapply` over 6.46M elements Ã— 5 vars | `data.table` grouped `by=` aggregation Ã— 5 vars |
| Memory for lookup | ~600 MB+ list overhead + index data | ~307 MB flat integer table |
| Estimated runtime | 86+ hours | **Minutes** (dominated by the 38.4M-row grouped aggregation, which `data.table` handles in C) |
| Output columns | `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` | Identical names and identical numerical values |
| Trained RF model | Untouched | Untouched |
| Numerical estimand | Preserved (same max/min/mean over same neighbor sets) | Preserved |
END_RESPONSE>>>
