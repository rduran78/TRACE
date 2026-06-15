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
run_id: Anthropic_F02_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core performance bottleneck is **`build_neighbor_lookup`**, which uses `lapply` over ~6.46 million rows, performing per-row character key construction and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector indexing in R is effectively a hash-table lookup, but doing it millions of times inside an `lapply` with dynamic `paste` calls is extremely slow. The second bottleneck is **`compute_neighbor_stats`**, which iterates over 6.46 million entries in the lookup list, extracting subsets of a numeric vector each time. Together, these two functions create:

1. **CPU bottleneck in `build_neighbor_lookup`**: ~6.46M iterations, each calling `paste`, indexing into a ~6.46M-length named vector, and filtering `NA`s. The character-based key construction and lookup dominate runtime.
2. **Memory bottleneck**: Storing 6.46 million list elements (each a vector of neighbor row indices) is memory-intensive. With an average of ~4 rook neighbors per cell and 28 years, the lookup list holds ~25.8 million integers, but the list overhead per element (each R list slot costs ~56+ bytes) alone is ~360 MB, and the character key vector is another several hundred MB.
3. **Repeated serial computation in `compute_neighbor_stats`**: Called 5 times (once per variable), each time looping over 6.46M entries.

Estimated breakdown of the ~86 hours: the vast majority is in `build_neighbor_lookup` (character key construction and named-vector lookup at O(n) per row with large constant factors).

---

## Optimization Strategy

The key insight is to **replace the row-level, character-key-based lookup with a vectorized `data.table` join**. Instead of building a 6.46M-element list, we:

1. **Expand the neighbor graph into an edge table** (cell_id â†’ neighbor_id) â€” only ~1.37M edges.
2. **Join by (neighbor_id, year)** using `data.table` to get the row index of each neighbor in each year â€” this produces ~1.37M Ã— 28 â‰ˆ 38.5M rows but is handled efficiently by `data.table`'s binary-search join.
3. **Compute aggregated neighbor statistics (max, min, mean) via grouped aggregation** in `data.table`, grouped by the focal cell's row index.

This eliminates the per-row `lapply`, eliminates character key construction, and replaces everything with vectorized `data.table` operations that run in seconds to minutes rather than days.

### Why this is correct and safe:
- The numerical results (max, min, mean of neighbor values) are identical because we use the same neighbor graph and the same variable values.
- The trained Random Forest model is untouched; we only change how input features are computed.
- Memory usage is bounded: the edge table Ã— years is ~38.5M rows Ã— a few columns, well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build an edge table from the spdep nb object (one-time)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  # id_order is the vector mapping index position -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ---------------------------------------------------------------
# Step 2: Compute neighbor stats for one variable via data.table
# ---------------------------------------------------------------
compute_neighbor_stats_dt <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must have columns: id, year, row_idx, and the variable
  # edge_dt must have columns: focal_id, neighbor_id

  # Create a keyed lookup: for each (neighbor_id, year) -> variable value
  # We join edges to cell_dt to get neighbor values, then aggregate by (focal_id, year)

  # Subset only needed columns for the neighbor side
  neighbor_vals <- cell_dt[, .(neighbor_id = id, year, nval = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)

  # Expand edges by year via join: for each (focal_id, neighbor_id) pair,
  # pull the neighbor's value in each year.
  # First, join edge_dt with neighbor_vals on neighbor_id and year.
  # We need the focal cell's row_idx to map results back.

  # Build focal side: (focal_id, year, row_idx)
  focal_info <- cell_dt[, .(focal_id = id, year, row_idx)]
  setkey(focal_info, focal_id, year)

  # Merge focal_info with edge_dt to get (focal_id, year, neighbor_id, row_idx)
  # This is the "expansion" step: each focal cell-year gets its list of neighbors
  expanded <- edge_dt[focal_info, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: focal_id, neighbor_id, year, row_idx

  # Now join to get neighbor values
  expanded[neighbor_vals, nval := i.nval, on = .(neighbor_id, year)]

  # Remove rows where neighbor value is NA
  expanded <- expanded[!is.na(nval)]

  # Aggregate by row_idx (unique per focal cell-year)
  agg <- expanded[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = row_idx]

  # Return aggregated result keyed by row_idx
  agg
}

# ---------------------------------------------------------------
# Step 3: Main pipeline
# ---------------------------------------------------------------
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table (in-place if already, otherwise copy)
  cell_dt <- as.data.table(cell_data)
  cell_dt[, row_idx := .I]

  # Build edge table once
  edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

  # Variables to process
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    agg <- compute_neighbor_stats_dt(cell_dt, edge_dt, var_name)

    # Define new column names (matching whatever naming convention you use)
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize with NA
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    # Fill in computed values by row_idx
    cell_dt[agg, (max_col)  := i.nb_max,  on = .(row_idx)]
    cell_dt[agg, (min_col)  := i.nb_min,  on = .(row_idx)]
    cell_dt[agg, (mean_col) := i.nb_mean, on = .(row_idx)]
  }

  # Drop helper column and return as data.frame if desired
  cell_dt[, row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(cell_dt)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement for original outer loop)
# ---------------------------------------------------------------
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
```

---

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M `lapply` iterations with `paste` + named vector lookup | Single vectorized `data.table` edge-table construction (<1 sec) |
| **`compute_neighbor_stats`** (per variable) | ~6.46M `lapply` iterations, subsetting vectors | One `data.table` join + one grouped aggregation (~30â€“90 sec per variable) |
| **Total estimated time** | ~86+ hours | **~5â€“15 minutes** |
| **Peak memory** | ~4â€“8 GB (6.46M-element list + character keys) | **~2â€“4 GB** (edge table expansion is transient and columnar) |

The optimization replaces all row-level character operations and R-list overhead with `data.table`'s binary-search joins and in-memory columnar aggregation, yielding roughly a **300â€“1000Ã— speedup** while producing numerically identical results and leaving the trained Random Forest model completely untouched.
END_RESPONSE>>>
