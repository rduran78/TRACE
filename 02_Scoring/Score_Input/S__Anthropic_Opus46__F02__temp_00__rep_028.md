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
run_id: Anthropic_F02_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. Each call to `idx_lookup[neighbor_keys]` does a linear-time hash probe on a 6.46M-element named character vector. The resulting `neighbor_lookup` list of 6.46M integer vectors also consumes enormous memory (estimated 10â€“15 GB with pointer overhead).
2. **`compute_neighbor_stats`:** `lapply` over 6.46M list elements with per-element subsetting, NA removal, and summary stats is slow but secondary to problem #1.
3. **Memory:** Storing 6.46M list elements (each a variable-length integer vector) plus the 6.46M-row data frame with 110+ columns pushes well past 16 GB RAM.

---

## Optimization Strategy

**Key insight:** The neighbor relationships are defined at the *cell* level (344K cells), not the *cell-year* level (6.46M rows). We should exploit this by:

1. **Replacing the per-row lookup with a vectorized join.** Instead of building a 6.46M-element list, we construct a flat `data.table` of `(row_index, neighbor_row_index)` pairs using fast equi-joins. This avoids all `paste`/named-vector lookups.
2. **Using `data.table` grouped aggregation** instead of `lapply` for computing neighbor stats. A single grouped `[, .(max, min, mean), by = row_index]` call replaces 6.46M R-level function calls.
3. **Building the edge list once at the cell level** (â‰ˆ1.37M directed edges), then joining to years â€” avoiding redundant per-year expansion inside a loop.
4. **Processing all 5 variables in one pass** over the edge table to amortize the join cost.

This reduces estimated runtime from 86+ hours to roughly **5â€“20 minutes** and peak memory to well under 16 GB. The trained Random Forest model and all numerical outputs are preserved exactly â€” we are only changing how features are computed, not what is computed.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat cell-level edge list from the nb object
#     rook_neighbors_unique is a list of length = length(id_order),
#     where element i contains integer indices into id_order of
#     neighbors of id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_cell_edge_list <- function(id_order, neighbors) {
  # For each cell index i, neighbors[[i]] gives neighbor indices into id_order
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0L)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
# ~ 1.37 M rows, two integer/numeric columns â€” trivial memory

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Create a row-index column so we can map back after aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Build the full (focal_row, neighbor_row) edge table via joins
#     This expands cell-level edges to cell-year-level edges by
#     joining on year, producing ~ 1.37M Ã— 28 â‰ˆ 38.5 M rows of
#     integer pairs â€” about 300 MB, well within budget.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Keyed lookup: for a given (id, year) â†’ .row_idx
row_map <- cell_data[, .(id, year, .row_idx)]

# Join focal side: attach focal .row_idx and year to each edge
setkey(row_map, id)
setkey(cell_edges, focal_id)

# Expand edges Ã— years via a join on focal_id â†’ id
#   result columns: neighbor_id, year, focal_row
edge_year <- cell_edges[row_map,
  .(neighbor_id, year, focal_row = .row_idx),
  on = .(focal_id = id),
  nomatch = NULL,
  allow.cartesian = TRUE
]

# Join neighbor side: attach neighbor .row_idx
setkey(edge_year, neighbor_id, year)
setkey(row_map, id, year)

edge_year[row_map,
  neighbor_row := i..row_idx,
  on = .(neighbor_id = id, year = year)
]

# Drop edges where the neighbor has no matching row (boundary / missing)
edge_year <- edge_year[!is.na(neighbor_row)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Compute neighbor stats for all variables in one vectorised pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_year, vars) {
  # Pre-extract the neighbor row indices (used for every variable)
  nr <- edge_year$neighbor_row
  fr <- edge_year$focal_row

  for (var_name in vars) {
    message("Processing neighbor features for: ", var_name)

    # Pull the variable values for every neighbor row
    vals <- cell_data[[var_name]][nr]

    # Build a temporary data.table for grouped aggregation
    tmp <- data.table(focal_row = fr, val = vals)

    # Remove rows where the neighbor value is NA (matches original logic)
    tmp <- tmp[!is.na(val)]

    # Grouped aggregation â€” one pass, fully vectorised in C
    agg <- tmp[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = focal_row]

    # Initialise result columns with NA (matches original for cells with
    # no valid neighbors)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    # Fill in computed values at the correct rows
    set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
  }

  invisible(cell_data)
}

compute_all_neighbor_features(cell_data, edge_year, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Clean up helper columns and temporary objects
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_idx := NULL]
rm(edge_year, row_map, cell_edges)
gc()

# cell_data now contains the same 15 neighbor feature columns
# (5 vars Ã— {max, min, mean}) as the original code, with identical
# numerical values.  Proceed to prediction with the trained RF model.
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste` + named-vector probe | One vectorized `data.table` join (~seconds) |
| **Neighbor stats** | 6.46M `lapply` iterations Ã— 5 vars | 5 grouped `data.table` aggregations (C-level) |
| **Peak memory** | ~15â€“20 GB (6.46M-element list of int vectors) | ~3â€“5 GB (flat integer edge table + data) |
| **Estimated runtime** | 86+ hours | **5â€“20 minutes** |
| **Numerical output** | Preserved exactly | Preserved exactly |
| **Trained RF model** | Unchanged | Unchanged |

The column naming convention (`{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`) should be adjusted to match whatever names the trained Random Forest model expects â€” simply rename the `max_col`, `min_col`, `mean_col` strings in the function to match the original `compute_and_add_neighbor_features` output.
END_RESPONSE>>>
