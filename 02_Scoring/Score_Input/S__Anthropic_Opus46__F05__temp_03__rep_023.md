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
run_id: Anthropic_F05_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is created once, but the `paste()` call to construct `neighbor_keys` and the named-vector lookup `idx_lookup[neighbor_keys]` are executed **once per row** â€” 6.46 million times. Each call to `paste()` allocates a new character vector, and each named-vector lookup performs repeated hash-table probes against a 6.46M-entry hash. This is O(N Ã— K) string operations where K is the average neighbor count.

### Broader Algorithmic Problem
The real issue is that the entire neighbor-lookup architecture is **row-centric** when it should be **vectorized and join-based**. The pattern is:

1. For each cell-year row, find which other rows share the same year and are spatial neighbors.
2. For each such neighbor row, pull variable values and compute max/min/mean.

This is fundamentally a **merge-aggregate** operation that can be expressed as a single vectorized join + grouped aggregation â€” no per-row `lapply`, no string keys, no 6.46M iterations.

### Root Cause Summary

| Layer | Problem | Cost |
|-------|---------|------|
| String key construction | `paste()` called 6.46M Ã— (1 + avg_neighbors) times | ~billions of string allocs |
| Named vector lookup | Hash probe into 6.46M-entry table, per row | O(N Ã— K) hash lookups |
| Row-wise `lapply` | R-level loop over 6.46M rows | Interpreter overhead |
| Repeated per variable | `compute_neighbor_stats` re-traverses all 6.46M neighbor lists 5 times | 5Ã— redundant traversal |
| Architecture | Row-centric instead of vectorized join-aggregate | Orders of magnitude slower |

## Optimization Strategy

1. **Build an integer edge list once** â€” expand the `nb` object into a two-column `data.table` of `(row_i, row_j)` where both rows share the same year. Use integer IDs throughout; no strings.
2. **Vectorized join** â€” for each variable, join the neighbor edge list to the data column, then aggregate with `data.table` grouped operations (`max`, `min`, `mean` by source row).
3. **Single pass for all variables** â€” optionally batch all 5 variables in one join.
4. **Memory-conscious** â€” the edge list is ~(6.46M Ã— avg_neighbors) â‰ˆ ~26M rows of two integers â‰ˆ ~200 MB, feasible on 16 GB.

Expected speedup: from ~86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ============================================================
# 0. Assumptions about inputs already in the environment:
#    - cell_data       : data.frame/data.table with columns id, year, and the 5 vars
#    - id_order        : integer vector of cell IDs in the order matching rook_neighbors_unique
#    - rook_neighbors_unique : an nb object (list of integer index vectors)
#    - The trained RF model object is untouched.
# ============================================================

# Convert to data.table if not already (non-destructive copy)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build a spatial edge list (id_from, id_to) from the nb object
#         This is done ONCE and uses only integer cell IDs.
# ============================================================

build_spatial_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[k]] contains integer indices into id_order that are neighbors of id_order[k]
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses 0L for "no neighbors"
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )
}

cat("Building spatial edge list...\n")
spatial_edges <- build_spatial_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed neighbor pairs\n", nrow(spatial_edges)))

# ============================================================
# STEP 2: Build the full row-level edge list by joining on year.
#         For each row i in cell_data, find all rows j that share
#         the same year AND whose cell id is a spatial neighbor.
#
#         We do this via keyed joins â€” no string pasting.
# ============================================================

# Add a row index to cell_data
cell_data[, .row_idx := .I]

# Create a lean lookup: (id, year) -> row_idx
# This replaces the old paste-based idx_lookup
id_year_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_lookup, id, year)

# For each row, get its (id, year), then find neighbor ids via spatial_edges,
# then find neighbor rows via id_year_lookup.
# We do this as a chain of data.table joins â€” fully vectorized.

cat("Building row-level neighbor edge list...\n")

# Start from cell_data rows: get (row_idx_from, id_from, year)
row_info <- cell_data[, .(.row_idx, id, year)]
setnames(row_info, c("row_from", "id_from", "year"))

# Join to spatial_edges to get neighbor cell IDs
setkey(row_info, id_from)
setkey(spatial_edges, id_from)

# This is the big join: for each row, expand to its spatial neighbors
# Result: (row_from, id_from, year, id_to)
edges_with_year <- spatial_edges[row_info, on = "id_from", allow.cartesian = TRUE,
                                  nomatch = NULL]
# edges_with_year now has columns: id_from, id_to, row_from, year

# Now find the actual row index of each (id_to, year) pair
setkey(id_year_lookup, id, year)
setkey(edges_with_year, id_to, year)

neighbor_edges <- id_year_lookup[edges_with_year,
                                  on = c("id" = "id_to", "year" = "year"),
                                  nomatch = NULL]
# neighbor_edges has: .row_idx (= row_to), id, year, row_from, id_from

# Keep only what we need
neighbor_edges <- neighbor_edges[, .(row_from, row_to = .row_idx)]
setkey(neighbor_edges, row_from)

cat(sprintf("  Row-level edge list: %d edges\n", nrow(neighbor_edges)))

# ============================================================
# STEP 3: For each neighbor source variable, compute max/min/mean
#         via vectorized grouped aggregation.
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_data, neighbor_edges, var_name) {
  cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

  # Extract the variable values for the "to" (neighbor) rows
  vals <- cell_data[[var_name]]
  neighbor_edges[, val := vals[row_to]]

  # Grouped aggregation â€” one pass, fully vectorized

  stats <- neighbor_edges[!is.na(val),
                           .(nmax  = max(val),
                             nmin  = min(val),
                             nmean = mean(val)),
                           by = row_from]

  # Allocate result columns (NA by default for rows with no valid neighbors)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]

  cell_data[stats$row_from, (col_max)  := stats$nmax]
  cell_data[stats$row_from, (col_min)  := stats$nmin]
  cell_data[stats$row_from, (col_mean) := stats$nmean]

  # Clean up temporary column
  neighbor_edges[, val := NULL]

  invisible(cell_data)
}

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features_fast(cell_data, neighbor_edges, var_name)
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done.\n")

# ============================================================
# STEP 4: Verify column names match what the trained RF expects.
#         Adjust the column naming pattern above if your original
#         code used a different convention (e.g., "ntl_neighbor_max").
#         The numerical values are identical â€” same estimand,
#         same max/min/mean over the same neighbor sets.
# ============================================================
```

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor definition | `rook_neighbors_unique` via `spdep::nb` | Same `nb` object, same directed pairs |
| Year matching | `paste(id, year)` string match | Integer join on `(id, year)` â€” identical match |
| Aggregation | `max`, `min`, `mean` of non-NA neighbor values | Identical `max`, `min`, `mean` with `!is.na` filter |
| NA handling | Rows with no valid neighbors get `NA` | Same â€” `stats` only contains rows with â‰¥1 valid neighbor; rest stay `NA` |
| RF model | Untouched | Untouched â€” only the feature columns are populated |

## Expected Performance

| Step | Estimated Time | Estimated Memory |
|------|---------------|-----------------|
| Spatial edge list construction | < 1 second | ~20 MB |
| Row-level edge join (the big step) | 1â€“3 minutes | ~2â€“3 GB peak |
| Per-variable grouped aggregation (Ã—5) | ~30 sec each | ~200 MB temp |
| **Total** | **~5â€“8 minutes** | **~4 GB peak** |

This replaces the estimated 86+ hours with a run time on the order of minutes, by eliminating all per-row string operations and R-level loops in favor of vectorized `data.table` keyed joins and grouped aggregations.
END_RESPONSE>>>
