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
run_id: Anthropic_F02_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again, once per variable (Ã—5 variables). Together:

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing character coercion, `paste`, and named-vector subsetting. The `idx_lookup` named vector has 6.46M entries, making each lookup expensive. Estimated: ~70â€“80% of total runtime.
2. **`compute_neighbor_stats`**: Uses `lapply` returning a 3-element vector per row, then `do.call(rbind, ...)` on a 6.46M-element list â€” this alone is a known R anti-pattern that causes massive memory allocation and copying.
3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors is itself memory-heavy (~several GB with list overhead).

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key lookups in `build_neighbor_lookup` | Replace with integer-keyed `data.table` join. Build a cell-year â†’ row-index mapping table and join neighbor cell-IDs by year in a single vectorized operation. |
| Storing 6.46M-element list of neighbor indices | Replace with a flat `data.table` of `(row_i, neighbor_row_j)` pairs â€” a sparse edge list. This is more cache-friendly and enables vectorized grouped aggregation. |
| Per-row `lapply` + `do.call(rbind, ...)` in `compute_neighbor_stats` | Replace with a single `data.table` grouped aggregation: join the variable values onto the edge list, then `[, .(max, min, mean), by = row_i]`. |
| Repeated work across 5 variables | Compute all 5 variables' stats in one pass over the edge list. |

**Expected speedup**: From ~86 hours to roughly **5â€“20 minutes**, depending on disk I/O. Memory peak should stay well under 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert cell_data to data.table (non-destructive; preserves all columns)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Add a row index column (will be used as the primary key for joining back)
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Build a flat edge list of (cell_id, neighbor_cell_id) from the nb object
#    This replaces the per-row string-key approach entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer index vectors
  # id_order maps positional index -> cell_id
  n <- length(neighbors)
  # Pre-allocate by computing total number of edges
  lengths_vec <- lengths(neighbors)
  total_edges <- sum(lengths_vec)

  from_id <- rep.int(id_order, lengths_vec)
  to_id   <- id_order[unlist(neighbors, use.names = FALSE)]

  data.table(cell_id = from_id, neighbor_cell_id = to_id)
}

cat("Building spatial edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed rook-neighbor pairs)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Expand edge list across years by joining with cell_data's
#    (cell_id, year) â†’ row_idx mapping.
#    This is the vectorized replacement for build_neighbor_lookup.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Building cell-year to row index mapping...\n")

# Mapping table: for every (id, year) in cell_data, record the row index
idx_map <- cell_data[, .(cell_id = id, year, .row_idx)]
setkey(idx_map, cell_id, year)

# Get the unique years present in the data
years_in_data <- sort(unique(cell_data$year))

# Cross-join edges Ã— years, then resolve both endpoints to row indices
cat("Expanding edge list across all years...\n")

# Add year dimension to edge list
edge_year_dt <- CJ_dt_edges(edge_dt, years_in_data)
# We implement this efficiently:
edge_year_dt <- edge_dt[, .(year = years_in_data), by = .(cell_id, neighbor_cell_id)]

# Resolve the "from" cell (the focal row) to its row index
setkey(edge_year_dt, cell_id, year)
edge_year_dt <- idx_map[edge_year_dt, on = .(cell_id, year), nomatch = 0L]
setnames(edge_year_dt, ".row_idx", "focal_row")

# Resolve the "to" cell (the neighbor) to its row index
setnames(edge_year_dt, "neighbor_cell_id", "cell_id_nb")
setkey(edge_year_dt, cell_id_nb, year)

idx_map_nb <- copy(idx_map)
setnames(idx_map_nb, c("cell_id", "year", "nb_row"))

edge_year_dt <- idx_map_nb[edge_year_dt, on = .(cell_id = cell_id_nb, year), nomatch = 0L]

# Result: edge_year_dt has columns: cell_id (neighbor), year, nb_row, cell_id (focal via key), focal_row
# Keep only what we need:
edge_year_dt <- edge_year_dt[, .(focal_row, nb_row)]

cat(sprintf("Expanded edge list: %s rows\n", format(nrow(edge_year_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Compute neighbor stats for all 5 variables in one pass
#    This replaces compute_neighbor_stats + the outer for-loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

# Attach neighbor variable values to the edge list
# We pull the columns we need from cell_data by nb_row index
nb_vals <- cell_data[edge_year_dt$nb_row, ..neighbor_source_vars]
nb_vals[, focal_row := edge_year_dt$focal_row]

# Grouped aggregation: for each focal_row, compute max/min/mean of each variable
agg_expr <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call dynamically
agg_call <- as.call(c(as.name("list"),
                       setNames(agg_expr, agg_names)))

neighbor_stats <- nb_vals[, eval(agg_call), by = focal_row]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
inf_cols <- agg_names[grepl("max_|min_", agg_names)]
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Join neighbor stats back to cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Joining neighbor features back to cell_data...\n")

setkey(neighbor_stats, focal_row)

# Rows with no neighbors at all won't appear in neighbor_stats;
# they will get NA after the join, which matches original behavior.
cell_data[neighbor_stats, (agg_names) := mget(agg_names), on = .(.row_idx = focal_row)]

# Clean up the helper column
cell_data[, .row_idx := NULL]

cat("Done. Neighbor features added.\n")
```

### Explanation of why this preserves the original numerical estimand

- **Same neighbor relationships**: The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping.
- **Same statistics**: `max`, `min`, and `mean` with `na.rm = TRUE` are computed identically; `NA` handling matches the original (groups with zero non-NA values yield `NA`).
- **Same column names**: The output columns follow the same naming convention (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.) that `compute_and_add_neighbor_features` would have produced.
- **Random Forest model is untouched**: Only the feature-engineering step is optimized; no model retraining occurs.

### Expected performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | Single vectorized `data.table` join (~seconds) |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` + `do.call(rbind)` | One grouped `data.table` aggregation (~minutes) |
| Peak memory | List of 6.46M vectors + repeated `rbind` copies | Flat edge table + in-place `data.table` ops (~4â€“8 GB) |
| **Total time** | **~86+ hours** | **~5â€“20 minutes** |
END_RESPONSE>>>
