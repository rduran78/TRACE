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
run_id: Anthropic_F04_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M string operations is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` with per-element R function calls over 6.46M entries, each allocating small vectors.

**Root causes, ranked:**

| # | Cause | Impact |
|---|-------|--------|
| 1 | `paste()`-based key construction inside a 6.46M-iteration `lapply` | Very High |
| 2 | Named character vector lookup (`idx_lookup[neighbor_keys]`) is slow at scale | Very High |
| 3 | Per-row anonymous function overhead in `lapply` (no vectorization) | High |
| 4 | `compute_neighbor_stats` uses R-level loop + per-row `max/min/mean` | Moderate |
| 5 | Repeated `do.call(rbind, ...)` on millions of 3-element vectors | Moderate |

## Optimization Strategy

**Core idea:** Replace all string-key lookups and row-level R loops with vectorized integer-indexed operations using `data.table`.

1. **Eliminate string keys entirely.** Create a direct integer mapping from `(id, year)` â†’ row index using a `data.table` keyed join, or even a 2D integer matrix if IDs and years are dense.
2. **Pre-expand the neighbor edge list** into a single long `data.table` of `(source_row, neighbor_row)` pairs (~25.8M rows). This is built once.
3. **Vectorize `compute_neighbor_stats`** by joining the edge list to the variable column and using `data.table` grouped aggregation (`[, .(max, min, mean), by = source_row]`), which is C-level fast.

**Expected speedup:** From ~86+ hours to **~2â€“10 minutes** depending on I/O, because all hot loops move from R interpreter to C (data.table internals).

**Memory:** The edge list is ~25.8M rows Ã— 2 integer columns â‰ˆ 200 MB. Comfortably fits in 16 GB.

## Optimized Working R Code

```r
library(data.table)

#
# STEP 1: Build a vectorized edge list (source_row -> neighbor_row)
#         This replaces build_neighbor_lookup entirely.
#         Run ONCE; reuse for all variables.
#

build_neighbor_edgelist <- function(cell_data, id_order, neighbors) {
  # cell_data must be a data.table (or will be converted)
  dt <- as.data.table(cell_data)
  
  # --- Map each cell id to its position in id_order (integer) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Map (id, year) -> row index using data.table keyed join ---
  dt[, row_idx := .I]
  setkey(dt, id, year)  # fast keyed lookup
  
  # --- Get unique years ---
  years <- sort(unique(dt$year))
  
  # --- Build the edge list in one vectorized pass ---
  # For each cell in id_order, get its neighbor cell ids
  # Then expand across all years
  
  # Construct cell-level neighbor edges (cell_id -> neighbor_cell_id)
  n_cells <- length(id_order)
  from_cell <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_cell   <- unlist(neighbors)
  
  # Map back to actual cell IDs
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]
  
  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  
  # Cross join with years to get cell-year level edges
  year_dt <- data.table(year = years)
  # Use CJ-style expansion: every edge Ã— every year
  cell_year_edges <- cell_edges[, .(year = years), by = .(from_id, to_id)]
  
  # Now join to get source row index
  setnames(cell_year_edges, c("from_id"), c("id"))
  cell_year_edges[dt, on = .(id, year), source_row := i.row_idx]
  
  # Now join to get neighbor row index
  setnames(cell_year_edges, c("id", "to_id"), c("from_id", "id"))
  cell_year_edges[dt, on = .(id, year), neighbor_row := i.row_idx]
  
  # Clean: keep only edges where both source and neighbor exist
  edges <- cell_year_edges[!is.na(source_row) & !is.na(neighbor_row),
                           .(source_row, neighbor_row)]
  
  # Clean up temporary column
  dt[, row_idx := NULL]
  
  return(edges)
}

#
# STEP 2: Compute neighbor stats for one variable â€” fully vectorized
#         Returns a data.table with columns: nb_max, nb_min, nb_mean
#         aligned to the rows of cell_data.
#

compute_neighbor_stats_fast <- function(cell_data, edges, var_name) {
  n <- if (is.data.table(cell_data)) nrow(cell_data) else nrow(cell_data)
  
  # Extract the variable values for neighbor rows
  vals <- cell_data[[var_name]]
  
  # Attach neighbor values to edge list
  edge_vals <- data.table(
    source_row = edges$source_row,
    val        = vals[edges$neighbor_row]
  )
  
  # Remove edges where the neighbor value is NA
  edge_vals <- edge_vals[!is.na(val)]
  
  # Grouped aggregation in C (data.table)
  stats <- edge_vals[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]
  
  # Allocate full-length result with NAs for rows with no valid neighbors
  result <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  result[stats$source_row, `:=`(
    nb_max  = stats$nb_max,
    nb_min  = stats$nb_min,
    nb_mean = stats$nb_mean
  )]
  
  # Name columns to match original pipeline convention
  prefix <- paste0("nb_", var_name, "_")
  setnames(result, c(
    paste0(prefix, "max"),
    paste0(prefix, "min"),
    paste0(prefix, "mean")
  ))
  
  return(result)
}

#
# STEP 3: Full replacement outer loop
#

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build edge list ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge list...")
edges <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge list: %s rows", format(nrow(edges), big.mark = ",")))

# Compute and attach neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  nb_stats <- compute_neighbor_stats_fast(cell_data, edges, var_name)
  cell_data <- cbind(cell_data, nb_stats)
}

message("Done. Neighbor features attached.")

# cell_data now has the same neighbor feature columns as the original pipeline.
# The trained Random Forest model can be applied directly with no retraining.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per row as the original code. The edge list encodes the identical rook-neighbor relationships, restricted to the same year. |
| **Trained RF model** | No model retraining occurs. The output columns have the same names and semantics. The model's `predict()` call is unchanged. |
| **Column naming** | Adjust the `prefix` pattern in `compute_neighbor_stats_fast` if your existing pipeline uses a different naming convention (e.g., `compute_and_add_neighbor_features` may name them differently). Match accordingly. |

## Expected Performance

| Stage | Original | Optimized | Reason |
|-------|----------|-----------|--------|
| Lookup build | ~hours (6.46M `paste` + char lookup) | ~1â€“3 min (keyed `data.table` join) | Integer join vs. string hashing |
| Stats (per var) | ~hours (6.46M `lapply` calls) | ~10â€“30 sec (`data.table` grouped agg in C) | Vectorized C vs. R interpreter |
| **Total (5 vars)** | **~86+ hours** | **~5â€“10 minutes** | **~500â€“1000Ã— speedup** |
END_RESPONSE>>>
