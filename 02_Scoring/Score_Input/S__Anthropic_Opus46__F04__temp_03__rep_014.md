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
run_id: Anthropic_F04_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs paste-based string keys for every neighbor Ã— year combination,
4. Looks those keys up in `idx_lookup` (a named character vector of length ~6.46M).

String construction (`paste`) and named-vector lookup on a 6.46M-element vector are **O(n)** or at best O(1)-with-high-constant for each of ~6.46M rows, each having ~4 neighbors on average (rook contiguity). That is roughly **25.8 million string constructions and hash lookups**. R's named vector lookup is not a true hash table â€” it degrades badly at this scale. `compute_neighbor_stats` is comparatively cheap (just numeric indexing), but the `lapply` + `do.call(rbind, ...)` pattern over 6.46M elements is also unnecessarily slow.

**Root causes, ranked by impact:**

1. **String-key lookup in a 6.46M named vector** â€” dominant cost.
2. **Row-level `lapply` in pure R** over 6.46M rows â€” interpreter overhead.
3. **`do.call(rbind, list-of-6.46M-vectors)`** â€” slow list-to-matrix coercion.

## Optimization Strategy

**Replace all string-key lookups with integer-arithmetic indexing, and vectorize both the lookup construction and the stats computation using `data.table`.**

Key ideas:

- Since years are contiguous (1992â€“2019, 28 years) and every cell has one row per year, we can compute a **direct integer row index** from `(cell_id, year)` using arithmetic: `row = (cell_position - 1) * 28 + (year - 1991)`. No strings, no hash lookups.
- Expand the neighbor list into a flat edge table `(row_i, neighbor_row_j)` using vectorized operations.
- Compute `max`, `min`, `mean` per row using `data.table` grouped aggregation on the flat edge table â€” this is highly optimized C code internally.

This reduces estimated runtime from **86+ hours to minutes**.

## Optimized Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure data is a data.table sorted by (id, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# id_order is the vector of unique cell IDs matching the nb object indexing.
# Ensure cell_dt is sorted so that row index can be computed arithmetically.
# Create an integer cell-position mapping:
cell_dt[, cell_pos := match(id, id_order)]  # integer position in id_order

# Sort by (cell_pos, year) so row index = (cell_pos - 1) * n_years + (year - min_year + 1)
setorder(cell_dt, cell_pos, year)

n_years  <- length(unique(cell_dt$year))       # 28
min_year <- min(cell_dt$year)                   # 1992

# Verify the arithmetic indexing assumption (balanced panel):
stopifnot(nrow(cell_dt) == length(id_order) * n_years)

# After sorting, row i corresponds to cell_pos = ((i-1) %/% n_years) + 1,
# year = min_year + ((i-1) %% n_years).
# Conversely: row_index(cell_pos, year) = (cell_pos - 1) * n_years + (year - min_year + 1)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build flat neighbor edge table (vectorized, no strings)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edges <- function(rook_neighbors, n_years, min_year) {
  # rook_neighbors is an nb object: list of integer vectors (neighbor positions)
  n_cells <- length(rook_neighbors)
  
  # Number of neighbors per cell
  n_nb <- vapply(rook_neighbors, length, integer(1))
  
  # Source cell positions, repeated for each neighbor
  src_pos <- rep(seq_len(n_cells), times = n_nb)
  # Destination cell positions
  dst_pos <- unlist(rook_neighbors, use.names = FALSE)
  
  # Remove the spdep "no-neighbor" sentinel (integer(0) produces nothing via unlist,
  # but a 0L element means no neighbors in some nb representations)
  valid <- dst_pos > 0L
  src_pos <- src_pos[valid]
  dst_pos <- dst_pos[valid]
  
  # Expand across all years: for each (src, dst) pair, create n_years rows
  n_edges <- length(src_pos)
  years   <- seq.int(min_year, min_year + n_years - 1L)
  
  # Repeat each edge n_years times; tile years
  src_pos_exp <- rep(src_pos, each = n_years)
  dst_pos_exp <- rep(dst_pos, each = n_years)
  year_exp    <- rep(years, times = n_edges)
  
  # Compute direct row indices (integer arithmetic, no strings)
  src_row <- (src_pos_exp - 1L) * n_years + (year_exp - min_year + 1L)
  dst_row <- (dst_pos_exp - 1L) * n_years + (year_exp - min_year + 1L)
  
  data.table(src_row = src_row, dst_row = dst_row)
}

cat("Building neighbor edge table...\n")
edges <- build_neighbor_edges(rook_neighbors_unique, n_years, min_year)
cat(sprintf("Edge table: %s rows\n", format(nrow(edges), big.mark = ",")))
# Expected: ~1,373,394 directed pairs Ã— 28 years â‰ˆ 38.5M rows

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for each variable (vectorized)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, edges, var_name) {
  vals <- cell_dt[[var_name]]
  
  # Attach neighbor values to edge table
  edges[, nb_val := vals[dst_row]]
  
  # Compute grouped stats â€” data.table does this in optimized C
  stats <- edges[!is.na(nb_val),
                 .(nb_max  = max(nb_val),
                   nb_min  = min(nb_val),
                   nb_mean = mean(nb_val)),
                 keyby = src_row]
  
  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign results by integer row index
  cell_dt[stats$src_row, (max_col)  := stats$nb_max]
  cell_dt[stats$src_row, (min_col)  := stats$nb_min]
  cell_dt[stats$src_row, (mean_col) := stats$nb_mean]
  
  # Clean up temporary column from edges
  edges[, nb_val := NULL]
  
  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Outer loop â€” identical interface, preserves all original values
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, edges, var_name)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Restore original row order and convert back to data.frame
#         so the trained Random Forest model sees identical input
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If the original cell_data had a specific row order, restore it.
# We kept all original columns; just re-sort to original order if needed.
setorder(cell_dt, cell_pos, year)  # or restore an original row-id if saved

cell_data <- as.data.frame(cell_dt)
# The trained RF model can now be applied with predict(rf_model, cell_data) as before.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values â€” only the indexing mechanism changed (integer arithmetic vs. string lookup). |
| **Trained RF model** | Untouched. The output `cell_data` data.frame has the same columns (including the 15 new neighbor-stat columns) with identical values, so `predict(rf_model, cell_data)` produces the same predictions. |
| **Column naming** | Uses the same `{var}_{nb_max|nb_min|nb_mean}` convention (adjust the suffix pattern to match your existing `compute_and_add_neighbor_features` if it differs). |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Lookup construction | ~86 hrs (string hashing) | ~30 sec (integer arithmetic, vectorized) |
| Stats per variable | ~minutes (numeric indexing OK, but `lapply` + `rbind` overhead) | ~10â€“20 sec (`data.table` grouped aggregation in C) |
| **Total (5 vars)** | **86+ hours** | **~3â€“5 minutes** |
| Peak RAM | Moderate | ~3â€“4 GB for the 38.5M-row edge table (fits in 16 GB) |

The dominant speedup comes from eliminating 25.8 million `paste` + named-vector lookups in favor of a single vectorized integer-arithmetic expansion, and replacing the row-level `lapply` with `data.table`'s grouped aggregation.
END_RESPONSE>>>
