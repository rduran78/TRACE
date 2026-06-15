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
run_id: Anthropic_F04_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Retrieves neighbor cell IDs from the `nb` object,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks up those keys in a named character vector (`idx_lookup`).

String allocation, `paste()`, and named-vector lookups are extremely slow in R when repeated millions of times. The `compute_neighbor_stats` function is comparatively lighter but still uses an R-level `lapply` over 6.46M elements, each calling `max`, `min`, `mean` on small vectors.

**Root causes (ranked by impact):**

1. **`build_neighbor_lookup`**: ~6.46M iterations Ã— multiple `paste()` and named-vector lookups per iteration. This is O(N Ã— avg_neighbors) string operations â€” roughly 50+ billion character operations.
2. **`compute_neighbor_stats`**: R-level loop over 6.46M elements, called 5 times (once per variable). Slow but secondary.
3. **No vectorization or use of `data.table`** â€” everything is scalar R.

## Optimization Strategy

1. **Replace the character-key lookup with integer-arithmetic indexing.** Since years are contiguous (1992â€“2019, 28 years) and cell IDs can be mapped to integers 1â€“344,208, every (cell, year) pair maps to a unique row via `(cell_index - 1) * 28 + (year - 1992 + 1)`. This eliminates all `paste()` and named-vector lookups.

2. **Pre-expand the neighbor list from cell-level to row-level using vectorized operations** with `data.table` and `rep()`/arithmetic, avoiding any per-row `lapply`.

3. **Compute neighbor stats via vectorized `data.table` grouped aggregation** instead of R-level `lapply`.

This reduces estimated runtime from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Ensure cell_data is a data.table sorted by (id, year)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Create integer cell index (1-based) and ensure year ordering
#   id_order is the vector of cell IDs matching the nb object
id_to_int <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_idx := id_to_int[as.character(id)]]

# Sort so that row number = (cell_idx - 1) * n_years + (year - min_year + 1)
min_year <- min(cell_dt$year)
max_year <- max(cell_dt$year)
n_years  <- max_year - min_year + 1L  # 28

setorder(cell_dt, cell_idx, year)

# Verify contiguous panel (every cell has every year)
stopifnot(nrow(cell_dt) == length(id_order) * n_years)

# Add a row_id that matches the sort order
cell_dt[, row_id := .I]

# ---------------------------------------------------------------
# 1.  Build edge list (cell-level) from the nb object
#     rook_neighbors_unique is a list of length n_cells;
#     element i contains integer indices of neighbors of cell i.
# ---------------------------------------------------------------
n_cells <- length(id_order)

# Expand nb object to an edge-list data.table: (from_cell, to_cell)
from_cell <- rep(seq_len(n_cells),
                 times = lengths(rook_neighbors_unique))
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove 0-neighbor entries (spdep uses 0L for no-neighbor)
valid <- to_cell != 0L
from_cell <- from_cell[valid]
to_cell   <- to_cell[valid]

edges <- data.table(from_cell = from_cell, to_cell = to_cell)

# ---------------------------------------------------------------
# 2.  Expand to row-level edges by crossing with years
#     row_id of cell c in year y = (c - 1) * n_years + (y - min_year + 1)
# ---------------------------------------------------------------
years_vec <- seq.int(min_year, max_year)

# Cross join edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
edge_rows <- edges[, .(year = years_vec), by = .(from_cell, to_cell)]

edge_rows[, from_row := (from_cell - 1L) * n_years + (year - min_year + 1L)]
edge_rows[, to_row   := (to_cell   - 1L) * n_years + (year - min_year + 1L)]

# ---------------------------------------------------------------
# 3.  Compute neighbor stats for each source variable (vectorized)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value to each edge
  edge_rows[, nbr_val := cell_dt[[var_name]][to_row]]

  # Aggregate: for each from_row, compute max/min/mean of non-NA neighbor values
  agg <- edge_rows[!is.na(nbr_val),
                    .(nbr_max  = max(nbr_val),
                      nbr_min  = min(nbr_val),
                      nbr_mean = mean(nbr_val)),
                    by = from_row]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign aggregated values back by row index
  cell_dt[agg$from_row, (max_col)  := agg$nbr_max]
  cell_dt[agg$from_row, (min_col)  := agg$nbr_min]
  cell_dt[agg$from_row, (mean_col) := agg$nbr_mean]
}

# Clean up helper column
edge_rows[, nbr_val := NULL]

# ---------------------------------------------------------------
# 4.  Convert back to data.frame if needed for predict()
# ---------------------------------------------------------------
cell_dt[, cell_idx := NULL]
cell_dt[, row_id   := NULL]
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# 5.  Predict with the already-trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` on the same non-NA neighbor values â€” the estimand is unchanged. |
| **Trained RF model** | Untouched; only the feature-construction step is replaced. Column names are identical. |
| **Memory** | The largest object is `edge_rows` (~38.5M rows Ã— 5 integer/numeric cols â‰ˆ 1.5 GB). Fits in 16 GB RAM. |
| **Speed** | All hot loops replaced by `data.table` vectorized joins and grouped aggregation. Expected runtime: **2â€“10 minutes** on a standard laptop (vs. 86+ hours). |

**Key insight:** The original code performed ~6.46 million R-level iterations each involving string construction and named-vector lookup. The optimized version eliminates all per-row R interpretation by (a) replacing string keys with integer arithmetic, (b) pre-expanding the cell-level neighbor graph to a row-level edge list via vectorized cross-join, and (c) computing grouped aggregations entirely within `data.table`'s C backend.
END_RESPONSE>>>
