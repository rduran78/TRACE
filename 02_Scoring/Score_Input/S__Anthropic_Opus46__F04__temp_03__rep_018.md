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
run_id: Anthropic_F04_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash-probe per call on a 6.46M-element named vector) inside a per-row `lapply` over 6.46 million iterations is catastrophically slow. The secondary bottleneck, `compute_neighbor_stats`, also uses a per-row `lapply` with subsetting, but is less severe since the index vectors are already integer.

**Root causes, ranked:**

| # | Cause | Impact |
|---|-------|--------|
| 1 | Per-row `paste` + named-vector string lookup in `build_neighbor_lookup` over 6.46M rows | Dominant (~95% of wall time) |
| 2 | R-level `lapply` loop with no vectorization for neighbor stats | Moderate |
| 3 | Repeated `do.call(rbind, ...)` on a 6.46M-element list | Minor but adds GC pressure |

## Optimization Strategy

1. **Eliminate all string key construction.** Replace the `paste(id, year)`â†’`idx_lookup` approach with a direct integer-indexed matrix/table join. Pre-sort data by `(id, year)` so that for a given cell `id` with `Y` years, its rows occupy a contiguous block. Then the row index for any `(neighbor_id, year)` pair can be computed arithmetically: `offset[neighbor_id] + (year - min_year)`. This turns the 6.46M string lookups into O(1) integer arithmetic.

2. **Vectorize neighbor stats with `data.table` grouping or matrix operations.** Expand the neighbor list into a long-form edge table `(row_i, row_j)`, then use `data.table` grouped aggregation to compute max/min/mean in one vectorized pass per variable.

3. **Process all 5 variables in one pass** over the edge table rather than 5 separate `lapply` calls.

## Optimized Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, neighbors, neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # ---- Step 1: Build an arithmetic row-index lookup ----
  # Ensure data is sorted by (id, year) so each cell's years form a contiguous block
  dt[, orig_row := .I]
  setkey(dt, id, year)
  dt[, sorted_row := .I]

  years <- sort(unique(dt$year))
  min_year <- min(years)
  n_years <- length(years)

  # For each unique id, record the starting row in the sorted table
  # Because data is keyed by (id, year) and panel is balanced (or we handle gaps),
  # the offset for cell c is: start_row[c] - 1, and row for (c, y) = offset[c] + (y - min_year + 1)
  id_starts <- dt[, .(start = min(sorted_row), count = .N), by = id]
  setkey(id_starts, id)

  # Build a fast integer-keyed lookup: id -> start_row
  # Use a named integer vector keyed by character id for O(1) amortized lookup via match
  all_ids <- id_starts$id
  start_vec <- id_starts$start  # start_vec[k] = first sorted_row for all_ids[k]

  # Map id_order indices to actual cell ids
  # neighbors[[k]] gives neighbor indices into id_order
  # id_order[k] gives the cell id

  # We need: for each id in id_order, its position in all_ids (sorted unique ids)
  id_to_pos <- match(id_order, all_ids)

  # ---- Step 2: Build long-form edge table (row_i, row_j) ----
  # For each cell i (index in id_order), its neighbors are neighbors[[i]] (indices in id_order).
  # For each year y, we need edge: (sorted_row of (id_order[i], y)) -> (sorted_row of (id_order[j], y))

  message("Building edge table...")

  # Expand neighbor list to edge list at the cell level: (cell_idx, neighbor_cell_idx)
  n_cells <- length(id_order)
  from_cell <- rep(seq_len(n_cells), lengths(neighbors))
  to_cell   <- unlist(neighbors)

  # Now expand across years: each cell-level edge becomes n_years row-level edges
  n_edges_cell <- length(from_cell)

  # Vectorized expansion
  # For cell c at position id_to_pos[c], start row = start_vec[id_to_pos[c]]
  # Row for year y (0-indexed offset) = start + (y - min_year)

  from_starts <- start_vec[id_to_pos[from_cell]]
  to_starts   <- start_vec[id_to_pos[to_cell]]

  # Check for NAs (cells in id_order not present in data)
  valid <- !is.na(from_starts) & !is.na(to_starts)
  from_starts <- from_starts[valid]
  to_starts   <- to_starts[valid]
  n_valid <- sum(valid)

  year_offsets <- seq(0L, n_years - 1L)

  # Use rep to expand: each valid cell-edge Ã— each year
  row_i <- rep(from_starts, each = n_years) + rep(year_offsets, times = n_valid)
  row_j <- rep(to_starts,   each = n_years) + rep(year_offsets, times = n_valid)

  # Filter to valid row indices (handles unbalanced panels)
  n_sorted <- nrow(dt)
  keep <- row_i >= 1L & row_i <= n_sorted & row_j >= 1L & row_j <= n_sorted
  row_i <- row_i[keep]
  row_j <- row_j[keep]

  edges <- data.table(row_i = row_i, row_j = row_j)

  message(sprintf("Edge table: %s row-level edges", format(nrow(edges), big.mark = ",")))

  # ---- Step 3: Vectorized grouped aggregation for all variables ----
  message("Computing neighbor statistics...")

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing: %s", var_name))

    vals <- dt[[var_name]][edges$row_j]
    edges[, val := vals]

    # Remove NA neighbor values before aggregation
    agg <- edges[!is.na(val), .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = row_i]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    dt[agg$row_i, (max_col)  := agg$nb_max]
    dt[agg$row_i, (min_col)  := agg$nb_min]
    dt[agg$row_i, (mean_col) := agg$nb_mean]
  }

  edges[, val := NULL]

  # ---- Step 4: Restore original row order ----
  setorder(dt, orig_row)
  dt[, c("orig_row", "sorted_row") := NULL]

  message("Done.")
  return(as.data.frame(dt))
}
```

**Usage (drop-in replacement for the original outer loop):**

```r
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged; predict as before:
# predictions <- predict(rf_model, newdata = cell_data[, predictor_cols])
```

## Why This Works and Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor value sets; NA handling is identical. |
| **Trained RF model untouched** | No retraining; only the feature-construction step is replaced. Column names are identical (`{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`). |
| **Memory on 16 GB laptop** | The edge table is ~38.5M rows Ã— 3 integer/numeric columns â‰ˆ ~900 MB. The `data.table` grouped aggregation is memory-efficient and processes one variable at a time. Peak usage stays well under 16 GB. |
| **Speed** | Eliminates 6.46M `paste` + string-lookup iterations. The `data.table` grouped `max`/`min`/`mean` on ~38.5M rows completes in seconds per variable. **Expected total: ~2â€“5 minutes** vs. the original 86+ hours. |

## Summary of Speedup Sources

| Original Operation | Cost | Replacement | Cost |
|---|---|---|---|
| `paste(id, year)` Ã— 6.46M rows Ã— ~4 neighbors | ~25.8M string constructions | Integer arithmetic on pre-sorted offsets | Vectorized, ~0 |
| Named-vector lookup on 6.46M-key vector | O(n) per probe Ã— 25.8M probes | `data.table` keyed join / direct integer indexing | O(1) amortized |
| `lapply` over 6.46M rows for stats | 6.46M R function calls Ã— 5 vars | `data.table` grouped aggregation | 1 vectorized pass Ã— 5 vars |

**Estimated speedup: ~1,000â€“2,000Ã—** (from ~86 hours to ~2â€“5 minutes).
END_RESPONSE>>>
