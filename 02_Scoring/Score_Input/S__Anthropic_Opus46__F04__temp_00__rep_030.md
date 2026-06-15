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
run_id: Anthropic_F04_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`) â€” fine individually, but done 6.46M times.
2. Subsets the `neighbors` list to get neighbor cell IDs.
3. Constructs character paste keys (`paste(id, year, sep="_")`) for every neighbor of every row.
4. Looks those keys up in `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) in the worst case for R named vectors, though internally hashed).
5. Filters `NA`s.

The result is **~6.46 million R-level iterations**, each allocating multiple small character vectors and performing multiple hash lookups. This is catastrophically slow in interpreted R. `compute_neighbor_stats` is a secondary bottleneck (another 6.46M-iteration `lapply` per variable), but lighter per iteration.

**Root cause summary:** Row-level `lapply` with per-row string construction and lookup over millions of rows in pure R.

## Optimization Strategy

**Core idea:** Eliminate the row-level loop entirely. Replace it with a vectorized join using `data.table`. Instead of building a per-row list of neighbor indices, build a **long-format edge table** (source_row â†’ neighbor_row) once, then compute all neighbor statistics via grouped aggregation â€” all in compiled C code under the hood.

### Steps

1. **Vectorized neighbor lookup construction:** Expand the `nb` object into a long `data.table` of `(cell_id, neighbor_cell_id)` pairs. Join with the panel data on `(neighbor_cell_id, year)` to get `(source_row, neighbor_row)` pairs. No per-row `lapply`.

2. **Vectorized neighbor stats:** For each variable, index into the values vector using the neighbor_row column, then group-by source_row to compute `max`, `min`, `mean`. This is a single `data.table` grouped aggregation â€” extremely fast.

3. **Memory:** The edge table will have ~6.46M rows Ã— (avg ~4 rook neighbors) â‰ˆ 25.8M rows Ã— a few integer/double columns â€” well within 16 GB.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 1: Build vectorized edge table (replaces build_neighbor_lookup) â”€â”€â”€â”€â”€â”€

build_neighbor_edges <- function(cell_data_dt, id_order, neighbors) {
  # Expand nb object into a long data.table of (cell_id, neighbor_cell_id)
  # neighbors is a list of integer index vectors into id_order
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edges <- data.table(
    cell_id          = id_order[from_idx],
    neighbor_cell_id = id_order[to_idx]
  )

  # Create a row-index column in the panel data
  cell_data_dt[, .row_idx := .I]

  # For each edge Ã— year combination, find the source row and neighbor row.
  # Source side: join edges with panel on cell_id to get (source_row, year, neighbor_cell_id)
  source_keys <- cell_data_dt[, .(cell_id = id, year, source_row = .row_idx)]
  setkey(source_keys, cell_id)
  setkey(edges, cell_id)

  # Merge: for every (cell_id, neighbor_cell_id) edge, replicate across all years
  # of that cell_id present in the panel
  edge_year <- merge(edges, source_keys, by = "cell_id", allow.cartesian = TRUE)
  # edge_year now has: cell_id, neighbor_cell_id, year, source_row

  # Neighbor side: find the row index of each (neighbor_cell_id, year)
  neighbor_keys <- cell_data_dt[, .(neighbor_cell_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_keys, neighbor_cell_id, year)
  setkey(edge_year, neighbor_cell_id, year)

  # Inner join: drops edges where the neighbor doesn't exist in that year
  edge_full <- merge(edge_year, neighbor_keys, by = c("neighbor_cell_id", "year"))
  # edge_full has: neighbor_cell_id, year, cell_id, source_row, neighbor_row

  # We only need source_row and neighbor_row
  edge_full[, .(source_row, neighbor_row)]
}

# â”€â”€ Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats) â”€

compute_neighbor_stats_vec <- function(cell_data_dt, edge_dt, var_name) {
  n <- nrow(cell_data_dt)
  vals <- cell_data_dt[[var_name]]

  # Attach neighbor values
  work <- edge_dt[, .(source_row, nval = vals[neighbor_row])]
  # Remove NA neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = source_row]

  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[agg$source_row]  <- agg$nb_max
  out_min[agg$source_row]  <- agg$nb_min
  out_mean[agg$source_row] <- agg$nb_mean

  data.table(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# â”€â”€ Step 3: Main pipeline (replaces outer loop) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_data_dt <- as.data.table(cell_data)

  message("Building edge table...")
  edge_dt <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, source_row)
  message(sprintf("Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_vec(cell_data_dt, edge_dt, var_name)

    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_data_dt, j = col_max,  value = stats$nb_max)
    set(cell_data_dt, j = col_min,  value = stats$nb_min)
    set(cell_data_dt, j = col_mean, value = stats$nb_mean)
  }

  # Clean up helper column
  cell_data_dt[, .row_idx := NULL]

  # Return as data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(cell_data_dt))
  }
  cell_data_dt
}

# â”€â”€ Usage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` over the identical neighbor sets produce identical values. The inner join replicates the original `!is.na` filter. |
| **Trained RF model untouched** | Only the feature-construction step is replaced; the model object and `predict()` call are unchanged. |
| **Column names** | Must match what the RF expects â€” adjust the `paste0` naming pattern to match your existing `compute_and_add_neighbor_features` output names. |
| **Memory** | Edge table â‰ˆ 25.8M rows Ã— 2 int columns â‰ˆ 0.4 GB. Temporary `work` table per variable â‰ˆ 0.6 GB peak. Well within 16 GB. |
| **Expected speedup** | The `merge` and grouped `data.table` operations run in compiled C. Expected wall time: **~2â€“10 minutes** total (vs. 86+ hours), roughly a **500â€“2500Ã— speedup**. |

The key insight is that the original code's per-row `lapply` with string construction is replaced by two bulk equi-joins and five grouped aggregations â€” all operations `data.table` executes at near-C speed via radix-based indexing.
END_RESPONSE>>>
