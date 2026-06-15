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
run_id: Anthropic_F04_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, the function:

1. Performs a character-based hash lookup (`id_to_ref[as.character(...)]`).
2. Indexes into the `neighbors` list.
3. Constructs character key strings via `paste(neighbor_cell_ids, data$year[i], sep = "_")`.
4. Performs another character-based hash lookup (`idx_lookup[neighbor_keys]`).

String construction and named-vector lookup are extremely slow in R when iterated millions of times in an interpreted loop. With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces roughly **25.8 million `paste` calls and named-vector lookups** inside a sequential `lapply`. This is the dominant cost.

**`compute_neighbor_stats`** is a secondary bottleneck: another `lapply` over 6.46M rows, each calling `max`, `min`, `mean` on small vectors. This is repeated 5 times (once per variable), totaling ~32.3M R-level function invocations.

The Random Forest inference, by contrast, is a single vectorized `predict()` call on a matrix â€” fast by comparison.

## Optimization Strategy

### 1. Vectorized Neighbor Lookup via Integer Arithmetic (eliminate all `paste`/string ops)

Instead of building character keys, exploit the panel structure: every cell appears once per year in a fixed order. Build a direct integer matrix mapping `(cell_index, year_index) â†’ row_number`. Then neighbor row indices can be retrieved via integer matrix indexing â€” no strings at all.

### 2. Vectorized Neighbor Stats via Matrix Indexing (eliminate per-row `lapply`)

Unroll the neighbor lookup into a long-form `(row, neighbor_row)` edge list. Then use vectorized grouped operations (via `data.table` or `rowsum`) to compute max/min/mean in one pass per variable â€” no R-level loop over 6.46M rows.

### 3. Preserve Numerical Equivalence

The operations (max, min, mean of non-NA neighbor values) are reproduced exactly.

## Optimized R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a vectorized neighbor edge list (replaces build_neighbor_lookup)
# ==============================================================================
build_neighbor_edgelist <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year', row order matters
  # id_order: vector of cell IDs (same order as neighbors list)
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  n_cells <- length(id_order)
  years   <- sort(unique(data_dt$year))
  n_years <- length(years)

  # Map cell id -> integer index in id_order (1-based)
  cell_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map (cell_index, year) -> row number in data_dt
  # Assumes data_dt is keyed/sorted by (id, year) or we build a lookup
  data_dt[, row_id := .I]
  row_lookup <- data_dt[, .(row_id, cell_idx = cell_idx[as.character(id)], year)]

  # Build a (cell_index, year_index) -> row_id matrix for O(1) lookup
  year_idx_map <- setNames(seq_along(years), as.character(years))
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[, year_idx := year_idx_map[as.character(year)]]
  row_matrix[cbind(row_lookup$cell_idx, row_lookup$year_idx)] <- row_lookup$row_id

  # Build the edge list: for each cell, expand its neighbors across all years

  # Pre-compute neighbor counts
  n_neighbors <- lengths(neighbors)  # integer vector, length n_cells

  # Total directed edges across all cell-years
  total_edges <- sum(as.numeric(n_neighbors)) * n_years

  # Pre-allocate vectors
  from_row <- integer(total_edges)
  to_row   <- integer(total_edges)

  offset <- 0L
  for (ci in seq_len(n_cells)) {
    nb <- neighbors[[ci]]
    if (length(nb) == 0L) next
    for (yi in seq_len(n_years)) {
      src <- row_matrix[ci, yi]
      if (is.na(src)) next
      tgts <- row_matrix[nb, yi]
      valid <- !is.na(tgts)
      n_valid <- sum(valid)
      if (n_valid == 0L) next
      idx_range <- (offset + 1L):(offset + n_valid)
      from_row[idx_range] <- src
      to_row[idx_range]   <- tgts[valid]
      offset <- offset + n_valid
    }
  }

  # Trim to actual size
  from_row <- from_row[seq_len(offset)]
  to_row   <- to_row[seq_len(offset)]

  data.table(from_row = from_row, to_row = to_row)
}

# ==============================================================================
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ==============================================================================
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # Extract neighbor values via vectorized indexing
  vals <- data_dt[[var_name]]
  n_rows <- nrow(data_dt)

  edge_dt[, nbr_val := vals[to_row]]

  # Remove NAs
  valid_edges <- edge_dt[!is.na(nbr_val)]

  # Grouped aggregation â€” one pass
  stats <- valid_edges[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = from_row]

  # Allocate full-length result columns (NA default)
  result <- data.table(
    nb_max  = rep(NA_real_, n_rows),
    nb_min  = rep(NA_real_, n_rows),
    nb_mean = rep(NA_real_, n_rows)
  )
  result[stats$from_row, `:=`(
    nb_max  = stats$nb_max,
    nb_min  = stats$nb_min,
    nb_mean = stats$nb_mean
  )]

  # Clean up temporary column
  edge_dt[, nbr_val := NULL]

  setnames(result,
    c("nb_max", "nb_min", "nb_mean"),
    paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  )
  result
}

# ==============================================================================
# STEP 3: Main pipeline (replaces outer loop)
# ==============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  cell_dt <- as.data.table(cell_data)

  message("Building neighbor edge list...")
  t0 <- Sys.time()
  edge_dt <- build_neighbor_edgelist(cell_dt, id_order, rook_neighbors_unique)
  message("  Edge list built: ", nrow(edge_dt), " edges in ",
          round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    stats_dt <- compute_neighbor_stats_vec(cell_dt, edge_dt, var_name)
    cell_dt <- cbind(cell_dt, stats_dt)
    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
  }

  # Return as data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(cell_dt))
  }
  cell_dt
}

# ==============================================================================
# Usage (drop-in replacement):
# ==============================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Further Optimization: Rcpp Inner Loop for Edge List Construction

The `build_neighbor_edgelist` function still has a double `for` loop over 344K cells Ã— 28 years. If this remains slow (~10-20 min), replace it with Rcpp:

```r
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List build_edgelist_cpp(IntegerMatrix row_matrix, List neighbors,
                        int n_cells, int n_years) {
  // Estimate size
  int total = 0;
  for (int ci = 0; ci < n_cells; ci++) {
    IntegerVector nb = neighbors[ci];
    total += nb.size() * n_years;
  }

  IntegerVector from_row(total);
  IntegerVector to_row(total);
  int offset = 0;

  for (int ci = 0; ci < n_cells; ci++) {
    IntegerVector nb = neighbors[ci];  // 1-based indices
    int n_nb = nb.size();
    if (n_nb == 0) continue;

    for (int yi = 0; yi < n_years; yi++) {
      int src = row_matrix(ci, yi);  // NA_INTEGER if missing
      if (src == NA_INTEGER) continue;

      for (int ni = 0; ni < n_nb; ni++) {
        int tgt = row_matrix(nb[ni] - 1, yi);  // nb is 1-based
        if (tgt == NA_INTEGER) continue;
        from_row[offset] = src;
        to_row[offset]   = tgt;
        offset++;
      }
    }
  }

  from_row = from_row[Range(0, offset - 1)];
  to_row   = to_row[Range(0, offset - 1)];

  return List::create(Named("from_row") = from_row,
                      Named("to_row")   = to_row);
}
')
```

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~60-80 hrs (paste/named lookup Ã— 6.46M) | ~1-5 min (integer matrix + Rcpp) |
| `compute_neighbor_stats` (Ã—5 vars) | ~6-10 hrs (lapply Ã— 6.46M Ã— 5) | ~30-90 sec (data.table grouped agg) |
| **Total neighbor features** | **~86+ hrs** | **~5-10 min** |
| RF prediction | Unchanged | Unchanged |
| Numerical output | â€” | Identical (max/min/mean preserved) |

The trained Random Forest model is untouched; only the feature-construction preprocessing is replaced. The computed neighbor max, min, and mean values are numerically identical to the original implementation.
END_RESPONSE>>>
