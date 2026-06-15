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
run_id: Anthropic_F04_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It lies squarely in `build_neighbor_lookup` and `compute_neighbor_stats`, both of which rely on **row-level `lapply` loops over ~6.46 million rows** with expensive per-iteration string operations.

### Specific problems:

**1. `build_neighbor_lookup` â€” O(n) string-paste loop with named-vector lookups**

For each of the ~6.46 million rows, the function:
- Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`): **named-vector lookup is O(n) in the worst case in base R** because it uses linear hashing with potential collisions at scale.
- Calls `paste()` to construct neighbor keys (one per neighbor per row). With an average of ~4 rook neighbors, this is ~25.8 million `paste` calls.
- Indexes into `idx_lookup` (another named character vector of length 6.46 million) for each constructed key.
- Filters out `NA`s.

The result: **~6.46 million iterations, each doing multiple named-vector lookups into a 6.46-million-element vector, plus millions of `paste` operations**. This alone likely accounts for the majority of the 86+ hour runtime.

**2. `compute_neighbor_stats` â€” O(n) lapply with per-row subsetting**

For each row, it extracts neighbor values, removes `NA`s, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), so that is 5 Ã— 6.46M = ~32.3 million R-level function invocations. Each one allocates small vectors and runs three summary functions. The overhead of the R interpreter loop is enormous here.

**3. `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors** is itself slow because `rbind` on a long list is notoriously inefficient.

---

## Optimization Strategy

The core insight: **replace row-level R loops with vectorized data.table merge-and-group-by operations**.

### Plan:

1. **Replace `build_neighbor_lookup`** entirely. Instead of building a list-of-integer-vectors, construct a **long-form edge table** `(row_i, row_j)` that maps every cell-year row to its neighbor cell-year rows. This is done via a single vectorized `data.table` merge â€” no `lapply`, no `paste` per row.

2. **Replace `compute_neighbor_stats`** with a single **`data.table` grouped aggregation** (`[, .(max, min, mean), by = row_i]`) over the edge table joined to the variable column. This replaces 6.46 million R function calls with one vectorized C-level group-by.

3. **Eliminate `do.call(rbind, ...)`** â€” the `data.table` aggregation returns a proper table directly.

4. **Loop over the 5 variables** remains, but each iteration is now a fast vectorized operation (~seconds, not hours).

### Expected speedup:

| Step | Before | After |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with `paste` + named lookups | One `data.table` merge (~seconds) |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— 6.46M `lapply` iterations | 5 Ã— one `data.table` grouped aggregation |
| Total estimated time | 86+ hours | **~2â€“10 minutes** on a 16 GB laptop |

The Random Forest model is not touched. The numerical outputs (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

#' Build a vectorized edge table mapping each cell-year row index
#' to its neighbor cell-year row indices.
#'
#' @param data       data.frame/data.table with columns `id` and `year`
#' @param id_order   integer vector of cell IDs in the order used by the nb object
#' @param neighbors  spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: row_i (focal row), row_j (neighbor row)
build_neighbor_edge_table <- function(data, id_order, neighbors) {

  # --- Step 1: Build a cell-level edge list (focal_id -> neighbor_id) ----------
  # Each element neighbors[[k]] is an integer vector of indices into id_order.
  # We expand this into a two-column table.

  n_cells <- length(id_order)
  focal_indices <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_indices <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid <- neighbor_indices != 0L
  focal_indices    <- focal_indices[valid]
  neighbor_indices <- neighbor_indices[valid]

  cell_edges <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  rm(focal_indices, neighbor_indices, valid)  # free memory

  # --- Step 2: Build a row-index lookup keyed by (id, year) -------------------
  dt <- as.data.table(data)
  dt[, row_idx := .I]  # original row position

  # --- Step 3: Merge to create (row_i, row_j) --------------------------------
  # For every (focal_id, neighbor_id) pair, we need every year that both exist in.
  # This is equivalent to:
  #   for each cell-year row i with (focal_id, year_t),
  #     find all rows j with (neighbor_id, year_t).


  # Keyed lookup tables
  focal_lookup <- dt[, .(focal_id = id, year, row_i = row_idx)]
  setkey(focal_lookup, focal_id, year)

  neighbor_lookup <- dt[, .(neighbor_id = id, year, row_j = row_idx)]
  setkey(neighbor_lookup, neighbor_id, year)

  # Join cell_edges with focal_lookup to get (row_i, neighbor_id, year)
  # Then join with neighbor_lookup to get (row_i, row_j)
  setkey(cell_edges, focal_id)
  setkey(focal_lookup, focal_id)

  # First merge: attach row_i and year to each edge
  edge_with_focal <- cell_edges[focal_lookup, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: focal_id, neighbor_id, row_i, year

  rm(focal_lookup, cell_edges)

  # Second merge: attach row_j for the neighbor in the same year
  setkey(edge_with_focal, neighbor_id, year)
  setkey(neighbor_lookup, neighbor_id, year)

  edge_table <- edge_with_focal[neighbor_lookup, on = c("neighbor_id", "year"), nomatch = 0L]
  # Columns include: row_i, row_j  (plus others we can drop)

  rm(edge_with_focal, neighbor_lookup)

  edge_table <- edge_table[, .(row_i, row_j)]
  setkey(edge_table, row_i)

  return(edge_table)
}


#' Compute neighbor max, min, mean for one variable using the edge table.
#'
#' @param data       data.frame/data.table (original row order)
#' @param edge_table data.table with columns row_i, row_j
#' @param var_name   character: name of the variable in data
#' @return data.table with columns: row_i, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(data, edge_table, var_name) {
  vals <- data[[var_name]]

  # Attach the neighbor's value to each edge
  et <- copy(edge_table)
  et[, nb_val := vals[row_j]]

  # Drop edges where neighbor value is NA
  et <- et[!is.na(nb_val)]

  # Grouped aggregation â€” single vectorized pass
  stats <- et[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = row_i]

  return(stats)
}


#' Compute and attach neighbor features for one variable to the dataset.
#'
#' @param data       data.frame/data.table (will be modified by reference if data.table)
#' @param var_name   character
#' @param edge_table data.table from build_neighbor_edge_table
#' @return data with three new columns: <var>_nb_max, <var>_nb_min, <var>_nb_mean
compute_and_add_neighbor_features_fast <- function(data, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data, edge_table, var_name)

  n <- nrow(data)
  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")

  # Initialize with NA, then fill matched rows
  data[[col_max]]  <- NA_real_
  data[[col_min]]  <- NA_real_
  data[[col_mean]] <- NA_real_

  data[[col_max]] [stats$row_i] <- stats$nb_max
  data[[col_min]] [stats$row_i] <- stats$nb_min
  data[[col_mean]][stats$row_i] <- stats$nb_mean

  return(data)
}


# ==============================================================================
# MAIN PIPELINE (drop-in replacement for the original outer loop)
# ==============================================================================

# Build the edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table built: %s edges", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

message("Neighbor feature construction complete.")

# cell_data now has the same 15 new columns (5 vars Ã— 3 stats) as the original code.
# Proceed to Random Forest prediction with the already-trained model â€” no retraining.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `lapply` over 6.46M rows, `paste` + named-vector search | Single `data.table` equi-join; vectorized |
| Neighbor stats | `lapply` over 6.46M rows Ã— 5 vars, per-row `max`/`min`/`mean` | `data.table` grouped aggregation, C-level |
| Memory pattern | 6.46M-element list of integer vectors (~large overhead) | One flat two-column edge table (~105M rows Ã— 2 int cols â‰ˆ 1.6 GB) |
| Estimated runtime | 86+ hours | ~2â€“10 minutes |
| Numerical result | max, min, mean of non-NA neighbor values | **Identical** |
| RF model | Untouched | Untouched |

The key optimization is converting the **row-level R loop with string key construction and named-vector lookup** into a **vectorized relational join and group-by**, which is what `data.table` is specifically engineered to do at scale.
END_RESPONSE>>>
