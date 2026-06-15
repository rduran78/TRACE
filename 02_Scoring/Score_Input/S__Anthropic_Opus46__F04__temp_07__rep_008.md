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
run_id: Anthropic_F04_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction functions, and specifically the interaction between `build_neighbor_lookup` and `compute_neighbor_stats`, both of which are implemented as **row-level `lapply` loops over ~6.46 million rows** with per-iteration string operations.

### `build_neighbor_lookup` â€” Primary Bottleneck

For each of the ~6.46M rows, this function:

1. **Converts an integer ID to character and looks it up in a named vector** (`id_to_ref`): O(1) amortized, but with per-call character coercion overhead Ã— 6.46M.
2. **Indexes into the `neighbors` list** to retrieve neighbor cell IDs: fast.
3. **Pastes neighbor cell IDs with the current year** (`paste(..., sep="_")`): this creates a character vector of length â‰ˆ average_degree (â‰ˆ4 for rook) per row. Over all rows this is ~25.8M string constructions.
4. **Looks up each pasted key in `idx_lookup`**, a named character vector of length 6.46M: named-vector lookup in R is **O(n) per probe** unless R internally hashes it (it does for `[.character` on named vectors, but the hash table is rebuilt or probed repeatedly). Still, ~25.8M hash probes into a 6.46M-entry table is expensive in R's interpreted loop.

The entire function wraps this in `lapply` over 6.46M iterations. **R-level `lapply` with non-trivial closures over millions of iterations is extremely slow** â€” estimated at 40â€“60+ hours alone.

### `compute_neighbor_stats` â€” Secondary Bottleneck

Another `lapply` over 6.46M rows, each computing `max`, `min`, `mean` on a small vector (~4 elements). The per-call overhead of R function dispatch dominates. This is called **5 times** (once per source variable), contributing another 20+ hours.

### Summary of Root Causes

| Cause | Location | Impact |
|---|---|---|
| 6.46M-iteration R-level `lapply` with string ops | `build_neighbor_lookup` | ~40â€“60 hrs |
| `paste()` + named-vector lookup per row | `build_neighbor_lookup` | Major |
| 6.46M-iteration `lapply` Ã— 5 variables | `compute_neighbor_stats` | ~20â€“30 hrs |
| No vectorization or use of integer arithmetic | Both functions | Fundamental |

---

## Optimization Strategy

### Core Idea: Replace string-key lookups with integer-indexed joins using `data.table`, and vectorize the neighbor stats computation.

**Three-part plan:**

1. **Replace `build_neighbor_lookup`** with a fully vectorized `data.table` equi-join. Instead of building a per-row list of neighbor row indices via string pasting and lookup, we:
   - Expand the neighbor graph into an edge-list `(cell_id, neighbor_id)`.
   - Join with the panel data on `(neighbor_id, year)` to get neighbor row indices.
   - Group by source-row index to collect neighbor row indices.
   
   This replaces 6.46M R-level iterations with a single `data.table` merge (~25.8M rows joined against a 6.46M-row keyed table â€” seconds, not hours).

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable. Instead of `lapply` over rows, we:
   - Build a long table of `(source_row, neighbor_value)`.
   - Compute `max`, `min`, `mean` grouped by `source_row`.
   
   This replaces 5 Ã— 6.46M R-level iterations with 5 vectorized group-by operations.

3. **Memory management**: The edge-list expansion (~25.8M rows Ã— a few integer columns) fits easily in 16 GB. We process one variable at a time and discard intermediates.

**Expected speedup**: From 86+ hours to **~5â€“15 minutes** total.

**Numerical equivalence**: The `max`, `min`, and `mean` computations are identical; only the iteration mechanism changes. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#' Drop-in replacement for the original build_neighbor_lookup +
#' compute_neighbor_stats + outer-loop pipeline.
#'
#' @param cell_data        data.frame (or data.table) with columns: id, year,
#'                         and all neighbor_source_vars columns.
#' @param id_order         integer vector: the cell IDs in the order matching
#'                         the spdep::nb object indices.
#' @param rook_neighbors   spdep::nb list (rook_neighbors_unique).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus neighbor feature columns.
build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors,
                                         neighbor_source_vars) {

  # --- Step 0: Convert to data.table if needed; add row index ---
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Step 1: Build the directed edge list from the nb object ---
  # rook_neighbors is a list of length = length(id_order).
  # rook_neighbors[[i]] contains integer indices (into id_order) of neighbors
  # of cell id_order[i].  0L entries mean no neighbors (spdep convention).
  message("Building edge list from nb object...")

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb_idx <- rook_neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id     = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1.37M rows (directed rook edges)

  message(sprintf("Edge list: %s directed edges.", format(nrow(edge_list), big.mark = ",")))

  # --- Step 2: Create a keyed lookup from (id, year) -> row index ---
  # This replaces the string-pasted idx_lookup.
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # --- Step 3: For each source row, find all (source_row_idx, neighbor_row_idx) pairs ---
  # Join edge_list with dt to get the year dimension.
  # For every row in dt, we know its cell_id and year.
  # Its neighbors are given by edge_list where edge_list$cell_id == dt$id.
  # The neighbor rows are those with (edge_list$neighbor_id, same year).

  message("Joining edges with panel years to build full neighbor map...")

  # Source rows: (cell_id, year, source_row_idx)
  source_rows <- dt[, .(cell_id = id, year, src_idx = .row_idx)]
  setkey(source_rows, cell_id)

  # Merge source rows with edge list on cell_id to get:
  # (cell_id, year, src_idx, neighbor_id)
  setkey(edge_list, cell_id)
  expanded <- edge_list[source_rows, on = "cell_id",
                        .(year       = i.year,
                          src_idx    = i.src_idx,
                          neighbor_id = x.neighbor_id),
                        allow.cartesian = TRUE,
                        nomatch = 0L]
  # expanded has ~6.46M * avg_degree â‰ˆ 25.8M rows

  message(sprintf("Expanded neighbor-year pairs: %s rows.",
                  format(nrow(expanded), big.mark = ",")))

  # Now look up the neighbor's row index for (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded[row_key, on = .(neighbor_id = id, year = year),
           nbr_idx := i..row_idx]

  # Drop rows where the neighbor doesn't exist in the panel (boundary / NA)
  expanded <- expanded[!is.na(nbr_idx)]

  message(sprintf("Valid neighbor links after join: %s.",
                  format(nrow(expanded), big.mark = ",")))

  # --- Step 4: Compute neighbor stats per variable, vectorized ---
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    # Attach neighbor values
    expanded[, nbr_val := dt[[var_name]][nbr_idx]]

    # Remove NAs in neighbor values for aggregation
    valid <- expanded[!is.na(nbr_val)]

    # Grouped aggregation
    agg <- valid[, .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ), by = src_idx]

    # Initialize columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results by row index
    dt[agg$src_idx, (max_col)  := agg$nb_max]
    dt[agg$src_idx, (min_col)  := agg$nb_min]
    dt[agg$src_idx, (mean_col) := agg$nb_mean]
  }

  # --- Cleanup ---
  dt[, .row_idx := NULL]

  message("Neighbor feature construction complete.")
  return(dt)
}


# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

# Original code (86+ hours):
# -------------------------------------------------------
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }
# -------------------------------------------------------

# Optimized replacement (~5-15 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the same neighbor feature columns as before.
# The trained Random Forest model can be applied directly â€” no retraining needed.
# predict(rf_model, newdata = cell_data)
```

---

## Why This Works and Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The grouped `max`, `min`, `mean` over the exact same set of non-NA neighbor values produces identical results. No approximation is introduced. |
| **Trained RF model untouched** | Only the feature construction step is replaced. The model object is never modified or retrained. Column names are preserved (assuming `compute_and_add_neighbor_features` used the same `nb_max_`, `nb_min_`, `nb_mean_` naming convention â€” adjust the prefix strings if your original code used different names). |
| **Memory on 16 GB laptop** | The largest intermediate (`expanded`) is ~25.8M rows Ã— 4 integer/numeric columns â‰ˆ 0.8 GB. The `data.table` in-place operations keep peak memory well under 10 GB. |
| **Speedup source** | String operations (`paste`, named-vector lookup) are eliminated entirely. The 6.46M-iteration R-level `lapply` is replaced by `data.table` keyed joins (radix-sort based, C-level) and vectorized grouped aggregation (also C-level). |

### Expected Timing Breakdown (16 GB laptop)

| Step | Estimated Time |
|---|---|
| Edge list construction (~344K list elements) | ~2â€“5 seconds |
| Keyed join to expand to ~25.8M rows | ~30â€“90 seconds |
| Neighbor row index lookup (keyed join) | ~30â€“60 seconds |
| Grouped aggregation Ã— 5 variables | ~2â€“5 minutes |
| **Total** | **~5â€“10 minutes** |

This represents a speedup of roughly **500â€“1000Ã—** over the original implementation.
END_RESPONSE>>>
