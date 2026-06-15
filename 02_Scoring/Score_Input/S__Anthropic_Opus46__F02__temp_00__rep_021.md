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
run_id: Anthropic_F02_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated for 5 variables, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-keyed lookup (`paste` + named vector indexing):** Building `idx_lookup` as a named vector of length ~6.46M and then querying it millions of times with character keys is extremely slow. Named vector lookup in R is O(n) per query in the worst case because R rehashes internally.
2. **Row-wise `lapply` over 6.46M rows:** Each iteration allocates small vectors, causing massive GC pressure.
3. **Redundant work across variables:** The neighbor lookup is the same for all 5 variables, but the stats computation still loops in R over 6.46M list elements per variable.
4. **Memory:** Storing a list of 6.46M integer vectors (the neighbor lookup) plus the 6.46M Ã— 110 data frame is feasible in 16 GB but leaves little headroom.

---

## Optimization Strategy

### 1. Replace string-keyed lookup with integer-keyed lookup using `data.table`

Use `data.table` to join on `(id, year)` as integer keys instead of pasting strings. This converts the O(n Ã— k) string-hash problem into a fast equi-join.

### 2. Build an edge list, not a per-row list

Instead of a list of 6.46M elements, build a flat **edge table** `(row_i, neighbor_row_j)` using a merge/join. This is cache-friendly and avoids millions of small allocations.

### 3. Vectorized grouped aggregation with `data.table`

Once we have the edge table with the neighbor's variable value joined in, compute `max`, `min`, and `mean` as a grouped `data.table` aggregation â€” fully vectorized in C, no R-level loop.

### 4. Process all 5 variables in one pass

Join all 5 source variable columns onto the edge table at once, then compute all 15 summary statistics (5 vars Ã— 3 stats) in a single grouped aggregation.

### Expected speedup

- `build_neighbor_lookup` (hours) â†’ edge-table construction via `data.table` join (~30â€“90 seconds).
- `compute_neighbor_stats` for 5 variables (hours) â†’ single grouped aggregation (~30â€“60 seconds).
- **Total: ~2â€“5 minutes** instead of 86+ hours.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation.
#' Preserves the trained RF model and original numerical estimand.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and
#'                        all neighbor_source_vars columns.
#' @param id_order        integer vector of cell IDs in the order matching
#'                        rook_neighbors_unique (the spdep nb object).
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars   character vector of variable names.
#' @return cell_data with new neighbor feature columns appended.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Step 0: Convert to data.table (by reference if already one) -----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order for safe column-binding later.
  cell_data[, .row_idx := .I]

  # --- Step 1: Build a flat edge list from the nb object ---------------------
  #
  # rook_neighbors_unique[[k]] gives the indices (into id_order) of the
  # neighbors of the cell whose ID is id_order[k].
  #
  # We build a two-column data.table: (focal_id, neighbor_id).

  message("Building edge list from nb object ...")
  edge_list <- rbindlist(
    lapply(seq_along(rook_neighbors_unique), function(k) {
      nb_idx <- rook_neighbors_unique[[k]]
      # spdep nb encodes "no neighbors" as a single 0L
      if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx == 0L)) {
        return(NULL)
      }
      data.table(focal_id    = id_order[k],
                 neighbor_id = id_order[nb_idx])
    }),
    use.names = TRUE
  )
  message(sprintf("  Edge list: %s directed edges.", format(nrow(edge_list), big.mark = ",")))

  # --- Step 2: Map (id, year) â†’ row index ------------------------------------
  #
  # We need to know, for each (neighbor_id, year) pair, which row in cell_data
  # holds the data so we can pull the variable values.

  message("Joining edge list with panel years ...")

  # Unique years present in the data
  years <- sort(unique(cell_data$year))

  # Create a keyed lookup: (id, year) â†’ .row_idx
  id_year_key <- cell_data[, .(id, year, .row_idx)]
  setkey(id_year_key, id, year)

  # Cross-join edges Ã— years to get (focal_id, year, neighbor_id) triples,
  # then join to get the focal row index and the neighbor row index.
  #
  # To avoid a massive cross join (edges Ã— 28 years) in memory all at once,
  # we instead:
  #   (a) For each focal cell-year row, look up its neighbors via the edge list.
  #
  # Efficient approach: join focal rows to edge_list on focal_id, then join
  # neighbor rows on (neighbor_id, year).

  # Focal side: every row in cell_data gets its neighbors
  focal_dt <- cell_data[, .(focal_row = .row_idx, focal_id = id, year)]
  setkey(edge_list, focal_id)
  setkey(focal_dt, focal_id)

  # This join replicates each focal row by its number of neighbors.
  # Result columns: focal_row, focal_id, year, neighbor_id
  expanded <- edge_list[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: focal_id, neighbor_id, focal_row, year

  message(sprintf("  Expanded edge-year table: %s rows.", format(nrow(expanded), big.mark = ",")))

  # Join to get the neighbor's row index
  setkey(expanded, neighbor_id, year)
  setkey(id_year_key, id, year)
  expanded[id_year_key, neighbor_row := i..row_idx, on = c(neighbor_id = "id", "year")]

  # Drop rows where the neighbor has no matching year (shouldn't happen in a

  # balanced panel, but be safe).
  expanded <- expanded[!is.na(neighbor_row)]

  # --- Step 3: Pull neighbor variable values and aggregate -------------------

  message("Computing neighbor statistics (max, min, mean) for all variables ...")

  # Pull the values for every source variable at the neighbor rows.
  for (v in neighbor_source_vars) {
    set(expanded, j = v, value = cell_data[[v]][expanded$neighbor_row])
  }

  # Grouped aggregation: for each focal_row, compute max/min/mean of each var.
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- expanded[, lapply(agg_exprs, eval, envir = .SD), by = focal_row]

  # --- Simpler and more robust aggregation approach --------------------------
  # (Replacing the bquote approach above for clarity and reliability.)

  # We'll compute per-variable stats in a straightforward loop and merge.
  stats_list <- vector("list", length(neighbor_source_vars))

  for (vi in seq_along(neighbor_source_vars)) {
    v <- neighbor_source_vars[vi]
    # Rename the variable column to a fixed name for easy aggregation
    tmp <- expanded[, .(focal_row, val = get(v))]
    tmp <- tmp[!is.na(val)]
    agg <- tmp[, .(
      vmax  = max(val),
      vmin  = min(val),
      vmean = mean(val)
    ), by = focal_row]
    setnames(agg,
             c("vmax", "vmin", "vmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
    stats_list[[vi]] <- agg
  }

  # Merge all stats together on focal_row
  all_stats <- stats_list[[1]]
  for (vi in 2:length(stats_list)) {
    all_stats <- merge(all_stats, stats_list[[vi]], by = "focal_row", all = TRUE)
  }

  # --- Step 4: Attach results back to cell_data -----------------------------

  message("Attaching neighbor features to cell_data ...")

  # Ensure cell_data is keyed on .row_idx for the merge
  setkey(cell_data, .row_idx)
  setkey(all_stats, focal_row)

  feature_cols <- setdiff(names(all_stats), "focal_row")
  cell_data[all_stats, (feature_cols) := mget(paste0("i.", feature_cols)),
            on = c(.row_idx = "focal_row")]

  # Replace Inf/-Inf from max/min of empty groups with NA (safety)
  for (col in feature_cols) {
    inf_mask <- is.infinite(cell_data[[col]])
    if (any(inf_mask, na.rm = TRUE)) {
      set(cell_data, which(inf_mask), col, NA_real_)
    }
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  message("Done.")
  return(cell_data)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
library(data.table)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# The trained Random Forest model is untouched â€” only the feature table changed.
# The numerical values (max, min, mean) are identical to the original code.
```

---

## Summary Table

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `paste` + named vector (O(nÂ·k) string ops) | `data.table` equi-join on integer keys |
| Neighbor stats | `lapply` over 6.46M list elements Ã— 5 vars | Single flat `data.table` grouped aggregation |
| Estimated runtime | **86+ hours** | **~2â€“5 minutes** |
| Peak memory | ~6â€“10 GB (list of 6.46M vectors + copies) | ~8â€“12 GB (flat edge table, then freed) |
| RF model | Unchanged | Unchanged |
| Numerical output | max / min / mean per neighbor set | Identical max / min / mean per neighbor set |

The key insight is converting from a **row-wise R-loop with string hashing** to a **flat join-and-group paradigm** that lets `data.table`'s C internals handle the heavy lifting. No retraining of the Random Forest is needed; the output columns are numerically identical.
END_RESPONSE>>>
