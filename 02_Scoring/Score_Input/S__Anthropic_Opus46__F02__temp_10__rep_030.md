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
run_id: Anthropic_F02_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear and two-fold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

- `paste(â€¦, sep="_")` is called **millions of times** inside the loop.
- Named-vector indexing (`idx_lookup[neighbor_keys]`) on a vector of ~6.46M names forces repeated hash-table lookups on very large character vectors.
- The result is a **list of 6.46 million integer vectors**, which is itself a large memory object (~hundreds of MB to several GB, depending on average neighbor count).
- Estimated time: **tens of hours** just for this step.

### 2. `compute_neighbor_stats` â€” Called 5 times, each iterating over the 6.46M-element list

- Each call to `lapply` loops 6.46M times in interpreted R, performing subsetting, NA removal, and three aggregations.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is notoriously slow (builds a huge temporary list of row vectors then binds them).

### Memory pressure

- The 6.46M-element neighbor lookup list, the 6.46M Ã— 110 data frame, and intermediate copies can easily exceed 16 GB, causing swapping and further slowdown.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste` and named-vector lookup in `build_neighbor_lookup` | Replace with a **`data.table` merge/join** approach. Encode (id, year) â†’ row index as a keyed `data.table`, then expand the neighbor list into a flat edge table and join in bulk. No per-row `paste`. |
| 6.46M-element R list for neighbor lookup | Replace with a **flat edge table** (`data.table` with columns `row_i`, `neighbor_row`). This is cache-friendly, column-oriented, and avoids R list overhead. |
| Interpreted `lapply` in `compute_neighbor_stats` | Replace with **`data.table` grouped aggregation**: join the flat edge table to the variable column, then `[, .(max, min, mean), by = row_i]`. This is vectorized C code under the hood. |
| `do.call(rbind, â€¦)` on millions of rows | Eliminated â€” `data.table` returns the result as a single table directly. |
| Repeated work across 5 variables | Process all 5 variables in **one pass** over the edge table (a single grouped aggregation over all 5 columns simultaneously). |
| General | Use `data.table` throughout to avoid copies and leverage in-place `:=` assignment. |

**Expected speed-up:** From 86+ hours to roughly **5â€“30 minutes** depending on disk I/O and exact machine specs. Memory peak should stay well under 16 GB.

---

## Working R Code

```r
# ==============================================================================
# Optimized neighbor-feature pipeline
# Preserves the trained Random Forest model and original numerical outputs.
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# 0. Convert cell_data to data.table (in-place, no copy)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure there is a sequential row identifier we can use as a join key.
# This column will be removed at the end if it did not already exist.
had_row_idx <- "..row_idx.." %in% names(cell_data)
cell_data[, `..row_idx..` := .I]

# --------------------------------------------------------------------------
# 1. Build a flat edge table (replaces build_neighbor_lookup)
#
#    Goal: for every row i in cell_data, find the rows that correspond to
#    cell i's rook neighbors in the SAME year.
# --------------------------------------------------------------------------
build_flat_edge_table <- function(dt, id_order, neighbors) {
  # Map: cell id  ->  position in id_order (reference index)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Expand the nb object into a flat (from_id, to_id) edge list ----------
  # Each element of `neighbors` is an integer vector of indices into id_order.
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  # --- Map (id, year) -> row index ------------------------------------------
  id_year_map <- dt[, .(id, year, `..row_idx..`)]
  setkey(id_year_map, id, year)

  # --- Cross-join edges with years via two keyed joins -----------------------
  # First, get (from_id, year, row_i) for every row that owns a "from" cell
  setnames(id_year_map, c("id", "year", "row_i"))
  edge_from <- merge(edge_ids, id_year_map, by.x = "from_id", by.y = "id",
                     allow.cartesian = TRUE)
  # edge_from columns: from_id, to_id, year, row_i

  # Now look up the neighbor's row in the same year
  setnames(id_year_map, c("id", "year", "neighbor_row"))
  edge_full <- merge(edge_from, id_year_map,
                     by.x = c("to_id", "year"),
                     by.y = c("id", "year"))
  # edge_full columns: to_id, year, from_id, row_i, neighbor_row

  # Keep only what we need
  edge_full <- edge_full[, .(row_i, neighbor_row)]
  setkey(edge_full, row_i)

  return(edge_full)
}

message("Building flat edge table â€¦")
edge_table <- build_flat_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge table: %s rows", formatC(nrow(edge_table), big.mark = ",")))

# --------------------------------------------------------------------------
# 2. Compute neighbor stats for ALL source variables in one grouped pass
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(dt, edge_tbl, var_names) {
  # Subset only the columns we need from dt (avoids copying everything)
  cols_needed <- c("..row_idx..", var_names)
  vals <- dt[, ..cols_needed]
  setnames(vals, "..row_idx..", "neighbor_row")
  setkey(vals, neighbor_row)

  # Join: attach variable values to the neighbor side of every edge
  joined <- merge(edge_tbl, vals, by = "neighbor_row")
  # joined columns: neighbor_row, row_i, <var_names...>

  # Grouped aggregation â€” compute max / min / mean per (row_i, variable)
  # Build the aggregation expression dynamically so it works for any var list.
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  names(agg_exprs) <- agg_names
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  message("  Computing grouped aggregations â€¦")
  stats <- joined[, eval(agg_call), by = row_i]

  # Replace -Inf / Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

message("Computing neighbor features â€¦")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# --------------------------------------------------------------------------
# 3. Merge the new features back into cell_data
# --------------------------------------------------------------------------
# Drop any old neighbor columns if re-running
old_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_cols)) {
  cell_data[, (old_cols) := NULL]
}

# Keyed join (fast, in-place)
setkey(cell_data, `..row_idx..`)
setkey(neighbor_stats, row_i)
cell_data <- merge(cell_data, neighbor_stats, by.x = "..row_idx..", by.y = "row_i", all.x = TRUE)

# Clean up helper column
if (!had_row_idx) cell_data[, `..row_idx..` := NULL]

# Free large temporaries
rm(edge_table, neighbor_stats)
gc()

message("Done. Neighbor features added to cell_data.")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same three aggregations (`max`, `min`, `mean`) are computed over exactly the same neighbor sets (same rook-neighbor list, same year matching, same NA handling). The output columns are identical in meaning and value (up to floating-point identity). |
| **Trained Random Forest model** | No model object is touched. The code only modifies the feature table (`cell_data`) that will be passed to `predict()`. Column names follow the same `neighbor_<var>_<stat>` convention. |
| **Edge cases** | Cells with no neighbors in a given year get `NA` for all stats (via `all.x = TRUE` in the final merge), matching the original `c(NA, NA, NA)` return. |

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `paste` + named-vector lookups (interpreted R) | Bulk `data.table` merge â€” vectorized C |
| Stats computation | 5 Ã— 6.46M `lapply` iterations | **Single** grouped `data.table` aggregation |
| Result binding | `do.call(rbind, <6.46M list>)` | Direct `data.table` output |
| Estimated runtime | 86+ hours | ~5â€“30 minutes |
| Peak RAM | Exceeds 16 GB (swapping) | ~4â€“10 GB (well within 16 GB) |
END_RESPONSE>>>
