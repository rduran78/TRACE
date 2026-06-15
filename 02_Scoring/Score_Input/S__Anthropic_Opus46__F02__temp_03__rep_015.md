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
run_id: Anthropic_F02_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates ~6.46 million R list elements, each produced by an anonymous function inside `lapply` that performs per-row string pasting, hash-table lookups, and NA filtering. This is an O(n) loop over millions of rows with expensive string operations at every iteration. Then **`compute_neighbor_stats`** iterates over that 6.46-million-element list again for *each* of the 5 variables (another ~32 million R-level function calls total). The `do.call(rbind, ...)` on a 6.46-million-element list of 3-vectors is also a known performance anti-pattern.

**Specific problems:**

| Component | Issue |
|---|---|
| `paste(data$id, data$year, sep="_")` inside `lapply` | Called per-row; string allocation is slow in a hot loop |
| `id_to_ref[as.character(...)]` per row | Named-vector lookup is O(n) in the worst case for large vectors |
| `idx_lookup[neighbor_keys]` per row | Same named-vector hash problem at scale |
| `lapply` over 6.46M rows | R-level loop overhead; no vectorization |
| `do.call(rbind, list_of_6.46M)` | Extremely slow list-to-matrix coercion |
| Entire neighbor lookup stored as an R list | ~6.46M list elements Ã— ~4 neighbors each â‰ˆ large memory footprint with R overhead per element |
| `compute_neighbor_stats` re-traverses the full list per variable | 5 Ã— 6.46M = 32.3M R function calls |

**Memory estimate for the current approach:** Each list element carries R overhead (~56+ bytes for a SEXP header + integer vector). 6.46M elements Ã— ~100 bytes â‰ˆ 646 MB just for the lookup list, before any computation. Combined with the 6.46M Ã— 110 data frame (~5.7 GB for doubles), this pushes a 16 GB laptop to its limits.

---

## Optimization Strategy

**Core idea: Replace all per-row R loops with vectorized joins using `data.table`, and compute all neighbor statistics in a single grouped aggregation pass.**

1. **Vectorized neighbor-row mapping via `data.table` keyed join** â€” Instead of building a per-row list, create a long-form edge table `(row_i, neighbor_row_j)` using a single merge. No `paste`, no named-vector lookups, no `lapply`.

2. **Single grouped aggregation for all variables at once** â€” Join the neighbor rows to their variable values, then compute `max`, `min`, `mean` per `(row_i, variable)` in one `data.table` grouped operation. This replaces 5 separate `lapply`-over-6.46M passes.

3. **Memory-conscious chunking is unnecessary** if we use `data.table`'s in-place reference semantics (`:=`), which avoids copying the main data frame.

4. **Preserve the trained RF model** â€” We only change how features are computed, not their values. The same neighbor definitions and the same `max`/`min`/`mean` aggregations produce numerically identical columns.

**Expected speedup:** From 86+ hours to roughly 5â€“15 minutes, depending on disk I/O and available RAM.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert the main data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure there is a sequential row identifier we can trace back to
cell_data[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a vectorised edge table from the nb object
#     rook_neighbors_unique is a list of length = number of unique cells
#     id_order is the vector of cell IDs in the same order as the nb list
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[k]] gives integer indices into id_order for cell id_order[k]
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbour" sentinel (0L)
  valid    <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Build the long-form (row_i  â†’  neighbor_row_j) mapping
#     by joining edges Ã— years in one vectorised step.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_map <- function(cell_data, edge_dt) {
  # Minimal keyed tables for the two joins
  # Map (id, year) â†’ .row_id
  id_year_key <- cell_data[, .(id, year, .row_id)]
  setkey(id_year_key, id, year)

  # For every row, attach its outgoing neighbor cell IDs
  # Step A: get (id_from, year, .row_id) for every row
  from_dt <- id_year_key[, .(id_from = id, year, row_i = .row_id)]
  setkey(from_dt, id_from)

  # Step B: join with edge table to explode each row into its neighbors
  #   result: (row_i, year, id_to)
  setkey(edge_dt, id_from)
  expanded <- edge_dt[from_dt, on = "id_from", allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(row_i, year, id_to = x.id_to)]

  # Step C: map each (id_to, year) to its row index
  setkey(expanded, id_to, year)
  expanded[id_year_key, on = c("id_to" = "id", "year" = "year"),
           row_j := i..row_id]

  # Drop rows where the neighbor didn't exist in that year
  expanded <- expanded[!is.na(row_j)]

  expanded[, .(row_i, row_j)]
}

neighbor_map <- build_neighbor_map(cell_data, edge_dt)
cat("Neighbor-map rows:", nrow(neighbor_map), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute max / min / mean for ALL neighbor source variables
#     in a single grouped pass over the neighbor map.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_data, neighbor_map,
                                          source_vars) {
  # Extract only the columns we need from cell_data for the neighbor rows
  needed_cols <- c(".row_id", source_vars)
  vals_dt <- cell_data[, ..needed_cols]

  # Attach neighbor values to each (row_i, row_j) pair
  # This is a column-bind by reference via row_j index
  neighbor_vals <- vals_dt[neighbor_map$row_j, ..source_vars]
  neighbor_vals[, row_i := neighbor_map$row_i]

  # Grouped aggregation: for each row_i compute stats across its neighbors
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- neighbor_vals[, lapply(agg_exprs, eval, envir = .SD), by = row_i]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# â”€â”€ Simpler, more robust aggregation (avoids bquote complexity) â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_data, neighbor_map,
                                          source_vars) {
  vals_dt <- cell_data[neighbor_map$row_j, ..source_vars]
  vals_dt[, row_i := neighbor_map$row_i]

  # Melt to long form: (row_i, variable, value)
  long <- melt(vals_dt, id.vars = "row_i",
               measure.vars = source_vars,
               variable.name = "var", value.name = "val")

  # Grouped aggregation â€“ one pass
  agg <- long[, .(nb_max  = max(val,  na.rm = TRUE),
                   nb_min  = min(val,  na.rm = TRUE),
                   nb_mean = mean(val, na.rm = TRUE)),
              by = .(row_i, var)]

  # Inf â†’ NA
  agg[is.infinite(nb_max), nb_max := NA_real_]
  agg[is.infinite(nb_min), nb_min := NA_real_]

  # Cast back to wide: one column per (stat, variable)
  wide <- dcast(agg, row_i ~ var,
                value.var = c("nb_max", "nb_min", "nb_mean"))

  return(wide)
}

stats_wide <- compute_all_neighbor_features(cell_data, neighbor_map,
                                            neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Merge the new columns back into cell_data by row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(stats_wide, row_i)

new_cols <- setdiff(names(stats_wide), "row_i")
cell_data[stats_wide, (new_cols) := mget(paste0("i.", new_cols)),
          on = c(".row_id" = "row_i")]

# Rows with no neighbors (e.g., islands) will already be NA from the
# non-match in the join â€” identical to the original code's behaviour.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  (Optional) Rename columns to match the original pipeline's naming
#     convention if compute_and_add_neighbor_features used a different
#     pattern.  Adjust the gsub as needed.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example: if original names were "neighbor_max_ntl", the dcast output
# produces "nb_max_ntl".  Rename:
old_names <- grep("^nb_", names(cell_data), value = TRUE)
new_names <- sub("^nb_", "neighbor_", old_names)
setnames(cell_data, old_names, new_names)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  Clean up temporary columns and free memory
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_id := NULL]
rm(neighbor_map, stats_wide, edge_dt)
gc()

cat("Done. Neighbor features added for:",
    paste(neighbor_source_vars, collapse = ", "), "\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same three aggregation functions (`max`, `min`, `mean`) are applied to exactly the same set of neighbor values identified by the same rook-neighbor definitions and the same `(id, year)` pairing logic. Output is bit-identical (modulo floating-point associativity of `mean`, which is negligible). |
| **Trained RF model** | No model retraining occurs. The code only reconstructs the *input feature columns* that the existing model expects. Column names are mapped to match the original naming convention (Step 5). |
| **Rows with no neighbors** | Produce `NA` for all three statistics, exactly as the original `if (length(idx) == 0) return(c(NA, NA, NA))` did. |

---

## Performance Expectations on a 16 GB Laptop

| Step | Estimated Time | Peak RAM |
|---|---|---|
| `build_edge_table` | < 1 second | ~22 MB |
| `build_neighbor_map` (keyed join) | 1â€“3 minutes | ~1.5 GB (the expanded table) |
| `compute_all_neighbor_features` (melt + grouped agg) | 3â€“8 minutes | ~3â€“4 GB transient |
| Column merge (`:=` join) | < 30 seconds | negligible (in-place) |
| **Total** | **~5â€“15 minutes** | **~8â€“10 GB peak** |

This is well within the 16 GB constraint and represents a roughly **350â€“1000Ã— speedup** over the original 86+ hour estimate, achieved entirely through vectorization and elimination of per-row R-level loops.
END_RESPONSE>>>
