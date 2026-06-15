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
run_id: Anthropic_F02_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` conversion and named-vector lookup (`id_to_ref`).
- `paste()` to build neighbor keys for that year.
- Named-vector lookup into `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) in the worst case for R's internal hashing).

This is the dominant cost. Building 6.46M list elements, each requiring string construction and hash-table lookups, is extremely slow in pure R.

### 2. `compute_neighbor_stats` â€” repeated `lapply` over 6.46M list elements, called 5 times

Each call iterates over the full 6.46M-element list, subsetting a numeric vector and computing `max/min/mean`. The list-of-vectors structure defeats vectorization and is cache-unfriendly. Called 5 times (once per variable), this multiplies the cost.

### Memory pressure

A 6.46M-element list of integer vectors, plus a data.frame with ~110+ columns of 6.46M rows, can easily exceed 16 GB when intermediate copies are created by `do.call(rbind, ...)` and column assignment inside a loop.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key construction and lookup in `build_neighbor_lookup` | Replace with a **`data.table` equi-join** on integer columns `(id, year)`. No strings, no hashing of character keys. |
| 6.46M-element R list for neighbor lookup | Eliminate entirely. Represent the neighbor graph as a **flat `data.table`** of `(row_i, neighbor_row_j)` pairs â€” a sparse edge list. |
| Per-element `lapply` in `compute_neighbor_stats` | Replace with a **single grouped `data.table` aggregation** over the edge list: `edges[, .(max, min, mean), by = row_i]`. Fully vectorized in C. |
| 5 separate passes (one per variable) | Compute **all 5 variables' neighbor stats in one pass** by joining all needed columns at once. |
| `do.call(rbind, ...)` on 6.46M rows | Eliminated â€” `data.table` aggregation returns a data.table directly. |
| Column assignment copies | Use **`:=` (set-by-reference)** to add new columns without copying the entire data.frame. |

**Expected speedup:** From ~86+ hours to roughly **5â€“20 minutes** on the same laptop, with peak RAM well under 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is untouched â€” we only change feature construction.
- The numerical results (max, min, mean of neighbor values) are identical to the original code.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert the working data to data.table (by reference if possible)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure there is a row-index column we can join back on.
# (This is a zero-copy integer column addition.)
cell_data[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat edge list  (row_i  â†”  neighbor_row_j)
#     This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edges <- function(cell_dt, id_order, neighbors) {
  # --- a. Map each cell id to its position in id_order ----------------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- b. Expand the nb object into a flat (cell_id, neighbor_id) table
  #        This is only ~1.37M rows â€” trivially small.
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    id          = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )

  # --- c. Build a row-index lookup:  (id, year) â†’ .row_id ------------
  row_lookup <- cell_dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # --- d. For every (id, year) row, find its neighbor rows via join ---
  #   i.  Attach the focal row's year and row_id to each edge.
  #       Join edge_ids to row_lookup on 'id' â€” this replicates each
  #       edge across all 28 years (â‰ˆ 1.37M Ã— 28 â‰ˆ 38.5M rows).
  #       data.table does this as a fast indexed join.
  setkey(edge_ids, id)
  edges <- row_lookup[edge_ids,
                      .(row_i = .row_id, neighbor_id, year),
                      on = "id",
                      nomatch = 0L,
                      allow.cartesian = TRUE]

  #  ii.  Now resolve each (neighbor_id, year) to its .row_id.
  setnames(row_lookup, c("id", "year", ".row_id"),
                       c("neighbor_id", "year", "row_j"))
  setkey(row_lookup, neighbor_id, year)
  setkey(edges, neighbor_id, year)

  edges <- row_lookup[edges,
                      .(row_i, row_j),
                      on = c("neighbor_id", "year"),
                      nomatch = 0L]

  # Clean up the temporary rename so cell_dt is unaffected
  # (row_lookup was a copy of selected columns, so cell_dt is safe.)

  edges
}

cat("Building neighbor edge list â€¦\n")
edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
setkey(edges, row_i)
cat(sprintf("Edge list: %s rows\n", format(nrow(edges), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute all neighbor statistics in one vectorised pass
#     This replaces compute_neighbor_stats + the outer for-loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edges, vars) {
  # Attach the neighbor values to each edge row (only the columns we need).
  # edges$row_j indexes directly into cell_dt rows.
  neighbor_vals <- cell_dt[edges$row_j, ..vars]
  neighbor_vals[, row_i := edges$row_i]

  # Grouped aggregation â€” one pass over the ~38.5M edge rows.
  # For each (row_i, variable) compute max, min, mean  (na.rm = TRUE).
  agg_exprs <- lapply(vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  })
  agg_exprs <- unlist(agg_exprs, recursive = FALSE)

  # Build readable column names:  neighbor_max_ntl, neighbor_min_ntl, â€¦
  agg_names <- unlist(lapply(vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Evaluate the aggregation
  agg <- neighbor_vals[,
    lapply(agg_exprs, eval, envir = .SD),
    by = row_i,
    .SDcols = vars
  ]

  # --- Simpler, equivalent approach that avoids bquote complexity: ----
  # (Overwrite the above block if preferred.)
  agg <- neighbor_vals[,
    {
      out <- vector("list", length(vars) * 3L)
      k <- 0L
      for (v in vars) {
        x <- get(v)
        x <- x[!is.na(x)]
        if (length(x) == 0L) {
          out[[k + 1L]] <- NA_real_
          out[[k + 2L]] <- NA_real_
          out[[k + 3L]] <- NA_real_
        } else {
          out[[k + 1L]] <- max(x)
          out[[k + 2L]] <- min(x)
          out[[k + 3L]] <- mean(x)
        }
        k <- k + 3L
      }
      names(out) <- agg_names
      out
    },
    by = row_i
  ]

  agg
}

cat("Computing neighbor features â€¦\n")
agg <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Join the aggregated features back onto cell_data by reference
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(agg, row_i)

# For rows with no valid neighbors (not present in agg), values stay NA.
new_cols <- setdiff(names(agg), "row_i")

cell_data[agg, (new_cols) := mget(new_cols), on = .(.row_id = row_i)]

# Handle Inf/-Inf produced by max/min on empty sets (safety net)
for (col in new_cols) {
  set(cell_data, which(is.infinite(cell_data[[col]])), col, NA_real_)
}

# Drop the helper column
cell_data[, .row_id := NULL]

cat("Done â€” neighbor features added.\n")
```

---

## Why This Works Within 16 GB RAM

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 numeric cols) | ~5.7 GB |
| `edges` (â‰ˆ38.5M Ã— 2 integer cols) | ~0.6 GB |
| `neighbor_vals` (38.5M Ã— 6 cols) | ~1.8 GB |
| `agg` (6.46M Ã— 16 cols) | ~0.8 GB |
| **Total peak** | **~9 GB** |

This fits comfortably in 16 GB. If memory is still tight, you can process `vars` in batches of 2â€“3 instead of all 5 at once, at negligible speed cost.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `paste` + named-vector lookups | Integer equi-join via `data.table` |
| Neighbor stats | 5 Ã— `lapply` over 6.46M list elements | Single grouped `data.table` aggregation |
| Column binding | `do.call(rbind, â€¦)` + `$<-` copies | `:=` set-by-reference |
| Estimated time | 86+ hours | ~5â€“20 minutes |
| Peak RAM | >16 GB (fails/swaps) | ~9 GB |
| Numerical results | max / min / mean | Identical max / min / mean |
| Trained RF model | Untouched | Untouched |
END_RESPONSE>>>
