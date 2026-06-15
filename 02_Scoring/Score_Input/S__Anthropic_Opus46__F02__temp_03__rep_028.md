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
run_id: Anthropic_F02_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation inside the loop creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, once per variable (Ã—5 variables), each time subsetting and computing summary statistics in pure R.

**Specific problems:**

1. **String-key lookups at scale:** `paste(id, year)` and `idx_lookup[neighbor_keys]` are called ~6.46M times inside `lapply`. String allocation and hashing dominate runtime.
2. **Redundant per-row work:** Neighbor cell IDs are the same across all years for a given cell. But the lookup is rebuilt per cell-year row, not factored into (cell â†’ neighbors) Ã— (year â†’ row index).
3. **`lapply` over 6.46M rows:** Returns a list of 6.46M integer vectors â€” massive memory allocation and GC pressure.
4. **`do.call(rbind, ...)` on 6.46M-element list:** This is notoriously slow; it creates a huge number of intermediate objects.
5. **Pure R loops for stats:** `max`, `min`, `mean` called individually per row in an interpreted loop.

**Estimated cost:** ~86+ hours is consistent with O(N Ã— avg_neighbors) string operations in interpreted R over ~6.46M rows.

---

## Optimization Strategy

The key insight: **factor the problem into a spatial dimension and a temporal dimension.**

- Each cell has a fixed set of rook neighbors (independent of year).
- For a given year, the neighbor rows are simply the neighbor cells' rows in that same year.

**Strategy:**

1. **Use `data.table` for fast indexed joins** instead of named-vector string lookups.
2. **Build an edge list once** (cell_id â†’ neighbor_cell_id from the `nb` object), then join on `(neighbor_cell_id, year)` to get neighbor row indices or values directly â€” a vectorized merge, not a per-row loop.
3. **Compute grouped statistics vectorized** using `data.table` grouping: `[, .(max, min, mean), by = .(id, year)]` over the joined edge table.
4. **Process all 5 variables in a single join pass** rather than looping over variables with separate lookups.
5. **Avoid materializing a 6.46M-element list** entirely.

This converts the problem from ~6.46M interpreted R iterations to a handful of vectorized `data.table` joins and grouped aggregations, which should run in **minutes, not days**, and stay well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert the spdep nb object into a two-column edge list (integer)
#    id_order maps positional index â†’ cell id.
#    rook_neighbors_unique[[i]] gives positional indices of neighbors of
#    the cell at position i.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate based on total number of directed neighbor links
  n_links <- sum(lengths(neighbors))          # ~1,373,394
  from_id <- integer(n_links)
  to_id   <- integer(n_links)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1L] == 0L)) next
    len <- length(nb_i)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_i]
    pos <- pos + len
  }
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute all neighbor features in one vectorized pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Convert to data.table (by reference if already, else copy) ----
  dt <- as.data.table(cell_data)

  # --- Build edge list -----------------------------------------------
  edges <- build_edge_list(id_order, rook_neighbors_unique)

  # --- Create a slim table of just the columns we need for the join --
  # Columns: id, year, and each source variable
  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..keep_cols]

  # --- Join: for every (from_id, year) get neighbor rows -------------
  # Merge edges with dt_slim on (to_id == id) to get neighbor values
  # Result: one row per (from_id, year, neighbor), with neighbor values
  setnames(dt_slim, "id", "to_id")          # rename for join
  setkeyv(dt_slim, c("to_id", "year"))
  setkeyv(edges, "to_id")                   # not strictly needed but helps

  # Expand edges Ã— years: each edge applies to every year.
  # Instead of a cross-join (which would be huge), we join edges onto
  # the data directly.
  # For each row in dt we know from_id = dt$id, year = dt$year.
  # We want: for each (from_id, year), find all to_id in edges, then
  # look up (to_id, year) in dt_slim.

  # Step A: join edges to get (from_id, to_id) pairs
  # Step B: join on (to_id, year) to get variable values

  # Efficient approach: join dt (as the "from" side) with edges,
  # then join result with dt_slim on (to_id, year).

  # Create from-side keyed on from_id
  dt_from <- dt[, .(from_id = id, year)]     # ~6.46M rows
  setkeyv(dt_from, "from_id")
  setkeyv(edges, "from_id")

  # Join: each (from_id, year) row gets expanded by its neighbors
  # This produces ~6.46M Ã— avg_neighbors â‰ˆ 6.46M Ã— (1373394/344208) â‰ˆ ~25.8M rows
  # (avg ~4 rook neighbors per cell)
  # 25.8M rows Ã— few columns is very manageable in RAM.
  expanded <- edges[dt_from, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: from_id, to_id, year

  # Now join to get neighbor variable values
  setkeyv(expanded, c("to_id", "year"))
  expanded <- dt_slim[expanded, on = c("to_id", "year"), nomatch = NA]
  # expanded now has: to_id, year, <var columns>, from_id

  # --- Compute grouped stats ----------------------------------------
  # Group by (from_id, year), compute max/min/mean for each variable
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Suppress -Inf/Inf warnings from max/min on all-NA groups
  stats <- suppressWarnings(
    expanded[, lapply(agg_exprs, eval, envir = .SD),
             by = .(from_id, year),
             .SDcols = neighbor_source_vars]
  )

  # Replace Inf/-Inf (from all-NA groups) with NA
  for (col_name in agg_names) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # --- Merge back onto original data ---------------------------------
  setnames(stats, "from_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- stats[dt, on = c("id", "year")]

  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }

  return(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Usage â€” drop-in replacement for the original outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used downstream as before â€”
# no retraining needed. The new columns are numerically identical
# to the originals (max, min, mean of the same neighbor values).
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named vector lookups in `lapply` | One vectorized `data.table` keyed join |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows, pure R `max/min/mean` | Single grouped `data.table` aggregation |
| **Result assembly** | `do.call(rbind, <6.46M-element list>)` | Already a `data.table`; merge by key |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** (depending on disk I/O) |
| **Peak RAM** | High (millions of small list elements + GC pressure) | ~3â€“5 GB (one ~26M-row intermediate table) |
| **Numerical output** | max, min, mean of neighbor values | **Identical** â€” same arithmetic, same values |
| **Trained RF model** | â€” | **Preserved unchanged** (no retraining) |

The optimization eliminates all per-row interpreted loops and string operations, replacing them with vectorized `data.table` joins and grouped aggregations. The numerical estimand (neighbor max, min, mean for each variable) is preserved exactly, and the trained Random Forest model requires no modification.
END_RESPONSE>>>
