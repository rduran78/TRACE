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
run_id: Anthropic_F02_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` and a named-vector lookup (`id_to_ref`) â€” O(1) amortised but with overhead.
- `paste()` to build neighbor keys â€” allocates a small character vector per row.
- Named-vector subsetting on `idx_lookup` (a 6.46M-element named character vector) â€” each lookup walks a hash, repeated for every neighbor of every row.

This creates **~6.46M Ã— avg_neighbors â‰ˆ 25â€“30 million** individual hash lookups inside a single-threaded R `lapply`, plus millions of small allocations. Estimated: **tens of hours** just for the lookup build.

### 2. `compute_neighbor_stats` â€” another O(n) `lapply` per variable

Each call iterates over 6.46M rows, subsetting a numeric vector by the index list, computing `max/min/mean`. This is called 5 times (once per source variable), yielding ~32M R-level function calls. The `do.call(rbind, ...)` at the end also copies a 6.46M Ã— 3 matrix from a list of 6.46M length-3 vectors â€” very slow.

### Memory
With 6.46M rows Ã— 110+ columns of doubles (8 bytes each), the data alone is ~5.7 GB. The neighbor lookup list adds another ~1â€“2 GB. Intermediate copies push peak usage well beyond 16 GB.

---

## Optimization Strategy

| Bottleneck | Fix | Speedup |
|---|---|---|
| Per-row `paste`/hash lookup in `build_neighbor_lookup` | Replace with a **merge/join via `data.table`**: explode the `nb` object into an edge-list, join on `(neighbor_id, year)` to get row indices, then split by source row. | ~100â€“500Ã— |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with **vectorised `data.table` grouped aggregation** on the edge-list: join neighbor values, then `[, .(max, min, mean), by = source_row]`. No R-level loop at all. | ~50â€“200Ã— |
| `do.call(rbind, list_of_vectors)` | Eliminated entirely â€” `data.table` returns a single matrix-like result. | large |
| Memory: copies of `cell_data` in loop | Assign columns **in-place** via `:=` (reference semantics of `data.table`). | saves GBs |
| 5 separate passes over edge-list | Compute all 5 variables' neighbor stats in **one pass** (wide join, then grouped summarise). | 5Ã— fewer scans |

**Estimated wall-clock time: 5â€“15 minutes** on a 16 GB laptop (down from 86+ hours).

The trained Random Forest model is untouched â€” we are only rewriting the feature-engineering step that feeds it. The numerical results (max, min, mean of rook-neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert cell_data to data.table (in-place conversion; no copy)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure a deterministic row-order column so we can map results back
cell_data[, .row_id := .I]

# ---------------------------------------------------------------
# 1.  Build a data.table edge-list from the nb object (one-time)
#
#     rook_neighbors_unique is an nb object of length
#     length(id_order) == 344,208.
#     Each element is an integer vector of neighbor positions in
#     id_order.
# ---------------------------------------------------------------
build_edge_dt <- function(id_order, nb_obj) {
  # Pre-allocate vectors the size of the total number of directed edges
  n_edges <- sum(lengths(nb_obj))            # ~1.37 M
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 0L
  for (i in seq_along(nb_obj)) {
    nb_i <- nb_obj[[i]]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    idx <- pos + seq_len(n_i)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  data.table(from_id = from_id, to_id = to_id)
}

cat("Building edge list ...\n")
edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---------------------------------------------------------------
# 2.  Cross-join edges with years to get (from_id, to_id, year),
#     then join to cell_data to pick up neighbour-row values.
#
#     This replaces build_neighbor_lookup + compute_neighbor_stats
#     entirely with vectorised data.table operations.
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {

  years <- sort(unique(cell_data$year))

  # Edge list Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
  # This is the "long" representation of every (source_row, neighbor_row) pair.
  cat("Expanding edge Ã— year table ...\n")
  ey <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  ey[, `:=`(from_id = edge_dt$from_id[edge_idx],
            to_id   = edge_dt$to_id[edge_idx])]
  ey[, edge_idx := NULL]

  # Key cell_data for fast join  (id, year) -> row values
  # We only need id, year, .row_id, and the source_vars columns.
  keep_cols <- c("id", "year", ".row_id", source_vars)
  cd_small <- cell_data[, ..keep_cols]
  setkey(cd_small, id, year)

  # Join to get the SOURCE row id (.row_id of the "from" cell-year)
  cat("Joining source rows ...\n")
  setnames(cd_small, "id", "from_id")
  # We only need .row_id from the source side
  ey <- cd_small[, .(from_id, year, .row_id)][ey, on = .(from_id, year), nomatch = 0L]
  setnames(ey, ".row_id", "src_row")
  setnames(cd_small, "from_id", "id")   # restore

  # Join to get the NEIGHBOR values
  cat("Joining neighbor values ...\n")
  setnames(cd_small, "id", "to_id")
  ey <- cd_small[, c("to_id", "year", source_vars), with = FALSE
                 ][ey, on = .(to_id, year), nomatch = 0L]
  setnames(cd_small, "to_id", "id")     # restore

  # Now ey has columns: src_row, and each of source_vars with the
  # neighbor's value.  Aggregate by src_row.
  cat("Computing grouped stats ...\n")

  # Build aggregation expression dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      sprintf("nb_%s_max  = fcase(all(is.na(%s)), NA_real_, default = max(%s, na.rm = TRUE))", v, v, v),
      sprintf("nb_%s_min  = fcase(all(is.na(%s)), NA_real_, default = min(%s, na.rm = TRUE))", v, v, v),
      sprintf("nb_%s_mean = fcase(all(is.na(%s)), NA_real_, default = mean(%s, na.rm = TRUE))", v, v, v)
    )
  }))
  agg_call <- paste0(".(", paste(agg_exprs, collapse = ", "), ")")

  stats <- ey[, eval(parse(text = agg_call)), by = src_row]

  # ---------------------------------------------------------------
  # 3.  Map aggregated stats back to cell_data rows.
  #     Rows with no neighbours (islands / boundary) get NA.
  # ---------------------------------------------------------------
  cat("Merging results back ...\n")
  setkey(stats, src_row)

  new_cols <- setdiff(names(stats), "src_row")
  # Pre-fill with NA
  for (col in new_cols) {
    set(cell_data, j = col, value = NA_real_)
  }
  # Assign computed values by row index
  cell_data[stats$src_row, (new_cols) := stats[, ..new_cols]]

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up the helper column
cell_data[, .row_id := NULL]

cat("Done â€” neighbor features added.\n")
```

---

### Why the numerical estimand is preserved

The aggregation expressions compute exactly `max`, `min`, and `mean` of the non-`NA` neighbor values for each cell-year â€” the same three statistics the original `compute_neighbor_stats` computes. No approximation, sampling, or rounding is introduced. Column names follow the pattern `nb_{var}_{stat}` (adjust to match whatever the existing `compute_and_add_neighbor_features` wrapper produced so the trained Random Forest's `predict()` call sees the expected feature names).

### Memory profile

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 115 doubles) | ~5.9 GB |
| `edge Ã— year` table (38.5M Ã— 7 cols) | ~2.1 GB |
| `stats` result (6.46M Ã— 16 cols) | ~0.8 GB |
| **Peak** | ~12â€“13 GB âœ” (fits in 16 GB) |

If memory is still tight, the `ey` join can be chunked by year (28 chunks of ~1.37M rows each) and the partial aggregations combined afterward, reducing peak usage by ~1.5 GB at the cost of a trivial amount of extra time.
END_RESPONSE>>>
