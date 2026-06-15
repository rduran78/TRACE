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
run_id: Anthropic_F02_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** constructs a list of 6.46 million elements, each produced by an anonymous function inside `lapply` that performs character coercion, string pasting, and named-vector lookups for every single row. This is O(n Ã— k) string operations (where n â‰ˆ 6.46M and k â‰ˆ average neighbor count â‰ˆ 4), producing tens of millions of transient character allocations. Then **`compute_neighbor_stats`** iterates over that 6.46M-element list five separate times (once per variable), each time extracting, subsetting, and summarizing neighbor values in pure R loops.

Specific problems:

1. **String-key lookups are slow.** `paste()` + named-vector indexing is far slower than integer-key joins or hash-table lookups at this scale.
2. **`lapply` over 6.46M rows is inherently slow in R.** Each iteration has R-level overhead (function call, environment creation).
3. **The neighbor lookup is a ragged list.** Storing 6.46M list elements, each a small integer vector, has enormous memory overhead (~50â€“100 bytes per list element header alone â‰ˆ 300â€“600 MB just for the list skeleton).
4. **`do.call(rbind, result)` on a 6.46M-element list** copies data repeatedly and is very slow.
5. **No vectorization or data.table/C-level acceleration** is used anywhere.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| String-key lookups | Replace with integer arithmetic: encode `(id, year)` â†’ row index via a `data.table` keyed join, done once. |
| Ragged list of 6.46M elements | Flatten into two parallel integer vectors (`from`, `to`) representing an edge list â€” a single `data.table` of directed neighbor-year edges. This is compact (~11M rows Ã— 2 cols â‰ˆ 170 MB). |
| Per-row `lapply` for stats | Use `data.table` grouped aggregation (`j` expressions by group) over the edge list, which runs at C speed. |
| Five separate passes | Compute all five variables' neighbor stats in a single grouped aggregation pass. |
| `do.call(rbind, ...)` | Eliminated entirely; `data.table` returns a single matrix/data.table directly. |

**Expected speedup:** From 86+ hours to roughly 5â€“15 minutes. Memory peak should stay well under 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is untouched (no retraining).
- The numerical outputs (neighbor max, min, mean per variable) are identical to the original code.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature pipeline
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data         data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order          integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars   character vector of variable names to summarize
#' @return cell_data (data.table) with new columns appended: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                               "def", "usd_est_n2")) {

  # --- Step 0: Convert to data.table if needed; add a row index ---
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
  cell_data[, .row_idx := .I]

  # --- Step 1: Build a flat edge list of (focal_cell_id, neighbor_cell_id) ---
  #     from the nb object.  This is year-independent.
  message("Building spatial edge list ...")
  n_cells <- length(id_order)
  from_id <- rep(id_order, times = lengths(rook_neighbors_unique))
  to_id   <- id_order[unlist(rook_neighbors_unique)]
  edges   <- data.table(focal_id = from_id, neighbor_id = to_id)
  # Remove any self-loops or entries from 0-neighbor cells (spdep uses 0L sentinel)
  edges <- edges[neighbor_id != 0L]
  rm(from_id, to_id)

  message(sprintf("  %s directed spatial edges.", format(nrow(edges), big.mark = ",")))

  # --- Step 2: Build a keyed lookup from (id, year) â†’ row index ---
  message("Building (id, year) â†’ row index lookup ...")
  id_year_key <- cell_data[, .(id, year, .row_idx)]
  setkey(id_year_key, id, year)

  # --- Step 3: Expand edges across years to get (focal_row, neighbor_row) ---
  #     Strategy: join edges to id_year_key twice â€” once for focal, once for neighbor â€”
  #     but do it year-by-year to limit peak memory.
  years <- sort(unique(cell_data$year))

  message("Expanding edges across years ...")
  edge_rows_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Rows for this year
    yr_key <- id_year_key[year == yr]
    setkey(yr_key, id)

    # Join focal side
    tmp <- edges[yr_key, on = .(focal_id = id), nomatch = 0L,
                 .(focal_row = i..row_idx, neighbor_id)]
    # Join neighbor side
    tmp <- tmp[yr_key, on = .(neighbor_id = id), nomatch = 0L,
               .(focal_row, neighbor_row = i..row_idx)]
    edge_rows_list[[yi]] <- tmp
  }
  edge_rows <- rbindlist(edge_rows_list)
  rm(edge_rows_list, id_year_key, tmp, yr_key)
  gc()

  message(sprintf("  %s focal-neighbor-year edges.", format(nrow(edge_rows), big.mark = ",")))

  # --- Step 4: Compute neighbor stats for all variables in one grouped pass ---
  message("Computing neighbor statistics ...")

  # Attach neighbor values for all source vars at once
  # We pull columns from cell_data by integer index for speed.
  var_cols <- neighbor_source_vars
  neighbor_vals <- cell_data[edge_rows$neighbor_row, ..var_cols]
  neighbor_vals[, focal_row := edge_rows$focal_row]

  # Grouped aggregation: max, min, mean per focal_row per variable
  agg_exprs <- unlist(lapply(var_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_cols, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  # Build the j-expression programmatically
  j_expr <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  stats_dt <- neighbor_vals[, eval(j_expr), by = focal_row]
  rm(neighbor_vals, edge_rows)
  gc()

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col_name in agg_names) {
    v <- stats_dt[[col_name]]
    set(stats_dt, i = which(is.infinite(v) | is.nan(v)), j = col_name, value = NA_real_)
  }

  # --- Step 5: Join stats back to cell_data by row index ---
  message("Joining neighbor features back to cell_data ...")
  setkey(stats_dt, focal_row)

  for (col_name in agg_names) {
    # Pre-allocate NA column, then fill matched rows
    set(cell_data, j = col_name, value = NA_real_)
    matched <- stats_dt$focal_row
    set(cell_data, i = matched, j = col_name, value = stats_dt[[col_name]])
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  message("Done. Added ", length(agg_names), " neighbor feature columns.")
  return(cell_data)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
library(data.table)

# cell_data, id_order, rook_neighbors_unique already in memory
# trained_rf_model is untouched â€” no retraining

cell_data <- add_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Proceed with prediction using the existing trained model
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Original operation | Optimized equivalent |
|---|---|
| `paste(id, year)` string key â†’ named vector lookup | Integer `(id, year)` keyed `data.table` join â€” same matching semantics, no floating-point ambiguity |
| `max/min/mean` of `vals[idx]` per row | `data.table` grouped `max/min/mean` over the same `(focal_row, neighbor_row)` pairs |
| `NA` when no neighbors or all-NA values | `na.rm = TRUE` produces `Â±Inf` / `NaN` which are explicitly replaced with `NA` |
| Column naming `{var}_neighbor_max/min/mean` | Identical naming convention |

The numerical estimand (predicted GDP from the already-trained Random Forest) is preserved because the feature values fed to the model are arithmetically identical.

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string-paste `lapply` iterations | One `data.table` join per year (28 joins) |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— 6.46M `lapply` iterations | Single grouped aggregation |
| Estimated wall time | 86+ hours | **5â€“15 minutes** |
| Peak RAM | >16 GB (list overhead) | ~4â€“8 GB (flat integer edge table + grouped agg) |
END_RESPONSE>>>
