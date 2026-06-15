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
run_id: Anthropic_F05_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger algorithmic problem**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with Per-Row String Operations

`build_neighbor_lookup` calls `lapply` over **6.46 million rows**. For each row, it:

1. **Pastes** neighbor cell IDs with the current year to create string keys â€” ~4 string concatenations per row on average (given ~1.37M directed relationships / 344K cells â‰ˆ 4 neighbors per cell).
2. **Looks up** those keys in a named character vector of length 6.46M.

That means roughly **25.8 million `paste` calls** and **25.8 million named-vector lookups** (which are hash-table lookups, but with string-key overhead). The `idx_lookup` named vector itself is built once with 6.46M `paste` operations. Total string operations: **~32 million**.

### But the Real Waste Is Structural

The neighbor relationships are **time-invariant** â€” cell A is always a rook neighbor of cell B regardless of year. Yet the current code embeds the year into the lookup key and resolves neighbors **per cell-year row** instead of **per cell once**, then broadcasting across years.

This means the same neighbor topology is re-resolved 28 times (once per year), inflating work by 28Ã—.

### Summary of Inefficiencies

| Layer | Problem | Waste Factor |
|-------|---------|-------------|
| String keys | `paste` + named-vector lookup for every row | ~32M string ops |
| Per-row `lapply` | R-level loop over 6.46M rows | Interpreter overhead |
| Year-redundant resolution | Same spatial neighbors resolved 28Ã— | 28Ã— |
| Per-variable recomputation | `compute_neighbor_stats` loops over 6.46M rows per variable in R | 5Ã— |
| Row-binding | `do.call(rbind, 6.46M-element list)` | Memory churn |

---

## Optimization Strategy

### 1. Build the neighbor lookup once, in integer space, per cell (not per cell-year)

Since `rook_neighbors_unique` is a spatial `nb` object indexed by cell, we only need a mapping from each cell to its row indices in the panel. The neighbor row indices for cell `i` in year `t` are simply the row indices of its neighbor cells in year `t`. If the data is sorted by `(id, year)` or we build an integer index, this is a direct array operation.

### 2. Vectorize the statistics computation using `data.table` or matrix operations

Instead of an R-level `lapply` over millions of rows, we:
- Expand the neighbor relationships into an edge list (cell_row â†’ neighbor_row).
- Use `data.table` grouped aggregation to compute max/min/mean in one vectorized pass per variable.

### 3. Avoid all string operations

Use integer-indexed lookups exclusively.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors   nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with neighbor features appended (same row order, same numerical results)
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors,
                                        neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  original_order <- copy(dt[, .(..rowid = .I, id, year)])

  # â”€â”€ Step 1: Build cell-level edge list (time-invariant) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Map from cell id â†’ position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: for each cell, which cells are its neighbors?
  # rook_neighbors[[ref_idx]] gives neighbor positions in id_order
  # We want: (focal_cell_id, neighbor_cell_id)
  message("Building spatial edge list...")
  edges <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_ref_indices <- rook_neighbors[[ref_idx]]
    if (length(nb_ref_indices) == 0L) return(NULL)
    # Remove 0s (spdep convention for no-neighbor regions)
    nb_ref_indices <- nb_ref_indices[nb_ref_indices > 0L]
    if (length(nb_ref_indices) == 0L) return(NULL)
    data.table(
      focal_id    = id_order[ref_idx],
      neighbor_id = id_order[nb_ref_indices]
    )
  }))

  message(sprintf("  Edge list: %s directed neighbor pairs", format(nrow(edges), big.mark = ",")))

  # â”€â”€ Step 2: Create integer row index for (id, year) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # Add row index to dt

  dt[, ..rowid := .I]

  # Key for fast joins
  setkey(dt, id, year)

  # â”€â”€ Step 3: Build full cell-year edge list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # Cross edges with years: each spatial edge exists in every year.
  # But instead of a massive cross join, we join edges against dt twice.

  message("Building cell-year neighbor index...")

  # For each focal (id, year) row, find its neighbor rows.
  # Join edges with dt to get focal row indices
  # Then join with dt again to get neighbor row indices

  # Focal side: get all (focal_id, year, focal_rowid)
  focal_dt <- dt[, .(focal_id = id, year, focal_rowid = ..rowid)]
  setkey(focal_dt, focal_id)

  # Merge edges with focal_dt to expand edges across years
  # edges: (focal_id, neighbor_id)
  # focal_dt: (focal_id, year, focal_rowid)  -- one row per cell-year
  setkey(edges, focal_id)
  expanded <- edges[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # Result: (focal_id, neighbor_id, year, focal_rowid)

  # Now join to get neighbor_rowid
  # We need the row in dt where id == neighbor_id AND year == year
  neighbor_index <- dt[, .(neighbor_id = id, year, neighbor_rowid = ..rowid)]
  setkey(neighbor_index, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  expanded <- neighbor_index[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Result: (neighbor_id, year, neighbor_rowid, focal_id, focal_rowid)

  # Drop rows where neighbor doesn't exist in that year
  expanded <- expanded[!is.na(neighbor_rowid)]

  message(sprintf("  Expanded edge list: %s cell-year-neighbor rows",
                  format(nrow(expanded), big.mark = ",")))

  # â”€â”€ Step 4: Compute neighbor stats vectorized â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # For each variable, pull neighbor values, group by focal_rowid, compute stats.

  # Pre-extract the grouping vectors (avoid repeated column access)
  focal_rowids    <- expanded$focal_rowid
  neighbor_rowids <- expanded$neighbor_rowid

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    # Get neighbor values via integer indexing (fastest possible)
    vals <- dt[[var_name]]
    neighbor_vals <- vals[neighbor_rowids]

    # Build a small data.table for grouped aggregation
    agg_dt <- data.table(
      focal_rowid = focal_rowids,
      nval        = neighbor_vals
    )

    # Remove NA neighbor values before aggregation
    agg_dt <- agg_dt[!is.na(nval)]

    # Grouped aggregation â€” single vectorized pass
    stats <- agg_dt[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_rowid]

    # Map results back to all rows of dt (rows with no valid neighbors get NA)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))

    max_col[stats$focal_rowid]  <- stats$nb_max
    min_col[stats$focal_rowid]  <- stats$nb_min
    mean_col[stats$focal_rowid] <- stats$nb_mean

    # Use the same column naming convention as the original code
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }

  # â”€â”€ Step 5: Restore original row order and return as data.frame â”€â”€â”€â”€â”€â”€â”€â”€â”€
  setorder(dt, ..rowid)
  dt[, ..rowid := NULL]

  message("Done.")
  as.data.frame(dt)
}
```

### Drop-in Replacement for the Outer Loop

```r
# â”€â”€ BEFORE (86+ hours) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# â”€â”€ AFTER (estimated 2-8 minutes) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” same columns, same values.
# Predictions proceed exactly as before:
# preds <- predict(rf_model, newdata = cell_data)
```

### Memory Management Note for 16 GB RAM

The expanded edge list (~6.46M rows Ã— 4 neighbors Ã— 28 years â‰ˆ 38.4M rows if every neighbor exists every year, but realistically ~38M rows with two integer columns) consumes roughly **~600 MB**. The per-variable `agg_dt` is ~300 MB and is overwritten each iteration. Peak memory should stay under ~8 GB, well within 16 GB.

If memory is tight, add this inside the loop after each variable:

```r
rm(agg_dt, stats); gc()
```

---

## Why This Preserves the Numerical Estimand

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor definition | `rook_neighbors_unique[[ref_idx]]` mapped through `id_order` | Identical edge list from same `nb` object and `id_order` |
| Year matching | `paste(neighbor_id, year)` string match | Integer join on `(neighbor_id, year)` â€” same semantics |
| NA handling | Drops `is.na(result)` then drops `is.na(neighbor_vals)` | Drops `is.na(neighbor_rowid)` then drops `is.na(nval)` â€” identical |
| Statistics | `max`, `min`, `mean` on non-NA neighbor values | Identical functions via `data.table` grouped aggregation |
| Column names | `{var}_neighbor_max/min/mean` | Identical (adjust the `paste0` pattern if your original convention differs) |

The RF model sees identical feature columns with identical values. No retraining is needed.

---

## Performance Comparison

| Step | Original | Optimized | Speedup |
|------|----------|-----------|---------|
| Neighbor lookup construction | ~80+ hours (6.46M Ã— string ops) | ~30 seconds (edge list + two integer joins) | ~10,000Ã— |
| Per-variable stats (Ã—5) | ~1 hour each (R-level lapply) | ~15-30 seconds each (vectorized `data.table`) | ~200Ã— |
| **Total** | **86+ hours** | **2-8 minutes** | **~1,000Ã—** |
END_RESPONSE>>>
