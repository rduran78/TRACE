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
run_id: Anthropic_F04_T03_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M string operations is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which uses an `lapply` over 6.46M elements returning 3-element vectors, then `do.call(rbind, ...)` on a 6.46M-element list â€” a notoriously slow pattern.

**Root causes, ranked:**
1. **Per-row string key construction and lookup** in `build_neighbor_lookup` (~6.46M iterations, each with `paste` and character-keyed lookup).
2. **`do.call(rbind, list_of_vectors)`** in `compute_neighbor_stats` on millions of rows.
3. **Five serial passes** over the full dataset in the outer loop (one per variable), each re-traversing the 6.46M-element lookup.

The Random Forest inference itself is comparatively fast (a single `predict` call on a pre-trained model using optimized C code).

---

## Optimization Strategy

### Key ideas:

1. **Replace string-keyed lookups with integer-indexed lookups.** Pre-sort data by `(id, year)` so that all 28 years for a given cell are contiguous and the row for cell `c` in year `y` can be found by arithmetic: `offset[c] + (y - 1992)`. This eliminates all `paste` and named-vector lookups.

2. **Build the neighbor lookup as a flat integer matrix** (CSR-like structure) once, using vectorized operations, not per-row `lapply`.

3. **Vectorize `compute_neighbor_stats`** using `data.table` grouping or, even better, direct C-level vectorized indexing. We can build a long-form edge table `(row_i, neighbor_row_j)` and use `data.table` to compute grouped `max/min/mean` in one pass for all variables simultaneously.

4. **Process all 5 variables in a single pass** over the edge table rather than 5 separate passes.

**Expected speedup:** From ~86+ hours to **~2â€“5 minutes** on the same laptop.

---

## Optimized Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars,
                                       year_range = 1992:2019) {
  # -----------------------------------------------------------
  # 0. Convert to data.table and record original row order
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]

  # -----------------------------------------------------------
  # 1. Build integer-indexed mapping: cell id -> contiguous block

  #    Ensure data is sorted by (id, year) for arithmetic indexing.
  # -----------------------------------------------------------
  setkey(dt, id, year)
  dt[, .sorted_row := .I]

  n_years  <- length(year_range)
  year_min <- min(year_range)

  # Map each unique cell id to its first row in the sorted table.
  # Because data is keyed by (id, year) and every cell has all 28 years,
  # cell id_order[k]'s row for year y is: first_row[k] + (y - year_min).
  cell_first_row <- dt[, .(.first = min(.sorted_row)), by = id]
  setkey(cell_first_row, id)

  # Fast integer lookup: id -> first_row
  id_to_first <- cell_first_row$.first
  names(id_to_first) <- as.character(cell_first_row$id)

  # -----------------------------------------------------------
  # 2. Build flat edge table (source_row, neighbor_row) â€” vectorized
  #    For each cell i and each neighbor j of i, and for each year y,
  #    source_row = first_row[i] + (y - year_min)
  #    neighbor_row = first_row[j] + (y - year_min)
  # -----------------------------------------------------------
  # Build cell-level edge list from nb object
  n_cells <- length(id_order)
  from_cell_idx <- rep(seq_len(n_cells),
                       times = lengths(rook_neighbors_unique))
  to_cell_idx   <- unlist(rook_neighbors_unique)

  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_cell_idx > 0L
  from_cell_idx <- from_cell_idx[valid]
  to_cell_idx   <- to_cell_idx[valid]

  n_edges_cell <- length(from_cell_idx)

  # Get first-row offsets for from and to cells
  from_first <- id_to_first[as.character(id_order[from_cell_idx])]
  to_first   <- id_to_first[as.character(id_order[to_cell_idx])]

  # Expand across all years: each cell-level edge becomes 28 row-level edges
  year_offsets <- 0L:(n_years - 1L)

  # Use outer-sum via rep + rep(each=...)
  from_rows <- rep(from_first, times = n_years) +
               rep(year_offsets, each = n_edges_cell)
  to_rows   <- rep(to_first, times = n_years) +
               rep(year_offsets, each = n_edges_cell)

  # Edge table: each row says "for sorted row `from_row`,
  # one of its spatial neighbors is sorted row `to_row`"
  edges <- data.table(from_row = from_rows, to_row = to_rows)

  # Free large temporaries
  rm(from_rows, to_rows, from_first, to_first,
     from_cell_idx, to_cell_idx)
  gc()

  # -----------------------------------------------------------
  # 3. Attach neighbor values for ALL source vars at once
  # -----------------------------------------------------------
  # Pull the variable columns from dt by sorted-row index
  for (v in neighbor_source_vars) {
    set(edges, j = v, value = dt[[v]][edges$to_row])
  }

  # -----------------------------------------------------------
  # 4. Compute grouped max / min / mean in one pass per variable
  # -----------------------------------------------------------
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_clean <- v  # column name
    agg_exprs[[paste0("n_max_", v)]]  <- call("max",  as.name(v_clean), na.rm = TRUE)
    agg_exprs[[paste0("n_min_", v)]]  <- call("min",  as.name(v_clean), na.rm = TRUE)
    agg_exprs[[paste0("n_mean_", v)]] <- call("mean", as.name(v_clean), na.rm = TRUE)
  }

  # Remove NA neighbor values before aggregation to match original logic
  # (original code filters NAs then computes; data.table na.rm=TRUE is equivalent)
  stats <- edges[, lapply(agg_exprs, eval, envir = .SD),
                 by = from_row]

  # Replace -Inf/Inf from max/min of zero-length groups with NA
  inf_cols <- grep("^n_max_|^n_min_", names(stats), value = TRUE)
  for (col in inf_cols) {
    vals <- stats[[col]]
    set(stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
  }

  # -----------------------------------------------------------
  # 5. Handle rows with NO neighbors (islands) â€” fill with NA
  # -----------------------------------------------------------
  all_sorted_rows <- seq_len(nrow(dt))
  missing_rows    <- setdiff(all_sorted_rows, stats$from_row)

  if (length(missing_rows) > 0L) {
    na_fill <- data.table(from_row = missing_rows)
    for (cn in setdiff(names(stats), "from_row")) {
      set(na_fill, j = cn, value = NA_real_)
    }
    stats <- rbindlist(list(stats, na_fill), use.names = TRUE)
  }

  setkey(stats, from_row)

  # -----------------------------------------------------------
  # 6. Join back to dt and restore original row order
  # -----------------------------------------------------------
  new_cols <- setdiff(names(stats), "from_row")
  dt[, (new_cols) := stats[.(.sorted_row), ..new_cols, on = "from_row"]]

  # Restore original row order
  setorder(dt, .orig_row)
  dt[, c(".orig_row", ".sorted_row") := NULL]

  # -----------------------------------------------------------
  # 7. Return as data.frame to preserve downstream compatibility
  # -----------------------------------------------------------
  as.data.frame(dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to: predictions <- predict(rf_model, cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| Lookup method | `paste` string keys, named-vector lookup per row | Integer arithmetic on sorted `(id, year)` blocks |
| Neighbor expansion | `lapply` over 6.46M rows | Vectorized `rep` + integer addition (one allocation) |
| Stat computation | `lapply` + `do.call(rbind, ...)` Ã— 5 vars | Single `data.table` grouped aggregation, all 5 vars at once |
| Passes over data | 5 (one per variable) | 1 |
| Estimated runtime | ~86+ hours | ~2â€“5 minutes |
| Peak RAM | Moderate but slow | ~3â€“4 GB for edge table (fits in 16 GB) |
| Numerical result | â€” | Identical (`max`, `min`, `mean` with NA removal preserved) |
| RF model | â€” | Untouched; no retraining |

The core insight is that the original code performed **~6.46 million interpreted R iterations**, each involving string allocation and hash-table probing. The optimized version replaces this with **bulk integer-vector operations** (vectorized `rep`, `+`, and `data.table` grouped aggregation), which execute in compiled C code and are orders of magnitude faster.
END_RESPONSE>>>
