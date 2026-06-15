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
run_id: Anthropic_F12_T00_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (only 5 calls total) and each call binds fixed-length-3 vectors â€” this is O(n) and completes in seconds.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookup at scale**: `build_neighbor_lookup` creates `idx_lookup`, a named vector of ~6.46 million elements keyed by `paste(id, year, sep="_")`. Then, for *each* of the 6.46 million rows, it:
   - Calls `as.character()` on a single id.
   - Looks up `id_to_ref[as.character(...)]` â€” a named-vector character lookup.
   - Extracts neighbor cell IDs from the `nb` object.
   - Calls `paste()` to create neighbor keys for that year.
   - Performs *multiple* named-character lookups into the 6.46M-element `idx_lookup` vector.

2. **Named character vector lookup is O(n) per probe in R** (R's named vectors use linear hashing with poor scaling). With ~6.46M rows, each doing ~4 neighbor lookups into a 6.46M-length named vector, this is catastrophically slow â€” on the order of **billions of character-match operations**.

3. **The `lapply` over 6.46M rows** with per-element R function calls adds massive interpreter overhead.

4. By contrast, `compute_neighbor_stats` does simple numeric indexing (`vals[idx]`) which is fast, and `do.call(rbind, result)` on length-3 vectors is a minor cost.

**Conclusion**: The bottleneck is the row-by-row character-key lookups in `build_neighbor_lookup()`, not the `rbind` or list operations in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Replace named-vector character lookups with `data.table` hash joins or `match()` on integer keys.** Avoid `paste()`-based string keys entirely by using a two-column integer key (id, year).

2. **Vectorize `build_neighbor_lookup`** â€” expand the neighbor relationships into a flat edge table, join to get row indices, and group by source row. This replaces 6.46M R-level function calls with a single vectorized join.

3. **Vectorize `compute_neighbor_stats`** â€” use `data.table` grouped aggregation on the edge table instead of `lapply` over 6.46M elements.

4. **Preserve the trained Random Forest model** â€” we only change feature-engineering code, producing numerically identical columns.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED build_neighbor_lookup + compute_neighbor_stats
# Replaces both functions with a single vectorized pipeline.
# Produces numerically identical results to the original code.
# ==============================================================

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Assign a row index to every cell-year row ---
  dt[, .row_idx := .I]

  # --- Step 2: Build an integer mapping from cell id -> ref index ---
  # id_order is the vector such that id_order[ref_idx] == cell_id
  id_to_ref <- data.table(
    cell_id = as.integer(id_order),
    ref_idx = seq_along(id_order)
  )

  # --- Step 3: Expand rook_neighbors_unique into a flat edge table ---
  # Each element of the nb list: rook_neighbors_unique[[ref_idx]] gives

  # the ref indices of neighbors of cell id_order[ref_idx].
  # We build: (source_ref_idx, neighbor_ref_idx)

  n_cells <- length(id_order)
  lens <- lengths(rook_neighbors_unique)
  edge_dt <- data.table(
    src_ref  = rep(seq_len(n_cells), times = lens),
    nbr_ref  = unlist(rook_neighbors_unique, use.names = FALSE)
  )

  # Convert ref indices to actual cell IDs
  edge_dt[, src_id := id_order[src_ref]]
  edge_dt[, nbr_id := id_order[nbr_ref]]
  edge_dt[, c("src_ref", "nbr_ref") := NULL]

  # --- Step 4: Build a row-index lookup keyed by (cell_id, year) ---
  row_lookup <- dt[, .(cell_id = id, year, .row_idx)]
  setkey(row_lookup, cell_id, year)

  # --- Step 5: For each source row, find all neighbor rows in the same year ---
  # First, get (src_row_idx, nbr_id, year) by joining source side
  src_rows <- dt[, .(src_row_idx = .row_idx, src_id = id, year)]

  # Join: for each source row, get its neighbor cell IDs
  setkey(src_rows, src_id)
  setkey(edge_dt, src_id)

  # This is a many-to-many join: each source cell has multiple neighbors,

  # and each source cell appears in multiple years.
  # Use allow.cartesian = TRUE
  edges_with_year <- edge_dt[src_rows,
    .(src_row_idx, nbr_id, year),
    on = "src_id",
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Now join to get neighbor row indices
  setkey(edges_with_year, nbr_id, year)
  edges_with_year[row_lookup,
    nbr_row_idx := i..row_idx,
    on = c("nbr_id" = "cell_id", "year")
  ]

  # Drop edges where neighbor row doesn't exist
  edges_final <- edges_with_year[!is.na(nbr_row_idx), .(src_row_idx, nbr_row_idx)]

  # Free memory
  rm(edges_with_year, src_rows, row_lookup, edge_dt, id_to_ref)
  gc()

  # --- Step 6: Compute neighbor stats for each variable ---
  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Get the values vector
    all_vals <- dt[[var_name]]

    # Attach neighbor values to edge table
    work <- copy(edges_final)
    work[, nbr_val := all_vals[nbr_row_idx]]

    # Remove NA neighbor values
    work <- work[!is.na(nbr_val)]

    # Grouped aggregation â€” the core computation, fully vectorized
    stats <- work[, .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ), by = src_row_idx]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values by row index
    dt[stats$src_row_idx, (max_col)  := stats$nb_max]
    dt[stats$src_row_idx, (min_col)  := stats$nb_min]
    dt[stats$src_row_idx, (mean_col) := stats$nb_mean]

    rm(work, stats)
    gc()
  }

  dt[, .row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ==============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimized_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original pipeline.
# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with `paste()` + named-vector char lookup | Single vectorized `data.table` join on integer keys |
| **Neighbor stats** | 6.46M `lapply` iterations + `do.call(rbind, ...)` | `data.table` grouped `max/min/mean` â€” single pass |
| **Estimated time** | 86+ hours | **Minutes** (typically 5â€“15 min on 16 GB laptop) |
| **Peak memory** | Moderate (but slow) | ~2â€“4 GB for edge table (fits in 16 GB) |
| **Numerical output** | Baseline | **Identical** â€” same max, min, mean per row |
| **RF model** | Preserved | **Preserved â€” untouched** |

The true bottleneck was the O(nÂ²)-behaving character-key lookups in `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`. The fix is to replace all character-keyed named-vector lookups with vectorized `data.table` integer-keyed joins.
END_RESPONSE>>>
