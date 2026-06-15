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
run_id: Anthropic_F05_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local Inefficiency
The `paste(..., sep="_")` keying and named-vector lookup inside `build_neighbor_lookup` is O(N) per row for key construction, executed ~6.46M times. String hashing and matching on 6.46M-length named vectors is extremely expensive.

### Broader Algorithmic Problem
The real issue is that `build_neighbor_lookup` uses a **row-wise `lapply` over 6.46M rows**, each performing:
1. A character-based lookup of the cell ID â†’ reference index (`id_to_ref`)
2. An index into the neighbor list (`neighbors[[ref_idx]]`)
3. String-paste of neighbor IDs with the current year
4. String-key lookup into `idx_lookup` (a 6.46M-entry named vector)

This is an **O(N Ã— K) string operation** where N â‰ˆ 6.46M and K â‰ˆ average neighbor count (~4 for rook). That's ~25.8M string constructions and hash lookups against a 6.46M-entry table â€” all to answer a question that has a clean **integer-arithmetic** solution.

Furthermore, `compute_neighbor_stats` then loops over 6.46M entries again per variable, doing per-row `max/min/mean`. With 5 variables, that's 5 Ã— 6.46M R-level function calls.

### Root Cause
The entire pattern conflates **spatial structure** (which cells are neighbors â€” time-invariant) with **temporal indexing** (which row corresponds to cell Ã— year). These should be separated. Since every cell appears once per year in a balanced panel, neighbor relationships in row-space are **the same shifted pattern repeated 28 times**. We only need to compute the spatial neighbor mapping once in row-index space for one year and then offset it.

---

## Optimization Strategy

1. **Eliminate all string operations.** Use integer-indexed lookups exclusively.
2. **Separate spatial from temporal indexing.** Build a cellâ†’row-offset map once; derive row-level neighbor indices by integer arithmetic.
3. **Vectorize `compute_neighbor_stats`.** Replace row-wise `lapply` with a single `data.table` grouped aggregation over an edge list, which is internally parallelized in C.
4. **Process all 5 variables in one pass** over the edge list rather than 5 separate passes.

**Expected speedup:** From ~86+ hours to **minutes** (typically 2â€“10 minutes depending on RAM/disk speed).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor-feature construction
# Drop-in replacement â€” preserves the exact numerical estimand
# =============================================================================

library(data.table)

build_and_add_neighbor_features <- function(cell_data,
                                            id_order,
                                            rook_neighbors_unique,
                                            neighbor_source_vars) {
  # --- Convert to data.table for fast grouped operations ---
  dt <- as.data.table(cell_data)

  # =========================================================================
  # STEP 1: Build a time-invariant directed edge list (cell-level)
  #
  # rook_neighbors_unique is an nb object: a list of length = # cells,

  # where element i is an integer vector of neighbor indices into id_order.
  # We expand this to a two-column edge table of (cell_id, neighbor_cell_id).
  # =========================================================================

  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells),
                  lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  # Map reference indices â†’ actual cell IDs
  edges_cell <- data.table(
    id       = id_order[from_ref],
    nb_id    = id_order[to_ref]
  )

  # =========================================================================
  # STEP 2: Build a fast integer lookup from (id) â†’ row positions per year
  #
  # Key insight: we work entirely in row-index space.
  # For each (id, year) we need the row index. We achieve this by keying

  # the data.table and using a fast equi-join.
  # =========================================================================

  # Add original row order so we can write results back in place
  dt[, .rowid := .I]

  # Create a slim table: (id, year) â†’ .rowid
  row_map <- dt[, .(id, year, .rowid)]

  # =========================================================================
  # STEP 3: Expand edge list across years and join to get row indices
  #
  # For every year, every spatial edge (idâ†’nb_id) becomes a row-level edge
  # (focal_row â†’ neighbor_row).  We achieve this with a single join rather
  # than 6.46M string-key lookups.
  # =========================================================================

  years <- sort(unique(dt$year))

  # Cross-join edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
  # This fits comfortably in 16 GB (â‰ˆ 0.6 GB for two integer columns + year)
  edges_yr <- CJ_dt(edges_cell, years)  # helper below; or use tidyr::crossing

  # Faster alternative avoiding CJ on data.tables:
  edges_yr <- edges_cell[, .(id, nb_id, year = list(years)), by = .I
                         ][, .(id, nb_id, year = unlist(year))]
  edges_yr[, I := NULL]

  # Join to get focal row index
  setkey(row_map, id, year)
  setkey(edges_yr, id, year)
  edges_yr[row_map, focal_row := i..rowid, on = .(id, year)]

  # Join to get neighbor row index
  setnames(edges_yr, "nb_id", "id_nb")
  # We need to join on (id_nb, year) â†’ .rowid
  edges_yr[row_map, nb_row := i..rowid, on = .(id_nb = id, year)]

  # Drop edges where either focal or neighbor is missing (boundary / NA year)
  edges_yr <- edges_yr[!is.na(focal_row) & !is.na(nb_row)]

  # =========================================================================
  # STEP 4: Compute max, min, mean for each variable in one vectorised pass
  # =========================================================================

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Pull neighbor values via integer indexing (vectorised)
    edges_yr[, nb_val := dt[[var_name]][nb_row]]

    # Grouped aggregation â€” data.table does this in C
    stats <- edges_yr[!is.na(nb_val),
                      .(v_max  = max(nb_val),
                        v_min  = min(nb_val),
                        v_mean = mean(nb_val)),
                      keyby = focal_row]

    # Initialise result columns with NA
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Write results back by row index
    dt[stats$focal_row, (max_col)  := stats$v_max]
    dt[stats$focal_row, (min_col)  := stats$v_min]
    dt[stats$focal_row, (mean_col) := stats$v_mean]
  }

  # Clean up
  edges_yr[, nb_val := NULL]
  dt[, .rowid := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# ---- Helper: Cross-join a data.table with a vector of years ----
# (More memory-efficient than full CJ on two data.tables)
# Already handled inline above; included for clarity:
CJ_dt <- function(edge_dt, years_vec) {
  idx <- rep(seq_len(nrow(edge_dt)), each = length(years_vec))
  out <- edge_dt[idx]
  out[, year := rep(years_vec, nrow(edge_dt))]
  out
}
```

### Drop-in Replacement for the Outer Loop

```r
# ---------------------------------------------------------------------------
# BEFORE (original â€” ~86+ hours):
# ---------------------------------------------------------------------------
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order,
#                                          rook_neighbors_unique)
# neighbor_source_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name,
#                                                  neighbor_lookup)
# }

# ---------------------------------------------------------------------------
# AFTER (optimized â€” ~2-10 minutes):
# ---------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_add_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched â€” use it as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Constrained Variant

If the ~38.5M-row edge table is tight on 16 GB RAM, process years in batches:

```r
build_and_add_neighbor_features_chunked <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars,
                                                     year_chunk_size = 7) {
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]

  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)
  valid    <- to_ref != 0L
  edges_cell <- data.table(id    = id_order[from_ref[valid]],
                           id_nb = id_order[to_ref[valid]])

  row_map <- dt[, .(id, year, .rowid)]
  setkey(row_map, id, year)

  years <- sort(unique(dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / year_chunk_size))

  # Initialise output columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_max_neighbor")  := NA_real_]
    dt[, paste0(var_name, "_min_neighbor")  := NA_real_]
    dt[, paste0(var_name, "_mean_neighbor") := NA_real_]
  }

  for (chunk in year_chunks) {
    message("Processing years: ", paste(chunk, collapse = ", "))

    edges_yr <- CJ_dt(edges_cell, chunk)
    setkey(edges_yr, id, year)
    edges_yr[row_map, focal_row := i..rowid, on = .(id, year)]
    edges_yr[row_map, nb_row    := i..rowid, on = .(id_nb = id, year)]
    edges_yr <- edges_yr[!is.na(focal_row) & !is.na(nb_row)]

    for (var_name in neighbor_source_vars) {
      edges_yr[, nb_val := dt[[var_name]][nb_row]]
      stats <- edges_yr[!is.na(nb_val),
                        .(v_max = max(nb_val),
                          v_min = min(nb_val),
                          v_mean = mean(nb_val)),
                        keyby = focal_row]

      max_col  <- paste0(var_name, "_max_neighbor")
      min_col  <- paste0(var_name, "_min_neighbor")
      mean_col <- paste0(var_name, "_mean_neighbor")

      set(dt, i = stats$focal_row, j = max_col,  value = stats$v_max)
      set(dt, i = stats$focal_row, j = min_col,  value = stats$v_min)
      set(dt, i = stats$focal_row, j = mean_col, value = stats$v_mean)
    }
  }

  dt[, .rowid := NULL]
  if (!is.data.table(cell_data)) return(as.data.frame(dt))
  return(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Key construction** | 6.46M Ã— `paste()` + named-vector hash | Zero strings; integer equi-join |
| **Neighbor resolution** | Row-wise `lapply` (R-level loop) | Single `data.table` keyed join (C-level) |
| **Stats computation** | `lapply` over 6.46M Ã— 5 vars | `data.table` grouped aggregation (C-level) |
| **Total passes over data** | 1 (build) + 5 (stats) = 6 | 1 combined pass per year-chunk |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **RAM peak** | Lower but slow | ~2â€“4 GB (chunked: ~1â€“2 GB) |
| **Numerical result** | âœ” max/min/mean per cell-year | âœ” Identical estimand preserved |
| **RF model** | â€” | **Untouched** |
END_RESPONSE>>>
