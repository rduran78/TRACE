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
run_id: Anthropic_F02_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still slow when called millions of times inside an interpreted loop. With ~6.46M rows this single function likely takes many hours.

### 2. `compute_neighbor_stats` â€” O(n) `lapply` over ~6.46M rows, called 5 times
Each call iterates over every row, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. The `do.call(rbind, result)` at the end materialises a list of 6.46M 3-element vectors into a matrix â€” this alone is very slow because `rbind` on a long list is quadratic in memory allocation.

### Compounding factors
- Everything is in base-R interpreted loops (`lapply` over millions of elements).
- String key construction (`paste(id, year)`) and named-vector lookup are repeated per row.
- Memory: 6.46M Ã— 110 columns is manageable (~5â€“8 GB as doubles), but intermediate list objects and string vectors created inside the loops cause heavy GC pressure on a 16 GB machine.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key construction & named-vector lookup in `build_neighbor_lookup` | Replace with a **vectorised join** using `data.table`. Build a `(cell_id, year) â†’ row_index` table once, then expand all neighbor pairs and join in one shot. No per-row loop. |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with a **grouped `data.table` aggregation** (`max`, `min`, `mean` by source row), fully vectorised in C. |
| `do.call(rbind, list_of_vectors)` | Eliminated entirely â€” `data.table` returns a single table. |
| 5 separate passes over the lookup | Compute all 5 variables' neighbor stats in a **single grouped aggregation** (or one pass per variable but each pass is now milliseconds, not hours). |
| Memory pressure from intermediate string vectors | Eliminated â€” joins are on integer keys. |

**Expected speedup:** from 86+ hours to roughly **2â€“10 minutes** on the same laptop, depending on disk I/O. Peak RAM usage ~6â€“10 GB (fits in 16 GB).

**Preservation guarantees:**
- The trained Random Forest model is untouched (we only change feature engineering).
- The output columns are numerically identical (`max`, `min`, `mean` of the same neighbor values with the same NA handling).

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optimised neighbor-feature pipeline using data.table
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)

build_neighbor_features_dt <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {
  # --- 0. Convert to data.table (by reference if already one) --------
  dt <- as.data.table(cell_data)

  # Ensure a deterministic row identifier so we can join results back
  dt[, .row_idx := .I]

  # --- 1. Build an edge list from the nb object ----------------------
  #
  # rook_neighbors_unique is a list of length = length(id_order).
  # Element k contains integer indices (into id_order) of the
  # neighbors of id_order[k].
  #
  # We expand this into a two-column data.table:
  #   (source_cell_id, neighbor_cell_id)

  edge_list <- rbindlist(
    lapply(seq_along(rook_neighbors_unique), function(k) {
      nb <- rook_neighbors_unique[[k]]
      # spdep uses 0L for "no neighbours"
      nb <- nb[nb != 0L]
      if (length(nb) == 0L) return(NULL)
      data.table(source_id = id_order[k],
                 neighbor_id = id_order[nb])
    })
  )
  # edge_list has ~1.37M rows â€” small and fast.

  # --- 2. Build a (cell_id, year) â†’ row_idx lookup -------------------
  #     This replaces the slow named-vector idx_lookup.

  lookup <- dt[, .(cell_id = id, year, .row_idx)]
  setkey(lookup, cell_id, year)

  # --- 3. Expand edges Ã— years in one vectorised join ----------------
  #
  # For every (source_id, year) we need the row indices of its
  # neighbors in that same year.
  #
  # Step A: get the unique years
  years <- sort(unique(dt$year))

  # Step B: cross-join edge_list Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows)
  #         This is the full set of (source_id, year, neighbor_id) triples.
  edges_by_year <- CJ_dt(edge_list, years)

  # Helper: memory-efficient cross join
  # (defined below if not yet available)

  # Step C: attach the SOURCE row index
  setnames(edges_by_year, c("source_id", "neighbor_id", "year"))
  edges_by_year[lookup,
                source_row := i..row_idx,
                on = .(source_id = cell_id, year)]

  # Step D: attach the NEIGHBOR row index
  edges_by_year[lookup,
                neighbor_row := i..row_idx,
                on = .(neighbor_id = cell_id, year)]

  # Drop edges where either side is missing (cell not observed that year)
  edges_by_year <- edges_by_year[!is.na(source_row) & !is.na(neighbor_row)]

  # --- 4. Pull neighbor values and aggregate -------------------------
  #
  # For each source row and each variable, compute max / min / mean
  # of the neighbor values.

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Attach the neighbor's value for this variable
    edges_by_year[, nbr_val := dt[[var_name]][neighbor_row]]

    # Grouped aggregation â€” fully vectorised in C
    agg <- edges_by_year[!is.na(nbr_val),
                         .(nbr_max  = max(nbr_val),
                           nbr_min  = min(nbr_val),
                           nbr_mean = mean(nbr_val)),
                         by = source_row]

    # Column names matching the original pipeline's convention
    col_max  <- paste0("nbr_max_",  var_name)
    col_min  <- paste0("nbr_min_",  var_name)
    col_mean <- paste0("nbr_mean_", var_name)

    # Write results back into dt by row index
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]

    dt[agg$source_row, (col_max)  := agg$nbr_max]
    dt[agg$source_row, (col_min)  := agg$nbr_min]
    dt[agg$source_row, (col_mean) := agg$nbr_mean]
  }

  # --- 5. Clean up and return ----------------------------------------
  dt[, .row_idx := NULL]

  # Return as data.frame if the caller expects one
  return(as.data.frame(dt))
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helper: memory-efficient cross join of edge_list Ã— years vector
# Avoids CJ() because edge_list is already a table.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CJ_dt <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times,
  # and tile the years vector nrow(edge_dt) times.
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx <- rep(seq_len(n_edges), each = n_years)
  out <- edge_dt[idx]
  out[, year := rep(years_vec, times = n_edges)]
  return(out)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# USAGE (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_dt(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as-is for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Numerically Identical

| Original logic | Optimised equivalent |
|---|---|
| For each row `i`, find neighbors of `data$id[i]` in the same `data$year[i]` | The cross join `edge_list Ã— years` + two keyed joins produce exactly the same `(source_row, neighbor_row)` pairs. |
| `max(neighbor_vals)`, `min(neighbor_vals)`, `mean(neighbor_vals)` after removing NAs | `edges_by_year[!is.na(nbr_val), .(max, min, mean), by = source_row]` â€” identical aggregation, identical NA handling. |
| Rows with no valid neighbors get `NA, NA, NA` | Columns are initialised to `NA_real_`; only rows present in `agg` are overwritten. |

---

## Resource Estimates on a 16 GB Laptop

| Object | Approximate size |
|---|---|
| `dt` (6.46M Ã— 115 cols) | ~6 GB |
| `edges_by_year` (~38.5M Ã— 5 int/num cols) | ~1.5 GB |
| `agg` (â‰¤ 6.46M Ã— 4 cols) | ~0.2 GB |
| **Peak total** | **~8â€“9 GB** (fits in 16 GB) |
| **Expected wall-clock time** | **2â€“10 minutes** |

If memory is still tight, the `for` loop over the 5 variables can be split into separate passes where `edges_by_year` is rebuilt each time (trading ~5Ã— more time for lower peak RAM), but this should not be necessary.
END_RESPONSE>>>
