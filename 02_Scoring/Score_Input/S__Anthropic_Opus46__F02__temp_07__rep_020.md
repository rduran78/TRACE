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
run_id: Anthropic_F02_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` + named-vector lookup to resolve the cell's reference index.
- A `paste()` call for every neighbor to build composite keys.
- A named-vector lookup (`idx_lookup[neighbor_keys]`) that R implements as repeated hash-table probes on a 6.46M-entry character vector.

Named-vector lookups in R are **O(k)** per probe (where k is key length) and are not vectorized internally. Doing this 6.46M Ã— ~4 neighbors â‰ˆ 25.8 billion character operations is catastrophically slow. The resulting `neighbor_lookup` list also stores ~25.8M integers in a 6.46M-element list, consuming significant memory.

### 2. `compute_neighbor_stats` â€” repeated per-variable `lapply` over the same 6.46M-element list

Each call to `compute_neighbor_stats` walks all 6.46M list elements again, extracting variable values. With 5 variables this is 32.3M R-level function calls, each allocating small vectors.

### Summary of cost drivers

| Component | Dominant cost | Estimated time share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + hash lookups | ~70â€“80% |
| `compute_neighbor_stats` (Ã—5 vars) | 6.46M `lapply` iterations Ã—5 | ~20â€“30% |

---

## Optimization Strategy

### Principle: Replace R-level row iteration and string hashing with vectorized `data.table` equi-joins and grouped aggregations.

| Original approach | Optimized approach |
|---|---|
| Build a 6.46M-element list of neighbor row indices via `paste` + named-vector lookup | Build an **edge table** (integer join on `id` + `year`) using `data.table` merge â€” fully vectorized, no string keys |
| `lapply` over list to compute `max/min/mean` per row per variable | Single `data.table` grouped aggregation `[, .(max, min, mean), by = row_idx]` per variable â€” columnar, cache-friendly, parallelizable |
| Memory: 6.46M-element list of integer vectors | Memory: one ~25.8M-row edge `data.table` of two integer columns (~200 MB) |

**Expected speedup:** From 86+ hours to roughly **10â€“30 minutes** on the same 16 GB laptop, depending on disk I/O. Memory peak stays well under 16 GB.

### Key design decisions

1. **No string composite keys.** We join on two integer columns (`neighbor_id`, `year`), which `data.table` handles via radix-based binary search.
2. **Edge table is built once**, then reused for all 5 variables â€” amortizing the join cost.
3. **The trained Random Forest model is never touched.** We only reproduce the same 15 derived columns (`{var}_{max|min|mean}`) with identical numerical values.
4. **`data.table` is used in-place** to avoid copying the 6.46M Ã— 110 column data frame.

---

## Working R Code

```r
# ============================================================
# Optimized neighbor-feature pipeline
# Requirements: install.packages("data.table") if not present
# ============================================================
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ----------------------------------------------------------
  # 0.  Convert to data.table (by reference if already one)
  # ----------------------------------------------------------
  if (!is.data.table(cell_data)) {
    setDT(cell_data)   # converts in place â€” no copy
  }

  # Preserve original row order so downstream predictions align
  cell_data[, .row_idx := .I]

  # ----------------------------------------------------------
  # 1.  Build a directed edge table  (cell_id -> neighbor_id)
  #     from the spdep nb object â€” pure integer, no strings.
  #     This replaces the per-row paste/hash in

  #     build_neighbor_lookup().
  # ----------------------------------------------------------
  # rook_neighbors_unique is a list of integer index vectors

  # referencing positions in id_order.
  edge_list <- rbindlist(
    lapply(seq_along(rook_neighbors_unique), function(ref) {
      nb <- rook_neighbors_unique[[ref]]
      # spdep convention: a single 0 means no neighbors
      if (length(nb) == 1L && nb == 0L) return(NULL)
      data.table(cell_id     = id_order[ref],
                 neighbor_id = id_order[nb])
    }),
    use.names = FALSE
  )
  # edge_list has ~1.37M rows (directed pairs), all integer.

  cat(sprintf("Edge table: %s directed pairs\n",
              formatC(nrow(edge_list), big.mark = ",")))

  # ----------------------------------------------------------
  # 2.  Expand edges across years via an equi-join.
  #     For every (cell_id, year) row in cell_data we need the
  #     row indices of its neighbors in the SAME year.
  #
  #     Instead of materializing the full ~25.8M-row expanded
  #     edge table up front, we join edge_list onto cell_data
  #     twice:
  #       a) join to get the focal row's year  (keyed on cell_id)
  #       b) join to get the neighbor row index (keyed on
  #          neighbor_id + year)
  #     data.table does both with binary search â€” no hashing.
  # ----------------------------------------------------------

  # Minimal lookup: row_idx, id, year  (avoids copying all 110 cols)
  row_ref <- cell_data[, .(.row_idx, id, year)]

  # 2a. For every edge, attach every year the focal cell appears in.
  #     Result: (cell_id, neighbor_id, year, focal_row_idx)
  setkey(row_ref, id)
  focal_edges <- edge_list[row_ref,
                           .(neighbor_id,
                             year      = i.year,
                             focal_row = i..row_idx),
                           on       = .(cell_id = id),
                           nomatch  = NULL,
                           allow.cartesian = TRUE]
  rm(edge_list)  # free memory

  # 2b. Attach the neighbor's row index for the same year.
  setkey(row_ref, id, year)
  focal_edges[row_ref,
              neighbor_row := i..row_idx,
              on = .(neighbor_id = id, year = year),
              nomatch = NA]

  # Drop edges where the neighbor has no data for that year
  focal_edges <- focal_edges[!is.na(neighbor_row)]

  cat(sprintf("Expanded edge table: %s cell-year-neighbor rows\n",
              formatC(nrow(focal_edges), big.mark = ",")))

  rm(row_ref)
  gc()

  # ----------------------------------------------------------
  # 3.  Compute max / min / mean for each variable.
  #     One grouped aggregation per variable â€” fully vectorized.
  #     This replaces compute_neighbor_stats().
  # ----------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Pull the variable values for neighbor rows
    focal_edges[, nval := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation â€” data.table radix groups on integer key
    stats <- focal_edges[!is.na(nval),
                         .(v_max  = max(nval),
                           v_min  = min(nval),
                           v_mean = mean(nval)),
                         keyby = .(focal_row)]

    # Allocate result columns (default NA for cells with no neighbors)
    col_max  <- paste0(var_name, "_max")
    col_min  <- paste0(var_name, "_min")
    col_mean <- paste0(var_name, "_mean")

    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    # Write results into the correct rows (vectorized assignment)
    set(cell_data, i = stats$focal_row, j = col_max,  value = stats$v_max)
    set(cell_data, i = stats$focal_row, j = col_min,  value = stats$v_min)
    set(cell_data, i = stats$focal_row, j = col_mean, value = stats$v_mean)

    # Clean up the temporary column
    focal_edges[, nval := NULL]
  }

  # ----------------------------------------------------------
  # 4.  Clean up helper columns and return
  # ----------------------------------------------------------
  cell_data[, .row_idx := NULL]
  rm(focal_edges)
  gc()

  return(cell_data)
}


# ============================================================
# Usage â€” drop-in replacement for the original outer loop
# ============================================================

# cell_data             : your existing data.frame / data.table
# id_order              : integer vector of cell IDs matching nb object
# rook_neighbors_unique : spdep nb list (loaded from disk)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns:
#   ntl_max, ntl_min, ntl_mean,
#   ec_max,  ec_min,  ec_mean,
#   pop_density_max, pop_density_min, pop_density_mean,
#   def_max, def_min, def_mean,
#   usd_est_n2_max, usd_est_n2_min, usd_est_n2_mean
#
# Feed directly to predict(trained_rf_model, cell_data) as before.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values for each `(cell_id, year)` pair. The join logic replicates the original `paste(id, year)` key matching but via integer equi-join. |
| **Trained RF model unchanged** | No model code is touched. The 15 derived columns have identical names and identical values, so `predict()` produces the same output. |
| **Column names** | `{var}_max`, `{var}_min`, `{var}_mean` â€” matching the pattern produced by `compute_and_add_neighbor_features()`. Adjust the suffix pattern if your wrapper uses a different naming convention. |

---

## Memory Budget Estimate (16 GB laptop)

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 numeric cols) | ~5.4 GB |
| `focal_edges` expanded (~25.8M Ã— 4 int cols) | ~0.8 GB |
| Temporary `stats` per variable | ~0.15 GB |
| R overhead + OS | ~2â€“3 GB |
| **Total peak** | **~8â€“9 GB** âœ” |

This fits comfortably within 16 GB. If memory is still tight (e.g., other objects in the session), process variables in chunks or call `gc()` more aggressively between variables â€” the code already does this.
END_RESPONSE>>>
