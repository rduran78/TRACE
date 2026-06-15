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
run_id: Anthropic_F04_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`) â€” fine individually, but done 6.46M times.
2. Subsets the `neighbors` list to get neighbor cell IDs.
3. Constructs character paste keys (`paste(id, year, sep="_")`) for every neighbor of every row.
4. Looks those keys up in `idx_lookup` (a named character vector of length 6.46M).

Named-vector lookup in R is **O(n)** per query on long vectors (linear scan of names), so ~6.46M lookups into a 6.46M-length named vector is effectively **O(nÂ²)** â€” roughly 4Ã—10Â¹Â³ character comparisons. This alone explains the 86+ hour estimate. `compute_neighbor_stats` is a secondary bottleneck (6.46M `lapply` iterations with subsetting), but far less severe.

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** â€” O(1) amortized per lookup.
2. **Vectorize `build_neighbor_lookup`** â€” expand all neighbor relationships into a single edge table, join once, then split.
3. **Vectorize `compute_neighbor_stats`** â€” use `data.table` grouped aggregation on the edge table instead of per-row `lapply`.
4. **Avoid materializing the full neighbor_lookup list entirely** â€” go straight from edge table to aggregated statistics.

This reduces runtime from ~86 hours to **minutes**.

## Optimized Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # Convert to data.table if not already; add a row index
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # ---- Step 1: Build a complete directed edge list (cell_id -> neighbor_cell_id) ----
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique[[i]] gives integer indices into id_order for neighbors of id_order[i]

  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(cell_id = id_order[i], neighbor_cell_id = id_order[nb])
  }))
  # edges has ~1,373,394 rows (directed rook-neighbor pairs, time-invariant)

  # ---- Step 2: Expand edges across years via join ----
  # For each (cell_id, year) row, we need the rows of all its neighbors in the same year.
  # Strategy: join edges to dt twice â€” once to get the focal row index, once to get neighbor row index.

  # Create a keyed lookup: (id, year) -> .row_id
  row_lookup <- dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # Get unique years
  years <- unique(dt$year)

  # Expand edges Ã— years: each edge applies to every year
  # To avoid a 1.37M Ã— 28 = 38.4M row table all at once, we can do it in one shot
  # (38.4M rows of 3 integer columns â‰ˆ 920 MB â€” fits in 16 GB)
  edge_year <- CJ_dt <- edges[, .(cell_id, neighbor_cell_id)]
  edge_year <- edge_year[rep(seq_len(.N), each = length(years))]
  edge_year[, year := rep(years, times = nrow(edges))]

  # Join to get focal row id
  setkey(edge_year, cell_id, year)
  edge_year[row_lookup, focal_row := i..row_id, on = .(cell_id = id, year = year)]

  # Join to get neighbor row id
  setkey(edge_year, neighbor_cell_id, year)
  edge_year[row_lookup, neighbor_row := i..row_id, on = .(neighbor_cell_id = id, year = year)]

  # Drop edges where either focal or neighbor is missing (cell-years not in data)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # ---- Step 3: Compute neighbor stats for each variable via grouped aggregation ----
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach the neighbor's value
    edge_year[, nval := dt[[var_name]][neighbor_row]]

    # Grouped aggregation: max, min, mean per focal row (excluding NAs)
    stats <- edge_year[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       by = focal_row]

    # Initialize columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]

    # Clean up temp column
    edge_year[, nval := NULL]
  }

  # ---- Step 4: Return as data.frame, preserving original structure ----
  dt[, .row_id := NULL]
  return(as.data.frame(dt))
}

# ---- Usage (drop-in replacement for the original outer loop) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as-is downstream â€” no retraining needed.
# predict(rf_model, newdata = cell_data)
```

## Memory-Constrained Variant

If the ~38.4M-row `edge_year` table strains the 16 GB laptop, process years in batches:

```r
build_neighbor_features_fast_lowmem <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # Build time-invariant edge list
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) return(NULL)
    data.table(cell_id = id_order[i], neighbor_cell_id = id_order[nb])
  }))

  row_lookup <- dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0("nb_max_", var_name)  := NA_real_]
    dt[, paste0("nb_min_", var_name)  := NA_real_]
    dt[, paste0("nb_mean_", var_name) := NA_real_]
  }

  years <- sort(unique(dt$year))

  for (yr in years) {
    message("Processing year: ", yr)

    ey <- copy(edges)
    ey[, year := yr]

    # Join focal
    setkey(ey, cell_id, year)
    ey[row_lookup, focal_row := i..row_id, on = .(cell_id = id, year)]

    # Join neighbor
    setkey(ey, neighbor_cell_id, year)
    ey[row_lookup, neighbor_row := i..row_id, on = .(neighbor_cell_id = id, year)]

    ey <- ey[!is.na(focal_row) & !is.na(neighbor_row)]

    for (var_name in neighbor_source_vars) {
      ey[, nval := dt[[var_name]][neighbor_row]]

      stats <- ey[!is.na(nval),
                   .(nb_max = max(nval), nb_min = min(nval), nb_mean = mean(nval)),
                   by = focal_row]

      if (nrow(stats) > 0L) {
        set(dt, i = stats$focal_row, j = paste0("nb_max_", var_name),  value = stats$nb_max)
        set(dt, i = stats$focal_row, j = paste0("nb_min_", var_name),  value = stats$nb_min)
        set(dt, i = stats$focal_row, j = paste0("nb_mean_", var_name), value = stats$nb_mean)
      }
      ey[, nval := NULL]
    }
  }

  dt[, .row_id := NULL]
  return(as.data.frame(dt))
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | Named vector (O(n) per query) | `data.table` keyed join (O(1) amortized) |
| Neighbor resolution | Per-row `lapply` with `paste` + named lookup Ã— 6.46M | Single vectorized join on edge table |
| Stat computation | Per-row `lapply` with subsetting | `data.table` grouped `by=` aggregation |
| Estimated runtime | 86+ hours | ~2â€“10 minutes |
| Peak memory | Lower but irrelevant at 86h | ~2â€“4 GB (batched: ~1.5 GB) |
| Numerical output | â€” | Identical (same max/min/mean) |
| RF model | Untouched | Untouched |

The core insight: replacing millions of R-level character-key lookups against a named vector with a single vectorized `data.table` equi-join eliminates the quadratic scaling and delivers identical numerical results in minutes.
END_RESPONSE>>>
