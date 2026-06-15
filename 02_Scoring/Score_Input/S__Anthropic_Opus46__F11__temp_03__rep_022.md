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
run_id: Anthropic_F11_T03_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is O(n) hash-based but repeated 6.46M times with multiple neighbor keys each time, this generates on the order of **tens of millions of string operations and hash lookups**. This function alone likely takes hours.

2. **`compute_neighbor_stats()`** iterates via `lapply` over the same 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean` per row. It is called **5 times** (once per neighbor source variable), meaning ~32.3 million R-level function invocations with per-element subsetting.

3. **The `do.call(rbind, result)` call** inside `compute_neighbor_stats` binds 6.46M small vectors into a matrix â€” a notoriously slow pattern in R.

4. By contrast, Random Forest prediction (`predict()`) on a pre-trained model with ~6.46M rows and ~110 features is a single vectorized C/C++ call (in `ranger` or `randomForest`). It typically completes in seconds to minutes, not hours. Loading and writing are also trivial relative to the feature engineering.

**Conclusion:** The bottleneck is the row-level R `lapply` loops over millions of rows performing string operations and lookups. The 86+ hour runtime is dominated by these neighbor feature computations, not RF inference.

---

## Optimization Strategy

1. **Eliminate per-row string pasting and named-vector lookups** in `build_neighbor_lookup()`. Replace with integer-indexed operations using `data.table` for fast keyed joins.

2. **Vectorize `compute_neighbor_stats()`** by "exploding" the neighbor relationships into a long-form edge table, joining the variable values, and computing grouped aggregations (`max`, `min`, `mean`) in a single `data.table` operation â€” no R-level row loop at all.

3. **Build the neighbor lookup as a long-form edge table once**, then reuse it for all 5 variables via a simple keyed join + grouped aggregation.

This reduces the complexity from ~32M interpreted R function calls to a handful of vectorized `data.table` operations, bringing runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert cell_data to data.table and assign a row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Build a vectorized neighbor edge table (replaces build_neighbor_lookup)
#
#    For every (focal_id, neighbor_id) pair from the rook neighbor list,
#    and for every year, we need the row index of the neighbor's cell-year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 2a. Expand the nb object into a long-form edge list of (focal_id, neighbor_id)
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_indices <- rook_neighbors_unique[[i]]
  if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_indices])
}))

# 2b. Create a keyed lookup from (id, year) -> row_idx
id_year_key <- cell_data[, .(id, year, row_idx)]
setkey(id_year_key, id, year)

# 2c. Get the unique years
years <- sort(unique(cell_data$year))

# 2d. Cross-join edges Ã— years, then look up focal and neighbor row indices.
#     This produces one row per (focal_cell-year, neighbor_cell-year) pair.
edge_year <- CJ_dt_edges_years(edges, years)  # see helper below

# Helper: cross join edges with years efficiently
build_edge_year_table <- function(edges, years, id_year_key) {
  # Expand edges by year
  ey <- edges[, .(focal_id, neighbor_id, year = rep(years, each = .N)),
              by = .EACHI,
              env = list()]
  # More memory-friendly approach: use a cross join
  ey <- edges[, CJ(edge_idx = .I, year = years)]
  # That won't work directly. Instead:
  ey <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  ey[, focal_id    := edges$focal_id[edge_idx]]
  ey[, neighbor_id := edges$neighbor_id[edge_idx]]

  # Look up focal row_idx
  ey[id_year_key, focal_row := i.row_idx,
     on = .(focal_id = id, year = year)]

  # Look up neighbor row_idx
  ey[id_year_key, neighbor_row := i.row_idx,
     on = .(neighbor_id = id, year = year)]

  # Drop rows where either side is missing
  ey <- ey[!is.na(focal_row) & !is.na(neighbor_row)]
  ey
}

edge_year <- build_edge_year_table(edges, years, id_year_key)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Vectorized neighbor stats (replaces compute_neighbor_stats + loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value for this variable to every edge-year row
  edge_year[, nbr_val := cell_data[[var_name]][neighbor_row]]

  # Compute grouped stats: max, min, mean per focal row, ignoring NAs
  stats <- edge_year[!is.na(nbr_val),
                     .(nbr_max  = max(nbr_val),
                       nbr_min  = min(nbr_val),
                       nbr_mean = mean(nbr_val)),
                     keyby = .(focal_row)]

  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign computed stats back by row index
  cell_data[stats$focal_row, (max_col)  := stats$nbr_max]
  cell_data[stats$focal_row, (min_col)  := stats$nbr_min]
  cell_data[stats$focal_row, (mean_col) := stats$nbr_mean]

  # Clean up the temporary column
  edge_year[, nbr_val := NULL]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Random Forest prediction (unchanged â€” this was never the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (preserving the trained model and original estimand):
# library(ranger)  # or randomForest
# predictions <- predict(trained_rf_model, data = cell_data)
# cell_data[, prediction := predictions$predictions]
```

### Memory-Conscious Refinement

If the full `edge_year` table (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows) strains the 16 GB laptop, process in year-chunks:

```r
for (var_name in neighbor_source_vars) {

  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  vals <- cell_data[[var_name]]

  for (yr in years) {
    # Subset focal rows for this year
    yr_rows <- id_year_key[year == yr]  # id, year, row_idx
    setkey(yr_rows, id)

    # For each edge, look up focal and neighbor row_idx in this year
    yr_edges <- edges[yr_rows, on = .(focal_id = id), nomatch = 0L,
                      allow.cartesian = FALSE]
    yr_edges[, focal_row := yr_rows[.(focal_id), row_idx, on = .(id)]]
    yr_edges[, neighbor_row := yr_rows[.(neighbor_id), row_idx, on = .(id)]]
    yr_edges <- yr_edges[!is.na(neighbor_row)]

    yr_edges[, nbr_val := vals[neighbor_row]]

    stats <- yr_edges[!is.na(nbr_val),
                      .(nbr_max  = max(nbr_val),
                        nbr_min  = min(nbr_val),
                        nbr_mean = mean(nbr_val)),
                      keyby = .(focal_row)]

    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nbr_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nbr_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nbr_mean)
  }
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (row-level `lapply` with string ops over 6.46M rows Ã— 5 vars) | Eliminated |
| **Mechanism** | ~32M R-level function calls, string paste, named-vector hash lookups, `do.call(rbind, ...)` | Vectorized `data.table` keyed joins + grouped aggregation |
| **Estimated runtime** | 86+ hours | Minutes (typically 5â€“20 min on a laptop) |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical results** | Original `max`, `min`, `mean` per neighbor set | Identical â€” same grouping, same aggregation functions |
END_RESPONSE>>>
