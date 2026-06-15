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
run_id: Anthropic_F07_T00_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character-key lookups inside an `lapply` over ~6.46M rows.** Each iteration does `paste`, named-vector indexing (`id_to_ref`, `idx_lookup`), and `is.na` filtering. Named-vector lookup in R is O(n) in the worst case for long vectors, and `paste` allocates a new string each call. Over 6.46M iterations this is catastrophically slow.

2. **`compute_neighbor_stats` is a second `lapply` over 6.46M rows**, subsetting a numeric vector each time. This is less severe but still slow in pure R.

3. **The entire pattern is repeated 5 times** (once per source variable), but the neighbor lookup is built only once â€” so the lookup construction dominates.

**Root cause:** The algorithm is O(NÂ·k) with enormous per-element constant factors due to R's string operations and named-vector hashing on millions of keys. On a laptop this yields the estimated 86+ hour runtime.

## Optimization Strategy

### 1. Vectorize `build_neighbor_lookup` entirely

Replace the row-by-row `lapply` with a **vectorized merge/join** approach:

- Expand the `nb` object into an edge list `(cell_index, neighbor_cell_index)`.
- Cross this with years using `data.table` joins (integer keys, no string pasting).
- The result is a single edge table: `(row_i, neighbor_row_j)` â€” the same information as the list, but in columnar form.

### 2. Vectorize `compute_neighbor_stats` via `data.table` grouped aggregation

Instead of iterating over 6.46M list elements, use the edge table to look up neighbor values, then `group by row_i` and compute `max`, `min`, `mean` in one vectorized pass.

### 3. Preserve the numerical estimand exactly

The aggregation functions (`max`, `min`, `mean` over non-NA neighbor values, returning `NA` when no valid neighbors exist) are identical. The trained Random Forest model is untouched â€” we only change how features are computed, not their values.

### Estimated speedup

- `data.table` join + grouped aggregation on ~6.46M rows Ã— ~4 neighbors â‰ˆ ~26M edge-rows.
- Expected runtime: **minutes**, not hours.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a vectorized edge table (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # --- Step A: expand nb object into a cell-level edge list ---
  # id_order[i] is the cell id for the i-th element of the nb object.
  # rook_neighbors_unique[[i]] gives integer indices of neighbors in id_order.

  from_ref <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0L)
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  cell_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- Step B: map cell ids to row indices via year ---
  # cell_data_dt must already have columns: id, year, and a row index.
  cell_data_dt[, row_idx := .I]

  # Create a keyed lookup: (id, year) -> row_idx
  id_year_lookup <- cell_data_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)

  # --- Step C: for every (from_id, to_id) pair, cross with all years ---
  # Get the unique years
  years <- sort(unique(cell_data_dt$year))

  # Cross edges Ã— years
  edge_years <- CJ_dt(cell_edges, years)

  # Now join to get row_idx for the "from" side (the focal cell)
  setnames(edge_years, "year", "year")
  edge_years[id_year_lookup, focal_row := i.row_idx,
             on = .(from_id = id, year = year)]

  # Join to get row_idx for the "to" side (the neighbor cell)
  edge_years[id_year_lookup, neighbor_row := i.row_idx,
             on = .(to_id = id, year = year)]

  # Drop edges where either side is missing (masked cells / boundary)
  edge_years <- edge_years[!is.na(focal_row) & !is.na(neighbor_row)]

  # Return lean table
  edge_years[, .(focal_row, neighbor_row)]
}

# Helper: cross join a data.table with a vector of years
CJ_dt <- function(dt, years) {
  yr_dt <- data.table(year = years)
  # keyed cross join
  res <- dt[, .(year = years), by = .(from_id, to_id)]
  res
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor stats vectorized (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_vec <- function(cell_data_dt, edge_table, var_name) {
  n <- nrow(cell_data_dt)

  # Pull the variable values for neighbor rows
  vals <- cell_data_dt[[var_name]]
  edge_table[, nval := vals[neighbor_row]]

  # Drop NA neighbor values
  valid_edges <- edge_table[!is.na(nval)]

  # Grouped aggregation
  agg <- valid_edges[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = .(focal_row)]

  # Allocate full-length result columns (NA default)
  res_max  <- rep(NA_real_, n)
  res_min  <- rep(NA_real_, n)
  res_mean <- rep(NA_real_, n)

  res_max[agg$focal_row]  <- agg$nb_max
  res_min[agg$focal_row]  <- agg$nb_min
  res_mean[agg$focal_row] <- agg$nb_mean

  # Clean up temporary column
  edge_table[, nval := NULL]

  data.table(nb_max = res_max, nb_min = res_min, nb_mean = res_mean)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Outer loop (replaces the original for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                  neighbor_source_vars) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building edge table â€¦")
  edge_table <- build_neighbor_edge_table(cell_data_dt, id_order,
                                          rook_neighbors_unique)
  setkey(edge_table, focal_row)
  message(sprintf("  Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_vec(cell_data_dt, edge_table, var_name)

    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_data_dt, j = col_max,  value = stats$nb_max)
    set(cell_data_dt, j = col_min,  value = stats$nb_min)
    set(cell_data_dt, j = col_mean, value = stats$nb_mean)
  }

  # Return as data.frame if the original was one
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(cell_data_dt))
  }
  cell_data_dt
}

# â”€â”€ Usage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” apply predict() as before.
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` over the identical set of non-NA rook-neighbor values per cell-year. NA returned when no valid neighbors exist â€” identical semantics. |
| **Irregular topology** | The `nb` object is expanded faithfully; no rectangular-grid assumption is made. Masked or missing cell-years are dropped by the join (`!is.na`), exactly as the original `idx_lookup` did. |
| **Trained RF model** | Untouched. Only the feature-engineering step is optimized. Column names and values are identical. |
| **Memory** | The edge table is ~26M rows Ã— 2 integer columns â‰ˆ 200 MB, well within 16 GB. `data.table` operates in-place via `set()`. |
| **Speed** | Two `data.table` keyed joins + 5 grouped aggregations. Expected wall-clock: **2â€“10 minutes** on a standard laptop (down from 86+ hours). |
END_RESPONSE>>>
