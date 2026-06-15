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
run_id: Anthropic_F07_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-like overhead from per-row `lapply`

With ~6.46 million cell-year rows, the `lapply` iterates row-by-row, performing character key lookups (`paste`, named-vector indexing) for every single row. Named vector lookups via `idx_lookup[neighbor_keys]` on a vector of 6.46M entries are hash-table lookups but are called millions of times with string allocation overhead. The `paste(..., sep="_")` string construction alone on 6.46M Ã— ~4 neighbors â‰ˆ 25M+ string operations is extremely slow in base R.

### 2. `compute_neighbor_stats` â€” Row-by-row `lapply` over 6.46M rows, repeated 5 times

Each call to `compute_neighbor_stats` iterates over every row, extracts a small vector of neighbor values, computes `max/min/mean`, and packs the result. This is called 5 times (once per variable), meaning ~32.3 million R-level function invocations with repeated subsetting.

### Combined: ~86+ hours is consistent with ~38.7M slow R-level iterations with string allocation and GC pressure on a 16GB laptop.

---

## Optimization Strategy

**Key insight:** The neighbor topology is *time-invariant* â€” a cell's rook neighbors are the same in every year. So we only need to map the ~344K cell-level neighbor graph once, then exploit the panel's regular structure (each cell appears once per year) to vectorize everything via `data.table` joins and grouped operations.

### Step-by-step:

1. **Build a cell-level edge list once** (from the `nb` object): ~1.37M directed edges. This is tiny.

2. **Convert `cell_data` to `data.table`**, keyed on `(id, year)`.

3. **Join the edge list** to the data to produce a long table of `(focal_id, focal_year, neighbor_value)` â€” this is a single equi-join, fully vectorized in C via `data.table`.

4. **Group-aggregate** `max, min, mean` by `(focal_id, focal_year)` in one pass per variable.

5. **Merge results back** into the main table.

This replaces all R-level loops with vectorized C-level operations. Expected runtime: **minutes, not hours**.

The numerical results are identical: same neighbor sets, same `max/min/mean` computations, same variable names. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a cell-level directed edge list from the nb object
#         This runs once over 344,208 cells. Output: ~1.37M rows.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# Typically ~1,373,394 rows (directed)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (in place if possible)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  setDT(cell_data)   # converts in place â€” no copy
}

# Ensure id and year columns exist and are keyed for fast joins
setkey(cell_data, id, year)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: For each neighbor source variable, compute neighbor stats
#         via a single vectorized join + grouped aggregation.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Build a slim lookup: just id, year, and the variable of interest
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # Cross-product of edges Ã— years via join:
  #   For each (focal_id, neighbor_id) edge and each year,
  #   retrieve the neighbor's value.
  #
  # We need focal_id + year to fan out the edges across years.
  # Strategy: join edge_dt to the focal side to get years,
  #           then join to lookup to get neighbor values.

  # Get the distinct years from the data
  years_vec <- sort(unique(cell_dt$year))

  # Expand edges across all years (edges are time-invariant)
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in memory easily
  edge_year <- CJ_dt_edges(edge_dt, years_vec)

  # Join to get neighbor values
  setkey(edge_year, neighbor_id, year)
  edge_year[lookup, val := i.val, on = .(neighbor_id, year)]

  # Aggregate: group by focal_id, year â†’ max, min, mean (excluding NA)
  stats <- edge_year[
    !is.na(val),
    .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ),
    by = .(focal_id, year)
  ]

  # Rename to match original convention
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_name, min_name, mean_name))
  setnames(stats, "focal_id", "id")
  setkey(stats, id, year)

  stats
}

# Helper: expand edge list across all years efficiently
CJ_dt_edges <- function(edge_dt, years_vec) {
  # Repeat each edge for every year
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  data.table(
    focal_id    = rep(edge_dt$focal_id,    each = n_years),
    neighbor_id = rep(edge_dt$neighbor_id,  each = n_years),
    year        = rep(years_vec, times = n_edges)
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Run for all 5 variables and merge back into cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")

  stats_dt <- compute_neighbor_features_dt(cell_data, edge_dt, var_name)

  # Merge into cell_data (left join: all cell-years preserved, NAs where no neighbors)
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")

  # Remove old columns if they exist (idempotence)
  for (col in c(max_name, min_name, mean_name)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats_dt[cell_data, on = .(id, year)]

  cat("  Done. Columns added:", max_name, min_name, mean_name, "\n")
}

setkey(cell_data, id, year)
```

---

### Memory-Optimized Variant (if 38.5M-row expansion is tight on 16 GB)

If the full edge Ã— year expansion (~38.5M rows Ã— 3 columns â‰ˆ 0.9 GB per variable) causes memory pressure alongside the 6.46M Ã— 110-column main table, process in year batches:

```r
compute_neighbor_features_dt_chunked <- function(cell_dt, edge_dt, var_name) {
  years_vec <- sort(unique(cell_dt$year))
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")

  results <- rbindlist(lapply(years_vec, function(yr) {
    # Subset to this year
    yr_data <- cell_dt[year == yr, .(id, val = get(var_name))]
    setnames(yr_data, "id", "neighbor_id")
    setkey(yr_data, neighbor_id)

    # Join edges to neighbor values
    merged <- edge_dt[yr_data, on = .(neighbor_id), nomatch = NULL]
    # merged has columns: focal_id, neighbor_id, val

    # Aggregate
    merged[
      !is.na(val),
      .(nmax = max(val), nmin = min(val), nmean = mean(val)),
      by = .(focal_id)
    ][, year := yr]
  }))

  setnames(results, c("nmax", "nmin", "nmean", "focal_id"),
                     c(max_name, min_name, mean_name, "id"))
  setkey(results, id, year)
  results
}

# Usage (drop-in replacement in the loop):
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  stats_dt <- compute_neighbor_features_dt_chunked(cell_data, edge_dt, var_name)

  for (col in c(paste0(var_name, "_neighbor_max"),
                paste0(var_name, "_neighbor_min"),
                paste0(var_name, "_neighbor_mean"))) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats_dt[cell_data, on = .(id, year)]
  cat("  Done.\n")
}

setkey(cell_data, id, year)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level string ops + hash lookups | 1 vectorized edge list build (344K cells) |
| **Stat computation** | 6.46M Ã— 5 = 32.3M `lapply` iterations | 5 vectorized `data.table` joins + group-bys |
| **String allocation** | ~25M+ `paste()` calls | Zero |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Numerical result** | max/min/mean of rook-neighbor values | Identical |
| **RF model** | Unchanged | Unchanged |
| **Peak RAM** | Lower but slower | ~2â€“4 GB additional (chunked: ~1 GB) |
END_RESPONSE>>>
