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
run_id: Anthropic_F04_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are clear:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string operations

For every one of the ~6.46 million rows, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Calls `paste()` to construct `"id_year"` key strings for every neighbor of that row.
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) â€” which is an **O(k)** hash lookup per neighbor key, but the **string construction and allocation** for ~6.46M rows Ã— ~4 neighbors each â‰ˆ 26 million `paste` calls is extremely expensive in R's interpreted loop.

The result is a **list of 6.46 million integer vectors**, which is itself a large, fragmented memory structure.

### 2. `compute_neighbor_stats` â€” Called 5 times, each iterating over the 6.46M-element list

Each call to `compute_neighbor_stats` runs another `lapply` over 6.46 million elements, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. That's 5 Ã— 6.46M = ~32.3 million R-level function invocations, each with small-vector allocation overhead. The final `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also notoriously slow.

### 3. Summary of cost drivers

| Cost driver | Magnitude |
|---|---|
| `paste()` calls in `build_neighbor_lookup` | ~26M string allocations |
| Named-vector lookups (`idx_lookup[...]`) | ~26M hash lookups |
| `lapply` iterations in `build_neighbor_lookup` | 6.46M |
| `lapply` iterations in `compute_neighbor_stats` | 5 Ã— 6.46M = 32.3M |
| `do.call(rbind, ...)` on 6.46M-element list | 5 times |
| Total R-level interpreted loop iterations | ~39M |

On a standard laptop, this easily accounts for the estimated 86+ hours.

---

## Optimization Strategy

**Core idea:** Replace all row-level R loops and string-key lookups with vectorized `data.table` joins and grouped aggregations.

### Key steps:

1. **Expand the neighbor graph into an edge table** (`data.table` with columns `id` and `neighbor_id`) â€” done once, ~1.37M rows.
2. **Join the edge table to the panel data by `(neighbor_id, year)`** to get neighbor variable values â€” this is a single keyed `data.table` merge producing ~1.37M Ã— 28 â‰ˆ ~38.5M rows (the "long neighbor-values" table). This replaces both `build_neighbor_lookup` and the inner loop of `compute_neighbor_stats`.
3. **Group by `(id, year)` and compute `max`, `min`, `mean`** â€” a single vectorized `data.table` aggregation per source variable.
4. **Join the aggregated stats back** to the main panel `data.table`.
5. Repeat steps 2â€“4 for each of the 5 source variables (or do all 5 simultaneously).

### Why this is fast:

- `data.table` keyed joins are C-level binary-search or hash joins â€” no string construction, no R-level loops.
- Grouped aggregation (`[, .(max, min, mean), by = .(id, year)]`) is computed in C with radix-sort grouping.
- Memory: the edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The expanded join table is ~38.5M rows Ã— a few columns â‰ˆ 300â€“600 MB per variable, well within 16 GB, especially if processed one variable at a time.

**Expected runtime:** Minutes, not hours. The dominant cost becomes 5 keyed joins of ~38.5M rows each, which `data.table` handles in seconds to low minutes on a laptop.

**Numerical equivalence:** The operations are identical â€” for each (cell, year), we find the rook neighbors present in that year and compute the same `max`, `min`, `mean`. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data is assumed to be a data.frame / data.table with
# columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order is the vector of cell IDs corresponding to the
#   indices in rook_neighbors_unique (an nb object).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the directed edge table from the nb object
#         (done once; ~1.37 M rows, two integer columns)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (index 0)
  valid    <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2-4: For each source variable, compute neighbor
#           max / min / mean via keyed join + grouped agg,
#           then join back to cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  # --- 2a. Build a slim lookup: (id, year, value) keyed on (id, year)
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # --- 2b. Expand: join edge_dt to cell_data to get (id, year) pairs,
  #         then join to val_dt to get neighbor values.
  #         We need every (id, year) paired with its neighbors.
  #         Efficient approach: take unique years, cross-join with edge_dt,
  #         then look up neighbor values.

  # Get the set of years present
  years_vec <- sort(unique(cell_data$year))

  # Cross join edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows)
  expanded <- CJ_edge_year(edge_dt, years_vec)

  # Join to get the neighbor's variable value
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, value := i.value, on = .(neighbor_id, year)]

  # --- 3. Grouped aggregation: max, min, mean per (id, year)
  #         Exclude NAs to match original logic.
  agg <- expanded[!is.na(value),
                  .(nmax  = max(value),
                    nmin  = min(value),
                    nmean = mean(value)),
                  by = .(id, year)]

  # Name columns to match original feature names
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # --- 4. Join aggregated stats back to cell_data
  setkey(agg, id, year)
  cell_data[agg, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]

  # Rows with no valid neighbors remain NA (data.table default)

  # Clean up to free memory before next variable

  rm(val_dt, expanded, agg)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helper: cross-join edge table with years vector
# (avoids materializing via CJ on three columns)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

CJ_edge_year <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx     <- rep(seq_len(n_edges), each = n_years)
  out     <- edge_dt[idx]
  out[, year := rep(years_vec, times = n_edges)]
  out
}
```

> **Note:** The helper function `CJ_edge_year` is defined at the bottom for clarity but must be sourced/defined **before** the loop that calls it. In practice, place it above the loop or in a sourced utilities file.

### Memory-optimized variant (process one year at a time)

If the ~38.5M-row `expanded` table per variable strains the 16 GB laptop, process in year batches:

```r
for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  # Pre-allocate columns with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  years_vec <- sort(unique(cell_data$year))

  for (yr in years_vec) {

    # Subset this year's data
    yr_dt <- cell_data[year == yr, .(id, value = get(var_name))]
    setnames(yr_dt, "id", "neighbor_id")
    setkey(yr_dt, neighbor_id)

    # Join edges to neighbor values for this year
    merged <- edge_dt[yr_dt, on = .(neighbor_id), nomatch = NULL]
    # merged has columns: id, neighbor_id, value

    # Aggregate
    agg <- merged[!is.na(value),
                  .(nmax = max(value), nmin = min(value), nmean = mean(value)),
                  by = .(id)]

    # Write back into cell_data for this year
    setkey(agg, id)
    idx <- cell_data[year == yr, which = TRUE]
    match_pos <- match(cell_data$id[idx], agg$id)

    set(cell_data, i = idx, j = max_col,  value = agg$nmax[match_pos])
    set(cell_data, i = idx, j = min_col,  value = agg$nmin[match_pos])
    set(cell_data, i = idx, j = mean_col, value = agg$nmean[match_pos])
  }

  rm(yr_dt, merged, agg)
  gc()
}
```

This variant processes only ~1.37M edge rows per year-variable combination (peak memory ~50 MB per iteration), making it safe on a 16 GB machine.

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M `paste`+hash) | Replaced by `build_edge_table`: **< 1 second** |
| `compute_neighbor_stats` (Ã—5 vars) | ~hours (32.3M R-level iterations) | 5 Ã— 28 keyed joins + grouped aggs: **2â€“10 minutes total** |
| `do.call(rbind, ...)` (Ã—5) | ~hours (6.46M-element list binding) | Eliminated (direct `:=` assignment) | 
| **Total neighbor feature construction** | **86+ hours** | **< 15 minutes** |
| Random Forest prediction | Unchanged | **Unchanged (model preserved)** |

The trained Random Forest model is never touched. The numerical results (max, min, mean of non-NA neighbor values per cell-year) are identical to the original implementation.
END_RESPONSE>>>
