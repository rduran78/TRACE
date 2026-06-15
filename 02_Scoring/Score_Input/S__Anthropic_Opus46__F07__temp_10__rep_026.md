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
run_id: Anthropic_F07_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements using nested `lapply` with per-row string-pasting, hash lookups, and NA filtering.** This is an O(n) loop in R with expensive character operations inside each iteration. Similarly, `compute_neighbor_stats` loops over 6.46 million entries, subsetting a numeric vector each time. With 5 variables, that's ~32 million R-level list iterations total, plus the initial ~6.46 million iteration build step.

**Root causes:**

1. **`build_neighbor_lookup`**: For each of 6.46M rows, it does `paste()`, character-keyed lookups (`idx_lookup[neighbor_keys]`), and NA removal. The string-key join is the most expensive â€” it converts a spatial-neighbor problem into millions of string operations.
2. **`compute_neighbor_stats`**: The inner `lapply` over 6.46M elements calls `max`, `min`, `mean` individually per neighborhood, each with subsetting and NA removal. This is death-by-a-million-small-calls.
3. **Memory**: Storing 6.46M list entries (each a small integer vector) is memory-heavy due to R's per-object overhead.

The entire pattern is essentially a **grouped aggregation / join** problem that `data.table` can solve in vectorized C code in minutes rather than days.

## Optimization Strategy

**Replace the row-level R loop with a vectorized `data.table` join-and-aggregate approach:**

1. **Expand the neighbor list into an edge table** (directed edges: `from_id â†’ to_id`). With ~1.37M directed rook-neighbor relationships per year Ã— 28 years â‰ˆ 38.4M edge-rows, this fits easily in RAM.
2. **Join** the edge table to the data on `(to_id, year)` to pull in neighbor values â€” one vectorized `data.table` merge.
3. **Aggregate** by `(from_id, year)` to compute `max`, `min`, `mean` â€” one vectorized `data.table` grouped operation.
4. **Join back** to the original data to add the new columns.

This replaces ~86 hours of R-level looping with a handful of vectorized operations that should complete in **minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a directed edge table from the nb object (one-time cost)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique: an nb object (list of integer vectors of neighbor indices)
# id_order: vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    n_i <- length(nb_i)
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  
  # Trim if any empty-neighbor cells caused over-allocation
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Convert cell_data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure key columns are proper types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
setkey(cell_data, id, year)

edge_dt[, from_id := as.integer(from_id)]
edge_dt[, to_id   := as.integer(to_id)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: For each source variable, compute neighbor max/min/mean
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Get all unique years once
all_years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Extract only the columns we need for the join
  # Columns: id (as to_id target), year, and the variable value
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)
  
  # Cross-join edges Ã— years, then join to get neighbor values
  # More memory-efficient: join edges to val_dt directly via CJ approach
  # Expand edges across all years
  edge_year_dt <- CJ_edge <- edge_dt[, .(from_id, to_id)]
  
  # Repeat for each year via a merge with years
  year_dt <- data.table(year = all_years)
  edge_year_dt <- edge_dt[, as.list(year_dt), by = .(from_id, to_id)]
  # This creates from_id, to_id, year for all combos
  
  setkey(edge_year_dt, to_id, year)
  
  # Join to get neighbor values
  edge_year_dt[val_dt, val := i.val, on = .(to_id, year)]
  
  # Aggregate: group by (from_id, year), compute stats (ignoring NAs)
  stats_dt <- edge_year_dt[!is.na(val),
    .(nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)),
    by = .(from_id, year)
  ]
  
  # Name columns to match original convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats_dt, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  setnames(stats_dt, "from_id", "id")
  setkey(stats_dt, id, year)
  
  # Join back to cell_data
  cell_data[stats_dt, (c(max_col, min_col, mean_col)) :=
    mget(paste0("i.", c(max_col, min_col, mean_col))),
    on = .(id, year)]
  
  # Clean up to free memory
  rm(val_dt, edge_year_dt, stats_dt)
  gc()
  
  cat("  Done:", var_name, "\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: The trained Random Forest model can now be applied as before.
# cell_data now has the same neighbor feature columns with identical
# numerical values as the original implementation would produce.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

### Memory-Optimized Variant

If the `CJ`-style expansion of edges Ã— years (~38.4M rows) causes memory pressure on a 16 GB laptop, process years in batches:

```r
for (var_name in neighbor_source_vars) {
  cat("Processing:", var_name, "\n")
  
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Pre-allocate columns with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  for (yr in all_years) {
    # Subset to this year
    yr_data <- cell_data[year == yr, .(id, val = get(var_name))]
    setnames(yr_data, "id", "to_id")
    setkey(yr_data, to_id)
    
    # Join edges to values
    edge_val <- edge_dt[yr_data, on = .(to_id), nomatch = NULL]
    
    # Aggregate
    stats_yr <- edge_val[!is.na(val),
      .(nmax = max(val), nmin = min(val), nmean = mean(val)),
      by = .(from_id)
    ]
    
    # Write back into cell_data
    idx <- cell_data[year == yr, which = TRUE]
    id_in_year <- cell_data$id[idx]
    
    # Create a keyed lookup
    setkey(stats_yr, from_id)
    matched <- stats_yr[J(id_in_year)]
    
    set(cell_data, i = idx, j = max_col,  value = matched$nmax)
    set(cell_data, i = idx, j = min_col,  value = matched$nmin)
    set(cell_data, i = idx, j = mean_col, value = matched$nmean)
  }
  
  gc()
  cat("  Done:", var_name, "\n")
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup build** | 6.46M R-level string pastes + hash lookups | One-time edge table from nb object (~1.37M rows) |
| **Stats computation** | 6.46M Ã— 5 = 32.3M `lapply` iterations | 5 vectorized `data.table` group-by-aggregations |
| **Estimated time** | 86+ hours | **5â€“15 minutes** |
| **Memory** | ~6.46M R list elements (high overhead) | ~38M-row data.table (compact) |
| **Numerical result** | max/min/mean of rook-neighbor values | **Identical** â€” same edges, same aggregation |
| **RF model** | â€” | **Untouched** â€” only feature columns are recomputed with identical values |

The edge-table approach is an exact isomorphism of the original neighbor lookup: every directed rook-neighbor pair is preserved, the same `NA`-removal logic applies, and `max`/`min`/`mean` are computed over precisely the same sets. The trained Random Forest model requires no retraining.
END_RESPONSE>>>
