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
run_id: Anthropic_F06_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine â€” O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for each neighbor.
3. Looks up those keys in `idx_lookup` (a named vector used as a hash map).

The `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow in R. String construction and matching at this scale dominates runtime.

**Key insight:** The neighbor topology is *purely spatial* â€” it does not change across years. Every cell has the same neighbors in every year. So the lookup can be factored into a spatial step (done once for 344,208 cells) and a temporal expansion step (mechanical, vectorized).

### Bottleneck 2: `compute_neighbor_stats` â€” per-row `lapply` with subsetting

For each of the 5 variables, another `lapply` iterates over 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is 5 Ã— 6.46M = 32.3M R-level function calls. Pure R loops over millions of rows are inherently slow.

**Key insight:** This is equivalent to a grouped aggregation. If we build a long-form edge table `(row_i, neighbor_row_j)` we can use `data.table` grouped operations to compute `max`, `min`, `mean` in a single vectorized pass per variable â€” or even all variables at once.

### Why raster focal/kernel operations are *not* the right analogy here

Focal operations assume a regular grid with a fixed rectangular kernel. The data here is an irregular spatial panel indexed by an `nb` object (which may have variable numbers of neighbors, boundary effects, missing cells, etc.). Forcing it into a raster would require padding, reindexing, and could introduce errors. The edge-table + `data.table` approach preserves the exact `nb` topology and the exact numerical results.

---

## Optimization Strategy

1. **Precompute a spatial-only neighbor edge list** â€” a two-column integer matrix `(cell_ref, neighbor_ref)` from the `nb` object. Done once for 344,208 cells.

2. **Expand to panel rows via vectorized merge** â€” join the spatial edge list to the panel data's row indices using `data.table` keyed joins. This produces an edge table `(row_i, row_j)` at the cell-year level, entirely vectorized.

3. **Compute all neighbor stats in one grouped aggregation** â€” for each variable, join the neighbor values via the edge table, then `data.table` grouped `max`, `min`, `mean` by `row_i`. This replaces 6.46M R-level `lapply` iterations with a single vectorized `data.table` operation.

4. **Memory management** â€” the edge table will have ~1.37M spatial edges Ã— 28 years â‰ˆ 38.5M rows Ã— 2 integer columns â‰ˆ 308 MB. With neighbor values joined, each variable adds ~308 MB of doubles temporarily. On 16 GB RAM this is feasible if we process variables sequentially and free intermediates.

**Expected speedup:** From 86+ hours to roughly 5â€“15 minutes.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 0: Convert to data.table, preserve original row order

# ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]  # preserve original row order

  # ---------------------------------------------------------------
  # STEP 1: Build spatial edge list from nb object (done ONCE)
  #         This replaces the per-cell-year string-key lookup.
  # ---------------------------------------------------------------
  # rook_neighbors_unique is a list of length = length(id_order)
  # where element [[i]] contains integer indices into id_order
  # representing the neighbors of id_order[i].

  n_cells <- length(id_order)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(rook_neighbors_unique))

  from_ref <- integer(n_edges)
  to_ref   <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    len  <- length(nb_i)
    if (len > 0L) {
      from_ref[pos:(pos + len - 1L)] <- i
      to_ref[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }

  # Map ref indices to actual cell IDs
  spatial_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  # ---------------------------------------------------------------
  # STEP 2: Build row-index lookup keyed on (id, year)
  # ---------------------------------------------------------------
  # This lets us expand spatial edges to cell-year row edges
  row_lookup <- dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # ---------------------------------------------------------------
  # STEP 3: Expand spatial edges to panel-level edges
  #         For each (from_id -> to_id) and each year,

  #         find the row indices of both the focal cell-year
  #         and the neighbor cell-year.
  # ---------------------------------------------------------------
  # Get unique years
  years <- sort(unique(dt$year))

  # Cross join spatial edges with years
  # To manage memory, we do this in a vectorized way:
  panel_edges <- spatial_edges[, .(year = years), by = .(from_id, to_id)]

  # Join to get focal row index
  setkey(panel_edges, from_id, year)
  panel_edges[row_lookup, focal_row := i..row_id, on = .(from_id = id, year)]

  # Join to get neighbor row index
  setkey(panel_edges, to_id, year)
  panel_edges[row_lookup, neighbor_row := i..row_id, on = .(to_id = id, year)]

  # Drop edges where either focal or neighbor row is missing
  panel_edges <- panel_edges[!is.na(focal_row) & !is.na(neighbor_row)]

  # Keep only what we need
  edge_dt <- panel_edges[, .(focal_row, neighbor_row)]
  rm(panel_edges, spatial_edges, row_lookup)
  gc()

  # ---------------------------------------------------------------
  # STEP 4: For each variable, compute neighbor max/min/mean
  #         via data.table grouped aggregation
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Attach neighbor values
    edge_dt[, nval := dt[[var_name]][neighbor_row]]

    # Grouped aggregation â€” only over non-NA neighbor values
    stats <- edge_dt[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = focal_row]

    # Initialize new columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]

    # Clean up
    edge_dt[, nval := NULL]
    rm(stats)
    gc()
  }

  # ---------------------------------------------------------------
  # STEP 5: Return as data.frame in original row order,
  #         preserving all original columns + new neighbor features
  # ---------------------------------------------------------------
  setorder(dt, .row_id)
  dt[, .row_id := NULL]

  as.data.frame(dt)
}

# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# Then proceed with prediction using the already-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Verification That Numerical Results Are Preserved

The optimized code computes **exactly** the same quantities:

| Original | Optimized | Equivalent? |
|---|---|---|
| `vals[idx]` subset per row | `dt[[var_name]][neighbor_row]` via edge table | âœ… Same values accessed |
| `neighbor_vals[!is.na()]` | `edge_dt[!is.na(nval), ...]` | âœ… Same NA filtering |
| `max(neighbor_vals)` | `max(nval)` grouped by `focal_row` | âœ… Identical |
| `min(neighbor_vals)` | `min(nval)` grouped by `focal_row` | âœ… Identical |
| `mean(neighbor_vals)` | `mean(nval)` grouped by `focal_row` | âœ… Identical |
| Rows with no valid neighbors â†’ `NA` | Rows absent from `stats` â†’ remain `NA` | âœ… Identical |

The trained Random Forest model is never touched â€” only the input feature matrix is constructed, identically to before, just faster.

---

## Performance Summary

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string pastes + hash lookups | One spatial edge list + vectorized `data.table` keyed join |
| Per-variable stats | 6.46M `lapply` iterations Ã— 5 vars | 5 grouped `data.table` aggregations over ~38.5M-row edge table |
| **Estimated total time** | **86+ hours** | **~5â€“15 minutes** |
| RAM peak | Lower but slow | ~4â€“6 GB (fits in 16 GB) |
END_RESPONSE>>>
