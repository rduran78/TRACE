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
run_id: Anthropic_F06_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`

The function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs `paste()`-based string keys for each neighbor Ã— year combination.
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]`.

String-key construction and named-vector lookup for 6.46M rows, each with ~4 neighbors on average (rook), means roughly **25+ million string operations and hash lookups**. The `lapply` returns a list of 6.46M integer vectors â€” this alone is extremely memory- and time-intensive.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated per-row `lapply`

For each of the 5 source variables, another `lapply` over 6.46M elements extracts neighbor values, removes NAs, and computes `max/min/mean`. That's **5 Ã— 6.46M = 32.3M R-level function calls**, each involving subsetting and aggregation. R's interpreted loop overhead makes this very slow.

### Why raster focal/kernel operations are *not* a direct substitute

Focal operations assume a regular rectangular grid with fixed kernel geometry. Here the data is a **panel** (cell Ã— year), neighbors are defined by an irregular `spdep::nb` object, and the computation is per-variable per-year. Focal convolutions would require reshaping into raster stacks per year, handling irregular boundaries, and would not naturally produce max/min. The analogy is useful conceptually (the neighbor stats *are* a spatial convolution), but the implementation should stay in tabular form to **preserve the exact numerical estimand** required by the pre-trained Random Forest.

### Root cause summary

| Component | Cost driver | Estimated time share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named vector lookups | ~40% |
| `compute_neighbor_stats` (Ã—5) | 32.3M interpreted R loops with per-element subsetting | ~60% |

---

## Optimization Strategy

### Principle: Replace row-level R loops with vectorized / `data.table` operations

1. **Eliminate string keys entirely.** Instead of `paste(id, year)` â†’ named lookup, use `data.table` keyed joins. Assign each row a simple integer row index. Build an edge list (a two-column integer matrix) of `(focal_row, neighbor_row)` once, then use vectorized subsetting.

2. **Build the edge list vectorized.** For each cell, we know its neighbors (from the `nb` object) and the years it appears in. Rather than iterating 6.46M rows, iterate over the 344K cells, expand neighbors, and join on year using `data.table` â€” a single merge replaces millions of string lookups.

3. **Compute stats via `data.table` grouped aggregation.** Once we have the edge list `(focal_row_idx, neighbor_row_idx)`, extract neighbor values by vectorized column subsetting, then `data.table::groupby` on `focal_row_idx` to compute `max`, `min`, `mean`. This replaces 6.46M R-level `lapply` calls with a single vectorized grouped operation per variable.

4. **Expected speedup:** From ~86 hours to **~2â€“10 minutes** on a 16 GB laptop.

5. **Numerical equivalence:** The same neighbor relationships, the same `max/min/mean` aggregations, the same column names are produced. The pre-trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with a row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a vectorized edge list (focal_row â†’ neighbor_row)
#
# id_order:              integer vector of cell IDs in the order
#                        matching rook_neighbors_unique (the nb object).
# rook_neighbors_unique: an nb object (list of integer index vectors
#                        referencing positions in id_order).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list_dt <- function(cell_dt, id_order, neighbors) {
  # --- 1a. Build cell-level neighbor edge list (cell_id â†’ neighbor_cell_id)
  n_cells <- length(id_order)
  # Pre-allocate: count total directed edges
  n_edges_cell <- sum(vapply(neighbors, function(x) {
    len <- length(x)
    # spdep::nb encodes "no neighbors" as a single 0
    if (len == 1L && x[1L] == 0L) 0L else len
  }, integer(1)))

  focal_cell   <- integer(n_edges_cell)
  neighbor_cell <- integer(n_edges_cell)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 1L && nb_i[1L] == 0L) next
    n_nb <- length(nb_i)
    focal_cell[pos:(pos + n_nb - 1L)]    <- id_order[i]
    neighbor_cell[pos:(pos + n_nb - 1L)] <- id_order[nb_i]
    pos <- pos + n_nb
  }

  cell_edges <- data.table(
    focal_id    = focal_cell,
    neighbor_id = neighbor_cell
  )

  # --- 1b. Map (cell_id, year) â†’ row_idx via keyed join
  # Build a small lookup: id, year â†’ row_idx
  id_year_lookup <- cell_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)

  # Get the unique years present
  years <- sort(unique(cell_dt$year))

  # Cross-join cell_edges Ã— years, then look up row indices for both

  # focal and neighbor.
  # To avoid a massive cross join in memory, we do two keyed joins.

  # Expand edges by year using CJ inside a merge:
  # But more memory-efficient: for each year, join edges â†’ row indices.
  edge_list_parts <- lapply(years, function(yr) {
    # For this year, get the row indices of all cells
    yr_lookup <- id_year_lookup[year == yr, .(id, row_idx)]
    setkey(yr_lookup, id)

    # Join focal side
    tmp <- cell_edges[yr_lookup, on = .(focal_id = id), nomatch = 0L,
                      .(focal_row = i.row_idx, neighbor_id = x.neighbor_id)]
    # Join neighbor side
    setkey(tmp, neighbor_id)
    tmp2 <- tmp[yr_lookup, on = .(neighbor_id = id), nomatch = 0L,
                .(focal_row = x.focal_row, neighbor_row = i.row_idx)]
    tmp2
  })

  edge_dt <- rbindlist(edge_list_parts)
  edge_dt
}

message("Building edge list...")
t0 <- proc.time()
edge_dt <- build_edge_list_dt(cell_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge list built: %d directed cell-year edges in %.1f seconds.",
                nrow(edge_dt), (proc.time() - t0)[3]))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(cell_dt, edge_dt, var_names) {
  # Attach neighbor values to edge list for all variables at once
  # by referencing column values via row index (vectorized).

  # Pre-allocate result columns in cell_dt (all NA)
  for (v in var_names) {
    cell_dt[, paste0("n_max_", v) := NA_real_]
    cell_dt[, paste0("n_min_", v) := NA_real_]
    cell_dt[, paste0("n_mean_", v) := NA_real_]
  }

  for (v in var_names) {
    message(sprintf("  Computing neighbor stats for: %s", v))
    t1 <- proc.time()

    # Vectorized extraction of neighbor values
    edge_dt[, nval := cell_dt[[v]][neighbor_row]]

    # Remove NA neighbor values before aggregation
    valid_edges <- edge_dt[!is.na(nval)]

    # Grouped aggregation â€” single pass
    stats <- valid_edges[, .(
      n_max  = max(nval),
      n_min  = min(nval),
      n_mean = mean(nval)
    ), by = focal_row]

    # Write results back into cell_dt by row index
    cell_dt[stats$focal_row, paste0("n_max_", v)  := stats$n_max]
    cell_dt[stats$focal_row, paste0("n_min_", v)  := stats$n_min]
    cell_dt[stats$focal_row, paste0("n_mean_", v) := stats$n_mean]

    message(sprintf("    Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  # Clean up temporary column
  edge_dt[, nval := NULL]

  cell_dt
}

message("Computing neighbor statistics...")
t0 <- proc.time()
cell_dt <- compute_all_neighbor_stats(cell_dt, edge_dt, neighbor_source_vars)
message(sprintf("All neighbor stats computed in %.1f seconds.", (proc.time() - t0)[3]))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Drop helper column, convert back if needed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, row_idx := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_dt)
# Otherwise keep as data.table for prediction speed.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_dt)
```

---

## Why This Preserves the Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order`. No neighbors are added or dropped. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same sets of non-NA neighbor values per focal cell-year. |
| **Same column names** | Output columns follow the same `n_max_<var>`, `n_min_<var>`, `n_mean_<var>` naming convention (adjust the prefix to match your `compute_and_add_neighbor_features` if it differs). |
| **No model retraining** | The Random Forest is loaded and used for `predict()` only. |
| **NA handling identical** | Neighbors missing from a given year (no row) produce no edge; NA variable values are excluded before aggregation; cells with zero valid neighbors get `NA` â€” identical to the original logic. |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build lookup / edge list | ~35 hours | ~30â€“90 seconds |
| Neighbor stats (5 vars) | ~51 hours | ~1â€“5 minutes |
| **Total** | **~86 hours** | **~2â€“10 minutes** |

The speedup comes from replacing **6.46 million interpreted R function calls** with vectorized `data.table` keyed joins and grouped aggregations that execute in compiled C code internally.
END_RESPONSE>>>
