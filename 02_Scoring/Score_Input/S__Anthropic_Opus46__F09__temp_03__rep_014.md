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
run_id: Anthropic_F09_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-ish in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” these are hash-table lookups on a 6.46-million-entry named vector, repeated ~4 times per row (rook neighbors), for every row.
- The result is a **list of 6.46 million integer vectors**, consuming enormous memory and time.

The fundamental flaw: **the neighbor topology is purely spatial and does not change across years**, yet the lookup is rebuilt per cell-year row, fusing spatial structure with temporal indexing in the most expensive possible way.

### 2. `compute_neighbor_stats` iterates over the 6.46M-element list
For each of the 5 variables, it walks the 6.46M-element list, subsets a numeric vector by index, removes NAs, and computes max/min/mean. This is called 5 times â†’ ~32.3 million R-level function calls, each allocating small vectors.

### Memory pressure
The `neighbor_lookup` list alone stores ~6.46M integer vectors (avg. length ~4) = ~25.8M integers + R list overhead â‰ˆ 1â€“2 GB. Combined with the 6.46M Ã— 110-column data.frame, this pushes a 16 GB laptop toward swapping.

---

## Optimization Strategy

**Core insight:** Build the neighbor table **once at the cell level** (344K cells, not 6.46M cell-years), then use a vectorized join-based approach to compute neighbor statistics per year.

### Step-by-step plan:

1. **Build a static edge table** from `rook_neighbors_unique` â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is done once.

2. **For each variable, join yearly cell attributes onto the edge table**, then group by `(cell_id, year)` and compute `max`, `min`, `mean` of the neighbor values â€” all in `data.table`, fully vectorized in C.

3. **Left-join** the resulting neighbor-stat columns back onto the main dataset.

This replaces 6.46M R-level list iterations with a handful of `data.table` grouped joins â€” expected runtime: **minutes, not days**.

### Complexity comparison:

| | Current | Proposed |
|---|---|---|
| Lookup build | 6.46M `paste` + hash lookups | 1.37M-row edge table (once) |
| Stats per variable | 6.46M `lapply` calls | 1 keyed `data.table` join + grouped aggregation |
| Total R-level iterations | ~38M | ~0 (vectorized C) |
| Expected wall time | 86+ hours | 5â€“15 minutes |

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a static spatial edge table ONCE from the nb object
#
#   rook_neighbors_unique : spdep nb object (list of integer vectors)
#   id_order              : vector mapping position â†’ cell id
#
#   Result: edges_dt with columns  (id, neighbor_id)
#           ~1,373,394 rows (directed), built in < 1 second
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "0" sentinel that marks cells with no neighbors
  valid    <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each source variable, compute neighbor max/min/mean
#         via a single keyed join + grouped aggregation, then attach
#         the results back to cell_data.
#
#   This replaces build_neighbor_lookup + compute_neighbor_stats +
#   compute_and_add_neighbor_features entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_data, id, year)

for (var in neighbor_source_vars) {

  message("Computing neighbor stats for: ", var)

  # --- 2a. Extract only the columns we need for the neighbor values ---
  #     This is a small subset: (id, year, <var>)
  val_dt <- cell_data[, .(id, year, val = get(var))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # --- 2b. Join neighbor values onto the edge table, by year ----------
  #     For every (id, neighbor_id) edge, we replicate across all years
  #     of the neighbor, then keep only the matching year of the focal cell.
  #
  #     Efficient approach: merge edges with val_dt on neighbor_id,
  #     which gives (id, neighbor_id, year, val). Then aggregate.
  merged <- merge(edges_dt, val_dt, by = "neighbor_id", allow.cartesian = TRUE)
  #     merged has columns: neighbor_id, id, year, val
  #     ~1.37M edges Ã— 28 years = ~38.4M rows (fits in RAM easily)

  # --- 2c. Aggregate: for each (id, year), compute stats over neighbors ---
  stats <- merged[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  by = .(id, year)]
  setkey(stats, id, year)

  # --- 2d. Name the new columns to match the original pipeline ---------
  col_max  <- paste0("neighbor_max_",  var)
  col_min  <- paste0("neighbor_min_",  var)
  col_mean <- paste0("neighbor_mean_", var)
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))

  # --- 2e. Left-join back onto cell_data --------------------------------
  #     Remove old columns if they exist (idempotent re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)

  # Clean up
  rm(val_dt, merged, stats)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the existing trained Random Forest
#         (model object is unchanged; column names are preserved)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adjust to your actual model object and predict call):
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 38.4M-row `merged` is too large)

If the full cartesian merge strains the 16 GB laptop, process **one year at a time** â€” the edge table stays the same, and each year's merge is only ~1.37M rows:

```r
compute_neighbor_stats_by_year <- function(cell_data, edges_dt, var) {

  col_max  <- paste0("neighbor_max_",  var)
  col_min  <- paste0("neighbor_min_",  var)
  col_mean <- paste0("neighbor_mean_", var)

  years <- sort(unique(cell_data$year))

  stats_list <- vector("list", length(years))

  for (i in seq_along(years)) {
    yr <- years[i]

    # Subset this year's values
    yr_vals <- cell_data[year == yr, .(neighbor_id = id, val = get(var))]
    setkey(yr_vals, neighbor_id)

    # Join onto edges
    merged <- yr_vals[edges_dt, on = "neighbor_id", nomatch = 0L]
    # merged columns: neighbor_id, val, id

    # Aggregate
    stats_list[[i]] <- merged[!is.na(val),
                              .(nmax = max(val), nmin = min(val), nmean = mean(val)),
                              by = .(id)][, year := yr]
  }

  stats <- rbindlist(stats_list)
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  setkey(stats, id, year)

  # Remove old columns if present
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  cell_data
}

# Usage:
for (var in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var)
  cell_data <- compute_neighbor_stats_by_year(cell_data, edges_dt, var)
  gc()
}
```

This year-by-year variant peaks at ~1.37M rows per merge iteration (trivial memory) and still completes in minutes because `data.table` keyed joins and grouped aggregations are executed in optimized C.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor structure | Rebuilt per cell-year (6.46M entries) | Static edge table (1.37M rows, built once) |
| Stats computation | R-level `lapply` over 6.46M list elements Ã— 5 vars | Vectorized `data.table` join + group-by Ã— 5 vars |
| Estimated runtime | 86+ hours | 5â€“15 minutes |
| Peak RAM | ~4â€“6 GB (lookup list + data) | ~2â€“3 GB (edge merge + data) |
| Trained RF model | âœ… Preserved, untouched | âœ… Preserved, untouched |
| Numerical results | âœ… Identical max/min/mean | âœ… Identical max/min/mean |
END_RESPONSE>>>
