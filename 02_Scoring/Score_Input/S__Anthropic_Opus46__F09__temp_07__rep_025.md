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
run_id: Anthropic_F09_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two fundamental bottlenecks**:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-like row-key construction
- It creates a **named character vector** `idx_lookup` of length ~6.46 million, keyed by `paste(id, year)`.
- Then, for **each of the 6.46 million rows**, it does character-key lookups into that vector via `idx_lookup[neighbor_keys]`.
- Named vector lookup in R is **O(n)** per access (linear scan of names), not O(1). With ~6.46M rows Ã— ~4 neighbors each, this is catastrophically slow.
- The `lapply` over 6.46M rows with per-row `paste` and named-vector subsetting is the primary time sink.

### 2. `compute_neighbor_stats` â€” Repeated R-level loops
- For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` one row at a time.
- This is 5 Ã— 6.46M = ~32.3M R-level function calls with small-vector operations â€” enormous overhead.

### Core Insight
The neighbor **topology** is time-invariant (rook adjacency depends only on spatial grid position). The current code rebuilds the full cell-year neighbor lookup every time, mixing spatial topology with temporal indexing in a single expensive step. This is unnecessary.

---

## Optimization Strategy

**Separate spatial topology from temporal attributes, then use vectorized joins.**

1. **Build a static neighbor edge table once** â€” a simple two-column `data.table` of `(id, neighbor_id)` derived from `rook_neighbors_unique`. This is ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** â€” for each year, join the cell-level attribute onto both the `id` and `neighbor_id` columns. This turns neighbor-stat computation into a grouped `data.table` aggregation: `group by (id, year)`, compute `max`, `min`, `mean` of neighbor values. This is **fully vectorized** and runs in seconds, not hours.

3. **Compute all 5 variables' neighbor stats in one pass per variable** â€” or even batch them.

4. **Join results back** to the main `cell_data` table.

**Expected speedup**: from ~86 hours to **~1â€“5 minutes** total.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial neighbor edge table (done once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique is an spdep::nb object (list of integer vectors).
# id_order is the vector mapping list-position â†’ cell id.

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the indices (into id_order) of cell i's neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (id, neighbor_id)
# This is time-invariant and reusable.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure keyed for fast joins
setkey(cell_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all variables â€” vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need to cross edge_table with all years, then join attributes.
# More memory-efficient: loop over variables, join per variable.

# Get the unique years
all_years <- sort(unique(cell_dt$year))

# Expand edge table Ã— years: each edge exists in every year
# ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB easily
# (2 int id cols + 1 int year col â‰ˆ 38.5M Ã— 12 bytes â‰ˆ 462 MB)

edges_by_year <- CJ(edge_idx = seq_len(nrow(edge_table)), year = all_years)
edges_by_year[, `:=`(
  id          = edge_table$id[edge_idx],
  neighbor_id = edge_table$neighbor_id[edge_idx]
)]
edges_by_year[, edge_idx := NULL]

# Key for joining neighbor attributes
setkey(edges_by_year, neighbor_id, year)

# Function to compute and attach neighbor features for one variable
compute_neighbor_features_fast <- function(cell_dt, edges_by_year, var_name) {

  # Extract only the columns we need for the join
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join neighbor cell's attribute value onto edges
  # edges_by_year is keyed on (neighbor_id, year)
  # attr_dt is keyed on (id, year)
  # We join: for each edge row, look up the neighbor's value
  edges_with_val <- merge(
    edges_by_year,
    attr_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  # Aggregate: for each (id, year), compute max/min/mean of neighbor values
  stats <- edges_with_val[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Rename columns to match expected output format
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  setkey(stats, id, year)
  return(stats)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Loop over variables, compute stats, merge back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  stats_dt <- compute_neighbor_features_fast(cell_dt, edges_by_year, var_name)

  # Merge onto cell_dt (left join â€” cells with no neighbors get NA)
  cell_dt <- merge(cell_dt, stats_dt, by = c("id", "year"), all.x = TRUE)
  rm(stats_dt)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Convert back to data.frame if needed for predict()
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object is preserved exactly as-is.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 16 GB is tight)

If the ~38.5M-row `edges_by_year` table causes memory pressure, process year-by-year:

```r
compute_neighbor_features_by_year <- function(cell_dt, edge_table, var_name) {
  all_years <- sort(unique(cell_dt$year))

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  results <- rbindlist(lapply(all_years, function(yr) {
    # Subset to this year
    yr_dt <- cell_dt[year == yr, .(id, value = get(var_name))]
    setkey(yr_dt, id)

    # Join neighbor values
    edges <- copy(edge_table)
    edges[yr_dt, value := i.value, on = .(neighbor_id = id)]

    # Aggregate
    stats <- edges[
      !is.na(value),
      .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
      by = .(id)
    ]
    stats[, year := yr]
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
    stats
  }))

  setkey(results, id, year)
  return(results)
}

# Usage:
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  stats_dt <- compute_neighbor_features_by_year(cell_dt, edge_table, var_name)
  cell_dt <- merge(cell_dt, stats_dt, by = c("id", "year"), all.x = TRUE)
  rm(stats_dt); gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M character-key lookups (named vector, O(n) each) | Static edge table built once (~1.37M rows, vectorized) |
| **Stat computation** | 5 Ã— 6.46M R-level `lapply` calls | 5 Ã— one vectorized `data.table` grouped aggregation |
| **Estimated time** | ~86+ hours | **~1â€“5 minutes** |
| **RAM peak** | Lower but irrelevant given runtime | ~3â€“6 GB (well within 16 GB) |
| **RF model** | Unchanged | Unchanged â€” no retraining |
| **Numerical output** | `max`, `min`, `mean` of neighbor values | Identical `max`, `min`, `mean` â€” same estimand preserved |
END_RESPONSE>>>
