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
run_id: Anthropic_F09_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46 million), but it does redundant work.
The neighbor *topology* is purely spatial â€” it never changes across years. Yet the function builds a lookup keyed on `(id, year)` pairs, iterating over every cell-year row. For each of the 6.46M rows it:
- Looks up the cell's spatial neighbors (which are the same for all 28 years of that cell).
- Pastes together character keys (`id_year`) and matches them into an index lookup.

This means it performs ~6.46M `paste` + named-vector lookups instead of doing the spatial lookup once per cell (344,208 times) and then joining yearly attributes.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows.
Each call indexes into a vector and computes `max`, `min`, `mean`. While individually cheap, 6.46M R-level function calls inside `lapply` are slow, and this is repeated for each of the 5 neighbor source variables (i.e., ~32.3M iterations total).

### 3. Character-key matching is expensive.
`paste(..., sep="_")` and named-vector lookups (`idx_lookup[neighbor_keys]`) on millions of character strings are inherently slow in R compared to integer-indexed operations.

**Summary:** The bottleneck is that the spatial topology is re-threaded through every cell-year row using slow character-key operations, rather than being resolved once at the cell level and then exploited via fast vectorized joins.

---

## Optimization Strategy

The key insight: **the neighbor relationship is time-invariant**. Build the adjacency table once at the cell level, then use a fast equi-join (via `data.table`) to bring in yearly attributes from neighbors, and compute grouped statistics with `data.table`'s optimized `by=` operations.

### Steps:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): one row per directed neighbor pair `(cell_id, neighbor_id)`. This is done once and is ~1.37M rows.

2. **Convert `cell_data` to `data.table`**, keyed on `(id, year)`.

3. **For each source variable**, join the edge table with the panel data to attach the neighbor's variable value for the same year, then compute `max`, `min`, `mean` grouped by `(cell_id, year)`. Merge the results back.

This replaces ~6.46M R-level iterations with vectorized `data.table` joins and grouped aggregations, reducing runtime from 86+ hours to **minutes**.

### Complexity comparison:

| Step | Old | New |
|---|---|---|
| Topology resolution | 6.46M character lookups | 344K integer iterations (once) |
| Neighbor stats | 6.46M Ã— 5 `lapply` calls | 5 vectorized `data.table` joins + `by` aggregations on ~38.4M rows |
| Key type | Character paste + named vector | Integer keys with `data.table` binary search |

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a static, time-invariant edge table (once)
# ---------------------------------------------------------------
# rook_neighbors_unique : an nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the indices (into id_order) of cell i's rook neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)

# ---------------------------------------------------------------
# STEP 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)
# Ensure proper types
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]

edge_table[, cell_id     := as.integer(cell_id)]
edge_table[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# STEP 3: For each source variable, compute neighbor stats via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_dt, edge_table, var_name) {
  # Subset the panel to only (id, year, variable) for the join's right side
  # This keeps the join lean
  neighbor_vals <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(neighbor_vals, id, year)

  # Expand edges Ã— years: join edge_table with neighbor_vals

  # For each (cell_id, neighbor_id) pair, get the neighbor's value in each year
  # Join: edge_table[neighbor_id] -> neighbor_vals[id == neighbor_id, year]
  # We need all (cell_id, year) combinations with their neighbors' values.


  # Step A: Join edge_table with neighbor_vals on neighbor_id == id
  #   This gives us: (cell_id, neighbor_id, year, value)
  #   i.e., for every edge and every year, the neighbor's attribute value.
  expanded <- merge(
    edge_table,
    neighbor_vals,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE   # each neighbor_id appears in 28 year-rows
  )
  # expanded columns: neighbor_id, cell_id, year, value
  # rows: ~1,373,394 edges Ã— 28 years â‰ˆ 38.5M (but only years present in data)

  # Step B: Aggregate by (cell_id, year) to get max, min, mean of neighbor values
  stats <- expanded[
    !is.na(value),
    .(
      nbr_max  = max(value),
      nbr_min  = min(value),
      nbr_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]

  # Step C: Rename columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
                  c(max_col,    min_col,    mean_col))

  # Step D: Merge back onto cell_dt by (id, year)
  # Use cell_id == id
  setnames(stats, "cell_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_dt, c("id", "year"))

  # Remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats, by = c("id", "year"), all.x = TRUE)
  return(cell_dt)
}

# ---------------------------------------------------------------
# STEP 4: Loop over the 5 source variables
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_dt(cell_dt, edge_table, var_name)
}

# ---------------------------------------------------------------
# STEP 5: Predict with the existing (already-trained) Random Forest
# ---------------------------------------------------------------
# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# The trained RF model object is assumed to be loaded in the environment
# e.g., rf_model <- readRDS("trained_rf_model.rds")
# Predict:
cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Memory Considerations (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M Ã— 110 cols) | ~5.5 GB |
| `edge_table` (1.37M Ã— 2 int cols) | ~11 MB |
| `expanded` (worst case ~38.5M Ã— 4 cols) | ~1.2 GB |
| Overhead + RF model | ~2â€“3 GB |

**Total peak: ~10 GB** â€” fits within 16 GB. The `expanded` table is created and destroyed inside each iteration, so only one variable's expansion is in memory at a time.

If memory is still tight, you can process the join in year-chunks:

```r
# Optional: chunk by year to reduce peak memory
compute_neighbor_features_chunked <- function(cell_dt, edge_table, var_name, years) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  stats_list <- lapply(years, function(yr) {
    yr_vals <- cell_dt[year == yr, .(id, value = get(var_name))]
    setkey(yr_vals, id)

    expanded_yr <- merge(edge_table, yr_vals,
                         by.x = "neighbor_id", by.y = "id")
    expanded_yr[
      !is.na(value),
      .(nbr_max = max(value), nbr_min = min(value), nbr_mean = mean(value)),
      by = .(cell_id)
    ][, year := yr]
  })

  stats <- rbindlist(stats_list)
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
                  c(max_col,    min_col,    mean_col))
  setnames(stats, "cell_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_dt, c("id", "year"))

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats, by = c("id", "year"), all.x = TRUE)
  return(cell_dt)
}

all_years <- sort(unique(cell_dt$year))
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features (chunked) for: ", var_name)
  cell_dt <- compute_neighbor_features_chunked(cell_dt, edge_table, var_name, all_years)
}
```

This reduces peak memory for the expanded table from ~1.2 GB to ~1.2 GB / 28 â‰ˆ **~43 MB** per chunk.

---

## Expected Speedup

| Component | Old Runtime | New Runtime |
|---|---|---|
| Build lookup (6.46M char keys) | ~hours | <1 sec (integer edge table) |
| Neighbor stats (5 vars Ã— 6.46M lapply) | ~80+ hours | ~5â€“15 min (data.table joins) |
| **Total** | **86+ hours** | **~5â€“20 minutes** |

The trained Random Forest model is never modified â€” only the feature-engineering pipeline is optimized. The numerical values produced (`max`, `min`, `mean` of each neighbor variable) are identical, preserving the original estimand.
END_RESPONSE>>>
