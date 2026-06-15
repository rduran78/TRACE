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
run_id: Anthropic_F09_T00_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Performs named-vector lookups via `idx_lookup[neighbor_keys]` â€” which is a **hash-table lookup on character keys repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8 million times**.

The real killer is that the neighbor topology is **year-invariant** â€” the same cell has the same rook neighbors in every year â€” yet the function rebuilds the mapping for every cell-year combination. With 344,208 cells Ã— 28 years, it does 28Ã— redundant work on the spatial graph.

### 2. `compute_neighbor_stats` uses per-row `lapply`
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46M rows with R-level loops is inherently slow. Each iteration incurs R interpreter overhead, subsetting, and `NA` handling.

### 3. Memory pressure from the lookup object
Storing 6.46 million integer vectors in a list consumes substantial RAM and creates GC pressure on a 16 GB laptop.

---

## Optimization Strategy

**Core insight:** The spatial neighbor structure is static across years. Build it once (344K cells), then use vectorized joins per year.

### Step-by-step plan:

1. **Build a cell-level edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and is tiny.

2. **For each variable, join yearly attributes onto the edge table** â€” use `data.table` keyed joins. For each cell-year, this gives all neighbor values in a flat table.

3. **Aggregate with `data.table` grouping** â€” compute `max`, `min`, `mean` per `(cell_id, year)` in one vectorized pass. This replaces millions of R-level `lapply` iterations with C-level `data.table` grouped aggregation.

4. **Join the aggregated stats back** onto the main dataset.

**Expected speedup:** From ~86 hours to **minutes**. The bottleneck becomes a handful of keyed `data.table` joins and grouped aggregations over ~25.8M edge-year rows â€” all executed in C.

**Memory:** The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The expanded edge-year table is ~1.37M Ã— 28 â‰ˆ 38.4M rows Ã— 3 columns â‰ˆ 920 MB at peak, which fits in 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static cell-level edge table ONCE from the nb object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # id_order is the vector of cell IDs corresponding to nb_obj positions
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]

  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, built in < 1 second

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure key columns have consistent types
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]
edge_dt[, cell_id     := as.integer(cell_id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Function to compute and attach neighbor stats for one variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  # --- 3a. Extract only the columns we need for the join ---
  # Columns: neighbor_id (to join on), year, and the variable value
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # --- 3b. Expand edges Ã— years by joining neighbor attributes ---
  # For every (cell_id, neighbor_id) edge, and every year,

  # look up the neighbor's value in that year.
  # We join edge_dt to val_dt on neighbor_id == id.
  # First, create the cross of edges Ã— years present in the data.
  # Efficient approach: join edge_dt with val_dt on neighbor_id == id.
  # This naturally expands to all (cell_id, neighbor_id, year) combos
  # where the neighbor has data.

  edge_vals <- merge(
    edge_dt,
    val_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE   # each neighbor appears in 28 years
  )
  # Result columns: neighbor_id, cell_id, year, val
  # Rows: ~1.37M edges Ã— 28 years â‰ˆ 38.4M (manageable)

  # --- 3c. Aggregate: max, min, mean per (cell_id, year) ---
  agg <- edge_vals[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(cell_id, year)
  ]

  # --- 3d. Rename columns to match original naming convention ---
  # Original code produces columns like: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # --- 3e. Join back onto the main table ---
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(
    cell_dt,
    agg,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  cell_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Loop over the 5 neighbor source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_and_add_neighbor_features_fast(cell_dt, edge_dt, var_name)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convert back to data.frame if the RF predict method requires it
cell_data <- as.data.frame(cell_dt)

# The trained model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Trained RF model** | Untouched â€” only the feature-engineering step is optimized. |
| **Numerical estimand** | Preserved â€” `max`, `min`, `mean` are computed on the identical neighbor sets with the identical values. The `merge` on `(neighbor_id, year)` replicates exactly the same cell-year neighbor lookups as the original `paste()`-key approach. |
| **RAM on 16 GB laptop** | Peak â‰ˆ main table (~3â€“4 GB) + largest edge-year expansion (~1 GB) + aggregation (~0.3 GB). Fits comfortably. `gc()` after each variable reclaims intermediates. |
| **Runtime** | ~2â€“5 minutes total for all 5 variables (vs. 86+ hours). Each variable involves one `data.table` merge (~38M rows) and one grouped aggregation â€” both executed in C. |
END_RESPONSE>>>
