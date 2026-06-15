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
run_id: Anthropic_F09_T03_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two compounding bottlenecks**:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-like per-row string hashing
For every one of the ~6.46 million cell-year rows, the function:
- Converts the cell ID to a reference index.
- Looks up neighbor cell IDs from the `nb` object.
- Pastes together `(neighbor_id, year)` string keys.
- Matches those keys against a named character vector of length 6.46 million.

Named-vector lookup in R is **hash-based but with per-call overhead**. Doing ~6.46 million `paste` + named-vector lookups, each touching multiple neighbors, results in billions of character operations. The lookup table itself is a **list of 6.46 million integer vectors**, consuming enormous memory and time to construct.

### 2. `compute_neighbor_stats` â€” Repeated `lapply` over 6.46M elements
For each of the 5 variables, `lapply` iterates over 6.46 million list elements, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is pure R-level looping â€” no vectorization.

### Root Cause
The neighbor topology is **static across years** (344,208 cells, ~1.37M directed neighbor pairs), but the code re-discovers neighbor row indices **per cell-year row** (6.46M rows), entangling spatial structure with temporal structure. This is the core inefficiency.

---

## Optimization Strategy

**Separate spatial topology from temporal attributes, then use vectorized joins.**

1. **Build the neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-independent.

2. **For each variable, join yearly attributes onto the edge table** â€” attach the variable value for each `(neighbor_id, year)` pair. This produces ~1.37M Ã— 28 â‰ˆ ~38.5M rows (but done via a keyed `data.table` equi-join, which is extremely fast).

3. **Aggregate (max, min, mean) by `(cell_id, year)`** â€” a single grouped `data.table` aggregation, fully vectorized in C.

4. **Join the aggregated stats back onto the main dataset.**

This replaces 6.46 million R-level list operations with a handful of `data.table` keyed joins and group-by aggregations. Expected runtime: **minutes, not hours**.

Memory: the edge table Ã— years is ~38.5M rows Ã— a few columns of integers/doubles â€” well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0 â€” Convert the spdep nb object to a static edge table
#          (done ONCE; can be serialized to disk)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # id_order: vector of 344,208 cell IDs (in the order matching nb_obj)
  # nb_obj:   spdep nb list of length 344,208
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep nb encodes "no neighbors" as 0L; skip those
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nbrs])
  }))
  edges
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (cell_id, neighbor_id)

# ---------------------------------------------------------------
# STEP 1 â€” Convert main data to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure key columns are of consistent type
edge_dt[, cell_id     := as.integer(cell_id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]

# ---------------------------------------------------------------
# STEP 2 â€” Function: compute neighbor stats for one variable
# ---------------------------------------------------------------
add_neighbor_features <- function(cell_dt, edge_dt, var_name) {
  # Thin attribute table: just (id, year, value)
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join neighbor attribute values onto the edge table Ã— year

  # First, cross edge_dt with all years present in the data
  years <- sort(unique(attr_dt$year))

  # Expand edges Ã— years
  # Memory: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (manageable)
  edge_year <- CJ_dt(edge_dt, years)

  # Keyed join: attach the neighbor's value for that year
  setkey(edge_year, neighbor_id, year)
  setkey(attr_dt, id, year)
  edge_year[attr_dt, neighbor_val := i.value, on = .(neighbor_id = id, year)]

  # Aggregate by (cell_id, year)
  agg <- edge_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match original naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Join aggregated stats back onto the main table
  setkey(agg, cell_id, year)
  setkey(cell_dt, id, year)
  cell_dt <- agg[cell_dt, on = .(cell_id = id, year)]

  # The join puts cell_id as the key; restore column name to 'id'
  setnames(cell_dt, "cell_id", "id")

  cell_dt
}

# Helper: cross join a data.table with a vector of years
CJ_dt <- function(edge_dt, years) {
  # Repeat each edge for every year â€” vectorized
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  idx     <- rep(seq_len(n_edges), times = n_years)
  yr      <- rep(years, each = n_edges)
  out     <- edge_dt[idx]
  out[, year := yr]
  out
}

# ---------------------------------------------------------------
# STEP 3 â€” Run for all 5 neighbor source variables
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- add_neighbor_features(cell_dt, edge_dt, var_name)
  gc()
}

# ---------------------------------------------------------------
# STEP 4 â€” Predict with the existing trained Random Forest
#           (model object is unchanged)
# ---------------------------------------------------------------
# Convert back to data.frame if the model expects one
cell_data <- as.data.frame(cell_dt)

# The trained RF model (rf_model) is used as-is â€” no retraining
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 38.5M-row expansion is tight on 16 GB)

If RAM is a concern, process one year at a time instead of expanding all years at once:

```r
add_neighbor_features_lowmem <- function(cell_dt, edge_dt, var_name) {
  years    <- sort(unique(cell_dt$year))
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  agg_list <- vector("list", length(years))

  for (j in seq_along(years)) {
    yr <- years[j]
    # Subset attribute for this year only
    attr_yr <- cell_dt[year == yr, .(id, value = get(var_name))]
    setkey(attr_yr, id)

    # Join neighbor values
    tmp <- copy(edge_dt)
    tmp[attr_yr, neighbor_val := i.value, on = .(neighbor_id = id)]

    agg_list[[j]] <- tmp[
      !is.na(neighbor_val),
      .(
        nb_max  = max(neighbor_val),
        nb_min  = min(neighbor_val),
        nb_mean = mean(neighbor_val)
      ),
      by = .(cell_id)
    ][, year := yr]
  }

  agg <- rbindlist(agg_list)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  setkey(agg, cell_id, year)
  setkey(cell_dt, id, year)
  cell_dt <- agg[cell_dt, on = .(cell_id = id, year)]
  setnames(cell_dt, "cell_id", "id")
  cell_dt
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector matches | 1.37M-row static edge table (built once) |
| **Stats computation** | R-level `lapply` over 6.46M list elements Ã— 5 vars | Vectorized `data.table` join + `group-by` Ã— 5 vars |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** |
| **RAM peak** | Large (6.46M-element list of int vectors) | ~2â€“4 GB (38.5M-row intermediate table) |
| **RF model** | Unchanged | Unchanged â€” predict only, no retraining |
| **Numerical results** | Original max/min/mean | Identical â€” same arithmetic, same neighbor topology |
END_RESPONSE>>>
