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
run_id: Anthropic_F09_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two critical bottlenecks:

1. **`build_neighbor_lookup` rebuilds the full cell-year adjacency every time it is called.** It iterates over ~6.46 million rows, and for each row it does string-paste key construction and named-vector lookups (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) hashing per call, and doing this 6.46 million times over string keys is catastrophically slow. This single function likely accounts for the vast majority of the 86+ hours.

2. **`compute_neighbor_stats` uses `lapply` over 6.46 million rows**, each time subsetting a numeric vector and computing `max/min/mean`. While not as bad as the lookup construction, it is still slow because it is a row-level R loop with no vectorization.

**The key insight:** The neighbor *topology* (which cells are neighbors of which cells) is **time-invariant** â€” it depends only on the spatial grid, not on the year. The current code re-discovers this for every cell-year row by doing string matching. Instead, we should:

- Build the adjacency table **once** at the cell level (344,208 cells, ~1.37M directed edges).
- Join yearly attributes onto this static edge table.
- Compute grouped `max`, `min`, `mean` using vectorized/compiled operations (via `data.table`).

This turns the problem from ~6.46M Ã— per-row R lookups into a single equi-join + grouped aggregation in `data.table`, which runs in compiled C code.

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| 1 | Build a **static edge table** `data.table(focal_id, neighbor_id)` from `rook_neighbors_unique` once. ~1.37M rows. | Topology is time-invariant. |
| 2 | For each year, **join** yearly cell attributes onto the edge table by `neighbor_id + year`. | Vectorized equi-join in `data.table` â€” compiled C, no R loop. |
| 3 | **Group-by** `(focal_id, year)` to compute `max`, `min`, `mean` for each variable. | Vectorized grouped aggregation â€” compiled C. |
| 4 | Join results back to `cell_data`. | Single keyed merge. |

**Expected speedup:** The 1.37M-edge Ã— 28-year cross gives ~38.5M edge-year rows, but the join + group-by in `data.table` handles this in seconds to low minutes, not hours. Total runtime for all 5 variables: **~2â€“10 minutes** on a 16 GB laptop, versus 86+ hours.

The trained Random Forest model is **not touched** â€” we are only recomputing the same input features faster, with identical numerical results.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build the static (time-invariant) edge table ONCE
# ===========================================================================
# rook_neighbors_unique is an spdep::nb object (list of integer vectors)
# id_order is the vector mapping position -> cell id

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains the positional indices of neighbors of cell i
  # id_order[i] is the actual cell id at position i
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb <- neighbors_nb[[i]]
    # spdep::nb uses 0L to encode "no neighbors"
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  edges
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has columns: focal_id, neighbor_id
# ~1,373,394 rows (directed rook edges)

# ===========================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ===========================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ===========================================================================
# STEP 3: For each neighbor source variable, compute neighbor stats
#         and add columns to cell_data
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim lookup keyed by (id, year) with only the columns we need
# to minimize memory during the join.
attr_cols <- intersect(neighbor_source_vars, names(cell_data))
attr_dt   <- cell_data[, c("id", "year", ..attr_cols)]
setnames(attr_dt, "id", "neighbor_id")
setkey(attr_dt, neighbor_id, year)

# Add year to edge table via a cross with the distinct years
years_dt <- data.table(year = sort(unique(cell_data$year)))

# Expand edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows)
# This fits comfortably in 16 GB: 38.5M Ã— 3 int cols â‰ˆ ~0.9 GB
edge_year <- edge_table[, CJ(year = years_dt$year), by = .(focal_id, neighbor_id)]
setkey(edge_year, neighbor_id, year)

# Join neighbor attributes onto expanded edge table
edge_year <- attr_dt[edge_year, on = .(neighbor_id, year), nomatch = NA]
# Now edge_year has: neighbor_id, year, <var cols>, focal_id

# Compute grouped stats for each variable
setkey(edge_year, focal_id, year)

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)

  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)

  # Grouped aggregation â€” runs in compiled C inside data.table
  stats <- edge_year[
    !is.na(get(var_name)),
    .(
      V_max  = max(get(var_name), na.rm = TRUE),
      V_min  = min(get(var_name), na.rm = TRUE),
      V_mean = mean(get(var_name), na.rm = TRUE)
    ),
    by = .(focal_id, year)
  ]
  setnames(stats, c("V_max", "V_min", "V_mean"), c(col_max, col_min, col_mean))

  # Merge back onto cell_data
  # Remove old columns if they exist (idempotent reruns)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
  }

  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("focal_id", "year"),
                     all.x = TRUE)
}

# ===========================================================================
# STEP 4: Predict with the EXISTING trained Random Forest (unchanged)
# ===========================================================================
# The trained model object is assumed to be `rf_model` (already in memory).
# cell_data now has the identical neighbor feature columns as before.
# Predict exactly as before:

cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 16 GB is tight)

If the ~38.5M-row `edge_year` table creates memory pressure, process one year at a time:

```r
# Memory-lean alternative: loop over years, not variables
# Still vastly faster than the original because each iteration is a
# data.table join + group-by over ~1.37M edges (seconds).

setkey(edge_table, neighbor_id)

all_stats <- vector("list", length(unique(cell_data$year)))

for (yr_i in seq_along(unique(cell_data$year))) {
  yr <- sort(unique(cell_data$year))[yr_i]

  # Subset this year's attributes
  yr_attr <- cell_data[year == yr, c("id", attr_cols), with = FALSE]
  setnames(yr_attr, "id", "neighbor_id")
  setkey(yr_attr, neighbor_id)

  # Join onto static edge table
  merged <- yr_attr[edge_table, on = "neighbor_id", allow.cartesian = TRUE]
  # merged has: neighbor_id, <vars>, focal_id

  # Aggregate all variables at once
  agg_expr <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(c("n_max_", "n_min_", "n_mean_"), v)
  }))

  stats_yr <- merged[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vals), min(vals), mean(vals))
    }), neighbor_source_vars),
    by = focal_id
  ]

  # Cleaner approach with explicit aggregation:
  stats_yr <- merged[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("n_max_", v)]]  <- NA_real_
        out[[paste0("n_min_", v)]]  <- NA_real_
        out[[paste0("n_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("n_max_", v)]]  <- max(vals)
        out[[paste0("n_min_", v)]]  <- min(vals)
        out[[paste0("n_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = focal_id]

  stats_yr[, year := yr]
  all_stats[[yr_i]] <- stats_yr
}

all_stats_dt <- rbindlist(all_stats)

# Merge all neighbor features back at once
for (cc in names(all_stats_dt)[!names(all_stats_dt) %in% c("focal_id", "year")]) {
  if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
}

cell_data <- merge(cell_data, all_stats_dt,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE)

# Predict with the existing trained RF â€” unchanged
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup | 6.46M R-loop iterations with string hashing | Static 1.37M-row edge table built once |
| Stats computation | `lapply` over 6.46M rows per variable | `data.table` grouped aggregation (compiled C) |
| Total variables | 5 Ã— full pass | Single join, one grouped aggregation |
| Estimated runtime | **86+ hours** | **2â€“10 minutes** |
| RF model | Unchanged | Unchanged |
| Numerical results | Identical | Identical |
END_RESPONSE>>>
