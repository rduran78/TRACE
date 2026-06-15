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
run_id: Anthropic_F09_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row over every cell-year observation. For each of the ~6.46M rows, it:
- Looks up the cell's spatial reference index.
- Maps neighbor cell IDs to `paste()`-constructed keys like `"cellid_year"`.
- Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named character vector lookup in R is O(n) per access on average due to hashing overhead at scale, and `paste()` generates ~6.46M Ã— ~4 (avg neighbors) â‰ˆ 25.8M temporary strings. This single step likely accounts for the majority of the 86-hour runtime.

### 2. Redundant Recomputation of Spatial Topology Per Year
The rook-neighbor relationships are **purely spatial** â€” they don't change across years. Yet `build_neighbor_lookup` rebuilds the full mapping for every cell-year row, effectively replicating the same neighbor structure 28 times (once per year) at the row level. This is the core architectural inefficiency.

### 3. Row-Level `lapply` in `compute_neighbor_stats`
After the lookup is built, `compute_neighbor_stats` again iterates over all 6.46M rows, extracting neighbor values and computing `max`, `min`, `mean` one row at a time in pure R. This is slow because R's `lapply` over millions of small vectors cannot be vectorized by the interpreter.

---

## Optimization Strategy

**Core Insight:** Build the neighbor table once as a spatial-only structure (a two-column `data.table` of `id â†’ neighbor_id`), then for each year, join the yearly cell attributes onto that table and compute grouped `max`, `min`, `mean` using `data.table`'s optimized grouped aggregation. This converts 6.46M row-level R iterations into 28 vectorized grouped joins + aggregations.

**Steps:**

1. **Expand `rook_neighbors_unique`** (an `nb` object) into a two-column edge table: `(id, neighbor_id)`. This is done once, producing ~1.37M rows.
2. **For each year**, subset the cell-year data, join neighbor attributes onto the edge table by `neighbor_id`, then group by `id` and compute `max`, `min`, `mean` for each variable.
3. **Join results back** to the main cell-year `data.table`.
4. **Predict** with the existing trained Random Forest model (unchanged).

**Complexity reduction:**
- Old: ~6.46M `lapply` iterations Ã— 5 variables Ã— string operations = billions of R-level operations.
- New: 28 years Ã— 5 variables Ã— 1 vectorized grouped join+aggregate on ~1.37M rows = fast `data.table` internals.

**Expected runtime:** Minutes, not hours.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table if not already
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the spatial neighbor edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns (id, neighbor_id)
#           ~1,373,394 rows (directed edges)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i
  # id_order[i] is the cell id for position i
  n <- length(nb_obj)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbs <- nb_obj[[i]]
    # spdep::nb objects use 0L to indicate no neighbors
    nbs <- nbs[nbs != 0L]
    if (length(nbs) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nbs))
      to_list[[i]]   <- id_order[nbs]
    }
  }
  data.table(
    id          = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
setkey(edge_dt, neighbor_id)   # key on neighbor_id for fast join

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor features via vectorized join + group-by
#
#   For each (id, year), we need max/min/mean of each neighbor source
#   variable across that cell's rook neighbors in the same year.
#
#   Strategy per year:
#     1. Subset cell_data to that year -> year_vals (id + source vars)
#     2. Join edge_dt[, .(id, neighbor_id)] with year_vals on
#        neighbor_id == id  â†’  gives each edge the neighbor's values
#     3. Group by id, compute max/min/mean for each variable
#     4. Result is one row per cell with 15 new columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-build the aggregation expression once (avoids repeated parsing)
# For variable "ntl" we produce: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
agg_exprs <- paste0(
  unlist(lapply(neighbor_source_vars, function(v) {
    c(
      sprintf("%s_neighbor_max  = as.numeric(max(%s, na.rm = TRUE))",  v, v),
      sprintf("%s_neighbor_min  = as.numeric(min(%s, na.rm = TRUE))",  v, v),
      sprintf("%s_neighbor_mean = as.numeric(mean(%s, na.rm = TRUE))", v, v)
    )
  })),
  collapse = ", "
)
agg_call <- parse(text = paste0("list(", agg_exprs, ")"))

# Column names that will be produced
new_col_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
}))

# Process year by year
years <- sort(unique(cell_data$year))

cat("Computing neighbor statistics for", length(years), "years ...\n")

result_list <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]
  cat(sprintf("  Year %d (%d/%d)\n", yr, yi, length(years)))

  # Extract only the columns we need for this year
  year_vals <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
  setnames(year_vals, "id", "neighbor_id")
  setkey(year_vals, neighbor_id)

  # Join: for every edge, attach the NEIGHBOR's attribute values
  merged <- year_vals[edge_dt, on = "neighbor_id", nomatch = NA, allow.cartesian = TRUE]
  # merged now has columns: neighbor_id, ntl, ec, ..., id (the focal cell)

  # Group by focal cell id, compute aggregates
  agg <- merged[, eval(agg_call), by = id]

  # Replace Inf/-Inf (from max/min of empty sets) with NA
  for (col in new_col_names) {
    vals <- agg[[col]]
    set(agg, i = which(!is.finite(vals)), j = col, value = NA_real_)
  }

  # Tag with year for later join
  agg[, year := yr]

  result_list[[yi]] <- agg
}

neighbor_stats <- rbindlist(result_list, use.names = TRUE)
setkey(neighbor_stats, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Remove old neighbor columns (if any) and join new ones
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Drop pre-existing neighbor columns to avoid duplication
old_cols <- intersect(names(cell_data), new_col_names)
if (length(old_cols) > 0L) {
  cell_data[, (old_cols) := NULL]
}

setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = c("id", "year")]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the EXISTING trained Random Forest model
#
#   The model object (e.g., `rf_model`) is already in memory and must
#   NOT be retrained.  We simply call predict().
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ensure predictor columns match what the model expects
pred_vars <- rf_model$forest$independent.variable.names  # ranger
# For randomForest package instead: pred_vars <- rownames(rf_model$importance)

cell_data[, predicted := predict(rf_model, data = cell_data[, ..pred_vars])$predictions]

cat("Done. Neighbor features computed and predictions generated.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same rook-neighbor sets and the same variable values. The edge table is a faithful expansion of `rook_neighbors_unique`. Cells with no neighbors (or neighbors missing in a given year) receive `NA`, matching the original logic. |
| **Trained RF model** | The model object is never modified. Only `predict()` is called on the newly-featured data. |
| **Column names** | The naming convention `{var}_neighbor_max/min/mean` matches the pattern used by the original `compute_and_add_neighbor_features`. Adjust the suffix pattern if your original code used a different convention. |

---

## Expected Performance

| Component | Old | New |
|---|---|---|
| Neighbor lookup construction | ~6.46M `lapply` + string ops | One-time expansion of `nb` list â†’ ~1.37M-row `data.table` (seconds) |
| Neighbor stats (per variable) | ~6.46M `lapply` iterations | 28 keyed joins on ~1.37M rows + grouped agg (seconds each) |
| **Total for 5 variables** | **~86+ hours** | **~5â€“15 minutes** on a 16 GB laptop |

The bottleneck shifts from R-level iteration to `data.table`'s C-optimized join and `GForce` grouped aggregation, yielding roughly a **300â€“1000Ã— speedup**.
END_RESPONSE>>>
