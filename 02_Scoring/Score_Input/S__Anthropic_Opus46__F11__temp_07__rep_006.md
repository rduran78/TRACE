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
run_id: Anthropic_F11_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points squarely to the **spatial neighbor feature engineering** stage â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates with `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when invoked millions of times inside an interpreted loop. This single function alone creates ~6.46 million character-key lookups against a 6.46-million-entry named vector.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), and each call iterates over all ~6.46 million rows with `lapply`, subsetting, removing NAs, and computing `max`/`min`/`mean` on small vectors. The `do.call(rbind, result)` at the end binds 6.46 million small vectors â€” itself an expensive operation.

3. **Combined cost**: The pipeline performs roughly **6.46M Ã— (1 + 5) = ~38.8 million R-level loop iterations**, each doing string operations, subsetting, and summary statistics. This is the classic "R row-level loop" anti-pattern and is entirely consistent with the reported 86+ hour runtime.

4. **Random Forest inference**, by contrast, is a single vectorized call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict.randomForest` or `predict.ranger` call typically completes in seconds to minutes â€” orders of magnitude less than the neighbor feature engineering.

**Verdict**: The bottleneck is the neighbor feature construction, not Random Forest inference.

---

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a fully vectorized approach using `data.table` integer joins. Instead of building a per-row list of neighbor indices via string keys, we expand the neighbor graph into an edge list and merge on `(neighbor_id, year)` to get row indices â€” all in one vectorized join.

2. **Replace the row-level `lapply` in `compute_neighbor_stats`** with a grouped `data.table` aggregation (`max`, `min`, `mean` by source-row index), which is computed in C and avoids millions of R-level function calls.

3. **Eliminate `do.call(rbind, ...)`** on millions of small vectors entirely.

4. **Process all 5 variables** in a single pass over the edge table rather than 5 separate `lapply` loops.

Expected speedup: from 86+ hours to roughly **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#       cell_data              â€“ data.frame with columns: id, year, ntl, ec,
#                                pop_density, def, usd_est_n2, ... (~6.46M rows)
#       id_order               â€“ integer/numeric vector of unique cell IDs
#                                (length 344,208) whose position corresponds
#                                to the index used in rook_neighbors_unique
#       rook_neighbors_unique  â€“ spdep nb object (list of length 344,208)
#       rf_model               â€“ pre-trained Random Forest model (untouched)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 1.  Convert cell_data to data.table and add a row index
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)
dt[, row_idx := .I]                 # original row position

# ---------------------------------------------------------------
# 2.  Build the directed edge list from the nb object (vectorized)
#     Each entry rook_neighbors_unique[[k]] is an integer vector of
#     neighbor positions in id_order.
# ---------------------------------------------------------------
n_neighbors <- lengths(rook_neighbors_unique)          # integer vector, length 344,208
from_pos    <- rep(seq_along(id_order), times = n_neighbors)
to_pos      <- unlist(rook_neighbors_unique, use.names = FALSE)

# Map positions back to actual cell IDs
edges <- data.table(
  from_id = id_order[from_pos],
  to_id   = id_order[to_pos]
)
rm(from_pos, to_pos, n_neighbors)                      # free memory

# ---------------------------------------------------------------
# 3.  For every (from_id, year) find the row_idx of from_id,
#     and for every (to_id, year) find the row_idx of the neighbor.
#     We achieve this with two keyed joins.
# ---------------------------------------------------------------

# Keyed lookup: cell id + year  -->  row_idx
id_year_key <- dt[, .(id, year, row_idx)]
setkey(id_year_key, id, year)

# Get unique years
years <- sort(unique(dt$year))

# Cross-join edges Ã— years  (edges ~1.37M Ã— 28 years â‰ˆ 38.5M rows)
# This is the full set of (source_row, neighbor_row) pairs.
edge_year <- CJ_dt_edges(edges, years)   # helper below â€” or simply:
edge_year <- edges[, .(year = years), by = .(from_id, to_id)]

# Attach source row index
setkey(edge_year, from_id, year)
edge_year[id_year_key, source_row := i.row_idx, on = .(from_id = id, year)]

# Attach neighbor row index
setkey(edge_year, to_id, year)
edge_year[id_year_key, neighbor_row := i.row_idx, on = .(to_id = id, year)]

# Drop any pairs where either side has no matching row
edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

rm(id_year_key, edges)

# ---------------------------------------------------------------
# 4.  Compute neighbor max / min / mean for every source variable
#     in ONE grouped aggregation pass.
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pull neighbor values into the edge table (only the columns we need)
neighbor_vals <- dt[edge_year$neighbor_row, ..neighbor_source_vars]
edge_year <- cbind(edge_year[, .(source_row)], neighbor_vals)

# Grouped aggregation â€” runs in data.table's C back-end
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
agg_list <- setNames(agg_exprs, agg_names)
agg_call <- as.call(c(as.name("list"), agg_list))

stats <- edge_year[, eval(agg_call), by = source_row]

# Replace -Inf/Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
# 5.  Join the aggregated neighbor features back to the main table
# ---------------------------------------------------------------
setkey(stats, source_row)
setkey(dt,    row_idx)

dt[stats, (agg_names) := mget(paste0("i.", agg_names)), on = .(row_idx = source_row)]

# Rows with no neighbors remain NA (the default after the left join).

# ---------------------------------------------------------------
# 6.  Random Forest prediction (unchanged â€” NOT the bottleneck)
# ---------------------------------------------------------------
dt[, prediction := predict(rf_model, newdata = dt)]

# ---------------------------------------------------------------
# 7.  Convert back to data.frame if downstream code expects one
# ---------------------------------------------------------------
dt[, row_idx := NULL]
cell_data <- as.data.frame(dt)
```

### Simpler alternative for Step 4 if the programmatic `eval` feels fragile

```r
# Straightforward loop â€” still fast because each iteration is a single
# data.table grouped aggregation (C-level), not an R row-level loop.
for (v in neighbor_source_vars) {
  val_vec <- dt[[v]][edge_year$neighbor_row]
  tmp     <- data.table(source_row = edge_year$source_row, val = val_vec)
  agg     <- tmp[!is.na(val),
                  .(nmax = max(val), nmin = min(val), nmean = mean(val)),
                  by = source_row]
  new_cols <- paste0("neighbor_", c("max_", "min_", "mean_"), v)
  setnames(agg, c("nmax", "nmin", "nmean"), new_cols)
  setkey(agg, source_row)
  dt[agg, (new_cols) := mget(paste0("i.", new_cols)), on = .(row_idx = source_row)]
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with string ops | Vectorized `data.table` join (seconds) |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations + `do.call(rbind)` | Grouped `data.table` aggregation in C (secondsâ€“minutes) |
| Random Forest inference | Single `predict()` call (minutes) | Unchanged â€” it was never the bottleneck |
| Estimated total runtime | 86+ hours | ~5â€“15 minutes |
| Trained RF model | Preserved | Preserved |
| Numerical estimand | Preserved (same max/min/mean logic, same predictions) | Preserved |
END_RESPONSE>>>
