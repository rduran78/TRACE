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
run_id: Anthropic_F09_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

### 1. `build_neighbor_lookup` is O(nÂ²)-scale in practice
The function iterates over **every cell-year row** (~6.46 million rows), and for each row it:
- Performs character-based key construction (`paste(id, year)`)
- Does named-vector lookups (`idx_lookup[neighbor_keys]`) â€” these are hash lookups but repeated millions of times with string allocation overhead

The critical insight is that **the neighbor topology is static across years**. Cell 42's rook neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the neighbor index mapping for every cell-year combination, doing ~6.46M iterations when only ~344K unique cell relationships exist, duplicated 28 times identically.

### 2. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Each call to `lapply` iterates 6.46M times, subsetting a numeric vector, removing NAs, and computing max/min/mean. This is done 5 times (once per variable), totaling ~32.3 million R-level function calls with per-element overhead.

### 3. No vectorization or data.table/matrix exploitation
Everything is done with base R lists, `lapply`, `paste`, and named vector lookups â€” the slowest possible idiom for this scale.

---

## Optimization Strategy

**Core idea: Build the neighbor table once (cell-level), then join yearly attributes onto it and compute grouped statistics using vectorized `data.table` operations.**

1. **Build a static edge table once** â€” a two-column `data.table` with `(cell_id, neighbor_id)` representing all ~1.37M directed rook-neighbor pairs. This never changes across years.

2. **Cross-join with years** â€” Expand the edge table by year (or, equivalently, join cell-year attributes onto both sides of the edge table by `(neighbor_id, year)`).

3. **Compute grouped aggregates** â€” For each `(cell_id, year)`, compute `max`, `min`, `mean` of each neighbor variable in a single vectorized `data.table` grouped aggregation. This replaces millions of R-level `lapply` calls with a single C-level grouped operation.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes** on the same laptop. The bottleneck shifts from millions of R function calls to a handful of vectorized grouped joins and aggregations over ~38M rows (1.37M edges Ã— 28 years).

**Preservation guarantees:**
- The trained Random Forest model is never touched or retrained.
- The numerical outputs (neighbor max, min, mean) are identical to the original, just computed faster.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0 â€” Convert cell_data to data.table (if not already) and key it
# ===========================================================================
cell_dt <- as.data.table(cell_data)
# Ensure original row order is preserved for downstream RF prediction
cell_dt[, .row_order := .I]

# ===========================================================================
# STEP 1 â€” Build static neighbor edge table ONCE (cell-level, year-agnostic)
#
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object
# ===========================================================================
build_static_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the integer indices (into id_order) of cell i's neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_static_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows and 2 columns: cell_id, neighbor_id
# This is the reusable topology table.

cat(sprintf(
  "Static edge table: %s directed neighbor pairs for %s cells\n",
  format(nrow(edge_dt), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ===========================================================================
# STEP 2 â€” Compute neighbor stats for all variables via vectorized join
#
# Strategy:
#   - For each (cell_id, year) we need max/min/mean of each variable across
#     its rook neighbors' values in that same year.
#   - We join cell_dt attributes onto the neighbor side of edge_dt by
#     (neighbor_id, year), then group by (cell_id, year).
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim lookup table: only the columns we need for the neighbor join
neighbor_cols <- c("id", "year", neighbor_source_vars)
neighbor_attr <- cell_dt[, ..neighbor_cols]
setnames(neighbor_attr, "id", "neighbor_id")
setkey(neighbor_attr, neighbor_id, year)

# Get unique years
years <- sort(unique(cell_dt$year))

# Expand edge table by year: every edge exists in every year
# ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits easily in 16 GB
edge_year_dt <- CJ_dt_year(edge_dt, years)

# Helper: cross join edges with years efficiently
# (We define this inline since CJ from data.table doesn't cross-join two tables directly)
edge_year_dt <- edge_dt[, .(year = years), by = .(cell_id, neighbor_id)]

cat(sprintf(
  "Edge-year table: %s rows (edges Ã— years)\n",
  format(nrow(edge_year_dt), big.mark = ",")
))

# Join neighbor attributes onto the expanded edge table
setkey(edge_year_dt, neighbor_id, year)
edge_year_dt <- neighbor_attr[edge_year_dt, on = .(neighbor_id, year)]

# Now edge_year_dt has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, cell_id

# ===========================================================================
# STEP 3 â€” Grouped aggregation: compute max, min, mean per (cell_id, year)
# ===========================================================================
# Build aggregation expressions programmatically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Execute the grouped aggregation in one pass
neighbor_stats <- edge_year_dt[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(cell_id, year)
]

# --- Alternative cleaner approach if the above bquote method is tricky: ---
# Build it as a single parseable string for robustness:

agg_str_parts <- unlist(lapply(neighbor_source_vars, function(v) {
  c(
    sprintf("neighbor_max_%s  = as.numeric(max(%s, na.rm = TRUE))", v, v),
    sprintf("neighbor_min_%s  = as.numeric(min(%s, na.rm = TRUE))", v, v),
    sprintf("neighbor_mean_%s = mean(%s, na.rm = TRUE)", v, v)
  )
}))

agg_call <- paste0(
  "edge_year_dt[, .(",
  paste(agg_str_parts, collapse = ",\n  "),
  "), by = .(cell_id, year)]"
)

neighbor_stats <- eval(parse(text = agg_call))

# Handle Inf/-Inf from max/min on all-NA groups â†’ set to NA
inf_cols <- grep("neighbor_max_|neighbor_min_", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

cat(sprintf(
  "Neighbor stats computed: %s rows Ã— %s new features\n",
  format(nrow(neighbor_stats), big.mark = ","),
  length(agg_names)
))

# ===========================================================================
# STEP 4 â€” Join neighbor stats back onto cell_dt
# ===========================================================================
setkey(cell_dt, id, year)
setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)

# Remove any old neighbor columns if they exist (from prior slow run)
old_neighbor_cols <- intersect(names(cell_dt), agg_names)
if (length(old_neighbor_cols) > 0) {
  cell_dt[, (old_neighbor_cols) := NULL]
}

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# Restore original row order (important for RF prediction alignment)
setorder(cell_dt, .row_order)
cell_dt[, .row_order := NULL]

# ===========================================================================
# STEP 5 â€” Convert back to data.frame for Random Forest prediction
# ===========================================================================
cell_data <- as.data.frame(cell_dt)

# ===========================================================================
# STEP 6 â€” Predict with the existing trained Random Forest (unchanged)
# ===========================================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (Use exactly as before â€” model is preserved, features are numerically identical)

cat("Done. Neighbor features attached. Ready for RF prediction.\n")
```

---

## Why This Is Equivalent and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm = TRUE` on the identical set of neighbor values produces bit-identical results. The `Inf`/`-Inf` â†’ `NA` handling mirrors the original's `length(neighbor_vals) == 0 â†’ NA` logic. |
| **Trained RF model** | Never touched. We only rebuild the predictor columns in `cell_data` and then call `predict()` with the existing model object. |
| **Neighbor correctness** | The static edge table is built from the same `rook_neighbors_unique` nb object and `id_order` vector. Each edge is replicated per year, and the join on `(neighbor_id, year)` ensures the correct year's attribute is used â€” exactly matching the original `paste(id, year)` key logic. |
| **Memory** | Edge-year table: ~38.5M rows Ã— 7 numeric cols â‰ˆ ~2.2 GB. Neighbor stats: ~6.46M rows Ã— 17 cols â‰ˆ ~0.9 GB. Total peak â‰ˆ ~5â€“6 GB, well within 16 GB. |
| **Speed** | One `data.table` grouped aggregation over ~38.5M rows with 15 summary expressions runs in **1â€“3 minutes** on a modern laptop. The join and edge expansion add another ~1â€“2 minutes. Total: **~2â€“5 minutes** vs. 86+ hours. |

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | Per cell-year (6.46M iterations, string hashing) | Static edge table built once (344K cells) |
| Stats computation | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R calls | Single `data.table` grouped aggregation |
| Time complexity | O(rows Ã— avg_neighbors Ã— vars) with R overhead | O(edges Ã— years) vectorized in C |
| Estimated runtime | **86+ hours** | **~2â€“5 minutes** |
| Numerical result | Identical | Identical |
| RF model | Preserved | Preserved |
END_RESPONSE>>>
