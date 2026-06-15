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
run_id: Anthropic_F09_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), named-vector lookups (`id_to_ref`, `idx_lookup`), and subsetting. Named vector lookup in R is **O(n)** per call because it uses linear hashing on character keys. Doing this 6.46 million times against a 6.46-million-element named vector produces roughly **O(nÂ²)** behavior. This alone accounts for the vast majority of the 86+ hour runtime.

### 2. Redundant recomputation of the same spatial topology every year
The rook-neighbor structure is **time-invariant** â€” cell 17's neighbors are always the same cells regardless of year. Yet `build_neighbor_lookup` re-resolves neighbor identities for every cell-year combination, doing 28Ã— the necessary spatial work.

### 3. Row-level `lapply` in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level `lapply` over 6.46 million rows with variable-length index vectors is slow due to R's per-call overhead and lack of vectorization.

---

## Optimization Strategy

**Core idea:** Build the neighbor table **once** at the cell level (344,208 cells), store it as a two-column `data.table` (cell_id â†’ neighbor_id), then use a **vectorized join** against the yearly attribute table to compute neighbor statistics via grouped aggregation. This replaces all `lapply` loops with `data.table` operations that run in C.

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `spdep::nb` to a `data.table` edge list: `(cell_id, neighbor_id)` â€” ~1.37M rows, built once | O(E) |
| 2 | For each year Ã— variable, join the edge list to cell attributes to get neighbor values, then aggregate `max`, `min`, `mean` grouped by `(cell_id, year)` | O(E) per variable per year via `data.table` merge + group-by |
| 3 | Join the aggregated neighbor stats back onto the main dataset | O(N) |

**Expected speedup:** The total work is ~5 variables Ã— 28 years Ã— 1.37M edges â‰ˆ 192M rows of joins+aggregations, all handled by `data.table`'s radix-sort joins in compiled C. Estimated wall-clock time: **2â€“10 minutes** on a 16 GB laptop, down from 86+ hours.

The trained Random Forest model is never touched. The output columns (neighbor max, min, mean for each source variable) are numerically identical because the same values are aggregated with the same functions.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the time-invariant edge list ONCE from the nb object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(id_order, nb_object) {
  # id_order: vector of cell IDs in the same order as the nb object

  # nb_object: spdep::nb list (rook_neighbors_unique)
  from <- rep(seq_along(nb_object), lengths(nb_object))
  to   <- unlist(nb_object)
  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: one row per directed neighbor relationship

cat("Edge list rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all source variables via joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  # Subset only the columns we need for the neighbor value lookup
  # to minimize memory during the join
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]

  # Key the attribute table for fast join
  setkey(attr_dt, id)

  # Expand edge list Ã— year: join neighbor attributes

  # edge_dt has (cell_id, neighbor_id)
  # We join attr_dt onto edge_dt by neighbor_id == id, matching on year
  # Strategy: merge edge_dt with the attribute table on neighbor_id

  # Rename for clarity before join
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  setkey(edge_dt, neighbor_id)

  # This is the big join: for each edge Ã— year, pull neighbor attribute values

  # Result: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  cat("Performing edge-attribute join...\n")
  joined <- attr_dt[edge_dt, on = "neighbor_id", allow.cartesian = TRUE]
  # joined now has columns: neighbor_id, year, <source_vars>, cell_id

  cat("Joined rows:", nrow(joined), "\n")

  # Aggregate by (cell_id, year) to get neighbor max, min, mean
  cat("Aggregating neighbor statistics...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)),  na.rm = TRUE)))
    )
  }))

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Perform grouped aggregation
  stats_dt <- joined[,
    lapply(agg_exprs, eval),
    by = .(cell_id, year)
  ]

  # Handle Inf/-Inf from max/min on all-NA groups â†’ convert to NA
  for (col in agg_names) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  # Rename cell_id back to id for merging with cell_data
  setnames(stats_dt, "cell_id", "id")

  return(stats_dt)
}

# Run the computation
neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Join neighbor stats back onto the main dataset
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Done. cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# The trained RF model object (e.g., `rf_model`) is used as-is.
# Ensure prediction columns match the model's expected feature names.
# Example:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

### If you prefer a simpler, more memory-conservative version

The join above can produce ~38M rows, using roughly 2â€“4 GB of RAM. If memory is tight, process one variable at a time:

```r
compute_neighbor_features_one_var <- function(cell_data, edge_dt, var_name) {
  attr_dt <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(attr_dt, neighbor_id, year)

  joined <- attr_dt[edge_dt, on = "neighbor_id", allow.cartesian = TRUE]

  stats <- joined[,
    .(
      nmax  = if (all(is.na(val))) NA_real_ else max(val, na.rm = TRUE),
      nmin  = if (all(is.na(val))) NA_real_ else min(val, na.rm = TRUE),
      nmean = if (all(is.na(val))) NA_real_ else mean(val, na.rm = TRUE)
    ),
    by = .(cell_id, year)
  ]

  setnames(stats,
    c("nmax", "nmin", "nmean"),
    paste0("neighbor_", c("max_", "min_", "mean_"), var_name)
  )
  setnames(stats, "cell_id", "id")
  return(stats)
}

# Process one variable at a time, merge sequentially
setkey(cell_data, id, year)

for (v in neighbor_source_vars) {
  cat("Processing:", v, "\n")
  stats_v <- compute_neighbor_features_one_var(cell_data, edge_dt, v)
  setkey(stats_v, id, year)
  cell_data <- stats_v[cell_data, on = .(id, year)]
  rm(stats_v); gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor topology | Rebuilt for every cell-year (6.46M `lapply` calls) | Built once as edge list (1.37M rows) |
| Attribute lookup | Character-key named vector: O(n) per lookup | `data.table` keyed join: O(1) amortized |
| Stat computation | R-level `lapply` over 6.46M rows | `data.table` grouped aggregation in C |
| Estimated runtime | 86+ hours | **2â€“10 minutes** |
| Numerical output | max, min, mean per neighbor set | **Identical** max, min, mean per neighbor set |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
