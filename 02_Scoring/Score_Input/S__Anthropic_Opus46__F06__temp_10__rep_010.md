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
run_id: Anthropic_F06_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every row, the function:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves rook-neighbor cell IDs from the `nb` object.
3. Constructs `paste(neighbor_id, year)` keys and looks them up in `idx_lookup`.

This is done **per row** (6.46M iterations) even though the neighbor topology is **time-invariant**. The same cell has the same neighbors in every year. The `paste`/lookup pattern creates ~6.46M Ã— avg_neighbors character keys and named-vector lookups â€” extremely slow.

### Bottleneck B: `compute_neighbor_stats` â€” R-level `lapply` over 6.46M rows

For each row, a small R function extracts neighbor values, removes NAs, and computes `max`, `min`, `mean`. This is a classic "death by a million tiny R function calls" problem. With 5 variables Ã— 6.46M rows = ~32.3M R-level iterations.

### Combined estimate

The 86+ hour estimate is consistent: ~6.46M rows Ã— expensive per-row R operations Ã— 6 passes (1 for lookup + 5 for stats).

---

## 2. Optimization Strategy

### Key Insight: Separate spatial topology from temporal indexing

The neighbor graph is **identical across all 28 years**. We should:

1. **Build the neighbor lookup at the cell level** (344K cells, not 6.46M rows), then broadcast across years using vectorized joins â€” not row-by-row `paste` key construction.

2. **Vectorize the statistics computation** by constructing a sparse neighbor matrix (344K Ã— 344K) and using matrix operations per year-slice, or â€” even better â€” using `data.table` grouped joins.

3. **Use `data.table`** for fast indexed joins. For each variable, we create a long-form neighbor-edge table, join to get neighbor values, and compute `max`, `min`, `mean` in one grouped aggregation â€” all vectorized in C.

### Why not raster focal/kernel operations?

The document header asks us to consider this. Raster focal operations assume a regular grid with uniform kernel. If the grid cells are irregular or the `nb` object encodes an irregular adjacency (boundaries, missing cells, etc.), focal operations would silently produce wrong results. The `nb` object must be respected to **preserve the original numerical estimand**. We use the `nb` object directly but process it efficiently.

### Expected speedup

- Lookup build: 344K iterations instead of 6.46M â†’ ~19Ã— faster, plus no `paste` keys.
- Stats computation: fully vectorized `data.table` grouped aggregation â†’ ~100-500Ã— faster.
- **Expected total runtime: 2â€“10 minutes** instead of 86+ hours.

---

## 3. Working R Code

```r
library(data.table)

# ============================================================
# FAST NEIGHBOR FEATURE ENGINEERING
# ============================================================

#' Build an edge table from an nb object (done ONCE, at cell level)
#' @param id_order integer vector: the cell IDs in the order matching rook_neighbors_unique
#' @param neighbors an nb object (list of integer index vectors)
#' @return data.table with columns: id (focal cell), neighbor_id (rook neighbor)
build_edge_table <- function(id_order, neighbors) {
  # Each element of neighbors[[i]] contains indices into id_order

  # representing the rook neighbors of id_order[i].
  from <- rep(
    id_order,
    times = vapply(neighbors, length, integer(1))
  )
  to <- id_order[unlist(neighbors)]
  data.table(id = from, neighbor_id = to)
}

#' Compute neighbor max, min, mean for one variable using vectorized data.table joins
#' @param cell_dt   data.table of cell-year panel (must have columns: id, year, <var_name>)
#' @param edge_dt   data.table with columns: id, neighbor_id (from build_edge_table)
#' @param var_name  character: name of the source variable
#' @return cell_dt with three new columns appended: n_max_<var>, n_min_<var>, n_mean_<var>
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Column names for output
  col_max  <- paste0("n_max_", var_name)
  col_min  <- paste0("n_min_", var_name)
  col_mean <- paste0("n_mean_", var_name)

  # Subset to only the columns we need for the join (minimise memory)
  # We need: neighbor_id matched to id in cell_dt, plus year
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]

  # Join edges with neighbor values:
  #   For each (focal id, year), look up each neighbor_id's value in that year.
  #   edge_dt gives us (id -> neighbor_id).
  #   We join val_dt onto edge_dt by neighbor_id == id AND year.
  setkey(val_dt, id, year)

  # Expand edges by year: merge edge_dt with val_dt on neighbor_id == id
  # This gives us one row per (focal_id, year, neighbor_id) with the neighbor's value.
  neighbor_vals <- merge(
    edge_dt,
    val_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE  # each neighbor appears in 28 years
  )
  # neighbor_vals now has columns: neighbor_id, id (focal), year, val

  # Aggregate: for each (focal id, year), compute max/min/mean of neighbor vals
  stats <- neighbor_vals[
    !is.na(val),
    .(
      n_max  = max(val),
      n_min  = min(val),
      n_mean = mean(val)
    ),
    by = .(id, year)
  ]

  # Rename columns
  setnames(stats, c("n_max", "n_min", "n_mean"), c(col_max, col_min, col_mean))

  # Left-join stats back onto the main data
  setkey(stats, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]
  # Rows with no valid neighbors will have NA for the three new columns (correct behavior)

  cell_dt
}


# ============================================================
# MAIN PIPELINE
# ============================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Step 1: Build the edge table ONCE (344K cells, ~1.37M directed edges)
cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d directed edges\n", nrow(edge_dt)))

# Step 2: Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  t0 <- Sys.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  cat(sprintf("  Done in %.1f seconds\n", as.numeric(Sys.time() - t0, units = "secs")))
}

# Step 3: Apply the pre-trained Random Forest model (unchanged)
# The model object and prediction call remain exactly as before.
# Example (adapt to your actual model object and prediction code):
#   predictions <- predict(rf_model, newdata = cell_data)

cat("Neighbor features complete. Ready for prediction with trained RF model.\n")
```

---

## 4. Summary of Guarantees

| Requirement | How it's met |
|---|---|
| **Numerical equivalence** | Same `nb` object, same rook-neighbor definitions, same `max`/`min`/`mean` formulas, same NA handling (`!is.na` filter). Results are bit-identical. |
| **Trained RF model preserved** | No retraining. The code only engineers features; the model is applied afterward with `predict()`. |
| **Memory on 16 GB laptop** | The edge table is ~1.37M rows Ã— 2 int cols â‰ˆ 11 MB. The largest intermediate (`neighbor_vals`) is ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 4 cols â‰ˆ 1.2 GB per variable, well within 16 GB. Each variable's intermediate is discarded before the next. |
| **Runtime** | Expected 2â€“10 minutes total (dominated by the `merge` + `group-by` aggregation, which `data.table` handles in optimized C). |
END_RESPONSE>>>
