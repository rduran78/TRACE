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
run_id: Anthropic_F06_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the cell's spatial neighbor IDs via `id_to_ref` and `neighbors`.
2. Pastes neighbor IDs with the current year to create keys (`paste(..., sep="_")`).
3. Looks up those keys in `idx_lookup` (a named vector of 6.46M elements).

This is repeated 6.46 million times. The `paste` calls and named-vector lookups are extremely slow in base R at this scale. Worse, the spatial neighbor structure is **identical across all 28 years** â€” only the year suffix changes â€” yet it's recomputed from scratch for every row.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` with per-row subsetting

For each of the 5 variables, the function loops over 6.46M rows, subsets the variable vector by neighbor indices, removes NAs, and computes `max`, `min`, `mean`. This produces ~32.3M individual R function calls (5 vars Ã— 6.46M rows), each with overhead from subsetting, `is.na`, and summary statistics.

### Why raster focal/kernel operations are tempting but wrong

Raster focal operations (e.g., `terra::focal`) assume a regular grid with uniform kernel geometry. This panel has an **irregular spdep::nb neighbor structure** (rook contiguity on potentially irregular polygons, with edge/boundary cells having fewer neighbors). A focal approach would either silently change the neighbor definitions or require padding/masking that negates the speed gain. We must **preserve the exact nb object** to preserve the numerical estimand.

### Summary of waste

| Source | Estimated share of 86h |
|---|---|
| 6.46M `paste` + named-vector lookups in `build_neighbor_lookup` | ~40% |
| 6.46M Ã— 5 per-row `lapply` calls in `compute_neighbor_stats` | ~55% |
| Memory pressure / GC | ~5% |

---

## 2. Optimization Strategy

### Strategy A: Vectorize the neighbor lookup via sparse matrix multiplication

Instead of row-by-row subsetting, represent the neighbor structure as a **sparse adjacency matrix W** (344,208 Ã— 344,208). Then for each year-slice, computing `neighbor_mean` is just a sparse matrix-vector product `W %*% x / row_degree`. `neighbor_max` and `neighbor_min` can be computed similarly using grouped operations.

However, sparse matrix multiplication only gives us the **sum** (and thus mean given known row-degrees). Max and min require a different approach.

### Strategy B: Expand to a long edge-list and use `data.table` grouped aggregation

Convert the nb object to an edge data.frame `(from_id, to_id)`. Join with the panel on `(to_id, year)` to get neighbor values. Then group by `(from_id, year)` and compute `max`, `min`, `mean` in one pass.

This is the **optimal approach** because:
- `data.table` grouped aggregation is vectorized in C.
- The join is a single hash-join, not 6.46M individual lookups.
- All three statistics are computed in one grouped call.
- Memory is manageable: the edge-list Ã— 28 years â‰ˆ 1.37M Ã— 28 â‰ˆ 38.4M rows of (from, to, year, value) â€” about 1.2 GB per variable, well within 16 GB.
- **The numerical results are identical** because the same neighbor pairs and the same aggregation functions are used.

### Strategy C: Hybrid â€” sparse matrix for mean, data.table for max/min

Use `Matrix::sparseMatrix` for the mean (fastest possible: one sparse mat-vec per year-variable), and `data.table` for max/min. This is faster but adds complexity.

**I recommend Strategy B (pure `data.table`)** for clarity and maintainability, with expected runtime of **2â€“5 minutes** instead of 86+ hours.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Convert spdep::nb object to a data.table edge list (one-time cost)
# ==============================================================================
# rook_neighbors_unique is an nb object indexed by position in id_order.
# id_order is the vector of cell IDs in the same order as the nb object.

nb_to_edge_dt <- function(nb_obj, id_order) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  
  for (i in seq_along(nb_obj)) {
    nbs <- nb_obj[[i]]
    # spdep uses 0 to denote "no neighbors"
    if (length(nbs) == 1L && nbs[1] == 0L) next
    n <- length(nbs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbs]
    pos <- pos + n
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)
# edges has columns: from_id, to_id
# Each row means: "to_id is a rook neighbor of from_id"

cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edges)))

# ==============================================================================
# STEP 1: Convert cell_data to data.table if not already
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 2: Compute neighbor features for all source variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal lookup table: only id, year, and the source variables
lookup_cols <- c("id", "year", neighbor_source_vars)
lookup_dt <- cell_data[, ..lookup_cols]

# Set key for fast join
setkey(lookup_dt, id, year)

# For each variable, join edges with neighbor values, aggregate, and merge back
for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # Build a small table with just id, year, and the variable of interest
  val_dt <- lookup_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join edges with the *neighbor's* values:
  # For each (from_id, to_id) pair, get all years of to_id
  # We need: for each (from_id, year), the values of all to_id neighbors in that year
  
  # Expand edges by year via join on to_id
  # edge_vals: from_id | to_id | year | val (val is the neighbor's value)
  edge_vals <- merge(
    edges,
    val_dt,
    by.x = "to_id",
    by.y = "id",
    allow.cartesian = TRUE  # each edge Ã— 28 years
  )
  # edge_vals columns: to_id, from_id, year, val
  
  # Aggregate: for each (from_id, year), compute max, min, mean of neighbor vals
  # Remove NAs before aggregation
  edge_vals <- edge_vals[!is.na(val)]
  
  agg <- edge_vals[,
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(from_id, year)
  ]
  
  # Rename columns to match expected output format
  max_col  <- paste0("max_neighbor_", var_name)
  min_col  <- paste0("min_neighbor_", var_name)
  mean_col <- paste0("mean_neighbor_", var_name)
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "from_id", "id")
  
  # Merge back into cell_data
  # First remove these columns if they already exist (idempotent re-runs)
  existing <- intersect(c(max_col, min_col, mean_col), names(cell_data))
  if (length(existing) > 0) {
    cell_data[, (existing) := NULL]
  }
  
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
  
  cat(sprintf("  Done. Added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# ==============================================================================
# STEP 3: Predict with existing Random Forest model (unchanged)
# ==============================================================================
# The trained RF model object is assumed to be loaded (e.g., `rf_model`).
# Prediction proceeds exactly as before â€” the feature columns have identical
# names and identical numerical values.

# Example (uncomment and adapt to your pipeline):
# cell_data$predicted <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete.\n")
```

---

## 4. Why This Preserves the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor pairs** | `nb_to_edge_dt` reads the identical `rook_neighbors_unique` nb object and `id_order`. Every `(from_id, to_id)` pair is preserved. |
| **Same aggregation functions** | `max`, `min`, `mean` are applied to the same sets of non-NA neighbor values per `(from_id, year)` group. |
| **NA handling** | `edge_vals[!is.na(val)]` before aggregation reproduces the `neighbor_vals[!is.na(neighbor_vals)]` filter. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` merge. |
| **No raster resampling** | No focal window or kernel is used. The exact irregular nb topology is preserved. |
| **RF model untouched** | The trained model is loaded and used as-is for prediction. No retraining occurs. |

## 5. Expected Performance

| Step | Estimated Time | Estimated Peak RAM |
|---|---|---|
| `nb_to_edge_dt` | < 1 second | ~20 MB |
| Per-variable `merge` + `aggregate` | ~20â€“40 seconds | ~2â€“3 GB |
| All 5 variables total | **2â€“4 minutes** | ~4 GB peak |
| **Speedup vs. 86 hours** | **~1,500Ã—** | â€” |

The dominant cost is the `merge` (cartesian join of 1.37M edges Ã— 28 years = 38.4M rows) followed by the grouped aggregation, both of which `data.table` executes in optimized C.
END_RESPONSE>>>
