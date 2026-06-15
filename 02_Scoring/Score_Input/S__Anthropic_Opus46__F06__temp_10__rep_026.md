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
run_id: Anthropic_F06_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is overwhelmingly in **`build_neighbor_lookup`**, which constructs a list of ~6.46 million elements, each built via an `lapply` iteration that performs character coercion, hash-table lookups (`setNames`/named-vector indexing), and `paste` operations **per row**. This means:

- **~6.46 million iterations** of `lapply`, each doing string concatenation, named-vector lookups, and `NA` filtering.
- The `id_to_ref` and `idx_lookup` named vectors are themselves large (~6.46M entries for `idx_lookup`), so each `[` lookup scans a hash table at scale, millions of times.
- `compute_neighbor_stats` then does another `lapply` over 6.46M elements **per variable** (Ã—5 variables = ~32.3M iterations), each extracting subsets from a numeric vector.

The raster/focal-kernel analogy is apt conceptually (each cell's neighborhood statistics are a "focal" operation), but here the geometry is an **irregular spatial grid with a precomputed `nb` object**, not a regular raster. So a true `terra::focal()` replacement is not directly applicable without confirming the grid is perfectly regular and re-encoding the data as a raster. The implementation below therefore stays in the tabular/vector domain but eliminates the R-level loops.

**Root causes of the 86+ hour runtime:**

| Component | Problem |
|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops and named-vector lookups |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M R-level iterations with subsetting |
| Memory | Storing a 6.46M-element list of integer vectors is allocation-heavy |

---

## 2. Optimization Strategy

### A. Replace `build_neighbor_lookup` with a vectorized `data.table` join

Instead of iterating row-by-row, we:

1. Expand the `nb` object into an **edge list** (cell_id â†’ neighbor_cell_id), which has ~1.37M directed edges.
2. Cross-join this edge list with the 28 years to get ~38.4M (edge Ã— year) rows.
3. Join against the data to get the **row index** of each neighbor in each year.

This replaces 6.46 million `lapply` iterations with a single vectorized merge.

### B. Replace `compute_neighbor_stats` with a `data.table` grouped aggregation

Instead of iterating over 6.46M list elements per variable, we:

1. Use the edge-list-with-row-indices from step A.
2. For each source variable, extract the neighbor values via vectorized indexing.
3. Compute `max`, `min`, `mean` in a single `data.table` grouped-by aggregation (`by = row_idx`).

This replaces all 32.3M R-level iterations with 5 vectorized `data.table` group-by operations.

### C. Memory management

- The edge list Ã— year table is ~38.4M rows Ã— 3 integer columns â‰ˆ ~440 MB, feasible within 16 GB.
- We avoid materializing a 6.46M-element list entirely.

**Expected speedup:** From 86+ hours to roughly **5â€“15 minutes**.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# INPUTS (assumed to already exist in the environment):
#   cell_data              : data.frame/data.table with columns id, year,
#                            ntl, ec, pop_density, def, usd_est_n2, ...
#   rook_neighbors_unique  : nb object (list of integer index vectors)
#   id_order               : integer vector; id_order[i] = cell id of
#                            the i-th element in the nb object
#   rf_model               : pre-trained Random Forest model (untouched)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already; add a row index column
setDT(cell_data)
cell_data[, .row_idx := .I]

# â”€â”€ Step 1: Build a directed edge list from the nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    nb object: rook_neighbors_unique[[i]] gives integer indices of
#    neighbors of the i-th cell; id_order[i] maps to cell id.
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_indices <- rook_neighbors_unique[[i]]
  if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id    = id_order[i],
             neighbor_id = id_order[nb_indices])
}))
# This table has ~1,373,394 rows (directed rook edges)

cat("Edge list rows:", nrow(edges), "\n")

# â”€â”€ Step 2: Cross-join edges with years, then join to get row indices â”€
years <- sort(unique(cell_data$year))

# Expand edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows)
edges_by_year <- edges[, CJ(year = years), by = .(focal_id, neighbor_id)]
# Columns: focal_id, neighbor_id, year

# Build a lookup from (id, year) â†’ row index in cell_data
id_year_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_lookup, id, year)

# Get the row index of the focal cell-year
setnames(id_year_lookup, ".row_idx", "focal_row_idx")
setkey(edges_by_year, focal_id, year)
edges_by_year <- id_year_lookup[edges_by_year, on = .(id = focal_id, year = year),
                                 nomatch = NA]
setnames(edges_by_year, "focal_row_idx", "focal_row_idx")

# Get the row index of the neighbor cell-year
setnames(id_year_lookup, "focal_row_idx", "neighbor_row_idx")
edges_by_year <- id_year_lookup[edges_by_year,
                                 on = .(id = neighbor_id, year = year),
                                 nomatch = NA]

# Keep only rows where both focal and neighbor exist
edges_by_year <- edges_by_year[!is.na(focal_row_idx) & !is.na(neighbor_row_idx)]

# Clean up: keep only what we need
edges_by_year <- edges_by_year[, .(focal_row_idx, neighbor_row_idx)]

# Restore lookup column name
setnames(id_year_lookup, "neighbor_row_idx", ".row_idx")

cat("Edges Ã— years rows (after join):", nrow(edges_by_year), "\n")

# â”€â”€ Step 3: Compute neighbor stats for each source variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")

  # Extract the neighbor values via vectorized indexing
  edges_by_year[, val := cell_data[[var_name]][neighbor_row_idx]]

  # Grouped aggregation: max, min, mean per focal row, dropping NAs
  stats <- edges_by_year[!is.na(val),
                         .(nb_max  = max(val),
                           nb_min  = min(val),
                           nb_mean = mean(val)),
                         by = focal_row_idx]

  # Initialize new columns to NA (for cells with no valid neighbors)
  max_col  <- paste0(var_name, "_max_nb")
  min_col  <- paste0(var_name, "_min_nb")
  mean_col <- paste0(var_name, "_mean_nb")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign computed values back to the correct rows
  cell_data[stats$focal_row_idx, (max_col)  := stats$nb_max]
  cell_data[stats$focal_row_idx, (min_col)  := stats$nb_min]
  cell_data[stats$focal_row_idx, (mean_col) := stats$nb_mean]

  # Drop temporary column
  edges_by_year[, val := NULL]

  cat("  Done:", max_col, min_col, mean_col, "\n")
}

# â”€â”€ Step 4: Clean up helper column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_idx := NULL]

# â”€â”€ Step 5: Predict using the pre-trained Random Forest model â”€â”€â”€â”€â”€â”€â”€â”€
# The model is NOT retrained. We just score.
# (Adjust the predict call to match your model framework, e.g. ranger/randomForest)

# Example for ranger:
# preds <- predict(rf_model, data = cell_data)$predictions

# Example for randomForest:
# preds <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete. Ready for prediction.\n")
```

---

## 4. Why This Preserves the Original Estimand

| Requirement | Guarantee |
|---|---|
| **Same max/min/mean values** | The grouped `data.table` aggregation computes `max`, `min`, `mean` on exactly the same set of non-NA neighbor values that the original `compute_neighbor_stats` extracted. The edge list is derived from the same `nb` object and `id_order`. |
| **Same column names** | Columns are named `{var}_max_nb`, `{var}_min_nb`, `{var}_mean_nb` â€” match these to whatever your `compute_and_add_neighbor_features` produced (adjust the suffix if your original code used a different naming convention). |
| **RF model untouched** | The trained model object is never modified; we only call `predict()` on the enriched data. |
| **No raster approximation** | We do **not** rasterize or use focal kernels, which could alter neighbor sets at grid boundaries or for irregular geometries. The exact `nb` object is used. |

---

## 5. Summary of Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations (~hours) | One `data.table` join (~1â€“2 min) |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M R-level iterations (~hours) | 5 grouped aggregations (~2â€“3 min) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |
| Peak RAM | Lists of 6.46M elements + overhead | ~38.5M-row edge table â‰ˆ 0.5â€“1 GB |
END_RESPONSE>>>
