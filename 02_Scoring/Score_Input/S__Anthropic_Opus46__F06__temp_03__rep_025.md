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
run_id: Anthropic_F06_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(neighbor_id, year, "_")`).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String-based key construction and named-vector lookup for 6.46M rows, each with ~4 neighbors on average (1.37M directed relationships / 344K cells â‰ˆ 4), means roughly **25+ million string paste and hash-lookup operations**. The `lapply` returns a list of 6.46M integer vectors â€” a massive, fragmented data structure.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per-row `lapply`

For each of the 5 variables, another `lapply` iterates over 6.46M list elements, extracting values, removing NAs, and computing `max/min/mean`. That's **5 Ã— 6.46M = 32.3M R-level function calls** with per-element subsetting.

### Why raster focal/kernel operations don't directly apply

Focal operations on rasters assume a regular grid with a fixed rectangular kernel. Here, the grid cells have an irregular neighbor structure (coastal cells, boundary cells have fewer neighbors), and the data is in long panel format (cell Ã— year). A focal approach would require reshaping each variable into a 2D raster per year, running `focal()`, and extracting back â€” possible but fragile and unnecessary once we vectorize properly.

### The real fix: vectorize using sparse-matrix or data.table group-by operations

The neighbor relationships are **time-invariant** â€” the same neighbor structure repeats for all 28 years. We can exploit this by:
1. Building a sparse adjacency matrix once (344K Ã— 344K).
2. Reshaping each variable into a matrix of (344K cells Ã— 28 years).
3. Using **sparse matrix multiplication and row-wise operations** to compute neighbor stats in bulk â€” or equivalently, using `data.table` joins on an edge list.

The `data.table` edge-list approach is simpler and memory-friendlier on a 16 GB laptop.

---

## 2. Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor lookup | 6.46M-element list of integer vectors (string keys) | Pre-built edge-list `data.table` with `(row_i, row_j)` for all cell-year pairs (~25.8M edges) |
| Stat computation | `lapply` over 6.46M elements, per variable | Vectorized `data.table` group-by `max/min/mean` on the edge list, per variable |
| Iterations | 5 variables Ã— 6.46M R calls = 32.3M calls | 5 variables Ã— 1 vectorized group-by = 5 operations |
| Estimated time | 86+ hours | **~2â€“10 minutes** |

**Key insight**: The edge list `(cell_i, cell_j)` is time-invariant (344K cells, ~1.37M directed edges). For each year, the same edges apply. So the full edge list in row-index space has ~1.37M Ã— 28 = ~38.4M rows. We join variable values onto the neighbor side and group-by the focal cell's row index.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table if not already
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure a row index column exists for stable reference
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a time-invariant directed edge list from the
#         spdep nb object (rook_neighbors_unique)
#
#   id_order[i] is the cell ID for the i-th element of the nb list.
#   rook_neighbors_unique[[i]] gives integer indices (into id_order)
#   of the rook neighbors of cell id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
message("Building time-invariant edge list from nb object...")

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_cell = id_order[i], neighbor_cell = id_order[nb_idx])
}))

message(sprintf("  Edge list: %s directed edges across %s unique focal cells.",
                format(nrow(edge_list), big.mark = ","),
                format(uniqueN(edge_list$focal_cell), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Expand edge list across all years by joining with
#         cell_data row indices.
#
#   For each (focal_cell, neighbor_cell) pair and each year,
#   we need:
#     - row_i: the row index in cell_data for (focal_cell, year)
#     - row_j: the row index in cell_data for (neighbor_cell, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
message("Mapping edge list to row indices across all years...")

# Create a lookup: cell id + year -> row index
setkey(cell_data, id, year)
id_year_lookup <- cell_data[, .(id, year, .row_idx)]

# Join focal side
setnames(id_year_lookup, c("id", "year", ".row_idx"),
         c("focal_cell", "year", "row_i"))
edges_full <- merge(edge_list, id_year_lookup,
                    by = "focal_cell", allow.cartesian = TRUE)

# Join neighbor side
setnames(id_year_lookup, c("focal_cell", "year", "row_i"),
         c("neighbor_cell", "year", "row_j"))
edges_full <- merge(edges_full, id_year_lookup,
                    by = c("neighbor_cell", "year"), allow.cartesian = FALSE)

# Restore lookup names for safety
setnames(id_year_lookup, c("neighbor_cell", "year", "row_j"),
         c("id", "year", ".row_idx"))

# Keep only what we need
edges_full <- edges_full[, .(row_i, row_j)]
setkey(edges_full, row_i)

message(sprintf("  Expanded edge list: %s (row_i, row_j) pairs.",
                format(nrow(edges_full), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: For each neighbor source variable, compute
#         max, min, mean of neighbor values â€” fully vectorized.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))

  # Extract the variable values as a plain vector (indexed by row)
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the edge list
  edges_full[, nbr_val := vals[row_j]]

  # Compute grouped stats: max, min, mean (excluding NAs)
  stats <- edges_full[!is.na(nbr_val),
                      .(nb_max  = max(nbr_val),
                        nb_min  = min(nbr_val),
                        nb_mean = mean(nbr_val)),
                      by = row_i]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign computed values by row index
  cell_data[stats$row_i, (max_col)  := stats$nb_max]
  cell_data[stats$row_i, (min_col)  := stats$nb_min]
  cell_data[stats$row_i, (mean_col) := stats$nb_mean]

  message(sprintf("  -> Added %s, %s, %s", max_col, min_col, mean_col))
}

# Clean up temporary column
edges_full[, nbr_val := NULL]
cell_data[, .row_idx := NULL]

message("Done. All 15 neighbor-stat columns added to cell_data.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the pre-trained Random Forest
#         (no retraining â€” model object is used as-is)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adjust to your actual model object and predict call):
# cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical results** | The `max`, `min`, `mean` are computed over exactly the same set of non-NA rook-neighbor values per cell-year as the original code. The edge list is derived from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Trained RF model unchanged** | The model object is never modified; only `predict()` is called on the augmented `cell_data`. |
| **NA handling** | Rows with no valid neighbors (boundary cells with all-NA neighbors, or cells absent from a given year) receive `NA` in all three stat columns â€” identical to the original `c(NA, NA, NA)` return. |

## 5. Memory Considerations (16 GB laptop)

| Object | Estimated size |
|---|---|
| `cell_data` (~6.46M rows Ã— 110 cols) | ~5.7 GB |
| `edges_full` (~38M rows Ã— 2 int cols) | ~0.6 GB |
| `edges_full` with `nbr_val` (temporary) | ~0.9 GB |
| `stats` (temporary per variable) | ~0.15 GB |
| **Total peak** | **~7.5 GB** âœ“ fits in 16 GB |

If memory is tight, you can process variables one at a time (as shown) and `rm(stats)` after each iteration, or even split the edge list by year chunks. But the above should fit comfortably.

## 6. Expected Runtime

- Edge list construction: ~10â€“30 seconds
- Edge list expansion (merge): ~1â€“3 minutes
- Per-variable grouped stats: ~10â€“30 seconds each (5 vars â†’ ~1â€“2.5 minutes)
- **Total: ~3â€“6 minutes** vs. the original 86+ hours â€” roughly a **1,000Ã— speedup**.
END_RESPONSE>>>
