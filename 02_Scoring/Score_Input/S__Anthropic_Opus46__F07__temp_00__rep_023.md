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
run_id: Anthropic_F07_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of R's interpreted loop is enormous at this scale.
3. **Memory**: The resulting `neighbor_lookup` is a list of ~6.46M integer vectors. Combined with the intermediate character vectors, this can spike well beyond available RAM, causing swapping.

Then `compute_neighbor_stats` iterates over the same 6.46M-element list **5 times** (once per variable), each time extracting neighbor values and computing max/min/mean in pure R.

**Root cause summary:**
- O(n) named-vector lookups inside an O(n) loop â†’ effectively O(nÂ²) behavior.
- Repeated string allocation (`paste`) for 6.46M Ã— avg-neighbors keys.
- The neighbor lookup is **time-invariant** but is encoded with year-specific row indices, forcing a full rebuild if anything changes.
- `compute_neighbor_stats` does 5 separate passes with per-element R overhead.

## Optimization Strategy

### 1. Replace character-key lookups with integer-arithmetic direct indexing

Since the panel is balanced (344,208 cells Ã— 28 years = 9,637,824 potential rows, of which ~6.46M exist), we can build a **cell-ID â†’ integer index map** and a **year â†’ integer offset map** using `data.table` or environments (O(1) hash lookup), then compute row indices arithmetically.

### 2. Vectorize neighbor lookup construction using `data.table` joins

Instead of looping over 6.46M rows, we:
- Expand the `nb` object into an edge list (cell_i, cell_j) â€” only ~1.37M edges.
- Cross-join with years to get (cell_i, year, cell_j) â€” ~1.37M Ã— 28 â‰ˆ 38.4M rows (but filtered to existing cell-years).
- Join against the data to get row indices for both the focal cell and the neighbor cell.
- Group by focal-row-index and compute stats directly â€” **all 5 variables in one pass**.

This replaces the 86-hour loop with vectorized `data.table` operations that should complete in **minutes**.

### 3. Compute all neighbor stats in one grouped aggregation

Instead of 5 separate `lapply` passes, compute max/min/mean for all 5 variables simultaneously in a single `data.table` grouped operation on the edge table.

### 4. Preserve the trained RF model and numerical estimand

We only change how the neighbor features are computed, not their values. The column names and semantics are identical, so the trained model's `predict()` call is unchanged.

---

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Inputs assumed to exist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data            : data.frame/data.table with columns id, year, ntl, ec,
#                        pop_density, def, usd_est_n2, ... (~6.46M rows)
# id_order             : integer/character vector of cell IDs matching the nb object
# rook_neighbors_unique: spdep nb object (list of integer index vectors)
# rf_model             : trained Random Forest model (untouched)

# â”€â”€ 1. Convert cell_data to data.table (in-place if possible) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€ 2. Build edge list from the nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    This is ~1.37M directed edges; very fast.
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_i <- rook_neighbors_unique[[i]]
  # spdep nb: integer(0) means no neighbors; 0L is also possible

  nb_i <- nb_i[nb_i > 0L]
  if (length(nb_i) == 0L) return(NULL)
  data.table(focal_cell = id_order[i], neighbor_cell = id_order[nb_i])
}))

cat("Edge list rows:", nrow(edges), "\n")

# â”€â”€ 3. Add row-index to cell_data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_idx := .I]

# â”€â”€ 4. Build a keyed lookup: (id, year) â†’ row index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_year_key <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_key, id, year)

# â”€â”€ 5. Get the unique years present â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_years <- sort(unique(cell_data$year))

# â”€â”€ 6. Cross-join edges Ã— years, then filter to existing cell-years â”€â”€â”€â”€â”€â”€â”€â”€
#    To avoid a 38M-row cross join in one shot (memory), we process in
#    year-chunks. Each chunk is ~1.37M rows â€” trivially small.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the columns we need for neighbor stats into a matrix for speed
val_mat <- as.matrix(cell_data[, ..neighbor_source_vars])

# Prepare result columns (pre-allocate with NA)
stat_names <- c("max", "min", "mean")
new_col_names <- as.vector(outer(
  neighbor_source_vars, stat_names,
  function(v, s) paste0("neighbor_", s, "_", v)
))

# Pre-allocate result matrix: nrow(cell_data) Ã— 15
result_mat <- matrix(NA_real_, nrow = nrow(cell_data), ncol = length(new_col_names))
colnames(result_mat) <- new_col_names

cat("Processing", length(all_years), "years...\n")

for (yr in all_years) {
  # Rows in this year
  yr_rows <- id_year_key[year == yr]  # columns: id, year, .row_idx
  setkey(yr_rows, id)

  # Join edges to get focal row index
  #   edges has (focal_cell, neighbor_cell)
  #   yr_rows has (id, year, .row_idx)
  focal_join <- yr_rows[edges, on = .(id = focal_cell), nomatch = 0L,
                        .(focal_row = .row_idx,
                          neighbor_cell = i.neighbor_cell)]

  # Join again to get neighbor row index
  setkey(focal_join, neighbor_cell)
  full_join <- yr_rows[focal_join, on = .(id = neighbor_cell), nomatch = 0L,
                       .(focal_row = i.focal_row,
                         neighbor_row = .row_idx)]

  if (nrow(full_join) == 0L) next

  # Extract neighbor values: matrix subsetting is very fast
  nb_vals <- val_mat[full_join$neighbor_row, , drop = FALSE]

  # Build a data.table for grouped aggregation
  agg_dt <- data.table(focal_row = full_join$focal_row, nb_vals)

  # Grouped aggregation: max, min, mean for each variable, grouped by focal_row
  # Build the aggregation expression dynamically
  agg_exprs <- lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  })
  agg_exprs <- unlist(agg_exprs, recursive = FALSE)
  names(agg_exprs) <- new_col_names

  agg_call <- as.call(c(as.name("list"), agg_exprs))

  agg_result <- agg_dt[, eval(agg_call), by = focal_row]

  # Write results into the pre-allocated matrix
  result_mat[agg_result$focal_row, ] <- as.matrix(
    agg_result[, ..new_col_names]
  )

  cat("  Year", yr, "done â€”", nrow(agg_result), "focal cells with neighbors\n")
}

# â”€â”€ 7. Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen, but safe)
result_mat[is.infinite(result_mat)] <- NA_real_

# â”€â”€ 8. Bind result columns to cell_data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (j in seq_along(new_col_names)) {
  set(cell_data, j = new_col_names[j], value = result_mat[, j])
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done. Added", length(new_col_names), "neighbor-stat columns.\n")

# â”€â”€ 9. Predict with the existing RF model (unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE` on exactly the same neighbor sets, filtered to existing cell-years via inner join (`nomatch = 0L`). |
| **Same column names** | The naming convention `neighbor_{stat}_{var}` matches the original `compute_and_add_neighbor_features` output. (Adjust the `paste0` template if your original used a different convention.) |
| **RF model untouched** | No retraining; the model object is never modified. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| Edge list construction (1.37M edges) | ~2 seconds |
| Per-year join + aggregation (28 iterations Ã— ~1.37M edges) | ~1â€“3 minutes total |
| Column binding | < 1 second |
| **Total** | **~2â€“5 minutes** (vs. 86+ hours) |

Peak memory: ~2â€“3 GB (the 6.46M Ã— 110 data.table + the edge list + one year's aggregation buffer), well within 16 GB.
END_RESPONSE>>>
