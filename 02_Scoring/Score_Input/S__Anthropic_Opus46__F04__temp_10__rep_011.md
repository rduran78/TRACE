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
run_id: Anthropic_F04_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The suspicion is correct: **the spatial neighbor feature construction is the computational bottleneck**, not Random Forest inference. Here is why:

### `build_neighbor_lookup` â€” O(N) character-key lookups via `lapply`

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to character, looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes together `(neighbor_id, year)` string keys â€” one `paste()` call per neighbor per row.
4. Looks up each key in a named character vector (`idx_lookup`) â€” effectively a hash-table probe per key, but wrapped in R's slow single-threaded `lapply` over 6.46M iterations.

With ~1.37M directed neighbor relationships spread across 28 years, the total number of string constructions and lookups is roughly **1,373,394 Ã— 28 â‰ˆ 38.5 million paste + match operations**, all inside an interpreted R loop. The named-vector lookup `idx_lookup[neighbor_keys]` is an O(1)-amortised hash probe per key, but the per-element R interpreter overhead across millions of iterations dominates.

### `compute_neighbor_stats` â€” repeated R-level loops

For each of the 5 source variables, `compute_neighbor_stats` iterates over all 6.46M rows again in `lapply`, subsetting a numeric vector and computing `max/min/mean`. That is **5 Ã— 6.46M = 32.3 million R-level function calls** with small-vector allocation overhead each time.

### Combined cost

The total is ~6.46M R-level iterations for the lookup build, plus ~32.3M R-level iterations for statistics â€” all sequential, all in interpreted R. On a laptop this easily reaches the estimated 86+ hours.

---

## Optimization Strategy

The core idea: **replace the row-level R `lapply` loops with vectorized `data.table` grouped joins and aggregations.** Specifically:

1. **Replace `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list of integer vectors, construct a long-form `data.table` edge table `(row_i, neighbor_row_j)` via a single vectorized merge. This eliminates millions of `paste` and named-vector lookups.

2. **Replace `compute_neighbor_stats` with a single grouped `data.table` aggregation.** Join the edge table to the source variable column, then `group by row_i` and compute `max`, `min`, `mean` in one vectorized pass â€” for all 5 variables together if desired.

3. **Memory is manageable.** The edge table has at most ~38.5M rows (1.37M neighbor pairs Ã— 28 years), each row being two integers (8 bytes each) â‰ˆ ~600 MB â€” fits within 16 GB alongside the 6.46M-row data.

Expected speedup: from 86+ hours to **minutes** (typically 5â€“20 minutes depending on disk/RAM speed), because `data.table` grouped operations are implemented in C and parallelised internally.

---

## Working R Code

```r
library(data.table)

#
# Step 0: Convert cell_data to data.table if not already;
#         assume columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#
cell_dt <- as.data.table(cell_data)

# Preserve original row order so the final result aligns with any downstream
# Random Forest predict() call that expects the same row order.
cell_dt[, .row_idx := .I]

# -----------------------------------------------------------------------
# Step 1: Build a vectorised edge table replacing build_neighbor_lookup
# -----------------------------------------------------------------------

# id_order is the vector whose positional index matches the nb object
# rook_neighbors_unique is the spdep nb list: rook_neighbors_unique[[k]]
# gives the positional indices (into id_order) of cell id_order[k]'s neighbors.

# 1a. Expand the nb list into a long-form edge list of (focal_id, neighbor_id).
#     This is ~1.37M rows â€” one per directed neighbor pair.
nb_lengths <- lengths(rook_neighbors_unique)
focal_pos  <- rep(seq_along(id_order), times = nb_lengths)
neigh_pos  <- unlist(rook_neighbors_unique)           # positional indices

edge_ids <- data.table(
  focal_id    = id_order[focal_pos],
  neighbor_id = id_order[neigh_pos]
)
rm(focal_pos, neigh_pos, nb_lengths)                  # free memory

# 1b. Cross-join with years to get (focal_id, year, neighbor_id) â€” the set of
#     all cell-year to neighbor-cell-year links.
#     Instead of an expensive explicit cross join (~38.5M rows built at once),
#     we merge through cell_dt which already has the (id, year) combinations.

# Create a keyed lookup: row index by (id, year)
row_lookup <- cell_dt[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# 1c. For every row in cell_dt, attach its neighbors via merge on focal_id == id.
#     Result: one row per (focal_row, neighbor_id, year).
focal_rows <- cell_dt[, .(focal_row = .row_idx, focal_id = id, year)]
setkey(edge_ids, focal_id)
setkey(focal_rows, focal_id)

# Merge: for each focal row, get all its neighbor cell IDs
edges <- edge_ids[focal_rows, on = "focal_id", allow.cartesian = TRUE,
                  nomatch = NULL]
# edges now has columns: focal_id, neighbor_id, focal_row, year

# 1d. Resolve neighbor_id + year â†’ neighbor_row via the row_lookup
setkey(edges, neighbor_id, year)
setkey(row_lookup, id, year)
edges <- row_lookup[edges, on = c(id = "neighbor_id", "year"), nomatch = NA]
# After this join, .row_idx is the neighbor's row index.
# Rename for clarity:
setnames(edges, ".row_idx", "neighbor_row")

# Drop rows where the neighbor didn't exist in that year
edges <- edges[!is.na(neighbor_row)]

# Keep only what we need to save memory
edges <- edges[, .(focal_row, neighbor_row)]
setkey(edges, focal_row)

cat(sprintf("Edge table: %s rows (%.1f MB)\n",
            formatC(nrow(edges), big.mark = ","),
            object.size(edges) / 1e6))

# Free intermediate objects
rm(focal_rows, row_lookup, edge_ids)
gc()

# -----------------------------------------------------------------------
# Step 2: Compute neighbor statistics for all 5 variables â€” vectorised
# -----------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pull the neighbor row's values into the edge table, compute grouped stats.
# We do this one variable at a time to limit peak memory.

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Attach the variable's value at the neighbor row
  edges[, val := cell_dt[[var_name]][neighbor_row]]

  # Grouped aggregation: max, min, mean â€” excluding NAs
  stats <- edges[!is.na(val),
                 .(nmax  = max(val),
                   nmin  = min(val),
                   nmean = mean(val)),
                 by = focal_row]

  # Initialise new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Fill in computed values by row index
  cell_dt[stats$focal_row, (max_col)  := stats$nmax]
  cell_dt[stats$focal_row, (min_col)  := stats$nmin]
  cell_dt[stats$focal_row, (mean_col) := stats$nmean]

  rm(stats)
}

# Clean up helper column
edges[, val := NULL]

# -----------------------------------------------------------------------
# Step 3: Convert back to data.frame, preserving original row order, and
#         remove the helper column.
# -----------------------------------------------------------------------

cell_dt[, .row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# -----------------------------------------------------------------------
# Step 4: Predict with the EXISTING trained Random Forest (unchanged)
# -----------------------------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (The model object and predict call remain exactly as before.)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per cell-year as the original code. The `data.table` grouped aggregation is algebraically identical. |
| **Same row alignment** | The `.row_idx` bookkeeping ensures every computed statistic is written back to the correct row. The final `cell_data` data.frame has the same row order and column values as the original pipeline would produce. |
| **Trained RF untouched** | No model retraining occurs. The same `predict()` call on the same feature matrix yields identical predictions. |

---

## Expected Performance

| Stage | Original | Optimised |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M R-level iterations with string ops) | ~1â€“3 min (vectorised `data.table` keyed joins) |
| `compute_neighbor_stats` (Ã—5 vars) | ~tens of hours (32.3M R-level `lapply` calls) | ~2â€“5 min (grouped C-level aggregation) |
| **Total neighbor feature construction** | **~86+ hours** | **~5â€“15 minutes** |
| Peak RAM | Moderate (6.46M-element list of int vectors) | ~1â€“3 GB (edge table + cell_dt) â€” fits in 16 GB |

The bottleneck is eliminated by moving from millions of interpreted R loop iterations with string allocation to a single vectorised join-then-aggregate pattern in `data.table`.
END_RESPONSE>>>
