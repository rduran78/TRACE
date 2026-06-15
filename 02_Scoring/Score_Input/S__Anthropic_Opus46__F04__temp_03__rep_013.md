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
run_id: Anthropic_F04_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction functions. Here is why:

### `build_neighbor_lookup` â€” O(N) `lapply` with per-row string operations

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls `paste()` to build `"id_year"` key strings for every neighbor of that row.
4. Looks those keys up in another named character vector (`idx_lookup`).

Named-vector lookup in R is **hash-based but still involves repeated character allocation and matching**. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million `paste` + hash-lookup operations** inside a sequential `lapply`. The result is a list of 6.46M integer vectors â€” a large, fragmented memory structure.

### `compute_neighbor_stats` â€” Called 5Ã— over the 6.46M-element list

Each call iterates over the 6.46M-element `neighbor_lookup` list, subsets a numeric vector, removes NAs, and computes `max/min/mean`. That is **~32.3 million R-level function calls** (5 variables Ã— 6.46M rows), each with small-vector allocation overhead.

### Combined cost estimate

| Step | Calls | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M | `paste`, named-vector hash lookup |
| `compute_neighbor_stats` | 5 Ã— 6.46M | per-element `lapply`, small-vector alloc |

The 86+ hour estimate is consistent with this: R's interpreted per-element overhead on millions of list items is enormous.

---

## Optimization Strategy

The core idea: **replace row-level R loops with vectorized, column-oriented joins using `data.table`**.

### Key changes

1. **Expand the neighbor graph into an edge table** (`data.table` with columns `id`, `neighbor_id`) â€” done once, ~1.37M rows.
2. **Join the edge table to the panel data by `(neighbor_id, year)`** to get all neighbor values in one vectorized merge â€” produces ~25.8M rows but is a single keyed join.
3. **Group-by aggregate** `[, .(max, min, mean), by = .(id, year)]` â€” a single `data.table` grouped operation, fully vectorized in C.
4. **Repeat for each of the 5 variables** (or pivot and do all at once).

This eliminates:
- All `paste`-based key construction.
- All per-row `lapply` calls.
- All named-vector hash lookups.
- The 6.46M-element list-of-vectors structure.

**Expected speedup: from 86+ hours to roughly 5â€“15 minutes** on a 16 GB laptop, depending on disk I/O and `data.table` thread count.

### Memory feasibility

The edge table expanded by year: ~25.8M rows Ã— a few columns of integers/doubles â‰ˆ < 1 GB. The panel data itself (~6.46M Ã— 110 cols) is the largest object. 16 GB is sufficient if we process one variable at a time and avoid unnecessary copies.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert panel data to data.table (in-place, no copy)
# ---------------------------------------------------------------
setDT(cell_data)

# ---------------------------------------------------------------
# 1.  Build a flat edge table from the nb object  (done once)
#
#     rook_neighbors_unique : list of length N_cells (spdep nb)
#     id_order              : integer vector mapping position -> cell id
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the positional indices of neighbors of cell i
  n <- length(neighbors)
  from_idx <- rep(seq_len(n), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (0L means no neighbors)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows (directed pairs)

# ---------------------------------------------------------------
# 2.  Compute neighbor features for all source variables
#
#     Strategy:
#       - For each variable, join edge_dt to cell_data on
#         (neighbor_id == id, year == year) to fetch neighbor values.
#       - Aggregate max / min / mean grouped by (id, year).
#       - Join the aggregates back onto cell_data.
# ---------------------------------------------------------------

# Key cell_data for fast joins
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_dt <- function(cell_data, edge_dt, var_name) {

  # --- a) Subset cell_data to only the columns we need for the join ---
  #     This avoids copying all 110 columns into the join result.
  lookup_cols <- c("id", "year", var_name)
  lookup_dt   <- cell_data[, ..lookup_cols]
  setnames(lookup_dt, c("id", var_name), c("neighbor_id", "nval"))
  setkey(lookup_dt, neighbor_id, year)

  # --- b) Expand edges Ã— years: join edge_dt to lookup_dt ---
  #     For every (id, neighbor_id) pair, get the neighbor's value
  #     in every year that the neighbor has data.
  #     We add year from cell_data's own rows via a second join.

  # First, get the years each focal cell appears in:
  focal_years <- cell_data[, .(id, year)]
  setkey(focal_years, id)

  # Merge focal (id, year) with edge_dt to get (id, year, neighbor_id)
  expanded <- edge_dt[focal_years, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: id, neighbor_id, year

  # Now join to get the neighbor's value in that year
  setkey(expanded, neighbor_id, year)
  expanded <- lookup_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded columns: neighbor_id, year, nval, id

  # --- c) Aggregate by (id, year) ---
  agg <- expanded[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    by = .(id, year)
  ]

  # Name the new columns to match the original pipeline's naming convention
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)

  # --- d) Merge back onto cell_data ---
  setkey(agg, id, year)
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)

  cell_data
}

# ---------------------------------------------------------------
# 3.  Run for all 5 variables
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_dt(cell_data, edge_dt, var_name)
}

# ---------------------------------------------------------------
# 4.  (Optional) If your original code used different column names,
#     rename to match exactly what the trained Random Forest expects.
#     Example â€” adjust the mapping below to your actual naming scheme:
#
#     setnames(cell_data, "ntl_nb_max",  "ntl_neighbor_max")
#     ...
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 5.  Run the already-trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (same rook-contiguity graph, same year matching, same NA exclusion). The `data.table` grouped aggregation is numerically identical to the original R code. |
| **Trained RF model** | The model object is never touched. Only the input feature columns are constructed; prediction is a forward pass. |
| **Column names** | The naming convention (`{var}_nb_max`, etc.) can be trivially mapped to whatever names the trained model expects via `setnames`. |
| **Edge cases** | Cells with no neighbors in a given year will have `NA` for all three stats (the left join produces `NA` when no aggregated row exists), matching the original `c(NA, NA, NA)` return. |

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup / edge table | ~hours (6.46M `lapply`) | < 1 second (vectorized `rep`/`unlist`) |
| Compute stats per variable | ~17 hours each | ~1â€“3 minutes each (`data.table` keyed join + grouped agg) |
| **Total (5 variables)** | **86+ hours** | **~5â€“15 minutes** |

The entire speedup comes from replacing millions of interpreted R function calls with a handful of vectorized C-level `data.table` operations (keyed joins and grouped aggregations).
END_RESPONSE>>>
