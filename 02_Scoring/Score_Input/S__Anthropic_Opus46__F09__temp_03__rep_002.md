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
run_id: Anthropic_F09_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” these are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times Ã— ~4 neighbors each â‰ˆ **25+ million hash lookups on a massive character vector**.

Character pasting and named-vector lookups at this scale are catastrophically slow in R. The resulting `neighbor_lookup` list itself consumes substantial memory (6.46M list elements, each an integer vector).

### 2. `compute_neighbor_stats` is called 5 times over 6.46M list elements
Each call iterates over 6.46M list entries, subsetting a numeric vector and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is slow due to R's per-element overhead and the final row-binding of 6.46M 3-element vectors.

### Core insight: The neighbor topology is **static across years**
The rook-neighbor relationships are purely spatial â€” cell A's neighbors are the same in 1992 as in 2019. The current code redundantly encodes this 28 times (once per year per cell). The entire lookup should be built **once at the cell level** (344K cells), and yearly attribute joins should be vectorized.

---

## Optimization Strategy

1. **Build a spatial adjacency edge-list once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-independent.

2. **For each variable, join yearly attributes onto the edge-list** using `data.table` keyed joins â€” this is vectorized C-level code, not R-level `lapply`.

3. **Aggregate neighbor stats with `data.table` grouped operations** â€” `[, .(max, min, mean), by = .(cell_id, year)]` runs in seconds on 1.37M Ã— 28 â‰ˆ 38M rows.

4. **Join aggregated stats back** to the main dataset.

This replaces ~6.46M R-level iterations with a handful of vectorized `data.table` joins and group-by aggregations. Expected runtime: **minutes, not days**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a static spatial edge-list ONCE
#
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns (id, neighbor_id), ~1.37M rows
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains positional indices of neighbors of id_order[i]
  n <- length(neighbors)
  from_idx <- rep(seq_len(n), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  # Remove zero-length / 0-coded "no neighbor" entries if present (spdep convention)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# ~1,373,394 rows â€” small and fast

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, compute neighbor stats
#         using vectorized data.table joins + grouped aggregation,
#         then attach results back to cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main data for fast joins
setkey(cell_data, id, year)

# Pre-expand edge list by year (all 28 years) â€” ~38.5M rows, but only 3 columns
# This is the "reusable neighbor table joined with year" concept.
years <- sort(unique(cell_data$year))
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year_dt[, `:=`(
  id          = edge_dt$id[edge_idx],
  neighbor_id = edge_dt$neighbor_id[edge_idx]
)]
edge_year_dt[, edge_idx := NULL]
setkey(edge_year_dt, neighbor_id, year)

# ~38.5M rows Ã— 3 columns â‰ˆ 0.9 GB â€” fits comfortably in 16 GB RAM

for (var_name in neighbor_source_vars) {

  message("Computing neighbor stats for: ", var_name)

  # Extract only the columns we need for the join
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join neighbor attribute values onto the edge-year table
  # For each (id, neighbor_id, year) row, get the neighbor's value
  edge_vals <- merge(
    edge_year_dt,
    attr_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  # Aggregate: for each (id, year), compute max/min/mean of neighbor values
  stats <- edge_vals[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Join stats back to cell_data
  setkey(stats, id, year)
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE, sort = FALSE)

  # Clean up per-iteration temporaries

  rm(attr_dt, edge_vals, stats)
}

# Re-sort to original order if needed
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the already-trained Random Forest
#         (model object unchanged â€” no retraining)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original pipeline produced, with identical numerical values.
#
# Example (adjust to your actual model object and predict call):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of neighbor values per (cell, year) as the original code. The edge-list encodes the identical rook-neighbor topology; the `data.table` join retrieves the identical year-matched attribute values. |
| **Trained RF model** | The model object is never touched. Only the input feature columns are recomputed (identically), then passed to `predict()`. |
| **Column names** | Output columns follow the same `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` naming convention. |
| **NA handling** | `!is.na(value)` in the aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with no valid neighbors get `NA` via the `all.x = TRUE` merge, matching the original `return(c(NA, NA, NA))`. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| `build_edge_list` | < 1 second |
| `CJ` expansion (38.5M rows) | ~5â€“10 seconds |
| Per-variable merge + aggregate (Ã—5) | ~30â€“90 seconds each |
| Total neighbor feature computation | **~3â€“8 minutes** |

This is a **~600â€“1700Ã— speedup** over the original 86+ hour estimate, achieved entirely through vectorization and eliminating redundant per-row R-level iteration.
END_RESPONSE>>>
