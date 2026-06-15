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
run_id: Anthropic_F04_T10_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"cellid_year"`).
4. Looks those keys up in a named character vector (`idx_lookup`).

**String concatenation (`paste`) and named-vector lookups on character keys are extremely slow in R when repeated millions of times.** The `idx_lookup` named vector has ~6.46M entries; each named-vector lookup is O(n) in the worst case because R's named vectors use linear hashing that degrades at scale. Multiply this by ~6.46M rows Ã— ~4 average neighbors = ~25.8 billion character operations. This is why the estimated runtime exceeds 86 hours.

`compute_neighbor_stats` is comparatively cheap (just numeric subsetting), but it is also called 5 times with redundant `lapply` overhead.

**Summary of root causes:**
1. Row-level `lapply` with `paste`-based string key construction over 6.46M rows.
2. Named-vector lookup on a 6.46M-element character vector (poor hash performance).
3. The neighbor lookup is year-invariant in structure but rebuilt per row per year.
4. `compute_neighbor_stats` is called in a loop over 5 variables, each time iterating over 6.46M entries separately.

---

## Optimization Strategy

### Key Insight
The spatial neighbor structure is **time-invariant**: cell A's neighbors are the same in every year. Only the *data rows* change by year. So we can:

1. **Vectorize the lookup using integer indexing instead of string keys.** Build a matrix mapping `(cell_index, year_index)` â†’ row number in `cell_data`. Then neighbor row indices are a direct integer matrix lookup â€” no strings, no paste, no named vectors.

2. **Use `data.table` for the statistics computation**, avoiding per-row `lapply` entirely. Expand the neighbor graph into an edge list, join to the data, and compute grouped `max/min/mean` in one vectorized pass per variable (or all variables at once).

3. **Compute all 5 variables' neighbor stats in a single pass** over the edge list rather than 5 separate `lapply` calls.

This reduces the complexity from ~6.46M Ã— k expensive string operations to a single vectorized join + grouped aggregation on ~25.8M edge-rows.

---

## Optimized R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table if not already; keep original row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # ---- Step 1: Build the directed edge list from the nb object (time-invariant) ----
  # rook_neighbors_unique is a list of integer vectors (indices into id_order)
  # id_order[i] is the cell id for the i-th spatial unit
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(from_id = id_order[i], to_id = id_order[nb_idx])
  }))
  # edges now has ~1,373,394 rows: (from_id, to_id) directed pairs

  # ---- Step 2: Expand edges across all years via join (vectorized) ----
  # We need, for every (from_id, year), the data-row indices of all (to_id, year).
  # First, create a keyed lookup: (id, year) -> .row_id
  dt_key <- dt[, .(id, year, .row_id)]
  setkey(dt_key, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Cross-join edges Ã— years: each spatial edge exists in every year
  # This produces ~1,373,394 Ã— 28 â‰ˆ 38.5M rows â€” fits in 16 GB easily
  edge_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_year[, `:=`(from_id = edges$from_id[edge_idx],
                    to_id   = edges$to_id[edge_idx])]
  edge_year[, edge_idx := NULL]

  # Join to get the focal row index (from_id, year)
  setkey(edge_year, from_id, year)
  setkey(dt_key, id, year)
  edge_year <- dt_key[edge_year, on = .(id = from_id, year = year),
                       nomatch = 0L]
  setnames(edge_year, ".row_id", "focal_row")

  # Join to get the neighbor row index (to_id, year)
  edge_year <- dt_key[edge_year, on = .(id = to_id, year = year),
                       nomatch = 0L]
  setnames(edge_year, ".row_id", "neighbor_row")

  # edge_year now has columns: focal_row, neighbor_row (plus id, year, to_id, etc.)
  # Keep only what we need
  edge_year <- edge_year[, .(focal_row, neighbor_row)]

  # ---- Step 3: Attach neighbor variable values and compute stats in one pass ----
  # Pull the variable columns we need for the neighbor rows
  var_cols <- neighbor_source_vars
  neighbor_vals <- dt[edge_year$neighbor_row, ..var_cols]
  neighbor_vals[, focal_row := edge_year$focal_row]

  # Compute max, min, mean per focal_row for each variable, all at once
  agg_exprs <- list()
  for (v in var_cols) {
    agg_exprs[[paste0("nb_max_", v)]] <- substitute(
      suppressWarnings(max(x[!is.na(x)])),
      list(x = as.name(v))
    )
    agg_exprs[[paste0("nb_min_", v)]] <- substitute(
      suppressWarnings(min(x[!is.na(x)])),
      list(x = as.name(v))
    )
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(x, na.rm = TRUE),
      list(x = as.name(v))
    )
  }

  # Single grouped aggregation
  stats <- neighbor_vals[, lapply(agg_exprs, eval), by = focal_row]

  # Replace -Inf/Inf (from max/min on all-NA groups) with NA
  inf_to_na <- function(x) { x[is.infinite(x)] <- NA; x }
  stat_cols <- setdiff(names(stats), "focal_row")
  stats[, (stat_cols) := lapply(.SD, inf_to_na), .SDcols = stat_cols]

  # ---- Step 4: Join back to the main data ----
  # Rows with no neighbors won't appear in stats; they get NA automatically
  setkey(stats, focal_row)
  dt[, .row_id := .I]
  dt <- stats[dt, on = .(focal_row = .row_id)]
  dt[, focal_row := NULL]

  # Convert back to data.frame if the original was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }
  return(dt)
}

# ---- Usage (drop-in replacement for the original outer loop) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm = TRUE` are computed on the identical neighbor value sets as the original code. No approximation is introduced. |
| **Trained RF model untouched** | No retraining. The output columns have the same names and values, so `predict(rf_model, newdata = ...)` works identically. |
| **RAM feasibility** | The edge-year table has ~38.5M rows Ã— 2 integer columns â‰ˆ 0.6 GB. The neighbor-values table adds ~38.5M Ã— 6 columns â‰ˆ 1.8 GB. Total peak overhead is ~3â€“4 GB, well within 16 GB. |
| **Speed** | Eliminates all per-row `paste`/string-lookup. `data.table` grouped aggregation on 38.5M rows with 5 numeric columns typically completes in **1â€“5 minutes** on a modern laptop â€” a reduction from 86+ hours to minutes (roughly **1,000Ã—** speedup). |

### Optional Further Speedups

- **Parallel aggregation:** `data.table` automatically uses `setDTthreads()` for grouped operations on multi-core machines.
- **Memory-constrained variant:** If RAM is tight, process one variable at a time instead of all 5 simultaneously (trades a small amount of speed for ~60% less peak memory).
- **Pre-cache to disk:** Save the resulting `cell_data` with neighbor features to `.fst` or `.parquet` so the computation never needs to be repeated.
END_RESPONSE>>>
