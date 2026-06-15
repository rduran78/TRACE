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
run_id: Anthropic_F04_T00_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated `paste` and named-vector lookups on a multi-million-element vector, is extremely slow. Named vector lookup in R degrades as the vector grows; at 6.46M entries the hash table is enormous and each probe is expensive. The total number of key lookups is roughly 6.46M Ã— average_neighbors â‰ˆ 6.46M Ã— 4 â‰ˆ 25.8M string-match lookups against a 6.46M-entry named vector.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 Ã— 6.46M â‰ˆ 32.3M times. The per-call overhead of anonymous function dispatch, `is.na`, and three summary functions dominates.

### Estimated cost breakdown

| Step | Calls | Estimated share |
|---|---|---|
| `build_neighbor_lookup` (paste + named lookup Ã—25.8M) | 1 | ~40â€“50% |
| `compute_neighbor_stats` (lapply Ã—5 vars) | 5 | ~45â€“55% |
| Random Forest `predict()` | 1 | ~1â€“5% |

---

## Optimization Strategy

**Principle: Replace row-level R loops and string-key lookups with vectorized integer-index operations using `data.table`.**

### A. `build_neighbor_lookup` â†’ Vectorized join

Instead of building a 6.46M-element named character vector and probing it row by row:

1. Create an integer-keyed `data.table` mapping `(id, year) â†’ row_index`.
2. Expand the neighbor list into an edge table: `(source_row, neighbor_id, year)`.
3. Perform a single keyed `data.table` join to resolve all neighbor row indices at once.

This replaces ~25.8M interpreted string lookups with one vectorized equi-join.

### B. `compute_neighbor_stats` â†’ Grouped `data.table` aggregation

Instead of `lapply` over 6.46M rows per variable:

1. Use the edge table from (A), which maps `source_row â†’ neighbor_row`.
2. For each variable, extract neighbor values vectorially, then `group by source_row` and compute `max`, `min`, `mean` in one `data.table` aggregation.

This replaces 5 Ã— 6.46M R function calls with 5 vectorized grouped aggregations.

### Expected speedup

| Component | Before | After | Factor |
|---|---|---|---|
| Neighbor lookup | ~40 hrs | ~1â€“3 min | ~1000Ã— |
| Neighbor stats (Ã—5) | ~45 hrs | ~2â€“5 min | ~500Ã— |
| **Total neighbor features** | **~86 hrs** | **~5â€“10 min** | **~500â€“1000Ã—** |

Memory: the edge table is ~25.8M rows Ã— 3 integer columns â‰ˆ 600 MB, which fits in 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Vectorized neighbor lookup via data.table join
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # data must be a data.table (or will be converted)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- Map each cell id to its neighbor cell ids (time-invariant) ---
  # id_to_ref: cell_id -> position in id_order (and in neighbors list)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: source_cell_id -> neighbor_cell_id
  # This is done once (not per year).
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref) {
    nb_refs <- neighbors[[ref]]
    if (length(nb_refs) == 0L) return(NULL)
    data.table(
      source_id   = id_order[ref],
      neighbor_id = id_order[nb_refs]
    )
  }))
  # edge_list has ~1.37M rows (directed rook edges, time-invariant)

  # --- Expand by year: cross-join edges with years ---
  years <- sort(unique(dt$year))
  # Cartesian expansion: each spatial edge exists in every year
  edge_list[, k := 1L]
  year_dt <- data.table(year = years, k = 1L)
  edge_year <- edge_list[year_dt, on = "k", allow.cartesian = TRUE]
  edge_year[, k := NULL]
  # edge_year now has ~1.37M Ã— 28 â‰ˆ 38.4M rows
  # (columns: source_id, neighbor_id, year)

  # --- Resolve source_row and neighbor_row via keyed join ---
  # Build lookup: (id, year) -> row_idx
  row_lookup <- dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # Resolve source rows
  edge_year[row_lookup, source_row := i.row_idx,
            on = .(source_id = id, year = year)]

  # Resolve neighbor rows
  edge_year[row_lookup, neighbor_row := i.row_idx,
            on = .(neighbor_id = id, year = year)]

  # Drop edges where either side is missing (boundary / missing year)
  edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

  # Return only the integer columns needed downstream
  edge_year[, .(source_row, neighbor_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Vectorized neighbor stats via grouped data.table aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_vec <- function(data, edge_dt, var_name) {
  # data can be data.frame or data.table
  vals <- data[[var_name]]
  n <- nrow(data)

  # Attach neighbor values to edge table (vectorized extraction)
  work <- copy(edge_dt)
  work[, nval := vals[neighbor_row]]

  # Drop NA neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation â€” one pass
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = source_row]

  # Allocate full-length result columns (NA default)
  res_max  <- rep(NA_real_, n)
  res_min  <- rep(NA_real_, n)
  res_mean <- rep(NA_real_, n)

  res_max[agg$source_row]  <- agg$nb_max
  res_min[agg$source_row]  <- agg$nb_min
  res_mean[agg$source_row] <- agg$nb_mean

  # Return a 3-column data.table with standard naming
  setNames(
    data.frame(res_max, res_min, res_mean),
    paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Drop-in replacement for the outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Build edge table once (replaces build_neighbor_lookup)
edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)

# Compute and attach neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  nb_feats <- compute_neighbor_stats_vec(cell_data, edge_dt, var_name)
  cell_data <- cbind(cell_data, nb_feats)
  message(sprintf("Done: %s neighbor features", var_name))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Random Forest prediction (unchanged â€” model is preserved)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
# (Use your existing predict() call exactly as before.)
```

---

## Verification Checklist

| Requirement | Status |
|---|---|
| Trained Random Forest model untouched | âœ… â€” No retraining; `predict()` call unchanged |
| Numerical estimand preserved | âœ… â€” Same `max`, `min`, `mean` computed on identical neighbor sets |
| Column names match original pipeline | âœ… â€” `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` (adjust suffix to match your `compute_and_add_neighbor_features` naming) |
| Fits in 16 GB RAM | âœ… â€” Edge table â‰ˆ 600 MB; `cell_data` â‰ˆ 5 GB; headroom remains |
| Expected runtime | ~5â€“10 minutes total (vs. 86+ hours) |

**Summary:** The 86-hour runtime was caused by 6.46 million interpreted R loop iterations performing string construction and named-vector hash lookups against a multi-million-entry table, repeated for both lookup construction and per-variable stats computation. Replacing these with vectorized `data.table` keyed joins and grouped aggregations eliminates the per-row interpreted overhead entirely, yielding an estimated ~500â€“1000Ã— speedup while producing numerically identical output columns for the downstream (unchanged) Random Forest prediction.
END_RESPONSE>>>
