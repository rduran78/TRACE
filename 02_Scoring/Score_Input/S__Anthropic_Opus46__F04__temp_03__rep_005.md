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
run_id: Anthropic_F04_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_005.md

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
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M â€” each lookup is **O(n)** in the worst case for R's hashed named vectors, but the sheer repetition across 6.46M rows is devastating).
- An `is.na` filter and `as.integer` coercion.

This produces ~6.46 million list elements. The dominant cost is the **per-row string construction and hash-table lookup repeated millions of times inside an interpreted R loop**.

### 2. `compute_neighbor_stats` â€” Another `lapply` over 6.46 million elements

For each row, it subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable). That is ~32.3 million R-level function invocations in total. The per-element overhead of the anonymous function, `is.na`, and three summary calls dominates.

### 3. Scaling arithmetic

- 6.46M rows Ã— average ~4 rook neighbors Ã— 5 variables = ~129M neighbor value lookups.
- The `lapply` loops are pure R interpreter loops with no vectorization.
- Estimated wall-clock: 86+ hours is consistent with this anti-pattern.

---

## Optimization Strategy

**Replace all row-level R loops with fully vectorized `data.table` grouped operations.**

The key insight: the neighbor lookup can be expressed as an **edge list join**. Instead of iterating row by row:

1. **Build an edge table** `(cell_id, neighbor_cell_id)` from the `nb` object â€” done once, ~1.37M rows.
2. **Cross-join with years** to get `(cell_id, year, neighbor_cell_id)` â€” ~1.37M Ã— 28 â‰ˆ 38.5M rows (fits in RAM).
3. **Join** the edge table to the data on `(neighbor_cell_id, year)` to pull neighbor values â€” one vectorized join per variable.
4. **Group-by** `(cell_id, year)` and compute `max`, `min`, `mean` â€” one vectorized aggregation per variable.

This eliminates all `lapply` loops and leverages `data.table`'s radix-based joins and grouped aggregation, which are C-level and cache-friendly.

**Expected speedup:** from 86+ hours to **~2â€“10 minutes** on the same laptop.

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical outputs (max, min, mean of neighbor values) are identical to the original code.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build the directed edge list from the nb object (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  
  data.table(
    cell_id          = id_order[from_idx],
    neighbor_cell_id = id_order[to_idx]
  )
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows: (cell_id, neighbor_cell_id)

# ---------------------------------------------------------------
# STEP 2: Convert cell_data to data.table and key it
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)

# Ensure id and year columns are keyed for fast joins
# (Adjust column names if yours differ; assumed "id" and "year")
setkey(dt, id, year)

# ---------------------------------------------------------------
# STEP 3: Expand edges Ã— years and compute neighbor features
# ---------------------------------------------------------------
# Get unique years once
all_years <- sort(unique(dt$year))

# Cross-join edges with all years: ~1.37M Ã— 28 â‰ˆ 38.5M rows
# This is the full set of (cell_id, year, neighbor_cell_id) triples
edges_by_year <- CJ_dt <- edges[, .(year = all_years), by = .(cell_id, neighbor_cell_id)]
setkey(edges_by_year, neighbor_cell_id, year)

# ---------------------------------------------------------------
# STEP 4: For each source variable, join + aggregate
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  
  # Subset only the columns we need from dt for the join
  # Rename id -> neighbor_cell_id so we can join on neighbor side
  lookup_cols <- c("id", "year", var)
  lookup_dt   <- dt[, ..lookup_cols]
  setnames(lookup_dt, old = "id", new = "neighbor_cell_id")
  setkey(lookup_dt, neighbor_cell_id, year)
  
  # Join: attach the neighbor's value of `var` to every edge-year row
  joined <- lookup_dt[edges_by_year, on = .(neighbor_cell_id, year), nomatch = NA]
  # joined has columns: neighbor_cell_id, year, <var>, cell_id
  
  # Aggregate by (cell_id, year), dropping NAs as the original code does
  agg <- joined[!is.na(get(var)),
                 .(nbr_max  = max(get(var)),
                   nbr_min  = min(get(var)),
                   nbr_mean = mean(get(var))),
                 by = .(cell_id, year)]
  
  # Name the new columns to match original naming convention
  new_names <- paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  setnames(agg, c("nbr_max", "nbr_min", "nbr_mean"), new_names)
  
  # Merge back into dt
  setkey(agg, cell_id, year)
  dt <- agg[dt, on = .(cell_id = id, year)]
  
  # The join above renames cell_id; fix it back
  setnames(dt, "cell_id", "id")
  setkey(dt, id, year)
  
  # Cells with zero valid neighbors get NA (same as original)
  # This is automatic because they won't appear in `agg`.
  
  message(paste0("Done: ", var))
}

# ---------------------------------------------------------------
# STEP 5: Convert back to data.frame if downstream code expects it
# ---------------------------------------------------------------
cell_data <- as.data.frame(dt)

# ---------------------------------------------------------------
# STEP 6: Run the already-trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Equivalent

| Original logic | Vectorized equivalent |
|---|---|
| `idx_lookup[neighbor_keys]` â€” find row of each neighbor in same year | `join on (neighbor_cell_id, year)` â€” identical match |
| `neighbor_vals[!is.na(neighbor_vals)]` â€” drop NAs before stats | `joined[!is.na(get(var)), ...]` â€” identical filter |
| `c(max(...), min(...), mean(...))` per row | `.(max(...), min(...), mean(...))` grouped by `(cell_id, year)` â€” identical aggregation |
| Cells with no valid neighbors â†’ `c(NA, NA, NA)` | Non-matching rows in the left join remain `NA` â€” identical |

The numerical estimand is preserved exactly.

---

## Memory Budget Check

| Object | Approximate size |
|---|---|
| `dt` (6.46M Ã— 110 cols, mostly numeric) | ~5.4 GB |
| `edges_by_year` (38.5M Ã— 3 cols, integer+integer+integer) | ~0.9 GB |
| `joined` (38.5M Ã— 4 cols, peak during loop) | ~1.2 GB |
| Overhead + R session | ~2 GB |
| **Total peak** | **~9.5 GB** (fits in 16 GB) |

If memory is tight, process one variable at a time (as the loop already does) and `rm(joined, agg)` with `gc()` at the end of each iteration.

---

## Summary

| Aspect | Before | After |
|---|---|---|
| Core mechanism | 6.46M-iteration `lapply` with string ops | Vectorized `data.table` keyed join + grouped aggregation |
| Estimated runtime | 86+ hours | ~2â€“10 minutes |
| RAM peak | Similar (list of 6.46M elements) | ~9.5 GB (fits 16 GB) |
| Numerical output | max, min, mean of neighbor values | Identical |
| Trained RF model | Untouched | Untouched |
END_RESPONSE>>>
