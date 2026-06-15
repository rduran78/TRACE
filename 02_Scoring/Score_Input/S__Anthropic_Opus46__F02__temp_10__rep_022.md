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
run_id: Anthropic_F02_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String operations**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called ~6.46M times, each producing a small character vector. This is extremely slow in R's single-threaded interpreted loop.
- **Named vector lookup (`idx_lookup[neighbor_keys]`)**: Named vector indexing in R is hash-table lookup, but doing it millions of times inside `lapply` has large per-call overhead.
- **Memory**: The resulting list of 6.46M integer vectors is itself a large, fragmented object.

### 2. `compute_neighbor_stats` â€” O(n) `lapply` with per-row subsetting, NA removal, and summary stats
- Called 5 times (once per variable), each iterating over 6.46M elements.
- `do.call(rbind, result)` on a 6.46M-element list is itself slow (row-binding millions of small vectors).

### Combined effect
The nested per-row R-level operations dominate. With ~6.46M rows and ~5 neighbor lookups per row, you are executing tens of millions of interpreted R function calls across the two functions, repeated for 5 variables. This easily accounts for the 86+ hour estimate.

---

## Optimization Strategy

**Replace row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup and neighbor-stat computation can be reformulated as a **single equi-join** followed by **grouped aggregation**, both of which `data.table` executes in optimized C.

**Steps:**

1. **Build an edge table** (`edge_dt`): one row per directed neighbor pair `(from_id, to_id)` from `rook_neighbors_unique`. This table has ~1.37M rows and is year-independent.

2. **Cross with years via join**: Join `edge_dt` to the main data on `(to_id, year)` to pull each neighbor's variable value. This is a keyed equi-join â€” O(n log n) in C, not interpreted R.

3. **Group-aggregate**: Group by `(from_id, year)` and compute `max`, `min`, `mean` in a single pass per variable.

4. **Merge back** to the main table.

This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` entirely. No list of 6.46M elements is ever created. Memory usage is dominated by the join intermediate (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows of integers/doubles), which fits comfortably in 16 GB.

**Expected speedup**: from 86+ hours to roughly 5â€“15 minutes total (all 5 variables).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the directed edge table from the nb object
#     rook_neighbors_unique is a list where element i contains
#     the indices (into id_order) of the neighbors of id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  # spdep nb objects use 0L to denote "no neighbours"

  nb <- nb[nb != 0L]
  if (length(nb) == 0L) return(NULL)
  data.table(from_id = id_order[i], to_id = id_order[nb])
}))
# edge_list now has columns: from_id, to_id   (~1.37 M rows)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Key the main table for fast joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(cell_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  For each neighbor source variable, compute max/min/mean
#     via a vectorised join + grouped aggregation, then merge.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # --- 3a. Build a slim table: only the columns we need for the join target

  #         (to_id == id in cell_dt, plus year and the variable)
  lookup_cols <- c("id", "year", var_name)
  neighbor_vals_dt <- cell_dt[, ..lookup_cols]
  setnames(neighbor_vals_dt, old = c("id", var_name),
           new = c("to_id", "val"))
  setkey(neighbor_vals_dt, to_id, year)

  # --- 3b. Expand edges Ã— years:
  #         Join edge_list to the main data on (from_id == id, year) to get
  #         one row per (from_id, year, to_id), then join to neighbor values.
  #
  #         Efficient approach: cross-join edges with the distinct years of
  #         each from_id, then look up neighbor values.
  #
  #         Even simpler: join edges directly to neighbor values, then bring

  #         back the from_id's year via the main data.
  #
  #         Simplest correct approach:
  #           For every (from_id, year) in cell_dt, find to_id neighbours
  #           and their values.  This is:
  #             cell_dt  â”€â”€[id == from_id]â”€â”€>  edge_list  â”€â”€[to_id, year]â”€â”€>  neighbor_vals_dt

  # Step A: attach from_id's years to edge_list
  from_years <- cell_dt[, .(id, year)]
  setnames(from_years, "id", "from_id")
  setkey(from_years, from_id)
  setkey(edge_list, from_id)

  # This is the big expansion: ~1.37 M edges Ã— 28 years â‰ˆ 38.4 M rows
  edges_by_year <- edge_list[from_years, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # columns: from_id, to_id, year

  # Step B: look up the neighbour's value
  edges_by_year_keyed <- edges_by_year  # already has to_id, year
  setkey(edges_by_year_keyed, to_id, year)
  edges_by_year_keyed <- neighbor_vals_dt[edges_by_year_keyed, on = c("to_id", "year")]
  # columns: to_id, year, val, from_id

  # Step C: aggregate per (from_id, year)
  stats <- edges_by_year_keyed[
    !is.na(val),
    .(nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)),
    keyby = .(from_id, year)
  ]

  # --- 3c. Rename columns to match the original feature names
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats,
           old = c("nb_max", "nb_min", "nb_mean"),
           new = new_names)
  setnames(stats, "from_id", "id")

  # --- 3d. Merge back to cell_dt

  #         Left join so rows with no valid neighbours get NA (preserving estimand).
  cell_dt <- stats[cell_dt, on = c("id", "year")]

  # Clean up heavy intermediates each iteration to stay within 16 GB

  rm(neighbor_vals_dt, from_years, edges_by_year, edges_by_year_keyed, stats)
  gc()

  message(sprintf("Done: %s", var_name))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Convert back to data.frame if needed downstream
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the already-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory Note

The largest intermediate is `edges_by_year` at ~38.4 M rows Ã— 3 integer/double columns â‰ˆ **0.9â€“1.2 GB**. After attaching the neighbor value it grows to ~38.4 M Ã— 4 columns â‰ˆ **1.2â€“1.5 GB**. Combined with `cell_dt` itself (~6.46 M Ã— 110 columns â‰ˆ 5â€“6 GB), peak usage stays **well under 16 GB**, especially since we `rm()` and `gc()` after each variable.

---

## Further Optional Speed-up

If even 5â€“15 minutes feels slow, you can additionally move `from_years` construction outside the loop (it is the same every iteration) and pre-allocate `edges_by_year` once:

```r
# Before the loop (avoids rebuilding the 38.4 M-row frame 5 times):
from_years <- cell_dt[, .(id, year)]
setnames(from_years, "id", "from_id")
setkey(edge_list, from_id)
edges_by_year <- edge_list[from_years, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
setkey(edges_by_year, to_id, year)
rm(from_years); gc()

for (var_name in neighbor_source_vars) {
  lookup_cols <- c("id", "year", var_name)
  neighbor_vals_dt <- cell_dt[, ..lookup_cols]
  setnames(neighbor_vals_dt, old = c("id", var_name), new = c("to_id", "val"))
  setkey(neighbor_vals_dt, to_id, year)

  merged <- neighbor_vals_dt[edges_by_year, on = c("to_id", "year")]

  stats <- merged[!is.na(val),
    .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
    keyby = .(from_id, year)
  ]

  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max","nb_min","nb_mean"), new_names)
  setnames(stats, "from_id", "id")

  cell_dt <- stats[cell_dt, on = c("id", "year")]
  rm(neighbor_vals_dt, merged, stats); gc()
  message(sprintf("Done: %s", var_name))
}

rm(edges_by_year); gc()
cell_data <- as.data.frame(cell_dt)
```

This version builds the 38.4 M-row edge frame **once** and reuses it, saving ~4 additional join operations. Expected total runtime: **2â€“5 minutes** on a modern laptop.

---

**Summary**: The original code's slowness stems from millions of interpreted R-level string operations and per-row list manipulations. By reformulating the problem as a `data.table` keyed join plus grouped aggregation (executed entirely in compiled C), we achieve orders-of-magnitude speedup while producing numerically identical features, preserving the trained Random Forest model and the original estimand.
END_RESPONSE>>>
