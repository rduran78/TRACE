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
run_id: Anthropic_F06_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys (`paste(id, year, sep="_")`).
4. Performs named-vector lookups in `idx_lookup`.

The string-key construction and named-vector lookup for 6.46M rows is extremely slow. Named vector lookup in R is O(n) in the worst case for each access because it uses linear hashing buckets, and `paste()` over millions of rows inside `lapply` generates massive garbage-collection pressure. The result is a **list of 6.46 million integer vectors** â€” itself a large memory object.

### Bottleneck B: `compute_neighbor_stats` â€” Called 5 times, each iterating over 6.46M rows

Each call does `lapply` over 6.46M entries, subsetting a numeric vector and computing `max/min/mean`. While each individual operation is trivial, the R-level loop overhead across 6.46M iterations Ã— 5 variables â‰ˆ 32.3M R-level function calls makes this very slow.

### Why raster focal/kernel operations are *not* a direct replacement

Focal operations assume a regular grid with a fixed rectangular window. Here, the grid cells have **irregular rook neighborhoods** (boundary cells, missing cells, NA handling). Forcing this into a raster focal operation would require: (a) confirming the grid is complete and regular, (b) handling NAs at boundaries identically, and (c) verifying numerical equivalence. The comment in the prompt is apt â€” it's a useful *analogy* but the safest path that **preserves the original numerical estimand** is to vectorize the existing logic using `data.table` joins rather than switching to raster focal.

### Summary

| Component | Current Complexity | Problem |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named lookups | ~hours of string ops |
| `compute_neighbor_stats` | 6.46M Ã— 5 R-level `lapply` iterations | ~hours of loop overhead |
| Total | ~86+ hours estimated | |

---

## 2. Optimization Strategy

### Strategy: Vectorized `data.table` join approach

1. **Expand the `nb` object into an edge table** (`cell_id â†’ neighbor_id`) â€” only ~1.37M edges, done once.
2. **Join the edge table to the panel data** on `(neighbor_id, year)` to pull neighbor values â€” this is a single keyed `data.table` merge, extremely fast.
3. **Group-by aggregate** `(cell_id, year)` to compute `max`, `min`, `mean` â€” fully vectorized in C via `data.table`.
4. **Left-join** the results back to the main table.
5. Repeat for each of the 5 variables (or do all 5 simultaneously).

This eliminates all R-level loops. Expected runtime: **minutes, not hours**.

### Why this preserves the numerical estimand

- The same rook-neighbor relationships are used (from the same `nb` object).
- The same `max`, `min`, `mean` aggregations are computed over the same non-NA neighbor values.
- The same NA propagation rules apply (no neighbors â†’ NA).
- The trained Random Forest model is not touched.

---

## 3. Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build edge table from the nb object (done once)
# ============================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: a list of integer vectors

  # id_order maps position -> cell_id
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # nb objects use 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(
      cell_id     = id_order[i],
      neighbor_id = id_order[nb_idx]
    )
  }))
  return(edges)
}

# ============================================================
# STEP 2: Compute neighbor stats for all variables at once
# ============================================================
compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          source_vars) {
  # Convert to data.table if needed (by reference if already)
  dt <- as.data.table(cell_data)

  # --- Build edge table ---
  message("Building edge table...")
  edges <- build_edge_table(id_order, neighbors_nb)
  # edges has columns: cell_id, neighbor_id
  # ~1.37M rows

  # --- Key the main table for fast joins ---
  # We need to look up neighbor values by (neighbor_id, year)
  # Create a lookup keyed on (id, year)
  setkey(dt, id, year)

  # --- For each variable, join + aggregate ---
  for (var_name in source_vars) {
    message(sprintf("Processing variable: %s", var_name))

    # Build a slim lookup: just id, year, and the variable
    lookup <- dt[, .(id, year, value = get(var_name))]
    setkey(lookup, id, year)

    # Expand edges Ã— years:
    # Instead of cross-joining edges with all years (expensive in memory),
    # we join edges to the main data to get (cell_id, year, neighbor_id),
    # then join again to get neighbor values.

    # Step A: Get all (cell_id, year) pairs that exist in data
    cell_years <- dt[, .(cell_id = id, year)]

    # Step B: Join cell_years with edges on cell_id
    #   Result: (cell_id, year, neighbor_id) â€” one row per neighbor per cell-year
    cell_year_neighbors <- edges[cell_years,
                                 on = .(cell_id),
                                 .(cell_id, year = i.year, neighbor_id),
                                 allow.cartesian = TRUE,
                                 nomatch = NA]

    # Drop rows where neighbor_id is NA (cells with no neighbors didn't
    # produce edges, so this is just defensive)
    cell_year_neighbors <- cell_year_neighbors[!is.na(neighbor_id)]

    # Step C: Join to get neighbor values
    cell_year_neighbors[lookup,
                        neighbor_val := i.value,
                        on = .(neighbor_id = id, year)]

    # Step D: Aggregate â€” drop NAs in neighbor_val, compute max/min/mean
    agg <- cell_year_neighbors[
      !is.na(neighbor_val),
      .(
        nb_max  = max(neighbor_val),
        nb_min  = min(neighbor_val),
        nb_mean = mean(neighbor_val)
      ),
      by = .(cell_id, year)
    ]

    # Rename columns to match expected output naming convention
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    # Step E: Left-join aggregated stats back to main table
    setkey(agg, cell_id, year)
    dt[agg, (max_col)  := get(paste0("i.", max_col)),  on = .(id = cell_id, year)]
    dt[agg, (min_col)  := get(paste0("i.", min_col)),  on = .(id = cell_id, year)]
    dt[agg, (mean_col) := get(paste0("i.", mean_col)), on = .(id = cell_id, year)]

    # Cells with no valid neighbors remain NA (the default for unmatched joins)

    # Clean up intermediate objects to manage memory
    rm(lookup, cell_years, cell_year_neighbors, agg)
    gc()
  }

  return(dt)
}

# ============================================================
# STEP 3: Main execution
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  neighbors_nb     = rook_neighbors_unique,
  source_vars      = neighbor_source_vars
)

# cell_data now has 15 new columns:
#   ntl_nb_max, ntl_nb_min, ntl_nb_mean,
#   ec_nb_max,  ec_nb_min,  ec_nb_mean,
#   pop_density_nb_max, pop_density_nb_min, pop_density_nb_mean,
#   def_nb_max, def_nb_min, def_nb_mean,
#   usd_est_n2_nb_max, usd_est_n2_nb_min, usd_est_n2_nb_mean

# ============================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + named-vector lookups | 1.37M-row edge table built once via `rbindlist` |
| Neighbor value retrieval | 6.46M R-level list element accesses per variable | Single keyed `data.table` join (~27M rows for cartesian expansion) |
| Aggregation | 6.46M `lapply` calls with `max/min/mean` per variable | Single `data.table` group-by (C-level) |
| Total loop iterations in R | ~32.3M | ~0 (fully vectorized) |
| **Estimated runtime** | **86+ hours** | **~5â€“15 minutes** |
| Peak memory | ~6.46M-element list of integer vectors + data | ~27M-row intermediate join table (managed per variable with `gc()`) |

### Memory safety on 16 GB RAM

The largest intermediate object is the cartesian join `cell_year_neighbors`. With ~1.37M edges Ã— 28 years Ã· 2 (directed, but some edges only exist for cells present in certain years) â‰ˆ ~19â€“27M rows Ã— 3 columns Ã— 8 bytes â‰ˆ ~650 MB. This is processed one variable at a time and freed with `rm(); gc()`. The main `data.table` (~6.46M Ã— ~125 columns) â‰ˆ ~6.5 GB. Total peak â‰ˆ ~8â€“9 GB, well within 16 GB.

### Numerical equivalence

The optimized code computes exactly `max`, `min`, and `mean` over the same set of non-NA rook-neighbor values for each `(cell_id, year)` pair, using the same `rook_neighbors_unique` nb object. The results are numerically identical to the original implementation. The pre-trained Random Forest model is loaded and applied without modification.
END_RESPONSE>>>
