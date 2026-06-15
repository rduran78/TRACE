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
run_id: Anthropic_F12_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable â€” only called 5 times total. This is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`lapply` over 6.46 million rows with per-row string operations**: For every single row, the function calls `as.character()`, performs character vector indexing into `id_to_ref`, calls `paste()` to construct neighbor keys, indexes into `idx_lookup`, and filters `NA`s. That's millions of `paste()` calls, each generating multiple key strings. With ~1.37M directed neighbor relationships spread across 28 years, the total number of key constructions is enormous.

2. **Redundant recomputation across years**: The neighbor *structure* is purely spatial â€” cell A's neighbors are the same cells regardless of year. Yet `build_neighbor_lookup()` recomputes the neighbor-cell-ID-to-row mapping **independently for every one of the 6.46 million rows**, re-doing the `paste` + lookup for every year. The spatial topology of 344,208 cells is being redundantly resolved 28 times (once per year per cell).

3. **Character key hashing at scale**: Using `paste(..., sep="_")` to build composite keys and named-vector lookup (`idx_lookup[neighbor_keys]`) is O(n) in the size of the name table for each lookup in the worst case, and the name table has 6.46 million entries.

`compute_neighbor_stats()` is comparatively cheap: it's just integer indexing into a numeric vector and computing `max/min/mean` on small neighbor sets. The `do.call(rbind, ...)` on a list of 6.46M length-3 vectors is a one-time cost per variable.

**Summary of bottleneck ranking:**
1. `build_neighbor_lookup()` â€” dominant cost (millions of paste + character lookups, redundant across years)
2. `compute_neighbor_stats()` â€” moderate cost, easily vectorizable
3. `do.call(rbind, ...)` â€” minor cost

---

## Optimization Strategy

1. **Exploit spatial-temporal separability**: Compute the spatial neighbor mapping once for 344,208 cells, then expand to cell-year rows via fast integer joins â€” not per-row string operations.

2. **Use `data.table` for fast equi-joins**: Replace the named-vector character lookup with `data.table` keyed joins, which are O(n log n) overall instead of O(nÂ²)-like repeated character hashing.

3. **Vectorize `compute_neighbor_stats()`**: Replace the per-row `lapply` + `do.call(rbind, ...)` with a grouped `data.table` aggregation that computes max/min/mean in one vectorized pass.

4. **Preserve the trained Random Forest model**: The output columns are numerically identical (same neighbor definitions, same max/min/mean computations), so the RF model remains valid with no retraining.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup + compute_neighbor_stats
# ============================================================

compute_and_add_all_neighbor_features <- function(cell_data, 
                                                   id_order, 
                                                   rook_neighbors_unique, 
                                                   neighbor_source_vars) {
  
  # Convert to data.table for speed; preserve original row order
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  
  # ----------------------------------------------------------
  # STEP 1: Build spatial neighbor edge list ONCE (344,208 cells)
  # ----------------------------------------------------------
  # id_order[i] is the cell id for the i-th entry in the nb object.
  # rook_neighbors_unique[[i]] gives integer indices into id_order
  # for the neighbors of cell id_order[i].
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  # ----------------------------------------------------------
  # STEP 2: Join edge list to cell-year data via data.table keys
  #         to get neighbor ROW indices for each focal row.
  # ----------------------------------------------------------
  # Create a mapping: (id, year) -> row index in dt
  # We join: for each focal row (focal_id, year), find all 
  # neighbor rows (neighbor_id, same year).
  
  # Keyed lookup table: neighbor_id + year -> row index
  neighbor_rows <- dt[, .(neighbor_id = id, year, neighbor_row = .row_order)]
  setkey(neighbor_rows, neighbor_id, year)
  
  # Focal rows joined to their spatial neighbors, then to neighbor data rows
  focal_edges <- dt[, .(focal_row = .row_order, focal_id = id, year)]
  
  # Merge focal rows with spatial edge list to get (focal_row, neighbor_id, year)
  setkey(edge_list, focal_id)
  setkey(focal_edges, focal_id)
  
  # This gives every (focal_row, neighbor_id, year) combination
  focal_neighbor <- edge_list[focal_edges, 
                              on = .(focal_id), 
                              .(focal_row, neighbor_id, year), 
                              allow.cartesian = TRUE, 
                              nomatch = NULL]
  
  # Now join to get the actual neighbor row in the data
  setkey(focal_neighbor, neighbor_id, year)
  focal_neighbor <- neighbor_rows[focal_neighbor, 
                                   on = .(neighbor_id, year), 
                                   .(focal_row, neighbor_row), 
                                   nomatch = NULL]
  
  # ----------------------------------------------------------
  # STEP 3: Vectorized neighbor stats per variable
  # ----------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    
    # Pull neighbor values via integer indexing (vectorized)
    focal_neighbor[, nval := dt[[var_name]][neighbor_row]]
    
    # Remove NA neighbor values
    valid <- focal_neighbor[!is.na(nval)]
    
    # Grouped aggregation â€” single vectorized pass
    stats <- valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed stats back by row index
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]
    
    # Clean up temp column
    focal_neighbor[, nval := NULL]
  }
  
  # ----------------------------------------------------------
  # STEP 4: Return as original data.frame, preserving row order
  # ----------------------------------------------------------
  dt[, .row_order := NULL]
  
  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” 
# the output columns are numerically identical to the original.
# Proceed directly to predict():
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Fixes the Bottleneck

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor topology resolution | 6.46M per-row `paste` + character lookups | One `rbindlist` over 344K cells + `data.table` keyed join |
| Year expansion | Implicit per-row string matching | Vectorized equi-join on `(neighbor_id, year)` |
| Stats computation | 6.46M `lapply` iterations + `do.call(rbind)` per variable | Single `data.table` grouped aggregation per variable |
| Estimated runtime | 86+ hours | **Minutes** (dominated by the `allow.cartesian` join, which `data.table` handles efficiently via radix-based merge) |
| Memory | Millions of small character vectors + intermediate lists | One edge `data.table` (~38M rows Ã— 2 integer cols â‰ˆ 0.6 GB) |
| Numerical output | max/min/mean of non-NA rook-neighbor values | Identical |
| RF model impact | â€” | None; no retraining needed |
END_RESPONSE>>>
