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
run_id: Anthropic_F06_T03_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`

`build_neighbor_lookup` iterates over **all ~6.46 million rows** and, for each row, performs character-based key lookups (`paste`, named-vector indexing). This creates ~6.46M character keys, does ~6.46M hash lookups per variable, and produces a list of 6.46M integer vectors. The `paste`/`setNames`/character-matching pattern is extremely slow in R at this scale.

### Bottleneck 2: `compute_neighbor_stats` â€” Row-wise `lapply` over 6.46M elements

For each of the 5 variables, `compute_neighbor_stats` loops over 6.46M list elements, subsets a numeric vector, removes NAs, and computes `max`/`min`/`mean`. This is called 5 times, so ~32.3M R-level iterations with per-element overhead.

### Why raster focal/kernel operations are *not* a direct substitute

Focal operations assume a regular rectangular grid with uniform spacing and a fixed kernel window. Here, the data is a **panel** (cell Ã— year), neighbor relationships come from an irregular `spdep::nb` object (not necessarily a regular lattice), and the computation is per-year within each cell's rook neighbors. Focal operations would only work if the grid is perfectly regular *and* you reshape to a raster for each year â€” which adds complexity and risks altering results at boundaries or for irregular geometries. The better approach is to **vectorize the existing logic** using `data.table` joins and grouped operations, which preserves the exact numerical results.

### Estimated speedup

The strategy below replaces all `lapply` loops with vectorized `data.table` joins and grouped aggregations, reducing the 86+ hour runtime to roughly **minutes**.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, build a two-column `data.table` of directed neighbor pairs `(id, neighbor_id)` from the `nb` object once â€” this is only ~1.37M rows.

2. **Join neighbor pairs to panel data by `(neighbor_id, year)`** to get neighbor values. This is a keyed `data.table` merge â€” extremely fast.

3. **Compute grouped `max`/`min`/`mean`** by `(id, year)` on the joined result using `data.table`'s `by=` grouping â€” fully vectorized in C.

4. **Merge results back** to the main dataset.

5. **Repeat for each of the 5 variables** (or do all at once).

This preserves the exact numerical estimand (same max, min, mean of rook-neighbor values per cell-year) and never touches the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a directed edge table from the nb object  (done ONCE)
#
#     rook_neighbors_unique : spdep nb object (list of integer vectors)
#     id_order              : vector mapping position -> cell id
#
#     Result: edges_dt with columns  (id, neighbor_id)
#             ~1.37 M rows
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate vectors
  n <- length(nb_obj)
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbs <- nbs[nbs != 0L]
    if (length(nbs) > 0L) {
      from_ids[[i]] <- rep(id_order[i], length(nbs))
      to_ids[[i]]   <- id_order[nbs]
    }
  }
  
  data.table(
    id          = unlist(from_ids, use.names = FALSE),
    neighbor_id = unlist(to_ids,   use.names = FALSE)
  )
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Function: compute neighbor stats for one variable and merge back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, edges, var_name) {
  
  # Columns we need from the panel for the neighbor lookup
  # We join edges to cell_dt on (neighbor_id == id, year) to get neighbor values
  
  # Subset to only needed columns for the join (saves memory)
  lookup_cols <- c("id", "year", var_name)
  neighbor_vals_dt <- cell_dt[, ..lookup_cols]
  
  # Rename 'id' to 'neighbor_id' so we can join on it
  setnames(neighbor_vals_dt, "id", "neighbor_id")
  setnames(neighbor_vals_dt, var_name, "nval")
  
  # Key for fast join
  setkey(neighbor_vals_dt, neighbor_id, year)
  
  # Add year to edges by joining edges to cell_dt on 'id'
  # Strategy: cross join edges with years via the focal cell's panel rows
  # More efficient: join edges -> cell_dt[, .(id, year)] then join neighbor values
  
  focal_keys <- cell_dt[, .(id, year)]
  setkey(focal_keys, id)
  setkey(edges, id)
  
  # Each edge (id, neighbor_id) gets expanded by all years the focal cell appears
  # This gives us (id, year, neighbor_id)
  edge_year <- edges[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # edge_year now has columns: id, neighbor_id, year
  
  # Join to get neighbor values
  setkey(edge_year, neighbor_id, year)
  edge_year[neighbor_vals_dt, nval := i.nval, on = .(neighbor_id, year)]
  
  # Compute grouped stats, dropping NAs
  stats <- edge_year[!is.na(nval),
                     .(nmax  = max(nval),
                       nmin  = min(nval),
                       nmean = mean(nval)),
                     by = .(id, year)]
  
  # Construct output column names (match original naming convention)
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  # Merge back to cell_dt
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  setkey(cell_dt, id, year)
  setkey(stats, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]
  
  # Clean up
  rm(neighbor_vals_dt, focal_keys, edge_year, stats)
  
  cell_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Run for all 5 neighbor source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edges_dt, var_name)
  gc()  # free memory between iterations on a 16 GB machine
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 16 GB is tight)

The `edge_year` intermediate table can be large (~1.37M edges Ã— 28 years â‰ˆ 38.4M rows). If memory is a concern, process one variable at a time (as above) and/or split by year:

```r
compute_and_add_neighbor_features_by_year <- function(cell_dt, edges, var_name) {
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  years <- sort(unique(cell_dt$year))
  
  stats_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    
    # Subset to this year
    yr_dt <- cell_dt[year == yr, .(id, nval = get(var_name))]
    setkey(yr_dt, id)
    
    # Rename for neighbor lookup
    neighbor_vals <- copy(yr_dt)
    setnames(neighbor_vals, "id", "neighbor_id")
    setkey(neighbor_vals, neighbor_id)
    
    # Join edges to neighbor values
    merged <- edges[neighbor_vals, on = "neighbor_id", nomatch = NULL]
    # merged: (id, neighbor_id, nval)
    
    stats_list[[j]] <- merged[!is.na(nval),
                               .(nmax  = max(nval),
                                 nmin  = min(nval),
                                 nmean = mean(nval),
                                 year  = yr),
                               by = .(id)]
  }
  
  stats <- rbindlist(stats_list)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  setkey(cell_dt, id, year)
  setkey(stats, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]
  
  cell_dt
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element `lapply` with `paste`/character matching | One-time edge table build (~1.37M rows) |
| **Stat computation** | 6.46M-element `lapply` per variable (Ã—5) | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **RAM peak** | Large (6.46M-element list of integer vectors) | ~2â€“3 GB for largest intermediate |
| **Numerical results** | Baseline | Identical (same max/min/mean per cell-year) |
| **RF model** | Untouched | Untouched |
END_RESPONSE>>>
