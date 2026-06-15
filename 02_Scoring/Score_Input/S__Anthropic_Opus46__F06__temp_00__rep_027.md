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
run_id: Anthropic_F06_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” dictionary lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for each neighbor (expensive string allocation Ã— ~4 neighbors Ã— 6.46M rows â‰ˆ 25M+ string constructions).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` â€” named character vector indexing is **O(n)** per lookup in base R because it does linear or hashed matching over a 6.46M-length named vector.

The result is a **list of 6.46M integer vectors**. This alone can take many hours and consumes substantial RAM.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M elements, repeated 5Ã—

Each call iterates over 6.46M list elements, subsets a numeric vector, removes NAs, and computes `max/min/mean`. The per-element overhead of R's `lapply` + anonymous function + subsetting is small individually but multiplied by 6.46M Ã— 5 variables = 32.3M R-level function calls.

### Why raster focal/kernel operations don't directly apply

The comment in the docstring is a red herring for this case. Focal operations assume a regular rectangular grid with a fixed kernel. Here, the grid cells have an **irregular neighbor structure** (coastal cells, boundary cells have fewer neighbors) stored in an `spdep::nb` object. Focal operations would require reconstructing a complete rectangular raster and handling NA masking â€” possible but fragile and not guaranteed to preserve the exact numerical results for irregular boundaries. The better approach is to **vectorize the neighbor computation directly using data.table joins and matrix operations**.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup factor |
|------|---------|----------|---------------|
| Neighbor lookup | 6.46M `paste` + named-vector lookups | Pre-build a **directed edge table** (`data.table` with `from_row, to_row`) via keyed joins â€” no string operations at runtime | ~100â€“500Ã— |
| Neighbor stats | `lapply` over 6.46M list elements Ã— 5 vars | **Vectorized grouped aggregation** via `data.table`: join edge table to values, group by `from_row`, compute `max/min/mean` in one pass per variable | ~50â€“200Ã— |
| Memory | 6.46M-element list of integer vectors (~2â€“4 GB) | Edge table: ~25M rows Ã— 2 integer cols (~200 MB) | ~10â€“20Ã— less RAM |

**Expected total runtime: 2â€“10 minutes** instead of 86+ hours.

### Key insight

Instead of building a row-level adjacency list, build an **edge data.table** where each row is `(focal_row_index, neighbor_row_index)`. Then for each variable, join the neighbor values onto this edge table and do a grouped `max/min/mean` by `focal_row_index`. This is a classic "graph aggregation via edge list + grouped reduction" pattern that `data.table` handles extremely efficiently.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table (non-destructive)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec,
# pop_density, def, usd_est_n2, and ~110 other predictor columns.
# Assumes rook_neighbors_unique is an spdep::nb list indexed by
# position in id_order, and id_order is the vector of unique cell IDs.

cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]  # preserve original row ordering

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the directed edge table (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(cell_dt, id_order, nb_list) {
  # 1a. Expand the nb object into a spatial edge list: (focal_id, neighbor_id)
  #     nb_list[[i]] contains integer indices into id_order for cell id_order[i]
  n_cells <- length(id_order)
  
  # Pre-allocate: count total edges
  n_edges_spatial <- sum(vapply(nb_list, function(x) {
    # spdep::nb uses 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  focal_ids    <- integer(n_edges_spatial)
  neighbor_ids <- integer(n_edges_spatial)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_list[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    nn <- length(nbrs)
    focal_ids[pos:(pos + nn - 1L)]    <- id_order[i]
    neighbor_ids[pos:(pos + nn - 1L)] <- id_order[nbrs]
    pos <- pos + nn
  }
  
  spatial_edges <- data.table(
    focal_id    = focal_ids[1:(pos - 1L)],
    neighbor_id = neighbor_ids[1:(pos - 1L)]
  )
  
  # 1b. Create a lookup from (id, year) -> row_idx
  id_year_lookup <- cell_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)
  
  # 1c. Get unique years
  years <- sort(unique(cell_dt$year))
  
  # 1d. Cross-join spatial edges with years, then map to row indices
  #     This creates the full (focal_row, neighbor_row) edge table
  edge_full <- spatial_edges[, .(year = years), by = .(focal_id, neighbor_id)]
  
  # Map focal (id, year) -> focal_row_idx
  setkey(edge_full, focal_id, year)
  edge_full[id_year_lookup, focal_row := i.row_idx, on = .(focal_id = id, year)]
  
  # Map neighbor (id, year) -> neighbor_row_idx
  setkey(edge_full, neighbor_id, year)
  edge_full[id_year_lookup, neighbor_row := i.row_idx, on = .(neighbor_id = id, year)]
  
  # Drop edges where either side is missing (cell-year not in data)
  edge_final <- edge_full[!is.na(focal_row) & !is.na(neighbor_row),
                          .(focal_row, neighbor_row)]
  setkey(edge_final, focal_row)
  
  return(edge_final)
}

message("Building edge table...")
t0 <- Sys.time()
edge_table <- build_edge_table(cell_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table built: %d edges in %.1f seconds",
                nrow(edge_table), difftime(Sys.time(), t0, units = "secs")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_table) {
  # Extract the variable values for neighbor rows
  vals <- cell_dt[[var_name]]
  
  # Attach neighbor values to edge table
  work <- copy(edge_table)
  work[, nval := vals[neighbor_row]]
  
  # Remove edges where neighbor value is NA
  work <- work[!is.na(nval)]
  
  # Grouped aggregation: max, min, mean by focal_row
  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]
  
  # Initialize result columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Fill in computed values
  cell_dt[stats$focal_row, (max_col)  := stats$nb_max]
  cell_dt[stats$focal_row, (min_col)  := stats$nb_min]
  cell_dt[stats$focal_row, (mean_col) := stats$nb_mean]
  
  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Outer loop â€” compute all 5 Ã— 3 = 15 neighbor features
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  t1 <- Sys.time()
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_table)
  message(sprintf("  Done in %.1f seconds", difftime(Sys.time(), t1, units = "secs")))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Convert back to data.frame and predict with trained RF
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, row_idx := NULL]  # remove helper column
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is unchanged â€” predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor relationships** | The edge table is built from the identical `rook_neighbors_unique` nb object and `id_order`. Every `(focal_id, neighbor_id, year)` triple is the same. |
| **Same aggregation functions** | `max`, `min`, `mean` are computed identically; `data.table` uses the same base R primitives. |
| **NA handling** | Edges with `NA` neighbor values are dropped before aggregation, exactly matching `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` for all three stats, matching `if (length(idx) == 0) return(c(NA, NA, NA))`. |
| **Trained RF model untouched** | No retraining occurs. The 15 new columns have identical names and identical values, so `predict()` produces identical results. |
| **Floating-point equivalence** | `mean()` on the same set of non-NA doubles in the same precision yields identical results. `data.table`'s grouped `mean` calls R's `mean` internally. For strict verification, you can run both implementations on a small subset and confirm `all.equal()`. |

---

## 5. Performance Estimate

| Phase | Estimated time | Estimated RAM |
|-------|---------------|---------------|
| Edge table construction | 30â€“90 seconds | ~1.5 GB peak (edge_full with years before filtering) |
| Per-variable neighbor stats (Ã—5) | 10â€“30 seconds each | ~400 MB working set per variable |
| **Total** | **~3â€“5 minutes** | **< 10 GB peak** (fits in 16 GB) |

This represents a speedup of roughly **1,000â€“1,500Ã—** over the original 86+ hour estimate.
END_RESPONSE>>>
