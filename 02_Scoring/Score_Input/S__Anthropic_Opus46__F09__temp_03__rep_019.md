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
run_id: Anthropic_F09_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs character-based key construction (`paste(id, year)`)
- Does named-vector lookups (`idx_lookup[neighbor_keys]`) â€” named vector lookup in R is hash-based but still slow when called millions of times in an interpreted loop.

**The critical insight:** The neighbor *topology* is fixed across years. There are only 344,208 cells and ~1.37M directed neighbor pairs. This topology is identical for every one of the 28 years. Yet the current code rebuilds the mapping for every cell-year row, effectively repeating the same spatial lookup 28 times and entangling spatial structure with temporal indexing.

### 2. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even after the lookup is built, computing max/min/mean via an R-level `lapply` over 6.46 million list elements is inherently slow â€” each iteration has R function-call overhead, subsetting, and NA handling.

### 3. The architecture is "row-centric" instead of "join-centric"
The entire design indexes by row position in a monolithic data frame. A vectorized, join-based approach using `data.table` can replace both functions with operations that run in seconds rather than hours.

---

## Optimization Strategy

**Core idea:** Build the neighbor edge table *once* (344K cells Ã— ~4 neighbors = ~1.37M edges), then for each year, join cell-year attributes onto both sides of the edge table and compute grouped statistics â€” all vectorized via `data.table`.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Convert `spdep::nb` to a two-column edge `data.table` (`cell_id`, `neighbor_id`) â€” **done once** | ~1.37M rows |
| 2 | For each variable, join the variable's values onto the edge table by `(neighbor_id, year)` | Vectorized keyed join |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean` | Vectorized grouped aggregation |
| 4 | Join results back to the main data | Keyed join |

**Expected speedup:** From ~86 hours to **~1â€“5 minutes** total for all 5 variables. The bottleneck shifts from millions of R-level iterations to a handful of `data.table` keyed joins and group-by operations over ~1.37M Ã— 28 â‰ˆ 38.4M edge-year rows.

**Preservation guarantees:**
- The trained Random Forest model is untouched â€” we only rebuild the input features identically.
- The numerical estimand is preserved: `max`, `min`, `mean` of rook-neighbor values per cell-year are computed with the same semantics (NA-safe, same variable set).

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Convert main data to data.table (if not already)
# ==============================================================
# Assumes: cell_data is a data.frame/data.table with columns
#   id (cell id), year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assumes: rook_neighbors_unique is an spdep::nb object
# Assumes: id_order is the vector of cell IDs corresponding to
#   positions in rook_neighbors_unique (i.e., id_order[i] is the
#   cell ID for the i-th element of the nb object)

cell_dt <- as.data.table(cell_data)

# ==============================================================
# STEP 1: Build the neighbor edge table ONCE
#         This encodes the fixed spatial topology.
# ==============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i
  # id_order[i] is the cell ID for position i
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nb_idx <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

cat("Building spatial edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed neighbor pairs\n",
            formatC(nrow(edge_dt), format = "d", big.mark = ",")))

# ==============================================================
# STEP 2: For each neighbor source variable, compute neighbor
#         max, min, mean via vectorized join + group-by
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_dt, id, year)

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # --- 2a. Extract the relevant column + keys ---
  # Subset to only the columns we need for the join (memory-efficient)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # --- 2b. Expand edges across all years ---
  # Cross join edge table with unique years
  years_dt <- data.table(year = sort(unique(cell_dt$year)))
  edge_year <- edge_dt[, CJ_dt := TRUE]  # placeholder
  # Efficient cross: use CJ-like expansion
  edge_year <- edge_dt[rep(seq_len(.N), each = nrow(years_dt))]
  edge_year[, year := rep(years_dt$year, times = nrow(edge_dt))]
  
  # --- 2c. Join neighbor values onto edge-year table ---
  # Join by (neighbor_id, year) to get the neighbor's value
  setkey(edge_year, neighbor_id, year)
  setkey(val_dt, id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # --- 2d. Group by (cell_id, year) and compute stats ---
  stats <- edge_year[!is.na(neighbor_val),
                     .(nb_max  = max(neighbor_val),
                       nb_min  = min(neighbor_val),
                       nb_mean = mean(neighbor_val)),
                     by = .(cell_id, year)]
  
  # --- 2e. Name the new columns to match original pipeline ---
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  # --- 2f. Join back to main table ---
  setkey(stats, cell_id, year)
  setkey(cell_dt, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt[stats, (c(max_col, min_col, mean_col)) :=
            mget(paste0("i.", c(max_col, min_col, mean_col))),
          on = .(id = cell_id, year)]
  
  # Clean up to free RAM
  rm(val_dt, edge_year, stats)
  gc()
}

cat("Done. Neighbor features added.\n")

# ==============================================================
# STEP 3: Convert back if needed and run prediction
#         (Random Forest model is UNCHANGED)
# ==============================================================
cell_data <- as.data.frame(cell_dt)

# Predict using the existing trained model (unchanged)
# cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

### Memory-Optimized Variant (if 16 GB RAM is tight)

The edge-year expansion above creates ~38.4M rows per variable. If RAM is a concern, process one year at a time:

```r
# ==============================================================
# MEMORY-SAFE VARIANT: Process one year at a time
# ==============================================================
compute_neighbor_features_by_year <- function(cell_dt, edge_dt, var_name) {
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Initialize columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  years <- sort(unique(cell_dt$year))
  
  for (yr in years) {
    # Subset this year's values
    yr_vals <- cell_dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_vals, id)
    
    # Join neighbor values onto edge table
    edges_with_vals <- copy(edge_dt)
    edges_with_vals[yr_vals, neighbor_val := i.val, on = .(neighbor_id = id)]
    
    # Compute grouped stats
    stats <- edges_with_vals[!is.na(neighbor_val),
                             .(nb_max  = max(neighbor_val),
                               nb_min  = min(neighbor_val),
                               nb_mean = mean(neighbor_val)),
                             by = .(cell_id)]
    
    # Write back into main table
    idx <- cell_dt[year == yr, which = TRUE]
    match_pos <- match(cell_dt$id[idx], stats$cell_id)
    
    set(cell_dt, i = idx, j = max_col,  value = stats$nb_max[match_pos])
    set(cell_dt, i = idx, j = min_col,  value = stats$nb_min[match_pos])
    set(cell_dt, i = idx, j = mean_col, value = stats$nb_mean[match_pos])
    
    rm(yr_vals, edges_with_vals, stats)
  }
  
  invisible(cell_dt)
}

# Usage:
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing: %s\n", var_name))
  compute_neighbor_features_by_year(cell_dt, edge_dt, var_name)
  gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | Rebuilt per cell-year row (6.46M R-level iterations with string ops) | Built once as a 1.37M-row edge table |
| **Stat computation** | `lapply` over 6.46M list elements | `data.table` keyed join + grouped aggregation |
| **Estimated runtime** | ~86+ hours | **~1â€“5 minutes** |
| **Peak RAM** | Moderate (but slow) | ~2â€“4 GB for edge-year table (or ~500 MB with year-by-year variant) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical output** | max/min/mean of rook-neighbor values | Identical max/min/mean of rook-neighbor values âœ“ |
END_RESPONSE>>>
