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
run_id: Anthropic_F06_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` character keys (expensive string allocation Ã— 6.46M rows Ã— ~4 neighbors each).
3. Indexes into `idx_lookup` (a named character vector â€” each lookup is O(1) amortized, but the sheer volume of string construction and named-vector indexing is slow).

The result is a **list of 6.46 million integer vectors**. This object alone consumes substantial memory and takes hours to build because of the per-row string operations.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsets values, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list is also slow.

### Why raster focal/kernel operations are *not* directly applicable

Focal operations assume a regular grid with a fixed rectangular kernel. Here the data is a panel (cell Ã— year), the neighbor structure is an irregular `spdep::nb` object (not all cells have 4 rook neighbors â€” boundary cells, missing cells), and the computation must respect the year dimension (neighbors only within the same year). A focal approach would require reshaping into a 3D array and handling irregular boundaries, risking numerical discrepancies. The better strategy is to **vectorize the existing logic using `data.table` joins and grouped aggregation**.

### Summary of time sinks

| Step | Estimated share | Root cause |
|---|---|---|
| `build_neighbor_lookup` | ~40% | 6.46M string-paste + named-vector lookups |
| `compute_neighbor_stats` (Ã—5) | ~55% | 6.46M R-level loops Ã— 5 variables |
| `do.call(rbind, ...)` (Ã—5) | ~5% | Binding 6.46M 3-element vectors |

---

## 2. Optimization Strategy

### Core idea: Replace row-level R loops with a single `data.table` equi-join + grouped aggregation.

**Step 1.** Build an **edge table** (a two-column data.table of `(cell_id, neighbor_id)`) from the `spdep::nb` object â€” done once, ~1.37M rows.

**Step 2.** Join the edge table to the panel data on `(neighbor_id, year)` to get neighbor variable values. This is a keyed `data.table` merge â€” extremely fast.

**Step 3.** Group by `(cell_id, year)` and compute `max`, `min`, `mean` for all 5 variables simultaneously in one pass.

**Step 4.** Join the aggregated neighbor stats back to the main data.

### Expected speedup

| Step | Old | New |
|---|---|---|
| Neighbor lookup | ~35 hours (string ops) | Eliminated â€” replaced by edge table join |
| Stats computation | ~50 hours (R loops Ã— 5) | ~30â€“90 seconds (data.table grouped aggregation) |
| Total | ~86+ hours | **~2â€“5 minutes** |

### Memory footprint

- Edge table: ~1.37M rows Ã— 2 int cols â‰ˆ 11 MB
- Join result: ~1.37M edges Ã— 28 years Ã— 5 vars â‰ˆ ~38.4M rows Ã— 7 cols â‰ˆ ~2.1 GB (fits in 16 GB)
- We process variables in a batch to avoid duplicating the join.

### Numerical equivalence

The `max`, `min`, `mean` computations use the same base R functions on the same neighbor sets, so results are identical to floating-point precision. The trained Random Forest model is never touched.

---

## 3. Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation for cell-year panel data.
#'
#' Replaces build_neighbor_lookup + compute_neighbor_stats loop
#' with a single data.table join + grouped aggregation.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, 
#'                         and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching
#'                         rook_neighbors_unique
#' @param rook_neighbors_unique  spdep::nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names to summarize
#'
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min,
#'         {var}_neighbor_mean for each var in neighbor_source_vars

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -------------------------------------------------------------------
  # Step 1: Build edge table from spdep::nb object
  # -------------------------------------------------------------------
  # Each element of rook_neighbors_unique is an integer vector of indices
  # into id_order (with 0L meaning no neighbors, per spdep convention).
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0L for cells with no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  message(sprintf("Edge table: %s directed neighbor relationships.", 
                  format(nrow(edge_list), big.mark = ",")))
  
  # -------------------------------------------------------------------
  # Step 2: Convert cell_data to data.table (if not already) and key it
  # -------------------------------------------------------------------
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  
  # Columns we need from the neighbor rows
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  dt_neighbor <- dt[, ..neighbor_cols]
  setnames(dt_neighbor, "id", "neighbor_id")
  
  # Key for fast join

  setkey(dt_neighbor, neighbor_id, year)
  
  # -------------------------------------------------------------------
  # Step 3: Join edge table with panel data to get neighbor values
  # -------------------------------------------------------------------
  # Add year from the focal cell: we need to join edges Ã— years.
  # Strategy: join edge_list to dt on cell_id to get years, then join
  # to dt_neighbor on (neighbor_id, year).
  
  # Get unique (cell_id, year) pairs â€” these are the focal observations
  focal <- dt[, .(cell_id = id, year)]
  setkey(focal, cell_id)
  setkey(edge_list, cell_id)
  
  # Expand: each focal cell-year gets its neighbor IDs
  # Result: (cell_id, year, neighbor_id)
  edges_by_year <- edge_list[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = NULL]
  
  message(sprintf("Edges Ã— years: %s rows.", 
                  format(nrow(edges_by_year), big.mark = ",")))
  
  # Now join to get neighbor variable values
  setkey(edges_by_year, neighbor_id, year)
  edges_with_vals <- dt_neighbor[edges_by_year, on = .(neighbor_id, year), nomatch = NA]
  
  # -------------------------------------------------------------------
  # Step 4: Grouped aggregation â€” compute max, min, mean per (cell_id, year)
  # -------------------------------------------------------------------
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))
  
  names(agg_exprs) <- agg_names
  
  # Evaluate
  stats <- edges_with_vals[, 
    lapply(agg_exprs, eval, envir = .SD), 
    by = .(cell_id, year)
  ]
  
  # Replace -Inf/Inf from max/min of all-NA groups with NA
  inf_cols <- grep("_neighbor_max|_neighbor_min", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # -------------------------------------------------------------------
  # Step 5: Merge stats back to main data
  # -------------------------------------------------------------------
  setkey(stats, cell_id, year)
  setkey(dt, id, year)
  
  # Rename for join
  setnames(stats, "cell_id", "id")
  
  dt <- stats[dt, on = .(id, year)]
  
  if (was_df) dt <- as.data.frame(dt)
  
  return(dt)
}
```

### Simpler alternative for Step 4 (avoids `bquote` complexity)

If the dynamic expression building feels fragile, here is a cleaner version of Step 4 that processes one variable at a time but still uses fully vectorized `data.table` grouped ops:

```r
compute_all_neighbor_features_v2 <- function(cell_data,
                                              id_order,
                                              rook_neighbors_unique,
                                              neighbor_source_vars) {
  
  # --- Step 1: Edge table ---
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  # --- Step 2: Prepare data ---
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  
  # Focal cell-year Ã— neighbor edges
  focal_years <- dt[, .(cell_id = id, year)]
  setkey(edge_list, cell_id)
  setkey(focal_years, cell_id)
  edges_by_year <- edge_list[focal_years, on = "cell_id",
                              allow.cartesian = TRUE, nomatch = NULL]
  setkey(edges_by_year, neighbor_id, year)
  
  # --- Step 3 & 4: Per-variable join + aggregate ---
  for (var_name in neighbor_source_vars) {
    
    # Subset neighbor data for this variable
    dt_sub <- dt[, .(neighbor_id = id, year, val = get(var_name))]
    setkey(dt_sub, neighbor_id, year)
    
    # Join
    joined <- dt_sub[edges_by_year, on = .(neighbor_id, year), nomatch = NA]
    
    # Aggregate
    agg <- joined[, .(
      nb_max  = {v <- val[!is.na(val)]; if(length(v)==0L) NA_real_ else max(v)},
      nb_min  = {v <- val[!is.na(val)]; if(length(v)==0L) NA_real_ else min(v)},
      nb_mean = {v <- val[!is.na(val)]; if(length(v)==0L) NA_real_ else mean(v)}
    ), by = .(cell_id, year)]
    
    # Rename columns
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))
    
    # Merge back
    setnames(agg, "cell_id", "id")
    setkey(agg, id, year)
    setkey(dt, id, year)
    dt <- agg[dt, on = .(id, year)]
    
    message(sprintf("Done: %s", var_name))
  }
  
  if (was_df) dt <- as.data.frame(dt)
  return(dt)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
library(data.table)

# Original objects assumed available:
#   cell_data                - data.frame, ~6.46M rows
#   id_order                 - integer vector, length 344,208
#   rook_neighbors_unique    - spdep::nb object, length 344,208

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- REPLACE the entire original block: ----
# neighbor_lookup <- build_neighbor_lookup(...)
# for (var_name in neighbor_source_vars) { ... }

# ---- WITH: ----
cell_data <- compute_all_neighbor_features_v2(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Then proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector index | One `data.table` keyed join on integer columns |
| **Stats computation** | 6.46M R-level `lapply` iterations Ã— 5 vars | `data.table` grouped `max/min/mean` â€” vectorized C |
| **Result binding** | `do.call(rbind, list_of_6.46M)` | Already a data.table â€” no binding needed |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **Peak RAM** | ~8â€“10 GB (6.46M-element list overhead) | ~4â€“6 GB (edge join intermediate) |
| **Numerical result** | Baseline | Identical (same `max/min/mean` on same neighbor sets) |
| **RF model** | Unchanged | Unchanged â€” not retrained |
END_RESPONSE>>>
