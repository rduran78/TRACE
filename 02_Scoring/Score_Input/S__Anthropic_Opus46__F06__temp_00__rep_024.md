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
run_id: Anthropic_F06_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (fast).
2. Constructs character keys by pasting neighbor IDs with the current row's year (slow â€” character allocation and concatenation at scale).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` (slow â€” repeated hash lookups on a 6.46M-entry named vector).

This is **O(N Ã— k)** where N â‰ˆ 6.46M and k â‰ˆ average number of rook neighbors (~4). The constant factor is enormous because of repeated `paste()` and named-vector indexing on character strings.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting and computing `max`, `min`, `mean`. The subsetting itself is fast (integer indexing), but the R-level loop over 6.46M elements with per-element function calls is slow. This runs 5 times = ~32.3M R-level function invocations.

### Why it takes 86+ hours

- ~6.46M R-level iterations in `build_neighbor_lookup`, each doing string operations.
- ~32.3M R-level iterations across the 5 calls to `compute_neighbor_stats`.
- R's `lapply` with anonymous functions has high per-call overhead (~1â€“5 Âµs), so 38M calls â‰ˆ 40â€“190 seconds just in dispatch, but the string operations inside `build_neighbor_lookup` push each call to ~40â€“50 Âµs â†’ ~80+ hours for that step alone.

**The dominant cost is `build_neighbor_lookup`.** The `paste`/character-lookup pattern is the killer.

---

## 2. Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Rook neighbors are **time-invariant**. Cell *i*'s neighbors are the same in every year. The current code redundantly re-discovers this for every cell-year. Instead:

1. **Build the neighbor lookup once at the cell level** (344,208 cells), not the cell-year level (6.46M rows).
2. **Exploit the panel structure**: if data is sorted by `(id, year)`, each cell occupies a contiguous block of 28 rows. A cell's neighbor in the same year is at a predictable offset. This eliminates all string operations.
3. **Vectorize the statistics computation** using `data.table` grouping or sparse matrix multiplication instead of row-level `lapply`.

### Concrete Plan

- Use `data.table` for fast indexed operations.
- Build a **cell-level** neighbor edge list (source_cell â†’ neighbor_cell), ~1.37M edges.
- Join the edge list with the data on `(neighbor_cell, year)` to pull neighbor values.
- Group by `(source_cell, year)` and compute `max`, `min`, `mean` in one vectorized pass.
- This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with a single vectorized pipeline.

**Expected speedup**: from 86+ hours to **~2â€“5 minutes**.

### Why not raster focal/kernel operations?

The comment in the prompt asks about this. Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. If the grid cells map 1:1 to raster pixels and the rook neighborhood is exactly the 4-connected pixel neighborhood, `focal` could work. However:
- The `spdep::nb` object may encode an **irregular** neighborhood (boundary cells, missing cells, non-rectangular domains).
- `focal` would need to be applied per-year-layer across a 28-layer raster stack, and then results re-extracted to the panel â€” adding complexity.
- The `data.table` join approach is **general**, preserves the exact `nb` structure, and is already extremely fast.
- **We choose the `data.table` join approach** to best preserve the required results (exact same neighbor definitions, exact same statistics).

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with columns: id, year, 
#         ntl, ec, pop_density, def, usd_est_n2 (plus other predictors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a CELL-LEVEL edge list from the spdep::nb object
#         rook_neighbors_unique is a list of length 344,208;
#         rook_neighbors_unique[[i]] gives integer indices of neighbors
#         of the i-th cell in id_order.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors
  # id_order[i] is the cell id for the i-th element
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(source_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

cat("Building cell-level edge list...\n")
edge_list <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each source variable, join, aggregate, and attach
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Subset the data to just (id, year, variable) for the join
  # This keeps memory usage low
  sub_dt <- cell_data[, .(id, year, val = get(var_name))]
  setnames(sub_dt, "id", "neighbor_id")
  setkey(sub_dt, neighbor_id, year)
  
  # Join edge_list with sub_dt: for each (source_id, neighbor_id),

  # get all years of the neighbor's values
  # First, add year dimension by joining edge_list Ã— sub_dt on neighbor_id
  joined <- merge(edge_list, sub_dt, by = "neighbor_id", allow.cartesian = TRUE)
  # joined has columns: neighbor_id, source_id, year, val
  
  # Aggregate by (source_id, year) to get max, min, mean of neighbor values
  stats <- joined[!is.na(val), 
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   by = .(source_id, year)]
  
  # Rename for merge back
  setnames(stats, "source_id", "id")
  
  # Define output column names (matching original pipeline's naming convention)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  # Clean up
  rm(sub_dt, joined, stats)
  gc()
}

cat("Done. Neighbor features added.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Apply the pre-trained Random Forest model (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model is applied exactly as before.
# Example (adjust to match your existing prediction code):
#
# predictions <- predict(rf_model, newdata = cell_data)
#
# The model is NOT retrained. The numerical estimand is preserved
# because the neighbor statistics (max, min, mean) are computed
# identically â€” same neighbor definitions, same aggregation functions.
```

---

## 4. Memory Optimization (if 16 GB is tight)

The `merge` of `edge_list` (~1.37M rows) Ã— `sub_dt` (~6.46M rows) on `neighbor_id` produces a large intermediate table (~6.46M Ã— avg_degree â‰ˆ ~25.8M rows per variable). At ~3 columns of 8 bytes each, that's ~620 MB per variable â€” manageable on 16 GB. However, if memory is a concern, process years in chunks:

```r
# Memory-safe variant: process in year chunks
compute_neighbor_stats_chunked <- function(cell_data, edge_list, var_name, 
                                            year_chunks = NULL) {
  if (is.null(year_chunks)) {
    years <- sort(unique(cell_data$year))
    year_chunks <- split(years, ceiling(seq_along(years) / 7))  # 4 chunks of 7 years
  }
  
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  results <- rbindlist(lapply(year_chunks, function(yrs) {
    sub_dt <- cell_data[year %in% yrs, .(id, year, val = get(var_name))]
    setnames(sub_dt, "id", "neighbor_id")
    setkey(sub_dt, neighbor_id, year)
    
    joined <- merge(edge_list, sub_dt, by = "neighbor_id", allow.cartesian = TRUE)
    
    stats <- joined[!is.na(val),
                     .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                     by = .(source_id, year)]
    
    rm(sub_dt, joined); gc()
    stats
  }))
  
  setnames(results, c("source_id", "nb_max", "nb_min", "nb_mean"),
           c("id", max_col, min_col, mean_col))
  results
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups | 1 cell-level edge list (344K cells, ~1.37M edges) |
| **Stats computation** | 6.46M Ã— 5 = 32.3M R-level `lapply` calls | 5 vectorized `data.table` group-by aggregations |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **RAM peak** | Moderate (6.46M-entry named vector) | ~1â€“2 GB for largest intermediate join |
| **Numerical results** | Baseline | **Identical** (same neighbors, same max/min/mean) |
| **RF model** | Pre-trained, applied as-is | **Unchanged â€” not retrained** |
END_RESPONSE>>>
