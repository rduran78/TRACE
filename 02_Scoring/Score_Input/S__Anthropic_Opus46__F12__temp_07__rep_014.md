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
run_id: Anthropic_F12_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not free â€” it builds a ~6.46M Ã— 3 matrix from a list. However, this is a **one-shot operation per variable** (5 times total). With modern R, `do.call(rbind, ...)` on uniform-length numeric vectors is reasonably fast (seconds, not hours). The `lapply` inside `compute_neighbor_stats` is a simple vectorized index-subset + three summary stats per row. This is **not the 86-hour bottleneck**.

### `build_neighbor_lookup()` â€” The True Bottleneck

The real bottleneck is **`build_neighbor_lookup()`**, which runs a `lapply` over **all 6.46 million rows**, and for each row:

1. Performs a **character key lookup** via `as.character(data$id[i])` into `id_to_ref` â€” 6.46M character conversions and named-vector lookups.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls **`paste()`** to construct `neighbor_keys` â€” creating character vectors per row.
4. Performs **named-vector lookup** `idx_lookup[neighbor_keys]` â€” on a named vector of length 6.46M.

Named vector lookup in R is **O(n)** per query in the worst case (it does linear hashing on names). Doing ~6.46 million lookups into a 6.46M-length named vector, each with ~4 neighbor keys (given ~1.37M directed relationships / 344K cells â‰ˆ ~4 neighbors per cell), means roughly **25.8 million name-based lookups into a 6.46M-entry named vector**. This is catastrophically slow.

Additionally, this entire structure is **redundant across years**: every cell has the same neighbors every year. The neighbor lookup for cell `c` at year `y` is the same set of neighbor cells at year `y`. The function recomputes this for all 28 years Ã— 344K cells instead of computing it once for 344K cells and replicating across years.

**Verdict: REJECT the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()` â€” specifically the repeated character-key construction and named-vector lookups across 6.46M rows, redundantly repeated across 28 identical yearly copies of the spatial topology.

---

## Optimization Strategy

1. **Eliminate year-redundant computation**: Build the spatial neighbor mapping once for 344K cells, not 6.46M cell-years.
2. **Replace named-vector lookups with integer indexing**: Use `match()` or `data.table` hash joins instead of named-vector subsetting.
3. **Vectorize `compute_neighbor_stats()`**: Replace the per-row `lapply` with a `data.table` grouped aggregation over a pre-built edge list, computing max/min/mean in one vectorized pass.
4. **Pre-build an edge table (cell-year â†’ neighbor-cell-year row indices)** using integer arithmetic, avoiding all `paste`/character operations.

This reduces the problem from ~25.8M character lookups into a 6.46M named vector to a single vectorized `data.table` join + grouped aggregation.

---

## Working R Code

```r
library(data.table)

#
# STEP 1: Build a fast edge list ONCE (spatial topology, no year dimension yet)
#
# Inputs:
#   cell_data           â€” data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order            â€” vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer neighbor index vectors)
#
# We build a data.table mapping each cell to its neighbor cells (integer indices into id_order).
#

build_neighbor_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] gives the positions (in id_order) of the neighbors of id_order[i]
  n <- length(neighbors)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    len  <- length(nb_i)
    if (len > 0L) {
      from_idx[pos:(pos + len - 1L)] <- i
      to_idx[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }
  
  data.table(from_cell_idx = from_idx, to_cell_idx = to_idx)
}

#
# STEP 2: Expand edge list to cell-year rows and compute stats vectorized
#

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  # --- Map cell IDs to spatial indices (position in id_order) ---
  id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
  dt <- merge(dt, id_map, by = "id", all.x = TRUE, sort = FALSE)
  
  # --- Create a unique row index for fast lookups ---
  dt[, row_id := .I]
  
  # --- Build spatial edge list (cell_idx -> neighbor_cell_idx), ~1.37M rows ---
  edges <- build_neighbor_edge_list(id_order, rook_neighbors_unique)
  
  # --- Build a lookup from (cell_idx, year) -> row_id in dt ---
  # This replaces the expensive named-vector lookup in the original code.
  setkey(dt, cell_idx, year)
  
  # --- Get unique years ---
  years <- sort(unique(dt$year))
  
  # --- Expand edges across years: CJ of edges Ã— years ---
  # For each year, the neighbor of cell i is cell j at the same year.
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16GB RAM easily.
  
  cat("Expanding edge list across years...\n")
  edge_year <- edges[, .(year = years), by = .(from_cell_idx, to_cell_idx)]
  
  # --- Join to get the ROW index of the neighbor (to_cell_idx, year) ---
  # We need the row_id of the neighbor in dt so we can pull variable values.
  neighbor_key <- dt[, .(cell_idx, year, row_id)]
  setnames(neighbor_key, c("cell_idx", "year", "row_id"),
           c("to_cell_idx", "year", "neighbor_row_id"))
  setkey(neighbor_key, to_cell_idx, year)
  setkey(edge_year, to_cell_idx, year)
  
  edge_year <- neighbor_key[edge_year, nomatch = NA]
  
  # --- Also get the row_id of the focal cell (from_cell_idx, year) ---
  focal_key <- dt[, .(cell_idx, year, row_id)]
  setnames(focal_key, c("cell_idx", "year", "row_id"),
           c("from_cell_idx", "year", "focal_row_id"))
  setkey(focal_key, from_cell_idx, year)
  setkey(edge_year, from_cell_idx, year)
  
  edge_year <- focal_key[edge_year, nomatch = NA]
  
  # Now edge_year has columns:
  #   from_cell_idx, year, focal_row_id, to_cell_idx, neighbor_row_id
  
  # --- For each variable, compute grouped stats vectorized ---
  cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")
  
  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    dt[, paste0("max_neighbor_", var_name) := NA_real_]
    dt[, paste0("min_neighbor_", var_name) := NA_real_]
    dt[, paste0("mean_neighbor_", var_name) := NA_real_]
  }
  
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    
    # Pull neighbor values via integer indexing (extremely fast)
    vals_vec <- dt[[var_name]]
    edge_year[, nval := vals_vec[neighbor_row_id]]
    
    # Grouped aggregation: for each focal_row_id, compute max/min/mean of neighbor values
    stats <- edge_year[!is.na(nval),
                       .(nmax  = max(nval),
                         nmin  = min(nval),
                         nmean = mean(nval)),
                       by = focal_row_id]
    
    # Write results back into dt by integer index
    max_col  <- paste0("max_neighbor_",  var_name)
    min_col  <- paste0("min_neighbor_",  var_name)
    mean_col <- paste0("mean_neighbor_", var_name)
    
    dt[stats$focal_row_id, (max_col)  := stats$nmax]
    dt[stats$focal_row_id, (min_col)  := stats$nmin]
    dt[stats$focal_row_id, (mean_col) := stats$nmean]
    
    # Clean up temp column
    edge_year[, nval := NULL]
  }
  
  # --- Clean up helper columns ---
  dt[, c("cell_idx", "row_id") := NULL]
  
  cat("Done.\n")
  return(as.data.frame(dt))
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The numerical estimand (max, min, mean of neighbor values per variable) is preserved exactly.
# Predictions proceed as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M character `paste` + named-vector lookups into a 6.46M-entry vector | Eliminated entirely |
| **Redundant computation** | Recomputes identical spatial topology for all 28 years | Builds topology once (344K cells), expands via integer join |
| **Key data structure** | Named character vectors (`idx_lookup`) | `data.table` keyed integer joins |
| **`compute_neighbor_stats`** | Per-row `lapply` (6.46M iterations) + `do.call(rbind)` | Single vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | **Minutes** (edge expansion is ~38.5M rows; grouped aggregation is a single pass) |
| **RAM** | Repeated character allocations cause GC pressure | ~38.5M-row integer edge table â‰ˆ ~1.5 GB; fits in 16 GB |
| **RF model** | Preserved | Preserved (no retraining) |
| **Numerical output** | max, min, mean per neighbor set | Identical values via same arithmetic |
END_RESPONSE>>>
