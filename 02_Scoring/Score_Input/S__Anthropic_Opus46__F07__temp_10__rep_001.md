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
run_id: Anthropic_F07_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²) effective cost
The function calls `lapply` over **~6.46 million rows**, and for each row it:
- Performs character-based key construction (`paste`)
- Does named-vector lookups (`idx_lookup[neighbor_keys]`) which are hash-table lookups repeated per row

With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates **~25M+ hash lookups** inside a serial R loop. The named-vector `idx_lookup` has 6.46M entries, so each lookup is nontrivial. Total wall time is dominated by this step.

### 2. `compute_neighbor_stats` â€” Repeated per variable, R-level loop
For each of 5 variables, a `lapply` iterates over 6.46M rows, subsetting a numeric vector by index each time. This is 5 Ã— 6.46M â‰ˆ 32M R-level function calls with memory allocation each.

### 3. Memory-safe but slow pattern
The `lapply` â†’ `do.call(rbind, ...)` pattern over millions of 3-element vectors creates millions of tiny objects, stressing R's garbage collector.

**Estimated breakdown**: ~80% of the 86 hours is in `build_neighbor_lookup`, ~20% in the repeated `compute_neighbor_stats` calls.

---

## Optimization Strategy

### Key insight: Separate the spatial topology from the temporal dimension

The neighbor relationships are **time-invariant**. Cell `i`'s neighbors are the same in every year. The current code re-discovers this for every cell-year row. Instead:

1. **Build a sparse adjacency structure once at the cell level** (344K cells, not 6.46M cell-years).
2. **Expand to cell-year using vectorized joins** â€” for each cell-year row, the neighbor rows are the neighbor-cells in the same year. This is a merge/join, not a per-row lookup.
3. **Compute neighbor stats using `data.table` grouped operations** â€” avoid R-level loops entirely.

### Specific approach:
- Convert the `nb` object to an edge list (cell_i â†’ cell_j) â€” ~1.37M directed edges.
- Join `cell_data` to itself on `(neighbor_id, year)` to get neighbor values â€” this produces ~1.37M Ã— 28 â‰ˆ ~38M edge-year rows (fits in RAM at ~2-3 GB).
- Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass.
- Repeat for each variable (or compute all simultaneously).

**Expected speedup**: From 86+ hours to **~2â€“5 minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Convert the nb object to a directed edge data.table
# ============================================================
# rook_neighbors_unique is an nb object (list of integer vectors)
# id_order is the vector mapping list index -> cell id

build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer index vectors
  from_idx <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb)
  
  # Remove the "no neighbors" sentinel (spdep uses 0L for no neighbors)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows: one row per directed rook-neighbor pair

cat("Edge list rows:", nrow(edges), "\n")

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ============================================================
# STEP 3: Compute neighbor stats for all variables at once
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(cell_data, edges, source_vars) {
  
  # Subset cell_data to only the columns we need for the join
  join_cols <- c("id", "year", source_vars)
  cd_subset <- cell_data[, ..join_cols]
  
  # Rename id to id_to for merging (we want to look up neighbor values)
  setnames(cd_subset, "id", "id_to")
  setkey(cd_subset, id_to, year)
  
  # Cross edges with years:
  # For each (id_from, id_to) edge and each year, get the neighbor's values.
  # Strategy: join edges to cd_subset on id_to, which gives us all
  #           (id_from, id_to, year, var_values) combinations.
  #
  # But we need to restrict to years where id_from also exists.
  # Since the panel is balanced (344208 cells Ã— 28 years), this is automatic.
  # If unbalanced, the final merge back handles it.
  
  # Merge: edges Ã— cd_subset on id_to â†’ gives neighbor values per year
  # This produces ~1.37M Ã— 28 â‰ˆ 38.4M rows (manageable)
  cat("Joining edges to cell data to get neighbor values...\n")
  edge_year <- merge(edges, cd_subset, by = "id_to", allow.cartesian = TRUE)
  # edge_year columns: id_to, id_from, year, ntl, ec, pop_density, def, usd_est_n2
  
  cat("Edge-year rows:", nrow(edge_year), "\n")
  
  # Group by (id_from, year) to compute max, min, mean of each variable
  cat("Computing grouped neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats <- edge_year[, eval(agg_call), by = .(id_from, year)]
  
  # Fix infinite values from max/min on all-NA groups (shouldn't happen but be safe)
  for (col in agg_names) {
    vals <- stats[[col]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats, j = col, value = vals)
  }
  
  setnames(stats, "id_from", "id")
  setkey(stats, id, year)
  
  return(stats)
}

neighbor_stats <- compute_all_neighbor_stats(cell_data, edges, neighbor_source_vars)

# ============================================================
# STEP 4: Merge neighbor stats back to cell_data
# ============================================================

# Remove any previously computed neighbor columns to avoid duplication
old_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

# Keyed merge
setkey(cell_data, id, year)
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

cat("Done. cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# ============================================================
# STEP 5: Verify column names match what the trained RF expects
# ============================================================
# The original code (compute_and_add_neighbor_features) presumably
# created columns like: neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.
# The code above uses exactly these names. Verify:

expected_new_cols <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

stopifnot(all(expected_new_cols %in% names(cell_data)))
cat("All 15 neighbor feature columns present and correctly named.\n")
cat("Trained Random Forest model can be applied without retraining.\n")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level iterations with character hashing | Vectorized `data.table` merge on integer keys |
| **Stats computation** | 5 Ã— 6.46M `lapply` calls | Single grouped aggregation over ~38M edge-year rows |
| **Memory pattern** | Millions of tiny 3-element vectors + `do.call(rbind)` | Columnar `data.table` operations, GC-friendly |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | Identical â€” `max/min/mean` with `na.rm=TRUE` on same edge set |
| **RF model** | N/A | Untouched â€” column names preserved exactly |
| **RAM peak** | ~variable, GC-heavy | ~3â€“5 GB for edge-year table (fits in 16 GB) |

The optimization exploits the key structural fact: **spatial topology is time-invariant**, so the neighbor lookup should be built at the cell level (344K) and broadcast across years via a vectorized join, not reconstructed at the cell-year level (6.46M) in a serial loop.
END_RESPONSE>>>
