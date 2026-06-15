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
run_id: Anthropic_F05_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of redundancy:

1. **O(N) string-key construction inside `build_neighbor_lookup`**: `idx_lookup` is built once (good), but the `lapply` over ~6.46M rows still calls `paste()` per row to construct `neighbor_keys`. Each call allocates character vectors and does hash-table lookups against a 6.46M-entry named vector. This is the innermost hot loop.

2. **The lookup is year-redundant**: For every row of cell `c` in year `t`, the neighbor *cell IDs* are identical across all 28 yearsâ€”only the year suffix changes. Yet the code re-derives neighbor cell IDs from the `neighbors` list for every row, not just every cell.

3. **The real bottleneck is the row-level `lapply` over 6.46M rows**: Even with the lookup built, iterating in R over millions of rows with per-row anonymous functions, character allocations, and named-vector subsetting is catastrophically slow. This is why the estimate is 86+ hours.

4. **`compute_neighbor_stats` is efficient in isolation** but is called 5 times, each time iterating over the 6.46M-entry `neighbor_lookup` list. The list itself consumes significant memory.

**Root cause**: The algorithm is structured as *row-level iteration in R* when it should be structured as *vectorized column-level operations on a pre-joined neighbor table*.

---

## Optimization Strategy

### Key insight
The neighbor relationship is **spatial only** (cell-to-cell), invariant across years. We can:

1. **Build a flat neighbor edge table once**: a two-column integer matrix `(focal_cell_id, neighbor_cell_id)` with ~1.37M rows.
2. **Join this edge table to the panel data by year** using `data.table` keyed merges â€” this is fully vectorized C-level code.
3. **Compute all neighbor stats (max, min, mean) in one grouped aggregation** per variable, or even for all 5 variables simultaneously.

This replaces:
- 6.46M R-level iterations â†’ 0
- 6.46M Ã— k string paste + hash lookups â†’ 0
- 5 separate passes over a 6.46M-element list â†’ 1 vectorized grouped aggregation

**Expected speedup**: from ~86 hours to **~1â€“3 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# 1. Build the spatial neighbor edge table (one-time, year-invariant)
#    rook_neighbors_unique: an nb object (list of integer index vectors)
#    id_order: vector of cell IDs in the order matching the nb object
# ===========================================================================
build_neighbor_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for the i-th cell
  n <- length(neighbors)
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_counts)
  
  focal_idx    <- rep(seq_len(n), times = edge_counts)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    focal_cell_id    = id_order[focal_idx],
    neighbor_cell_id = id_order[neighbor_idx]
  )
}

# ===========================================================================
# 2. Compute neighbor features for all variables in one pass
# ===========================================================================
compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already (non-destructive copy)
  dt <- as.data.table(cell_data)
  
  # --- Step 1: Build edge table (spatial only, ~1.37M rows) ---------------
  edges <- build_neighbor_edge_table(id_order, neighbors)
  
  # --- Step 2: Join edges Ã— years to get neighbor row references ----------
  # We need: for each (focal_cell_id, year) â†’ all (neighbor_cell_id, year)
  # This is achieved by joining edges to dt on neighbor_cell_id + year
  
  # Subset dt to only the columns we need for the neighbor side
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  dt_neighbor <- dt[, ..neighbor_cols]
  setnames(dt_neighbor, "id", "neighbor_cell_id")
  
  # Keyed join: edges + dt_neighbor on (neighbor_cell_id)
  # Result: (focal_cell_id, neighbor_cell_id, year, var1, var2, ...)
  # For each year, every edge gets the neighbor's variable values
  setkeyv(dt_neighbor, "neighbor_cell_id")
  setkeyv(edges, "neighbor_cell_id")
  
  # This is the main join: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows
  # data.table handles this efficiently
  joined <- edges[dt_neighbor, on = "neighbor_cell_id", allow.cartesian = TRUE, nomatch = NULL]
  # joined now has columns: focal_cell_id, neighbor_cell_id, year, ntl, ec, ...
  
  # --- Step 3: Grouped aggregation ----------------------------------------
  # Group by (focal_cell_id, year), compute max/min/mean for each variable
  
  # Build aggregation expressions programmatically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))
  
  names(agg_exprs) <- agg_names
  
  # Execute the grouped aggregation
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  stats <- joined[, eval(agg_call), by = .(focal_cell_id, year)]
  
  # Replace -Inf/Inf from max/min on all-NA groups with NA
  inf_cols <- grep("_(max|min)$", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # --- Step 4: Merge back to original data --------------------------------
  setnames(stats, "focal_cell_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))
  
  # Remove any pre-existing neighbor columns to avoid duplication
  existing_ncols <- intersect(agg_names, names(dt))
  if (length(existing_ncols) > 0) {
    dt[, (existing_ncols) := NULL]
  }
  
  dt <- stats[dt, on = c("id", "year")]
  
  # Convert back to data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }
  
  dt
}

# ===========================================================================
# 3. Usage â€” drop-in replacement for the original outer loop
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” it consumes the same
# 15 neighbor feature columns (5 vars Ã— {max, min, mean}) with identical
# numerical values as the original implementation.
```

---

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[i]]` | Same nb object, same edges |
| Year matching | `paste(id, year)` key match | Equi-join on `(neighbor_cell_id, year)` |
| Aggregation functions | `max`, `min`, `mean` with `na.rm` | Identical functions, identical `na.rm` |
| Output columns | `n_{var}_{max,min,mean}` | Identical names and semantics |
| RF model | Untouched | Untouched â€” same input features |

The join-then-aggregate approach computes **exactly the same quantity**: for each `(focal_cell, year)`, gather the variable values of all rook neighbors present in that year, then take `max`, `min`, `mean` ignoring `NA`s.

---

## Performance Estimate

| Step | Rows | Time estimate |
|---|---|---|
| Edge table construction | ~1.37M | < 1 sec |
| Keyed join (edges Ã— 28 years) | ~38.4M | ~10â€“30 sec |
| Grouped aggregation (5 vars Ã— 3 stats) | ~38.4M â†’ ~6.46M | ~30â€“90 sec |
| Merge back | ~6.46M | ~5â€“10 sec |
| **Total** | | **~1â€“3 minutes** |

Memory peak: the `joined` table at ~38.4M rows Ã— ~8 columns â‰ˆ **2.5 GB**, well within the 16 GB laptop constraint.
END_RESPONSE>>>
