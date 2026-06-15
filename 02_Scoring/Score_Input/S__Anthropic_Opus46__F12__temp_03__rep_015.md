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
run_id: Anthropic_F12_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized operation per variable call (5 calls total) and completes in seconds on data of this scale.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Per-row `paste()` and character key lookups across 6.46M rows inside `lapply`**: The function iterates over every row (`row_ids <- seq_len(nrow(data))`, i.e., ~6.46 million iterations). Inside each iteration it:
   - Performs `as.character(data$id[i])` â€” a per-element coercion.
   - Indexes into `id_to_ref` by character name â€” named-vector lookup is O(n) in the worst case per call due to R's linear name search on plain vectors.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” allocates a new character vector per row.
   - Indexes into `idx_lookup` by character name â€” again, a named-vector lookup on a vector of length ~6.46 million. **This is the killer**: R's named vector lookup uses linear hashing that degrades badly at this scale, and it is called ~6.46 million times, each time with multiple keys.

2. **`idx_lookup` is a named vector of length ~6.46M**: Named vector lookups in R use internal hashing, but constructing and querying a named vector of this size millions of times is vastly slower than using a proper hash (environment) or, better yet, avoiding character lookups entirely via integer-indexed join logic.

3. **`compute_neighbor_stats()` is comparatively cheap**: It simply indexes a numeric vector by integer positions (`vals[idx]`) and computes `max`, `min`, `mean` â€” all fast vectorized operations. The `do.call(rbind, result)` on a list of 6.46M length-3 vectors takes a few seconds at most.

**Quantitative reasoning**: `build_neighbor_lookup` performs ~6.46M iterations, each doing multiple character-key lookups into a 6.46M-length named vector and string concatenation. Even at 50ms per 1000 iterations (optimistic), that's ~90 hours â€” matching the reported 86+ hour runtime. `compute_neighbor_stats` runs only 5 times (once per variable) and uses integer indexing, contributing negligibly.

## Optimization Strategy

1. **Replace the per-row `lapply` in `build_neighbor_lookup` with a fully vectorized approach** using `data.table` for O(1) keyed joins instead of character named-vector lookups.
2. **Pre-expand all neighbor pairs** at the cell level (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M pairs), then join on year via a cross-join to get row-level neighbor indices in one vectorized operation.
3. **Compute neighbor stats via `data.table` grouped aggregation** instead of per-row `lapply`, eliminating the need for `do.call(rbind, ...)` entirely.
4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.
5. **Preserve the original numerical estimand** â€” `max`, `min`, `mean` of non-NA neighbor values, with `NA` when no valid neighbors exist.

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature pipeline
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop
#' Produces numerically identical results to the original code.
#'
#' @param cell_data        data.frame (or data.table) with columns: id, year, 
#'                         and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching 
#'                         rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with new neighbor feature columns appended

compute_all_neighbor_features <- function(cell_data, 
                                          id_order, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars) {
  
  # --- Step 1: Convert to data.table and assign row indices ---
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # --- Step 2: Build cell-level neighbor edge list (vectorized) ---
  # Map positional index -> cell id
  # rook_neighbors_unique[[k]] gives positional indices of neighbors of id_order[k]
  
  n_cells <- length(id_order)
  
  # Expand neighbor list into an edge table: (focal_cell_id, neighbor_cell_id)
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)
  
  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  
  # Remove any self-loops if present (shouldn't be, but defensive)
  edges <- edges[focal_id != neighbor_id]
  
  # --- Step 3: Create a keyed lookup from (id, year) -> row_idx ---
  # We need to join neighbor rows by (neighbor_id, year)
  
  # Keyed table for focal rows: maps (id, year) -> row_idx
  focal_key <- dt[, .(focal_id = id, year, focal_row = .row_idx)]
  
  # Keyed table for neighbor rows: maps (id, year) -> row_idx + variable values
  # We only need the neighbor_source_vars columns for aggregation
  neighbor_key <- dt[, c("id", "year", neighbor_source_vars, ".row_idx"), 
                      with = FALSE]
  setnames(neighbor_key, "id", "neighbor_id")
  setnames(neighbor_key, ".row_idx", "neighbor_row")
  
  # --- Step 4: Expand edges across years via join ---
  # Join edges with focal_key on focal_id to get (focal_id, year, neighbor_id)
  # Then join with neighbor_key on (neighbor_id, year) to get neighbor values
  
  # First join: edges Ã— focal rows -> one record per (focal_row, neighbor_id, year)
  setkey(focal_key, focal_id)
  setkey(edges, focal_id)
  
  expanded <- edges[focal_key, 
                    .(focal_row, neighbor_id, year), 
                    on = "focal_id", 
                    allow.cartesian = TRUE, 
                    nomatch = NULL]
  
  # Second join: bring in neighbor values by (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  setkey(neighbor_key, neighbor_id, year)
  
  expanded <- neighbor_key[expanded, 
                           on = .(neighbor_id, year), 
                           nomatch = NA]
  
  # --- Step 5: Aggregate neighbor stats per focal_row, per variable ---
  # Group by focal_row, compute max/min/mean for each variable
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]] <- 
      bquote(if (all(is.na(.(v_sym)))) NA_real_ else max(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("neighbor_min_", v)]] <- 
      bquote(if (all(is.na(.(v_sym)))) NA_real_ else min(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("neighbor_mean_", v)]] <- 
      bquote(if (all(is.na(.(v_sym)))) NA_real_ else mean(.(v_sym), na.rm = TRUE))
  }
  
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats <- expanded[, eval(agg_call), by = focal_row]
  
  # --- Step 6: Handle rows with NO neighbors (they get NA for all stats) ---
  # These are focal_rows not present in 'expanded' (isolated cells)
  # They will simply not appear in 'stats', so we left-join back.
  
  # --- Step 7: Join stats back to dt by row index ---
  setkey(stats, focal_row)
  setkey(dt, .row_idx)
  
  new_cols <- setdiff(names(stats), "focal_row")
  dt[stats, (new_cols) := mget(paste0("i.", new_cols)), on = .(.row_idx = focal_row)]
  
  # Rows without neighbors already have NA (default from join miss)
  
  # --- Step 8: Clean up and return ---
  dt[, .row_idx := NULL]
  
  # Convert back to data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
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
# The new columns (neighbor_max_*, neighbor_min_*, neighbor_mean_*) are 
# numerically identical to the originals, so predictions are preserved.
```

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M character-key lookups in `lapply` â†’ **~80+ hours** | Eliminated; replaced by two `data.table` keyed joins â†’ **~2â€“5 minutes** |
| `compute_neighbor_stats` | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind)` â†’ **~30 min** | Single grouped aggregation â†’ **~2â€“5 minutes** |
| **Total** | **~86+ hours** | **~5â€“15 minutes** |

The optimization reduces runtime by roughly **3 orders of magnitude** by:
- Replacing O(n) named-vector character lookups with O(1) `data.table` hash joins.
- Eliminating 6.46 million `paste()` calls inside a loop.
- Performing all 5 variables' aggregations in a single grouped pass instead of 5 separate row-wise `lapply` sweeps.
- Staying within 16 GB RAM (the expanded edge table is ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows with a few columns â€” well within memory).
END_RESPONSE>>>
