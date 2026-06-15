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
run_id: Anthropic_F11_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is hash-based but still carries significant overhead when called millions of times in an interpreted loop. With ~1.37 million neighbor relationships spread across 344,208 cells Ã— 28 years, this function builds millions of string keys and performs millions of name-matching operations sequentially.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, extracting neighbor values, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations just for the stats, each with subsetting, NA filtering, and three summary computations.

3. The final `do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is itself expensive â€” it creates a massive matrix from millions of tiny row vectors.

4. **In contrast**, Random Forest prediction (`predict()`) on a pre-trained model with ~6.46 million rows and ~110 predictors is a single vectorized C/C++ call (in `randomForest` or `ranger`). It typically completes in seconds to a few minutes, even on a laptop. Loading a serialized model (`readRDS`) and writing predictions (`fwrite`/`write.csv`) are also fast I/O operations.

**Quantitative reasoning for the 86+ hour estimate:**

- `build_neighbor_lookup`: ~6.46M iterations, each doing string operations and named lookups â†’ estimated 30-50+ hours alone.
- `compute_neighbor_stats`: 5 variables Ã— 6.46M iterations â†’ estimated 20-35+ hours.
- RF predict: single vectorized call â†’ minutes.

The bottleneck is overwhelmingly in the row-level R loops with string-key operations.

---

## Optimization Strategy

1. **Eliminate string-key lookups entirely.** Replace the `paste(id, year, sep="_")` â†’ named-vector lookup pattern with direct integer indexing. Since the data has a regular panel structure (344,208 cells Ã— 28 years), we can compute row indices arithmetically.

2. **Vectorize `build_neighbor_lookup`** using `data.table` for fast group-based joins, or precompute an integer-indexed neighbor-row mapping using the panel structure.

3. **Vectorize `compute_neighbor_stats`** by replacing the per-row `lapply` with a single `data.table` grouped aggregation over an edge list of (row, neighbor_row) pairs.

4. **Process all 5 variables simultaneously** in one pass over the edge list rather than 5 separate loops.

These changes reduce the complexity from millions of interpreted R-loop iterations with string operations to a handful of vectorized/compiled operations.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE ENGINEERING
# ==============================================================
# Assumptions (consistent with the pipeline facts):
#   - cell_data is a data.frame/data.table with columns: id, year, 
#     and the neighbor source variables.
#   - id_order is the vector of unique cell IDs (length 344,208)
#     in the same order as rook_neighbors_unique.
#   - rook_neighbors_unique is an nb object (list of length 344,208),
#     where each element is an integer vector of neighbor indices 
#     into id_order.
#   - cell_data is sorted (or will be sorted) by (id, year).
# ==============================================================

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, 
                                         neighbor_source_vars) {
  
  # ---- Step 0: Convert to data.table and ensure sorted ----
  dt <- as.data.table(cell_data)
  
  # Create a mapping from cell id to its position in id_order (1-indexed)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Assign each cell id its reference index
  dt[, ref_idx := id_to_ref[as.character(id)]]
  
  # Sort by (ref_idx, year) so we can compute row positions arithmetically
  setkey(dt, ref_idx, year)
  
  # Verify the panel is balanced and complete
  unique_years <- sort(unique(dt$year))
  n_years      <- length(unique_years)
  n_cells      <- length(id_order)
  stopifnot(nrow(dt) == n_cells * n_years)
  
  # Create a year-to-offset mapping (0-indexed offset within each cell's block)
  year_to_offset <- setNames(seq_along(unique_years) - 1L, as.character(unique_years))
  
  # After sorting by (ref_idx, year), the row for cell i (1-based), year t is:
  #   row = (i - 1) * n_years + offset_t + 1
  # where offset_t = year_to_offset[as.character(t)]
  
  # ---- Step 1: Build edge list (source_row, neighbor_row) ----
  # For each cell ref_idx i, its neighbors are rook_neighbors_unique[[i]].
  # We need to expand this across all years.
  
  message("Building edge list...")
  
  # Build cell-level edge list: (cell_ref, neighbor_ref)
  # rook_neighbors_unique is a list; each element is an integer vector of neighbor indices
  cell_edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0 || (length(nb) == 1 && nb[1] == 0L)) {
      return(data.table(cell_ref = integer(0), neighbor_ref = integer(0)))
    }
    data.table(cell_ref = i, neighbor_ref = as.integer(nb))
  }))
  
  message(sprintf("  Cell-level edges: %d", nrow(cell_edges)))
  
  # Expand across all years: for each year, the row index is computed arithmetically
  # source_row = (cell_ref - 1) * n_years + year_offset + 1
  # neighbor_row = (neighbor_ref - 1) * n_years + year_offset + 1
  
  # To avoid a massive cross-join in memory, we use vectorized arithmetic:
  # Repeat cell_edges for each year
  n_edges_per_year <- nrow(cell_edges)
  
  # Create the full edge list using outer-product logic but vectorized
  offsets <- year_to_offset  # named integer vector, 0-indexed
  offset_vals <- as.integer(offsets)  # length n_years
  
  # Replicate cell_edges n_years times, adding the year offset each time
  message("Expanding edge list across years...")
  
  # Pre-allocate
  total_edges <- as.numeric(n_edges_per_year) * n_years
  message(sprintf("  Total directed row-edges: %.0f", total_edges))
  
  # Vectorized construction
  # rep each column n_years times, and rep-each the offset
  src_cell_rep  <- rep(cell_edges$cell_ref, times = n_years)
  nbr_cell_rep  <- rep(cell_edges$neighbor_ref, times = n_years)
  offset_rep    <- rep(offset_vals, each = n_edges_per_year)
  
  source_rows   <- (src_cell_rep - 1L) * n_years + offset_rep + 1L
  neighbor_rows <- (nbr_cell_rep - 1L) * n_years + offset_rep + 1L
  
  # Free intermediates
  rm(src_cell_rep, nbr_cell_rep, offset_rep, cell_edges)
  gc()
  
  # ---- Step 2: Compute neighbor stats for all variables at once ----
  message("Computing neighbor statistics...")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing variable: %s", var_name))
    
    vals <- dt[[var_name]]
    neighbor_vals <- vals[neighbor_rows]
    
    # Build a data.table with source_row and neighbor value
    edge_dt <- data.table(
      src = source_rows,
      nval = neighbor_vals
    )
    
    # Remove edges where neighbor value is NA
    edge_dt <- edge_dt[!is.na(nval)]
    
    # Compute grouped stats
    stats <- edge_dt[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = src]
    
    # Initialize result columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed values by row index
    set(dt, i = stats$src, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$src, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$src, j = mean_col, value = stats$nb_mean)
    
    rm(edge_dt, stats, neighbor_vals)
    gc()
  }
  
  # Clean up helper column
  dt[, ref_idx := NULL]
  
  # Free the large index vectors
  rm(source_rows, neighbor_rows)
  gc()
  
  message("Neighbor feature engineering complete.")
  return(dt)
}

# ==============================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimized_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# ==============================================================
# RANDOM FOREST PREDICTION (unchanged â€” not the bottleneck)
# ==============================================================
# The trained RF model is loaded and used as-is.
# This is already fast (vectorized C/C++ call).

rf_model    <- readRDS("trained_rf_model.rds")
predictions <- predict(rf_model, newdata = cell_data)
cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (interpreted R loops over 6.46M rows with string ops, called 5Ã—) | Eliminated |
| **Key technique** | `lapply` + `paste` + named vector lookup per row | Arithmetic row indexing + vectorized `data.table` grouped aggregation |
| **Estimated time** | 86+ hours | ~5â€“15 minutes (depending on RAM/disk) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical output** | Same max/min/mean per neighbor set | Identical values â€” same estimand preserved |
| **Memory** | Lower peak but extreme wall time | Higher peak (~2â€“4 GB for edge vectors) but well within 16 GB |

The optimization replaces millions of interpreted R-level iterations and string operations with vectorized integer arithmetic and `data.table`'s compiled C-level grouped aggregation, reducing runtime by roughly **three orders of magnitude** while producing numerically identical results.
END_RESPONSE>>>
