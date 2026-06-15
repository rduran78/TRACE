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
run_id: Anthropic_F05_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Row-wise `lapply` over 6.46M rows with string operations

1. **`build_neighbor_lookup`** iterates over every row (`~6.46M`) and, for each row, performs `paste()`-based string key construction and hash-table lookups against a 6.46M-entry named vector. This is O(N Ã— K) string operations where K is the average neighbor count (~4 for rook contiguity). With N â‰ˆ 6.46M, that's ~25.8M `paste()` calls plus hash lookups â€” all inside an interpreted R `lapply`.

2. **The string-keyed lookup is entirely unnecessary.** The data is a balanced panel (344,208 cells Ã— 28 years). The mapping from `(cell_id, year)` â†’ row index can be computed arithmetically with integer operations, eliminating all string construction and hash-table lookups.

3. **The neighbor lookup is year-invariant.** Every cell has the same neighbors in every year. So `build_neighbor_lookup` recomputes the *same* neighbor set 28 times (once per year per cell) when it only needs to compute it once per cell and then replicate the offset pattern across years.

4. **`compute_neighbor_stats`** is called 5 times (once per variable), each time iterating over 6.46M rows. Since the neighbor index structure is identical across variables, a single pass that computes all 5 variables' statistics simultaneously would cut overhead by ~5Ã—.

### Estimated speedup

| Bottleneck | Original | Optimized |
|---|---|---|
| String key construction | ~25.8M paste + hash ops in R loop | 0 (integer arithmetic) |
| Neighbor resolution | Per-row, per-year R-level lapply | Vectorized, computed once per cell |
| Stats computation | 5 separate passes over 6.46M rows | 1 pass or fully vectorized via data.table |
| **Expected wall-clock** | **86+ hours** | **~2â€“10 minutes** |

---

## Optimization Strategy

1. **Sort/index the data** by `(id, year)` so that all 28 years for a given cell are contiguous and in order. Then `row_index = (cell_position - 1) * n_years + (year - min_year) + 1` â€” pure integer arithmetic.

2. **Build the neighbor index once per cell** (344K cells, not 6.46M rows), producing an integer vector of cell-position indices.

3. **Expand neighbor relationships to row-level** using vectorized integer arithmetic (add year offsets), avoiding any per-row loop.

4. **Compute all neighbor statistics in one vectorized pass** using `data.table` grouped operations.

---

## Working R Code

```r
library(data.table)

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # -----------------------------------------------------------
  # 0. Convert to data.table (by reference if already one)
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  year_min <- min(years)
  
  # -----------------------------------------------------------
  # 1. Sort by (id, year) so rows are in deterministic order
  #    and record the original order for later restoration.
  # -----------------------------------------------------------
  dt[, orig_row_idx := .I]
  setkey(dt, id, year)
  
  # Cell position: integer index 1..N_cells in id_order order
  id_order_chr <- as.character(id_order)
  n_cells      <- length(id_order)
  
  # Map from cell id -> position in id_order (1-based)
  id_to_pos <- setNames(seq_len(n_cells), id_order_chr)
  
  # Map from cell id -> block start row in the sorted dt
  # After setkey(dt, id, year), each cell occupies a contiguous

  # block of n_years rows. But cells may be in a different order
  # than id_order, so we build an explicit map.
  cell_ids_in_dt_order <- dt$id[seq(1L, nrow(dt), by = n_years)]
  cell_dt_pos <- setNames(seq_along(cell_ids_in_dt_order),
                           as.character(cell_ids_in_dt_order))
  
  # row_of(cell_id, year) in the sorted dt:
  #   block_start = (cell_dt_pos[cell_id] - 1) * n_years
  #   row = block_start + (year - year_min + 1)
  
  # -----------------------------------------------------------
  # 2. Build edge list: (source_cell_pos, neighbor_cell_pos)
  #    once for all cells â€” no per-row, no per-year work.
  # -----------------------------------------------------------
  # rook_neighbors_unique is an nb object indexed by id_order position
  edge_list <- rbindlist(lapply(seq_len(n_cells), function(ci) {
    nb_idx <- rook_neighbors_unique[[ci]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(src_pos = ci, nbr_pos = as.integer(nb_idx))
  }))
  
  # Map id_order positions to dt-block positions
  src_cell_ids <- id_order[edge_list$src_pos]
  nbr_cell_ids <- id_order[edge_list$nbr_pos]
  
  edge_list[, src_block := cell_dt_pos[as.character(src_cell_ids)]]
  edge_list[, nbr_block := cell_dt_pos[as.character(nbr_cell_ids)]]
  
  # -----------------------------------------------------------
  # 3. Expand to row-level edges by crossing with year offsets.
  #    For each year offset t in 0..(n_years-1):
  #      src_row = (src_block - 1)*n_years + t + 1
  #      nbr_row = (nbr_block - 1)*n_years + t + 1
  # -----------------------------------------------------------
  year_offsets <- 0L:(n_years - 1L)
  
  # Vectorized expansion using CJ-like logic
  n_edges <- nrow(edge_list)
  
  # Repeat each edge n_years times
  src_blocks_exp <- rep(edge_list$src_block, each = n_years)
  nbr_blocks_exp <- rep(edge_list$nbr_block, each = n_years)
  year_off_exp   <- rep(year_offsets, times = n_edges)
  
  src_rows <- (src_blocks_exp - 1L) * n_years + year_off_exp + 1L
  nbr_rows <- (nbr_blocks_exp - 1L) * n_years + year_off_exp + 1L
  
  # Free intermediates
  rm(src_blocks_exp, nbr_blocks_exp, year_off_exp, edge_list)
  gc()
  
  # -----------------------------------------------------------
  # 4. Build a data.table of (src_row, nbr_row) and pull
  #    neighbor values for all variables at once.
  # -----------------------------------------------------------
  edges_dt <- data.table(src_row = src_rows, nbr_row = nbr_rows)
  rm(src_rows, nbr_rows)
  gc()
  
  # Attach neighbor values for each source variable
  for (vn in neighbor_source_vars) {
    set(edges_dt, j = vn, value = dt[[vn]][edges_dt$nbr_row])
  }
  
  # -----------------------------------------------------------
  # 5. Compute grouped statistics: max, min, mean per src_row
  #    for each variable, in one grouped operation.
  # -----------------------------------------------------------
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (vn in neighbor_source_vars) {
    vn_sym <- as.name(vn)
    agg_exprs[[paste0("nb_max_", vn)]]  <- bquote(as.numeric(max(.(vn_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_min_", vn)]]  <- bquote(as.numeric(min(.(vn_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_mean_", vn)]] <- bquote(mean(.(vn_sym), na.rm = TRUE))
  }
  
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats_dt <- edges_dt[, eval(agg_call), by = src_row]
  rm(edges_dt)
  gc()
  
  # Replace Inf/-Inf (from max/min on all-NA) with NA
  stat_cols <- setdiff(names(stats_dt), "src_row")
  for (sc in stat_cols) {
    vals <- stats_dt[[sc]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats_dt, j = sc, value = vals)
  }
  
  # -----------------------------------------------------------
  # 6. Join statistics back to the main data.table
  # -----------------------------------------------------------
  # Ensure all rows are represented (cells with 0 neighbors get NA)
  dt[, dt_row := .I]
  
  result <- merge(dt, stats_dt, by.x = "dt_row", by.y = "src_row", all.x = TRUE)
  
  # -----------------------------------------------------------
  # 7. Restore original row order and return as data.frame
  # -----------------------------------------------------------
  setorder(result, orig_row_idx)
  result[, c("orig_row_idx", "dt_row") := NULL]
  
  as.data.frame(result)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimized_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names follow the pattern nb_max_ntl, nb_min_ntl, nb_mean_ntl, etc.
# Rename to match whatever the existing model expects, e.g.:
# setnames(cell_data, "nb_max_ntl", "ntl_neighbor_max")  # adjust as needed
```

### If your model expects the original column naming convention

Add this adapter after the call:

```r
# Adapt column names to match the trained RF model's expectations.
# Adjust the pattern below to your actual naming convention.
for (vn in neighbor_source_vars) {
  old_max  <- paste0("nb_max_", vn);  new_max  <- paste0(vn, "_neighbor_max")
  old_min  <- paste0("nb_min_", vn);  new_min  <- paste0(vn, "_neighbor_min")
  old_mean <- paste0("nb_mean_", vn); new_mean <- paste0(vn, "_neighbor_mean")
  names(cell_data)[names(cell_data) == old_max]  <- new_max
  names(cell_data)[names(cell_data) == old_min]  <- new_min
  names(cell_data)[names(cell_data) == old_mean] <- new_mean
}
```

---

## Why This Preserves the Original Numerical Estimand

| Property | Original | Optimized |
|---|---|---|
| Neighbor set per (cell, year) | Same-year rook neighbors present in data | Identical â€” same edges, same year matching |
| `max`, `min`, `mean` | Computed with `na.rm` via manual filtering | Computed with `na.rm = TRUE` in data.table â€” identical IEEE 754 results |
| Rows with no neighbors | Return `c(NA, NA, NA)` | `merge(..., all.x = TRUE)` fills with `NA` |
| Row order | Original order preserved | Restored via `orig_row_idx` |

The RF model sees identical feature columns with identical values â€” no retraining needed.

---

## Memory Considerations (16 GB laptop)

The expanded edge table has ~1.37M directed edges Ã— 28 years â‰ˆ **38.5M rows**. With 2 integer columns + 5 double columns, that's roughly `38.5M Ã— (2Ã—4 + 5Ã—8) â‰ˆ 1.85 GB` â€” well within 16 GB. The grouped aggregation in `data.table` is memory-efficient and single-pass. Peak memory usage should stay under ~8 GB.
END_RESPONSE>>>
