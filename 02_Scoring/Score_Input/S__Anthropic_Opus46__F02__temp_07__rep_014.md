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
run_id: Anthropic_F02_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core performance bottleneck is in **`build_neighbor_lookup`**, which iterates over ~6.46 million rows with an `lapply` loop, performing per-row string pasting and named-vector lookups. Named vector lookups in R are O(n) hash lookups but doing ~6.46M of them, each with multiple neighbor keys, is extremely slow. The secondary bottleneck is **`compute_neighbor_stats`**, which also loops over 6.46M entries with per-element R function calls (`max`, `min`, `mean`), and the `do.call(rbind, ...)` on a 6.46M-element list is itself a major memory and time sink.

**Specific problems:**

1. **String-key lookups (`paste` + named vector indexing):** `build_neighbor_lookup` creates ~6.46M string keys and then, for each row, creates more string keys for all neighbors and looks them up. This is O(N Ã— avg_neighbors) string operations â€” roughly 40â€“50 million paste + hash lookups.
2. **R-level `lapply` over millions of rows:** Pure R loops/lapply over millions of elements are inherently slow due to interpreter overhead.
3. **`do.call(rbind, list_of_6.46M_vectors)`:** This is notoriously slow and memory-hungry in R.
4. **Redundant recomputation:** The neighbor lookup is the same structural relationship for every variable, which is correctly shared, but the stats computation is done in pure R per-element loops.

---

## Optimization Strategy

### 1. Replace string-key lookups with integer arithmetic
Instead of `paste(id, year, sep="_")` â†’ named vector lookup, use direct integer indexing. Since years are contiguous (1992â€“2019, 28 years), we can compute a row index directly:  
`row_index = (cell_position - 1) * n_years + (year - min_year + 1)`  
This requires that the data is sorted by `(id, year)` and complete (every cell has every year). This eliminates all string operations.

### 2. Vectorize neighbor lookup construction using `data.table`
Build an edge list of `(row_i, row_j)` pairs for all neighbor relationships across all years in one vectorized operation, avoiding any per-row R loop.

### 3. Vectorize `compute_neighbor_stats` using `data.table` grouped operations
Instead of looping over 6.46M entries, create a long-form edge table `(focal_row, neighbor_row)`, join the variable values, and compute `max/min/mean` grouped by `focal_row` â€” all in C-level `data.table` code.

### 4. Memory management
Process one variable at a time and only keep the columns needed. The edge table of ~38M rows (6.46M rows Ã— ~6 avg neighbors) of two integer columns is ~300 MB, which fits in 16 GB RAM.

**Expected speedup:** From 86+ hours to roughly 10â€“30 minutes.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Prepare data as data.table, sorted by (id, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
setkeyv(cell_dt, c("id", "year"))

# Confirm years are contiguous 1992-2019
min_year <- 1992L
max_year <- 2019L
n_years  <- max_year - min_year + 1L  # 28

# id_order is the vector of unique cell IDs aligned with rook_neighbors_unique
# Ensure every cell has every year (panel is balanced)
# If not balanced, the direct-index approach needs a fallback (see note below).

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a global edge list (focal_row, neighbor_row) â€” vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list_vectorized <- function(cell_dt, id_order, neighbors, 
                                       min_year, n_years) {
  # Map each id to its position in id_order (1-based)
  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map each id to its starting row in cell_dt (sorted by id, year)
  # For cell at position p in id_order, its rows in cell_dt are at
  # positions determined by the sort order. We need the actual mapping.
  
  # Build a lookup: for each id, what is its first row in cell_dt?
  cell_dt[, row_idx := .I]
  first_rows <- cell_dt[, .(first_row = min(row_idx), n = .N), by = id]
  
  # Merge with id_order position
  first_rows[, pos := id_to_pos[as.character(id)]]
  
  # For balanced panel: row for (cell_pos p, year y) = first_row[p] + (y - min_year)
  # We need: for each cell p and each neighbor q of p, 
  #          for each year t: edge (first_row[p] + t_offset, first_row[q] + t_offset)
  
  # Build neighbor edge list at the cell level (not year level)
  # Each element of neighbors[[i]] gives positions in id_order of neighbors of cell i
  cat("Building cell-level edge list...\n")
  
  # Flatten the nb object into a data.table of (focal_pos, neighbor_pos)
  focal_pos_list <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_pos_list <- unlist(neighbors)
  
  # Remove 0-entries (spdep::nb uses 0 for no-neighbor regions)
  valid <- neighbor_pos_list > 0L
  cell_edges <- data.table(
    focal_pos    = focal_pos_list[valid],
    neighbor_pos = neighbor_pos_list[valid]
  )
  rm(focal_pos_list, neighbor_pos_list, valid)
  
  cat(sprintf("  Cell-level edges: %d\n", nrow(cell_edges)))
  
  # Map positions to first_row in cell_dt
  # first_rows is keyed by id; we need pos -> first_row
  pos_to_first_row <- integer(n_cells)
  pos_to_first_row[first_rows$pos] <- first_rows$first_row
  
  pos_to_n <- integer(n_cells)
  pos_to_n[first_rows$pos] <- first_rows$n
  
  cell_edges[, focal_first    := pos_to_first_row[focal_pos]]
  cell_edges[, neighbor_first := pos_to_first_row[neighbor_pos]]
  cell_edges[, focal_n        := pos_to_n[focal_pos]]
  cell_edges[, neighbor_n     := pos_to_n[neighbor_pos]]
  
  # Only expand for cells that actually exist in the data
  cell_edges <- cell_edges[focal_first > 0L & neighbor_first > 0L]
  
  cat("Expanding to year-level edges...\n")
  
  # For balanced panels (every cell has n_years rows), expand with year offsets
  # For unbalanced panels, we need a merge-based approach (see fallback below)
  
  # Check if panel is balanced
  balanced <- all(pos_to_n[pos_to_n > 0] == n_years)
  
  if (balanced) {
    cat("  Panel is balanced â€” using fast integer arithmetic.\n")
    year_offsets <- 0L:(n_years - 1L)
    
    # Cross join: each cell_edge Ã— each year_offset
    # To avoid a massive cross join in memory, do it in chunks if needed
    n_cell_edges <- nrow(cell_edges)
    
    # Resulting edges: n_cell_edges * n_years rows
    total_edges <- as.numeric(n_cell_edges) * n_years
    cat(sprintf("  Total year-level edges: %.0f\n", total_edges))
    
    # Allocate
    focal_rows    <- integer(n_cell_edges * n_years)
    neighbor_rows <- integer(n_cell_edges * n_years)
    
    ff <- cell_edges$focal_first
    nf <- cell_edges$neighbor_first
    
    for (t_off in year_offsets) {
      start_idx <- as.integer(t_off) * n_cell_edges + 1L
      end_idx   <- start_idx + n_cell_edges - 1L
      focal_rows[start_idx:end_idx]    <- ff + t_off
      neighbor_rows[start_idx:end_idx] <- nf + t_off
    }
    
    edge_dt <- data.table(
      focal_row    = focal_rows,
      neighbor_row = neighbor_rows
    )
    rm(focal_rows, neighbor_rows, ff, nf)
    
  } else {
    cat("  Panel is unbalanced â€” using merge-based approach.\n")
    # Fallback: expand cell_edges by actual years present for each focal cell
    # This is slower but correct for unbalanced panels
    
    year_dt <- cell_dt[, .(year, row_idx, id)]
    
    # Map focal_pos and neighbor_pos to id
    cell_edges[, focal_id    := id_order[focal_pos]]
    cell_edges[, neighbor_id := id_order[neighbor_pos]]
    
    # Merge to get focal rows
    edge_dt <- merge(
      cell_edges[, .(focal_id, neighbor_id)],
      year_dt[, .(focal_id = id, year, focal_row = row_idx)],
      by = "focal_id",
      allow.cartesian = TRUE
    )
    
    # Merge to get neighbor rows for same year
    edge_dt <- merge(
      edge_dt,
      year_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)],
      by = c("neighbor_id", "year"),
      allow.cartesian = FALSE
    )
    
    edge_dt <- edge_dt[, .(focal_row, neighbor_row)]
  }
  
  gc()
  cat(sprintf("  Final edge table: %d rows\n", nrow(edge_dt)))
  return(edge_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Vectorized neighbor stats using data.table grouping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  
  # Extract the variable values indexed by row
  vals <- cell_dt[[var_name]]
  
  # Attach neighbor values to edge table
  edge_dt[, neighbor_val := vals[neighbor_row]]
  
  # Remove edges where neighbor value is NA
  valid_edges <- edge_dt[!is.na(neighbor_val)]
  
  # Group by focal_row, compute max, min, mean
  stats <- valid_edges[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = focal_row]
  
  # Create full result aligned to all rows
  n <- nrow(cell_dt)
  result_max  <- rep(NA_real_, n)
  result_min  <- rep(NA_real_, n)
  result_mean <- rep(NA_real_, n)
  
  result_max[stats$focal_row]  <- stats$nb_max
  result_min[stats$focal_row]  <- stats$nb_min
  result_mean[stats$focal_row] <- stats$nb_mean
  
  # Clean up temp column
  edge_dt[, neighbor_val := NULL]
  
  # Name columns to match original pipeline expectations
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  cell_dt[, (max_col)  := result_max]
  cell_dt[, (min_col)  := result_min]
  cell_dt[, (mean_col) := result_mean]
  
  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Run the optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ensure row_idx exists
cell_dt[, row_idx := .I]

# Build edge list once (replaces build_neighbor_lookup)
cat("=== Building edge list ===\n")
edge_dt <- build_edge_list_vectorized(
  cell_dt, id_order, rook_neighbors_unique, min_year, n_years
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("=== Computing neighbor features ===\n")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)
  gc()  # free memory between variables
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Convert back to data.frame if the RF predict method requires it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)
cell_dt[, row_idx := NULL]  # clean up helper column

# The trained Random Forest model is unchanged.
# Predictions proceed as before:
# predictions <- predict(rf_model, newdata = cell_data)

cat("=== Done ===\n")
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level `lapply` iterations with `paste` + named vector lookup | One vectorized edge-list construction via integer arithmetic |
| **Stats computation** | 6.46M R-level `lapply` iterations + `do.call(rbind, ...)` | `data.table` grouped aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~10â€“30 minutes |
| **Peak memory** | Excessive due to 6.46M-element list of vectors + `do.call(rbind)` | ~2â€“3 GB for edge table + variable columns |
| **Numerical results** | `max`, `min`, `mean` of non-NA neighbor values | Identical â€” same `max`, `min`, `mean` of same non-NA neighbor values |
| **Trained RF model** | â€” | **Untouched.** No retraining. Column names match original convention. |

The key insight is replacing **millions of R-interpreter-level iterations with string operations** by a single **vectorized integer-indexed edge table** and **`data.table` grouped aggregations** that execute in compiled C code. This eliminates the two dominant bottlenecks while producing numerically identical results.
END_RESPONSE>>>
