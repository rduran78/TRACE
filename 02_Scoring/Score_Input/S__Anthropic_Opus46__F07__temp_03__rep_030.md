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
run_id: Anthropic_F07_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²) list construction via string-key lookups
- For each of the **~6.46 million rows**, the function:
  - Converts cell IDs to character strings.
  - Pastes cell ID + year into a string key.
  - Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which are hash-table lookups on character keys.
- The `idx_lookup` named vector has **6.46 million entries**. Building it and querying it millions of times is extremely slow.
- The `lapply` over 6.46M rows with per-element R-level logic is inherently slow (no vectorization).

### 2. `compute_neighbor_stats` â€” Repeated R-level loops
- Another `lapply` over 6.46M elements, each extracting a variable-length subset of values, removing NAs, and computing max/min/mean.
- Called **5 times** (once per source variable), so ~32.3 million R-level iterations total.
- Each iteration allocates small vectors and calls three summary functions.

### Combined effect
The nested R-level loops with string operations over millions of rows produce the estimated **86+ hour** runtime. The algorithm itself is correct â€” the problem is purely implementation efficiency.

---

## Optimization Strategy

### A. Replace string-keyed lookup with integer-indexed join (vectorized)

Instead of building a named character vector and doing per-row pastes and lookups, we:

1. **Sort `cell_data` by `(id, year)`** (or ensure a known order) and build a fast integer-indexed mapping: a matrix where `row_map[cell_index, year_index]` gives the row number in `cell_data`. This is O(n) to build and O(1) to query.
2. **Expand the neighbor list to a two-column edge table** (source_cell_index, neighbor_cell_index) â€” only ~1.37M edges.
3. **Cross-join edges Ã— years** to get all (source_row, neighbor_row) pairs â€” ~1.37M Ã— 28 â‰ˆ 38.4M pairs, which is large but manageable as integer vectors.
4. Use `data.table` grouped aggregation on the edge table to compute max, min, mean in one vectorized pass per variable.

### B. Use `data.table` for grouped aggregation

`data.table` computes grouped statistics (max, min, mean) in optimized C code. One grouped operation over 38.4M rows is far faster than 6.46M R-level `lapply` iterations.

### C. Compute all 5 variables in a single pass (or 5 fast passes)

Each variable requires one `data.table` grouped aggregation â€” trivially fast once the edge table exists.

### Expected speedup
- **Build phase**: from ~hours to ~seconds (integer matrix indexing replaces millions of string operations).
- **Stats phase**: from ~hours per variable to ~seconds per variable (vectorized C-level grouping).
- **Total**: from 86+ hours to **minutes**.

### Preservation guarantees
- The neighbor topology is identical (same rook-neighbor relationships).
- The statistics (max, min, mean of non-NA neighbor values) are numerically identical.
- `cell_data` gains the same columns with the same names.
- The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # Convert to data.table for speed (preserves all columns)
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure we can map back to original row order
  dt[, .orig_row := .I]
  
  # ---------------------------------------------------------------
  # Step 1: Build integer cell-index mapping

  # id_order is the vector of cell IDs in the order matching

  # rook_neighbors_unique (an nb object).
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  
  # Map each cell ID to its index in id_order (1-based)
  cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  # ---------------------------------------------------------------
  # Step 2: Build edge table from nb object
  # Each element of rook_neighbors_unique[[i]] is a vector of
  # neighbor indices (into id_order). 0 means no neighbors.
  # ---------------------------------------------------------------
  # Preallocate by computing total edges
  edge_from <- integer(0)
  edge_to   <- integer(0)
  
  # Vectorized construction of edge list
  lengths_nb <- lengths(rook_neighbors_unique)
  # nb objects use 0 to indicate no neighbors
  has_neighbors <- sapply(rook_neighbors_unique, function(x) !(length(x) == 1 && x[0+1] == 0L))
  
  # More robust: filter out 0-entries
  from_list <- rep(seq_len(n_cells), lengths_nb)
  to_list   <- unlist(rook_neighbors_unique)
  
  # Remove entries where neighbor index is 0 (no-neighbor sentinel in nb objects)
  valid <- to_list != 0L
  edge_from <- from_list[valid]
  edge_to   <- to_list[valid]
  
  edges <- data.table(from_cell_idx = edge_from, to_cell_idx = edge_to)
  
  cat(sprintf("Edge table: %d directed rook-neighbor edges\n", nrow(edges)))
  
  # ---------------------------------------------------------------
  # Step 3: Build row-lookup matrix: row_map[cell_idx, year_offset]
  # This gives the row index in dt for each (cell, year) combination.
  # ---------------------------------------------------------------
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_offset <- setNames(seq_len(n_years), as.character(years))
  
  # Map cell IDs in dt to cell indices
  dt[, .cell_idx := cell_id_to_idx[as.character(id)]]
  dt[, .year_off := year_to_offset[as.character(year)]]
  
  # Build the lookup matrix (NA where a cell-year doesn't exist)
  row_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_map[cbind(dt$.cell_idx, dt$.year_off)] <- dt$.orig_row
  
  cat(sprintf("Row map: %d cells x %d years\n", n_cells, n_years))
  
  # ---------------------------------------------------------------
  # Step 4: Expand edges x years to get (source_row, neighbor_row)
  # ---------------------------------------------------------------
  # For each year offset, look up source and neighbor rows
  # This produces ~edges * years pairs, but we do it vectorized per year
  # to control memory.
  
  # Preallocate result columns in dt
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = NA_real_)
  }
  
  # Strategy: process year-by-year to limit memory to ~edges rows per iteration
  # (~1.37M rows per year, very fast)
  
  n_edges <- nrow(edges)
  
  for (yr_idx in seq_len(n_years)) {
    yr <- years[yr_idx]
    
    # Source rows and neighbor rows for this year
    src_rows <- row_map[edges$from_cell_idx, yr_idx]
    nbr_rows <- row_map[edges$to_cell_idx,   yr_idx]
    
    # Filter to valid pairs (both source and neighbor exist in this year)
    valid_pair <- !is.na(src_rows) & !is.na(nbr_rows)
    
    if (sum(valid_pair) == 0L) next
    
    src_valid <- src_rows[valid_pair]
    nbr_valid <- nbr_rows[valid_pair]
    
    for (var_name in neighbor_source_vars) {
      # Get neighbor values
      nbr_vals <- dt[[var_name]][nbr_valid]
      
      # Build a small data.table for grouped aggregation
      agg_dt <- data.table(src = src_valid, val = nbr_vals)
      
      # Remove NA values before aggregation
      agg_dt <- agg_dt[!is.na(val)]
      
      if (nrow(agg_dt) == 0L) next
      
      # Compute grouped stats
      stats <- agg_dt[, .(
        nmax  = max(val),
        nmin  = min(val),
        nmean = mean(val)
      ), by = src]
      
      # Write results back into dt
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")
      
      set(dt, i = stats$src, j = col_max,  value = stats$nmax)
      set(dt, i = stats$src, j = col_min,  value = stats$nmin)
      set(dt, i = stats$src, j = col_mean, value = stats$nmean)
    }
    
    if (yr_idx %% 5 == 0 || yr_idx == n_years) {
      cat(sprintf("  Completed year %d (%d/%d)\n", yr, yr_idx, n_years))
    }
  }
  
  # ---------------------------------------------------------------
  # Step 5: Clean up helper columns and restore original order
  # ---------------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c(".orig_row", ".cell_idx", ".year_off") := NULL]
  
  # Convert back to data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    result <- as.data.frame(dt)
  } else {
    result <- dt
  }
  
  return(result)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# cell_data now contains the same 15 new columns
# (5 vars Ã— {_neighbor_max, _neighbor_min, _neighbor_mean})
# with numerically identical values to the original implementation.
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named-vector hash lookups | Integer matrix `row_map[cell, year]` â€” O(1) per query |
| **Neighbor stats** | 6.46M R-level `lapply` iterations Ã— 5 vars | `data.table` grouped aggregation per year (~1.37M rows Ã— 28 years Ã— 5 vars), all in C |
| **Peak memory** | ~6.46M-element list of integer vectors | ~1.37M-row edge table + year-slice aggregation tables (~tens of MB) |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** on a 16 GB laptop |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | **Identical** â€” same edges, same aggregation functions, same NA handling |
| **RF model** | Trained | **Untouched** â€” only the feature-engineering step is optimized |
END_RESPONSE>>>
