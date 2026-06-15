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
run_id: Anthropic_F08_T03_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code recomputes string-key lookups (`paste(id, year)`) for every single row, producing ~6.46 million list entries instead of ~344,208.

2. **String hashing is expensive at scale.** `paste()` and named-vector lookups over 6.46M keys are O(n) in memory and slow in R.

3. **`compute_neighbor_stats` iterates over 6.46M list entries** per variable, doing redundant subsetting. With 5 variables, that's ~32.3 million list traversals.

4. **The lookup is rebuilt identically for every call** if the outer loop were to rebuild it (though here it's built once, the lookup itself is the bottleneck).

**Root cause:** The code conflates the *static spatial structure* (which cell is neighbor to which) with the *dynamic panel structure* (which row holds a given cell's data in a given year). The redesign must separate these two concerns.

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a list of length ~344K mapping each cell's index (in `id_order`) to the indices of its neighbors. This is just a reformatting of `rook_neighbors_unique` and is essentially free.

2. **Organize the data so that year-specific slicing is O(1).** Sort/group the data by `(id, year)` and create a matrix or indexed structure where row `[cell, year]` gives the data row. Since we have 344,208 cells Ã— 28 years, we can build a cellÃ—year index matrix (~9.6M integers, ~38 MB).

3. **Vectorize the neighbor-stat computation per variable.** For each variable, extract values into a cellÃ—year matrix (344,208 Ã— 28). Then for each cell, gather its neighbor rows from this matrix and compute column-wise (i.e., year-wise) max, min, mean. This turns the inner loop from 6.46M iterations to 344K iterations, each operating on small numeric vectors across 28 years simultaneously.

4. **Further vectorize using `vapply` or C++-backed operations.** We can avoid the 344K-iteration R loop entirely by using matrix indexing: build a ragged-to-CSR representation of the neighbor graph, then use `rowsum`-style operations or `data.table` grouping.

The numerical results are identical â€” we are simply reordering the same arithmetic. The trained Random Forest model is untouched.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) cell attributes.
# Preserves the original numerical estimand exactly.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # STEP 0: Convert to data.table for speed (non-destructive)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure consistent ordering: by id then year
  # Create integer keys for fast indexing
  dt[, cell_idx := match(id, id_order)]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # -------------------------------------------------------------------------
  # STEP 1: STATIC â€” Build cell-level neighbor lookup (once, ~344K entries)
  # rook_neighbors_unique is an nb object: a list of integer vectors
  # where element i contains the neighbor indices (into id_order) of cell i.
  # We just need to clean it (remove 0s that spdep uses for no-neighbor cells).
  # -------------------------------------------------------------------------
  cat("Building static cell-level neighbor lookup...\n")
  
  # nb objects use 0L to indicate no neighbors; filter those out
  cell_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb > 0L]
    as.integer(nb)
  })
  # cell_neighbor_idx[[i]] = integer vector of neighbor cell indices for cell i
  
  # -------------------------------------------------------------------------
  # STEP 2: Build a cell Ã— year index matrix for O(1) row lookup
  # idx_matrix[cell_idx, year_col] = row number in dt
  # -------------------------------------------------------------------------
  cat("Building cell x year index matrix...\n")
  
  # Pre-allocate matrix with NA
  idx_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  
  # Fill using vectorized assignment
  yr_col_vec <- year_to_col[as.character(dt$year)]
  idx_matrix[cbind(dt$cell_idx, yr_col_vec)] <- seq_len(nrow(dt))
  
  # -------------------------------------------------------------------------
  # STEP 3: For each variable, build cell Ã— year value matrix, then compute
  #         neighbor max/min/mean via the static neighbor graph.
  # -------------------------------------------------------------------------
  
  # Pre-build CSR-like structure for vectorized neighbor gathering
  # "Expand" the neighbor list into a flat edge list for matrix indexing
  cat("Building flat edge list from neighbor graph...\n")
  
  n_neighbors_per_cell <- lengths(cell_neighbor_idx)
  total_edges <- sum(n_neighbors_per_cell)
  
  # Source cell index (repeated for each neighbor)
  src_cell <- rep.int(seq_len(n_cells), n_neighbors_per_cell)
  # Destination cell index (the actual neighbor)
  dst_cell <- unlist(cell_neighbor_idx, use.names = FALSE)
  
  cat(sprintf("  Total directed edges: %d\n", total_edges))
  cat(sprintf("  Cells with neighbors: %d / %d\n",
              sum(n_neighbors_per_cell > 0), n_cells))
  
  # For grouping operations: create a factor for src_cell
  src_factor <- src_cell  # integer, will use with rowsum or data.table
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))
    
    # --- Build cell Ã— year value matrix for this variable ---
    val_vec <- dt[[var_name]]
    val_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_matrix[cbind(dt$cell_idx, yr_col_vec)] <- val_vec
    
    # --- For each year, gather neighbor values and compute stats ---
    # We'll compute column-by-column (year-by-year) using the flat edge list
    # This avoids a 344K R-level loop.
    
    # Pre-allocate result matrices (cell Ã— year)
    max_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (yr_idx in seq_len(n_years)) {
      # Values for all cells in this year
      yr_vals <- val_matrix[, yr_idx]  # length n_cells
      
      # Gather neighbor values using the edge list
      neighbor_vals <- yr_vals[dst_cell]  # length = total_edges
      
      # We need max, min, mean grouped by src_cell
      # Handle NAs: we need to ignore them
      
      # Use data.table for fast grouped aggregation
      edge_dt <- data.table(
        src  = src_cell,
        nval = neighbor_vals
      )
      
      # Remove NA neighbor values
      edge_dt <- edge_dt[!is.na(nval)]
      
      if (nrow(edge_dt) > 0) {
        stats_dt <- edge_dt[, .(
          nmax  = max(nval),
          nmin  = min(nval),
          nmean = mean(nval)
        ), by = src]
        
        max_matrix[stats_dt$src, yr_idx]  <- stats_dt$nmax
        min_matrix[stats_dt$src, yr_idx]  <- stats_dt$nmin
        mean_matrix[stats_dt$src, yr_idx] <- stats_dt$nmean
      }
    }
    
    # --- Write results back into dt using the index matrix ---
    # Flatten cellÃ—year matrices back to row-level vectors
    valid <- !is.na(as.vector(idx_matrix))
    row_indices <- as.vector(idx_matrix)[valid]
    
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)
    
    # Initialize columns with NA
    set(dt, j = max_col_name,  value = NA_real_)
    set(dt, j = min_col_name,  value = NA_real_)
    set(dt, j = mean_col_name, value = NA_real_)
    
    # Assign values
    set(dt, i = row_indices, j = max_col_name,  value = as.vector(max_matrix)[valid])
    set(dt, i = row_indices, j = min_col_name,  value = as.vector(min_matrix)[valid])
    set(dt, i = row_indices, j = mean_col_name, value = as.vector(mean_matrix)[valid])
    
    cat(sprintf("  Done: %s, %s, %s\n", max_col_name, min_col_name, mean_col_name))
  }
  
  # -------------------------------------------------------------------------
  # STEP 4: Clean up helper column and return
  # -------------------------------------------------------------------------
  dt[, cell_idx := NULL]
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup size** | ~6.46M list entries (row-level) | ~344K list entries (cell-level) |
| **String key operations** | ~6.46M `paste()` + named vector lookups | **Zero** â€” pure integer indexing |
| **Inner loop iterations per variable** | ~6.46M (one per row) | 28 (one per year), vectorized over edges |
| **Aggregation engine** | Base R `lapply` with per-element subsetting | `data.table` grouped aggregation (C-backed) |
| **Memory for lookup** | ~6.46M-element list of integer vectors | One integer matrix 344K Ã— 28 (~38 MB) + flat edge list (~11 MB) |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |

### Why the numerical results are identical

The original code, for each row `i`, finds the rows of cell `i`'s rook neighbors in the same year, extracts `var_name` values, removes NAs, and computes `max`, `min`, `mean`. The optimized code does exactly the same arithmetic â€” it gathers the same neighbor cells' values for the same year, removes NAs, and computes the same three aggregates â€” but organizes the work by (1) separating the static topology from the dynamic values, (2) using matrix indexing instead of string lookups, and (3) leveraging `data.table`'s C-level grouped aggregation instead of R-level `lapply`. The trained Random Forest model is never modified or retrained; only its input features are computed more efficiently.
END_RESPONSE>>>
