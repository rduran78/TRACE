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
run_id: Anthropic_F08_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is in `build_neighbor_lookup`. It constructs a lookup mapping **per cell-year row** (~6.46 million entries), even though the neighbor *topology* is purely spatial and identical across all 28 years. Specifically:

1. **Redundant work Ã— 28**: The function builds ~6.46M entries in `neighbor_lookup`, one per row. But the neighbor graph has only ~344K cells. The topology is duplicated 28 times â€” once per year â€” doing string-pasting, hash lookups, and allocation for every cell-year combination.

2. **String-key hash lookups at scale**: `paste(id, year, sep="_")` creates ~6.46M string keys, and then for each row, neighbor keys are looked up in that hash. This is O(n Ã— avg_neighbors) string operations on millions of rows.

3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the 6.46M-element `neighbor_lookup` list. The lookup list itself consumes substantial memory (millions of integer vectors).

4. **The result is numerically identical** to: for each cell, find its spatial neighbors (fixed); for each year, pull that year's variable values for those neighbors and compute max/min/mean. This is a **spatial join per year**, not a per-row operation.

**Summary**: The static neighbor graph is being re-expanded into a year-specific row-index lookup at enormous cost. The fix is to separate the static topology from the year-varying data.

---

## Optimization Strategy

1. **Build the neighbor topology once** over the 344K cells (not 6.46M rows). Store it as a simple list: `cell_neighbors[[cell_index]] â†’ vector of neighbor cell indices`. This is just a reformatting of `rook_neighbors_unique` and is done once.

2. **Organize data by year**. For each year, extract the variable columns into a matrix indexed by cell position. Since cells repeat in the same order each year (or can be sorted to do so), neighbor indexing is direct integer subscripting into a vector â€” the fastest possible R operation.

3. **Compute neighbor stats per year per variable** using vectorized operations. For each cell, its neighbors are known; pull their values from the year-slice vector, compute max/min/mean. With `data.table` or matrix operations, this can be heavily vectorized.

4. **Optional further speedup**: Use a CSR (compressed sparse row) representation of the neighbor graph and a single vectorized pass per variable-year via `collapse::fmax`, `fmin`, `fmean` grouped operations, or manual vectorization with `rep()`/`unlist()` tricks.

**Expected speedup**: From ~86 hours to **minutes**. The dominant cost drops from ~6.46M list elements Ã— 5 variables to ~344K cells Ã— 28 years Ã— 5 variables with vectorized arithmetic.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from year-varying data
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, 
                                          id_order, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars) {
  
  # ------------------------------------------------------------------
  # STEP 1: Build the static neighbor topology ONCE (344K cells)
  # ------------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
  # rook_neighbors_unique[[i]] gives the indices (into id_order) of neighbors of cell i.
  
  n_cells <- length(id_order)
  
  # Map cell IDs to their positional index in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute CSR-style representation for vectorized lookups
  # neighbor_of[[i]] = integer vector of positional indices of neighbors of cell i
  # This is essentially rook_neighbors_unique itself, but we ensure integer indexing.
  neighbor_of <- rook_neighbors_unique  # nb object: list of integer vectors
  
  # Build CSR (compressed sparse row) arrays for fully vectorized computation
  # "from" cell index repeated for each neighbor, "to" = neighbor cell index
  n_neighbors <- lengths(neighbor_of)
  
  # Cell indices that have at least one neighbor
  has_neighbors <- which(n_neighbors > 0)
  
  # CSR vectors
  csr_from <- rep.int(seq_along(neighbor_of), n_neighbors)
  csr_to   <- unlist(neighbor_of, use.names = FALSE)
  
  # Remove 0-entries that spdep::nb uses to indicate no neighbors
  valid <- csr_to > 0L
  csr_from <- csr_from[valid]
  csr_to   <- csr_to[valid]
  
  n_edges <- length(csr_from)
  
  message(sprintf("Static topology: %d cells, %d directed edges", n_cells, n_edges))
  
  # ------------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table and ensure sort order
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure the data has an 'id' and 'year' column
  stopifnot(all(c("id", "year") %in% names(dt)))
  
  # Create a cell position column for fast indexing
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  message(sprintf("Processing %d variables Ã— %d years", 
                  length(neighbor_source_vars), n_years))
  
  # ------------------------------------------------------------------
  # STEP 3: Pre-allocate output columns
  # ------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # ------------------------------------------------------------------
  # STEP 4: For each year, compute neighbor stats vectorized
  # ------------------------------------------------------------------
  # Key the data.table for fast subsetting
  setkey(dt, year, cell_pos)
  
  for (yr in years) {
    
    # Extract this year's slice, ordered by cell_pos
    # After setkey(dt, year, cell_pos), rows for each year are contiguous
    # and sorted by cell_pos.
    yr_idx <- dt[.(yr), which = TRUE]
    
    # Get cell positions for this year (should be 1..n_cells if complete panel,
    # but we handle incomplete panels too)
    yr_cell_pos <- dt$cell_pos[yr_idx]
    
    # Build a fast map: cell_pos -> row index within yr_idx
    # For a complete balanced panel this is identity, but we handle gaps
    pos_to_yr_row <- integer(n_cells)
    pos_to_yr_row[] <- NA_integer_
    pos_to_yr_row[yr_cell_pos] <- seq_along(yr_idx)
    
    for (var_name in neighbor_source_vars) {
      
      # Extract the variable values for this year, ordered by cell_pos
      vals_yr <- dt[[var_name]][yr_idx]  # length = number of cells this year
      
      # Look up neighbor values using CSR representation
      # csr_from and csr_to are in cell_pos space
      # Map to this year's row space
      from_yr_row <- pos_to_yr_row[csr_from]
      to_yr_row   <- pos_to_yr_row[csr_to]
      
      # Filter edges where both endpoints exist this year
      edge_valid <- !is.na(from_yr_row) & !is.na(to_yr_row)
      e_from <- from_yr_row[edge_valid]
      e_to   <- to_yr_row[edge_valid]
      
      # Get neighbor values
      neighbor_vals <- vals_yr[e_to]
      
      # Also filter out NA variable values
      val_valid <- !is.na(neighbor_vals)
      e_from_v  <- e_from[val_valid]
      nv        <- neighbor_vals[val_valid]
      
      # Compute grouped max, min, mean using data.table's fast grouping
      # or base R tapply / collapse package
      if (length(e_from_v) > 0) {
        
        # Use data.table for fast grouped aggregation
        edge_dt <- data.table(from_row = e_from_v, nval = nv)
        
        agg <- edge_dt[, .(nmax  = max(nval), 
                           nmin  = min(nval), 
                           nmean = mean(nval)), 
                       by = from_row]
        
        # Write results back into dt
        target_rows <- yr_idx[agg$from_row]
        
        max_col  <- paste0("neighbor_max_", var_name)
        min_col  <- paste0("neighbor_min_", var_name)
        mean_col <- paste0("neighbor_mean_", var_name)
        
        set(dt, i = target_rows, j = max_col,  value = agg$nmax)
        set(dt, i = target_rows, j = min_col,  value = agg$nmin)
        set(dt, i = target_rows, j = mean_col, value = agg$nmean)
      }
    }
    
    if (which(years == yr) %% 5 == 0 || yr == years[n_years]) {
      message(sprintf("  Completed year %d (%d/%d)", yr, which(years == yr), n_years))
    }
  }
  
  # ------------------------------------------------------------------
  # STEP 5: Remove helper column and restore original row order
  # ------------------------------------------------------------------
  dt[, cell_pos := NULL]
  
  # Restore original row order (setkey changed it)
  # If caller needs original order, they should pass a row-id.
  # We restore by (id, year) to match the input.
  setkey(dt, NULL)
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# USAGE â€” Drop-in replacement for the original outer loop
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

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `rook_neighbors_unique` is used identically; no edges added or removed. |
| **Same variables** | The same 5 source variables are read; no transformations applied. |
| **Same aggregation functions** | `max()`, `min()`, `mean()` are applied to exactly the same sets of non-NA neighbor values per cell-year. |
| **Same output columns** | `neighbor_max_X`, `neighbor_min_X`, `neighbor_mean_X` â€” identical names and semantics. |
| **Random Forest untouched** | The trained model is only used for `predict()` afterward; no retraining. |

## Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-key entries | 344K-cell CSR built once |
| Per-variable computation | 6.46M list iterations Ã— 5 | 28 year-slices Ã— 5 vars, vectorized `data.table` grouped aggregation |
| String operations | ~billions of `paste()` + hash lookups | Zero |
| Memory for lookup | ~6.46M-element list of integer vectors | Two integer vectors of length ~1.37M (CSR) |
| **Estimated wall time** | **86+ hours** | **~5â€“15 minutes** |

The speedup comes from recognizing that the neighbor topology is **static across years** and should be encoded once as a sparse graph, then applied per year-slice using vectorized integer indexing and `data.table` grouped aggregation â€” eliminating millions of redundant string operations and list allocations.
END_RESPONSE>>>
