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
run_id: Anthropic_F08_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates the static spatial topology with the dynamic temporal data.**

### Specific bottlenecks:

1. **`build_neighbor_lookup` operates on the full 6.46M-row cell-year panel.** It builds a lookup that maps each of the ~6.46M rows to its neighbor *rows* â€” but the neighbor *relationship* is purely spatial (344,208 cells Ã— ~4 neighbors each). The year dimension is irrelevant to topology. By embedding year into the lookup via `paste(id, year)` keys, the function creates ~6.46M list entries instead of ~344K, doing ~28Ã— redundant work.

2. **String-based key lookups (`paste` + named vector indexing) are extremely slow at this scale.** Creating and hashing ~6.46M string keys, then performing ~6.46M Ã— ~4 = ~26M named lookups, is a major bottleneck. Named vector lookup in R is O(n) per probe in the worst case.

3. **`compute_neighbor_stats` iterates row-by-row over 6.46M rows via `lapply`.** Each iteration subsets a numeric vector, removes NAs, and computes three aggregates. The overhead of 6.46M R function calls plus `do.call(rbind, ...)` on 6.46M three-element vectors is enormous.

4. **The outer loop repeats this entire process 5 times** (once per variable), multiplying the cost.

### The key insight:

- **Static:** The neighbor graph (which cell borders which cell) never changes across years. There are only ~344K cells with ~1.37M directed neighbor edges.
- **Dynamic:** The variable values (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`) change by year.

The correct design is: **build the neighbor structure once over cells (not cell-years), then for each year, slice the data, use the cell-level neighbor structure to gather neighbor values, and compute stats â€” all vectorized.**

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a simple list of length 344,208 where each element contains the integer indices of that cell's neighbors within the cell ID vector. No year, no string keys. This runs in milliseconds.

2. **Process year-by-year.** For each of the 28 years, subset the data to that year's ~344K rows. Within a single year, every cell appears exactly once, so the cell-level neighbor indices directly map to row indices (after aligning cell order).

3. **Vectorize the neighbor aggregation using `data.table` or matrix operations.** Instead of `lapply` over millions of rows, "explode" the neighbor list into an edge table (cell_index, neighbor_index), join on variable values, and use `data.table` grouped aggregation (`max`, `min`, `mean` by cell) â€” which is C-level fast.

4. **Process all 5 variables simultaneously** within the same year pass to avoid redundant subsetting.

### Expected speedup:
- Neighbor lookup build: from ~hours to <1 second.
- Neighbor stats: from ~17 hours per variable to ~seconds per variable per year.
- Total: from ~86+ hours to **~2â€“5 minutes**.

The Random Forest model is never touched â€” only the feature-engineering step is redesigned. The numerical results (neighbor max, min, mean) are identical.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the STATIC cell-level neighbor lookup (once, independent of year)
# ==============================================================================
# Inputs:
#   id_order           â€” vector of 344,208 cell IDs in the order matching
#                         rook_neighbors_unique (i.e., id_order[i] is the cell
#                         whose neighbors are rook_neighbors_unique[[i]])
#   rook_neighbors_unique â€” spdep::nb object (list of 344,208 integer vectors,
#                            each giving positional indices of neighbors within
#                            id_order)
#
# Output:
#   cell_neighbor_idx  â€” list of length 344,208; cell_neighbor_idx[[i]] gives
#                         the positional indices (within id_order) of cell i's
#                         rook neighbors. This is EXACTLY rook_neighbors_unique
#                         with the spdep zero-neighbor convention handled.

build_cell_neighbor_lookup <- function(rook_neighbors_unique) {
  # spdep::nb objects encode "no neighbors" as a single 0L.
  # We convert those to integer(0) for clean downstream indexing.
  lapply(rook_neighbors_unique, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })
}

cell_neighbor_idx <- build_cell_neighbor_lookup(rook_neighbors_unique)


# ==============================================================================
# STEP 2: Pre-build the exploded edge table (once, static)
# ==============================================================================
# This is a two-column data.table: (cell_pos, neighbor_pos)
# where both are positional indices into id_order (1..344208).
# ~1,373,394 rows â€” trivially small.

build_edge_table <- function(cell_neighbor_idx) {
  n_neighbors <- vapply(cell_neighbor_idx, length, integer(1))
  cell_pos     <- rep(seq_along(cell_neighbor_idx), times = n_neighbors)
  neighbor_pos <- unlist(cell_neighbor_idx, use.names = FALSE)
  data.table(cell_pos = cell_pos, neighbor_pos = neighbor_pos)
}

edge_dt <- build_edge_table(cell_neighbor_idx)


# ==============================================================================
# STEP 3: Compute neighbor stats for all variables, all years â€” vectorized
# ==============================================================================
# Inputs:
#   cell_data  â€” data.frame/data.table with columns: id, year, and the
#                 neighbor_source_vars. ~6.46M rows.
#   id_order   â€” the 344,208 cell IDs in positional order.
#   edge_dt    â€” from Step 2.
#   neighbor_source_vars â€” character vector of variable names.
#
# Output:
#   cell_data with new columns: <var>_neighbor_max, <var>_neighbor_min,
#   <var>_neighbor_mean for each var in neighbor_source_vars.

compute_all_neighbor_features <- function(cell_data, id_order, edge_dt,
                                          neighbor_source_vars) {
  
  # Convert to data.table if needed (by reference if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Create a mapping: cell_id -> positional index in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add positional index to cell_data
  cell_data[, cell_pos := id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns with NA_real_
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # Process each year
  for (yr in years) {
    
    cat(sprintf("Processing year %d ...\n", yr))
    
    # Row indices in cell_data for this year
    yr_row_idx <- which(cell_data$year == yr)
    
    # Extract the sub-table for this year: cell_pos and variable values
    # We need a fast mapping: cell_pos -> row index within yr_row_idx
    yr_cell_pos <- cell_data$cell_pos[yr_row_idx]
    
    # Map: positional_index_in_id_order -> index_within_yr_row_idx
    # Not all 344K cells may be present every year, so use a sparse approach.
    pos_to_yr_row <- integer(length(id_order))  # vector of length 344K
    pos_to_yr_row[] <- 0L
    pos_to_yr_row[yr_cell_pos] <- seq_along(yr_row_idx)
    
    # For the edge table, find which edges have both cell and neighbor present
    # this year. Map cell_pos and neighbor_pos to yr_row indices.
    edge_cell_yr     <- pos_to_yr_row[edge_dt$cell_pos]
    edge_neighbor_yr <- pos_to_yr_row[edge_dt$neighbor_pos]
    valid_edges      <- (edge_cell_yr > 0L) & (edge_neighbor_yr > 0L)
    
    # Filtered edge mapping (indices into yr_row_idx)
    e_cell     <- edge_cell_yr[valid_edges]
    e_neighbor <- edge_neighbor_yr[valid_edges]
    
    # Actual row indices in cell_data
    e_cell_row     <- yr_row_idx[e_cell]
    e_neighbor_row <- yr_row_idx[e_neighbor]
    
    for (var_name in neighbor_source_vars) {
      
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")
      
      # Get neighbor values
      neighbor_vals <- cell_data[[var_name]][e_neighbor_row]
      
      # Build a small data.table for grouped aggregation
      agg_dt <- data.table(
        yr_idx = e_cell,
        nval   = neighbor_vals
      )
      
      # Remove NA neighbor values before aggregation
      agg_dt <- agg_dt[!is.na(nval)]
      
      if (nrow(agg_dt) > 0L) {
        # Compute grouped stats â€” very fast in data.table
        stats <- agg_dt[, .(
          nb_max  = max(nval),
          nb_min  = min(nval),
          nb_mean = mean(nval)
        ), by = yr_idx]
        
        # Write results back into cell_data at the correct rows
        target_rows <- yr_row_idx[stats$yr_idx]
        set(cell_data, i = target_rows, j = col_max,  value = stats$nb_max)
        set(cell_data, i = target_rows, j = col_min,  value = stats$nb_min)
        set(cell_data, i = target_rows, j = col_mean, value = stats$nb_mean)
      }
    }
  }
  
  # Clean up helper column
  cell_data[, cell_pos := NULL]
  
  return(cell_data)
}


# ==============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  edge_dt              = edge_dt,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the 15 new columns (5 vars Ã— 3 stats).
# Proceed directly to Random Forest prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Guarantees

| Requirement | Status |
|---|---|
| **Numerical equivalence** | âœ… Identical `max`, `min`, `mean` per cell-year-variable. The aggregation logic is mathematically equivalent; only the execution strategy changes. |
| **Trained RF model preserved** | âœ… No model retraining. The code only engineers features, then feeds them to `predict()`. |
| **Same output columns** | âœ… `<var>_neighbor_max`, `<var>_neighbor_min`, `<var>_neighbor_mean` for all 5 variables. |
| **RAM feasibility (16 GB)** | âœ… Peak additional memory is the edge table (~1.37M Ã— 2 integers â‰ˆ 11 MB) plus one year-slice of aggregation (~1.37M Ã— 2 doubles â‰ˆ 22 MB). The main `cell_data` table (~6.46M Ã— 125 columns) is the dominant allocation and was already in memory. |
| **Speed** | âœ… ~28 years Ã— 5 variables Ã— ~1.37M edges of `data.table` grouped aggregation â‰ˆ **2â€“5 minutes** total vs. 86+ hours. |
END_RESPONSE>>>
