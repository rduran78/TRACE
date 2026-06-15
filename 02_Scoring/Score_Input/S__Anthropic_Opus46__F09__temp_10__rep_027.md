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
run_id: Anthropic_F09_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is slow (~86+ hours) for three compounding reasons:

### A. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
For every one of the ~6.46 million cell-year rows, the function performs:
- A character-based lookup into `id_to_ref` (hash lookup).
- Extraction of neighbor cell IDs from the `nb` object.
- Construction of `paste()`-based keys for every neighbor Ã— that row's year.
- A named-vector lookup via `idx_lookup[neighbor_keys]`.

This means ~6.46M iterations in R-level `lapply`, each allocating small character vectors and performing named-vector subsetting. Named-vector subsetting in R is O(n) per lookup when the vector is large (6.46M entries), making this effectively **O(rows Ã— avg_neighbors)** with a large constant factor. The total key lookups are ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M named-vector lookups against a 6.46M-length named vector, which is extremely slow.

### B. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
For each variable, another 6.46M-iteration `lapply` computes max/min/mean on small neighbor-value vectors. This is called 5 times (one per variable), so ~32.3M R-level iterations with per-element allocation.

### C. The Fundamental Design Flaw: Rebuilding Neighbor Context Per Cell-Year
The neighbor **topology** is constant across all 28 years â€” cell A is always the rook neighbor of cell B regardless of year. But `build_neighbor_lookup` re-resolves neighbor relationships at the cell-year level, exploding a 344K-cell topology problem into a 6.46M-row problem. The lookup should be built **once at the cell level** (344K entries) and then joined by year.

---

## 2. Optimization Strategy

### Step 1: Build a Static Cell-Level Neighbor Edge Table (Once)
Convert the `spdep::nb` object into a two-column `data.table` of directed edges: `(cell_id, neighbor_id)`. This has ~1.37M rows and is year-invariant.

### Step 2: Join Yearly Attributes onto the Edge Table
For each year, join the cell-level attribute values onto the edge table using `data.table` keyed joins. This turns the neighbor-value resolution into a vectorized equi-join.

### Step 3: Compute Grouped Aggregates
Group by `(cell_id, year)` and compute `max`, `min`, `mean` in a single vectorized `data.table` aggregation â€” no R-level loops.

### Step 4: Join Back to the Main Dataset
Left-join the aggregated neighbor stats back to the main `cell_data`.

**Expected speedup**: The dominant cost moves from ~32M R-level `lapply` iterations with named-vector lookups to a handful of `data.table` keyed joins and grouped aggregations over ~38M rows (1.37M edges Ã— 28 years), which should complete in **minutes, not days**.

**Preserves**: The trained Random Forest model (untouched) and the original numerical estimand (same max/min/mean computed over the same neighbor sets, same variable names appended to `cell_data`).

---

## 3. Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build a static, year-invariant neighbor edge table (run ONCE)
# ===========================================================================
# Inputs:
#   id_order            â€” vector of 344,208 cell IDs, in the same order as the nb object
#   rook_neighbors_unique â€” spdep::nb object (list of integer index vectors)

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total directed edges
  n_edges <- sum(lengths(neighbors_nb))  # ~1,373,394
  
  from_id    <- integer(n_edges)
  to_id      <- integer(n_edges)
  pos        <- 1L
  
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb_idx) == 1L && nb_idx == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }
  
  # Trim if any cells had zero neighbors
  edge_dt <- data.table(
    cell_id     = from_id[1:(pos - 1L)],
    neighbor_id = to_id[1:(pos - 1L)]
  )
  
  return(edge_dt)
}

edge_table <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1.37M rows, two integer columns. Tiny in memory.

# ===========================================================================
# STEP 2â€“4: Compute all neighbor features and attach to cell_data
# ===========================================================================
# Inputs:
#   cell_data  â€” data.frame/data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, plus other columns
#   edge_table â€” from Step 1
#
# Output:
#   cell_data  â€” same object with new columns appended:
#                neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...
#                (3 columns Ã— 5 variables = 15 new columns)

compute_all_neighbor_features <- function(cell_data, edge_table, 
                                          neighbor_source_vars) {
  
  # Convert to data.table if needed (modifies in place to save memory)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # --- Build the cross-year edge table by joining on year ----
  # Take just the columns we need from cell_data for the neighbor side
  # to keep the join table small.
  
  # Unique years
  years <- sort(unique(cell_data$year))
  
  # Expand edge_table Ã— years  (~1.37M Ã— 28 = ~38.4M rows)
  # But we do NOT need all years in memory at once if RAM is tight.
  # Strategy: process year-by-year to stay under 16 GB.
  
  # Ensure keys for fast join
  setkey(cell_data, id, year)
  
  # Pre-allocate result columns in cell_data (filled with NA)
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
  }
  
  # Create a row-index column for fast assignment
  cell_data[, .row_idx := .I]
  
  # Columns to extract from cell_data for the neighbor lookup
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  
  # Process one year at a time to control peak memory
  for (yr in years) {
    
    # Subset cell_data rows for this year (neighbor attribute source)
    # This is the "attribute table" for the neighbor cells
    yr_attrs <- cell_data[year == yr, ..neighbor_cols]
    setnames(yr_attrs, "id", "neighbor_id")
    setkey(yr_attrs, neighbor_id)
    
    # Join edge_table with neighbor attributes for this year
    # Result: one row per (cell_id, neighbor_id) with neighbor's var values
    edges_yr <- merge(edge_table, yr_attrs, by = "neighbor_id", 
                      all.x = FALSE, allow.cartesian = FALSE)
    # edges_yr has columns: neighbor_id, cell_id, year, ntl, ec, ...
    # ~1.37M rows for this year
    
    # Aggregate by cell_id: compute max, min, mean for each variable
    agg_exprs <- list()
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_",  var_name)
      min_col  <- paste0("neighbor_min_",  var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      agg_exprs[[max_col]]  <- call("max",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[min_col]]  <- call("min",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[mean_col]] <- call("mean", as.name(var_name), na.rm = TRUE)
    }
    
    # Build and evaluate the aggregation in one grouped pass
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    yr_stats <- edges_yr[, eval(agg_call), by = cell_id]
    
    # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen, but safe)
    for (col_name in names(yr_stats)) {
      if (col_name == "cell_id") next
      v <- yr_stats[[col_name]]
      v[is.infinite(v)] <- NA_real_
      set(yr_stats, j = col_name, value = v)
    }
    
    # Join aggregated stats back to cell_data for this year
    # Use the row index for direct assignment (fastest)
    yr_stats[, year := yr]
    setkey(yr_stats, cell_id, year)
    
    # Get row indices in cell_data for this year
    idx_dt <- cell_data[year == yr, .(cell_id = id, year, .row_idx)]
    setkey(idx_dt, cell_id, year)
    
    matched <- merge(idx_dt, yr_stats, by = c("cell_id", "year"), 
                     all.x = TRUE)
    
    # Assign each new column directly by row index
    result_cols <- setdiff(names(yr_stats), c("cell_id", "year"))
    for (rc in result_cols) {
      set(cell_data, i = matched$.row_idx, j = rc, value = matched[[rc]])
    }
    
    # Free memory
    rm(yr_attrs, edges_yr, yr_stats, idx_dt, matched)
  }
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  return(cell_data)
}

# ===========================================================================
# EXECUTION
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  edge_table, 
  neighbor_source_vars
)

# --- Prediction with the existing trained Random Forest (unchanged) ---
# The model object (`rf_model`) and predict call remain exactly as before.
# cell_data now has the same 15 neighbor columns with identical values.
# Example:
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor topology resolution** | Per cell-year (6.46M R-level iterations) | Per cell, once (344K cells â†’ 1.37M edge rows) |
| **Attribute lookup mechanism** | Named character vector subsetting â€” O(n) per lookup | `data.table` keyed equi-join â€” O(log n) per row, vectorized in C |
| **Stat computation** | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R iterations | Grouped `data.table` aggregation â€” vectorized C, one pass per year |
| **Peak memory** | ~6.46M-element named character vector + 6.46M-element list of integer vectors | ~1.37M-row edge table + ~1.37M-row per-year join result (processed and freed per year) |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** on the same laptop |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` over the same neighbor sets, same column names |
| **RF model** | Trained | **Untouched â€” not retrained** |
END_RESPONSE>>>
