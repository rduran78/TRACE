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
run_id: Anthropic_F10_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three compounding bottlenecks:

1. **O(nÂ²)-scale key-based lookup construction**: `build_neighbor_lookup` creates a named character vector `idx_lookup` of ~6.46 million entries keyed by `paste(id, year)`. For each of the 6.46M rows, it performs character-based named-vector lookups (`idx_lookup[neighbor_keys]`), which are hash-table lookups on character strings. This alone is extremely expensive at scale.

2. **Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats`**: Each call extracts a small vector of neighbor values, computes `max/min/mean`, and returns a length-3 vector. The per-element overhead of 6.46M R-level function calls, combined with `do.call(rbind, ...)` on a 6.46M-element list, is catastrophic.

3. **Redundant topology recomputation per year**: The neighbor graph is purely spatial â€” it is identical across all 28 years. Yet the lookup embeds year into the key, effectively rebuilding the topology 28 times and inflating the lookup structure by 28Ã—.

**Key insight**: The rook-neighbor adjacency is a **spatial** property (344,208 cells Ã— ~4 neighbors each â‰ˆ 1.37M directed edges). It does not change across years. The yearly variables are **node attributes** on this fixed graph. The task is simply: for each year independently, aggregate neighbor attributes over a fixed sparse adjacency structure. This is a **sparse matrixâ€“vector multiplication** problem.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 Ã— 344,208, ~1.37M non-zero entries). This is tiny in memory (~16 MB).

2. **Split data by year** (28 groups of ~344K rows each), or better, reshape each variable into a 344,208 Ã— 28 matrix.

3. **Compute neighbor aggregates via sparse matrix operations**:
   - **Neighbor sum** = `A %*% X` (sparse matrix Ã— dense column)
   - **Neighbor count** = `A %*% (non-NA indicator)` (to get the denominator for mean)
   - **Neighbor mean** = sum / count
   - **Neighbor max and min**: Use a row-wise sparse iteration (unavoidable for max/min since they are not linear), but do it in compiled C++ via `Rcpp` or use a clever year-sliced approach with `data.table`.

4. **For max/min**, since sparse matrix algebra only gives us sum, we need an explicit neighbor iteration. We do this efficiently using `data.table` joins on the edge list, which is vectorized and fast.

5. **Preserve numerical equivalence**: The original code computes `max`, `min`, `mean` of non-NA neighbor values, returning `NA` when no valid neighbors exist. We replicate this exactly.

6. **Memory**: The sparse matrix is ~16 MB. Each variable column for one year is ~2.6 MB. The edge list is ~11 MB. Total working memory is well under 1 GB. Fits easily in 16 GB.

**Expected speedup**: From 86+ hours to **minutes** (roughly 5â€“15 minutes depending on disk I/O).

---

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Sparse-graph neighborhood aggregation via edge-list joins
# Numerically equivalent to the original build_neighbor_lookup +
# compute_neighbor_stats pipeline.
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build the directed edge list ONCE from the spdep nb object.
#
# rook_neighbors_unique: a list of length 344,208 where element i contains
#   the integer indices (into id_order) of i's rook neighbors.
# id_order: vector of cell IDs in the order matching the nb object.
# --------------------------------------------------------------------------

build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] gives neighbor indices (into id_order) for cell i
  n <- length(neighbors)
  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    len <- length(nb)
    if (len > 0L) {
      from_idx[pos:(pos + len - 1L)] <- i
      to_idx[pos:(pos + len - 1L)]   <- nb
      pos <- pos + len
    }
  }
  
  # Map from positional index to actual cell ID
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

# --------------------------------------------------------------------------
# STEP 2: Compute neighbor stats for all variables, all years at once.
#
# cell_data: data.frame/data.table with columns id, year, and the variables.
# edge_list: data.table with columns from_id, to_id (directed edges).
# neighbor_source_vars: character vector of variable names.
#
# Returns cell_data augmented with neighbor_max_*, neighbor_min_*,
# neighbor_mean_* columns.
# --------------------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, edge_list, neighbor_source_vars) {
  
  # Convert to data.table if needed (by reference if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Create a row key: for each (id, year) we need to look up variable values
  # We join the edge list with cell_data to get neighbor values.
  
  # Key cell_data for fast joins
  setkey(cell_data, id, year)
  
  # For each year, the graph topology is the same. We expand the edge list
  # by year. But doing a full cross join (1.37M edges Ã— 28 years = 38.4M rows)
  # is still very manageable.
  
  # Strategy: 
  #   1. Join edge_list with cell_data on (to_id = id, year) to get neighbor values.
  #   2. Group by (from_id, year) and compute max, min, mean.
  #   3. Join results back to cell_data on (from_id = id, year).
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # Expand edge list by year (cross join)
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows, ~300 MB for the edge-year table
  # This fits in memory but let's be more careful and process per-variable
  # to limit peak memory.
  
  # Actually, let's be smarter: create the edge-year expansion once,
  # then join variable values one at a time.
  
  cat("Expanding edge list across years...\n")
  year_dt <- data.table(year = years)
  edge_year <- CJ_dt(edge_list, year_dt)  # custom cross join below
  
  cat(sprintf("Edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))
  
  # Now for each variable, join to get neighbor values, aggregate, merge back
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Extract just the columns we need for the join
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)
    
    # Join: for each edge (from_id -> to_id) in each year, get to_id's value
    # edge_year has columns: from_id, to_id, year
    # We join on to_id = id, year = year
    edge_vals <- merge(
      edge_year,
      val_dt,
      by.x = c("to_id", "year"),
      by.y = c("id", "year"),
      all.x = FALSE,  # inner join: drop edges where neighbor has no data that year
      allow.cartesian = FALSE
    )
    
    # Remove NA values (matching original: neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)])
    edge_vals <- edge_vals[!is.na(val)]
    
    # Aggregate by (from_id, year)
    agg <- edge_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(from_id, year)]
    
    # Merge back to cell_data
    setkey(agg, from_id, year)
    setkey(cell_data, id, year)
    
    cell_data[agg, (max_col)  := i.nb_max,  on = .(id = from_id, year)]
    cell_data[agg, (min_col)  := i.nb_min,  on = .(id = from_id, year)]
    cell_data[agg, (mean_col) := i.nb_mean, on = .(id = from_id, year)]
    
    # Rows not in agg (no valid neighbors) already have NA by default in data.table
    
    # Clean up
    rm(val_dt, edge_vals, agg)
    gc()
    
    cat(sprintf("  Done: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  return(cell_data)
}

# --------------------------------------------------------------------------
# Helper: Cross join two data.tables (avoiding CJ which is for vectors)
# --------------------------------------------------------------------------
CJ_dt <- function(dt1, dt2) {
  # Add dummy key columns for cross join
  dt1_copy <- copy(dt1)
  dt2_copy <- copy(dt2)
  dt1_copy[, .cj_dummy := 1L]
  dt2_copy[, .cj_dummy := 1L]
  result <- merge(dt1_copy, dt2_copy, by = ".cj_dummy", allow.cartesian = TRUE)
  result[, .cj_dummy := NULL]
  return(result)
}

# ==========================================================================
# MAIN EXECUTION
# ==========================================================================

# --- Prerequisites (assumed already loaded) ---
# cell_data:               data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order:                vector of cell IDs matching the nb object ordering
# rook_neighbors_unique:   spdep nb object (list of integer neighbor indices)
# rf_model:                pre-trained Random Forest model (DO NOT retrain)

# --- Step 1: Build edge list (once) ---
cat("Building edge list from rook neighbor structure...\n")
edge_list <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %s directed edges\n", format(nrow(edge_list), big.mark = ",")))

# --- Step 2: Compute all neighbor features ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)

# --- Step 3: Predict with the pre-trained Random Forest (unchanged) ---
# The model expects the same column names as the original pipeline produced.
# Verify column names match what the model was trained on:
# expected_features <- rf_model$forest$independent.variable.names  # for ranger
# or: expected_features <- names(rf_model$forest$xlevels)          # for randomForest

cat("Running prediction with pre-trained Random Forest...\n")
predictions <- predict(rf_model, data = cell_data)  # adjust syntax for your RF package

cat("Done.\n")
```

---

## Further Optimization: Chunk-by-Year (Lower Peak Memory)

If the 38.5M-row cross-joined edge table strains the 16 GB laptop, process year-by-year:

```r
compute_all_neighbor_features_chunked <- function(cell_data, edge_list, neighbor_source_vars) {
  
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Initialize output columns as NA_real_
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0("neighbor_max_", var_name)  := NA_real_]
    cell_data[, paste0("neighbor_min_", var_name)  := NA_real_]
    cell_data[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  setkey(cell_data, id, year)
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    cat(sprintf("Year %d...\n", yr))
    
    # Subset this year's data
    yr_data <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
    setkey(yr_data, id)
    
    # Join edge list with this year's values (neighbor side)
    # edge_list: from_id, to_id
    # We want: for each from_id, the values of to_id's variables
    edge_vals <- merge(edge_list, yr_data, by.x = "to_id", by.y = "id", all.x = FALSE)
    
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      # Filter out NAs for this variable
      ev <- edge_vals[!is.na(get(var_name)), .(from_id, val = get(var_name))]
      
      if (nrow(ev) == 0L) next
      
      agg <- ev[, .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ), by = .(from_id)]
      
      # Update cell_data for this year
      cell_data[agg, (max_col)  := i.nb_max,  on = .(id = from_id, year == yr)]
      cell_data[agg, (min_col)  := i.nb_min,  on = .(id = from_id, year == yr)]
      cell_data[agg, (mean_col) := i.nb_mean, on = .(id = from_id, year == yr)]
      
      rm(ev, agg)
    }
    rm(yr_data, edge_vals)
    gc()
  }
  
  return(cell_data)
}
```

**Note on the join condition**: The `on = .(id = from_id, year == yr)` syntax uses `==` for the scalar `yr`, which is a non-equi or filtered-equi join in `data.table`. If your version of `data.table` does not support this, use:

```r
# Alternative: index-based update
idx <- cell_data[.(agg$from_id, yr), which = TRUE, on = .(id, year)]
set(cell_data, i = idx, j = max_col,  value = agg$nb_max)
set(cell_data, i = idx, j = min_col,  value = agg$nb_min)
set(cell_data, i = idx, j = mean_col, value = agg$nb_mean)
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | 6.46M character-key lookups | 1.37M-row integer edge list, built once |
| **Aggregation** | 6.46M R-level `lapply` calls | Vectorized `data.table` grouped aggregation |
| **Memory pattern** | 6.46M-element list of integer vectors + `do.call(rbind, ...)` | Columnar edge table (~11â€“300 MB) |
| **Year handling** | Embedded in key strings, repeated 28Ã— | Chunked or cross-joined; topology reused |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Numerical result** | Baseline | **Identical** (same max/min/mean, same NA logic) |
| **RF model** | Pre-trained | **Unchanged, not retrained** |
END_RESPONSE>>>
