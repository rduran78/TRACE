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
run_id: Anthropic_F10_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector â€” this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` removal dominates.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt monolithically across all cell-years, entangling spatial topology with temporal indexing. This prevents vectorized sparse-matrix operations.

**Key insight:** Since the neighbor graph is fixed across years, the aggregation `max/min/mean of neighbor attributes` for a given year is simply a **sparse matrix operation** on the 344,208-node graph applied independently to each year's column vector. This can be expressed as sparse matrixâ€“vector products and sparse-structure iterations via `Matrix` package or `data.table` grouped operations â€” reducing 6.46M list traversals to 28 sparse-matrix operations per variable.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 Ã— 344,208, ~1.37M nonzeros). This is tiny in memory (~20 MB as a `dgCMatrix`).

2. **For each year and each variable**, extract the 344,208-length attribute vector, then:
   - **Mean**: Sparse matrixâ€“vector multiply `A %*% x` divided by row-degree vector â†’ exact mean.
   - **Min/Max**: Use the sparse structure to do grouped min/max via compiled C++ code (`dgCMatrix` column pointers) or `data.table` edge-list aggregation.

3. **Avoid all string-key lookups.** Map cell IDs to integer indices once; use integer indexing throughout.

4. **Process year-by-year** to keep memory bounded (one 344K vector at a time rather than 6.46M).

5. **Use `data.table` for the edge-list aggregation** of min/max (vectorized, compiled C internals), and sparse matrix multiply for mean.

This reduces complexity from O(6.46M Ã— average_string_lookup_cost) to O(28 Ã— 1.37M) per variable â€” roughly a **200â€“500Ã— speedup**, bringing runtime from 86+ hours to **minutes**.

## Optimized R Code

```r
# =============================================================================
# Optimized Neighborhood Aggregation for Spatial Panel Data
# Preserves numerical equivalence with original compute_neighbor_stats output
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build sparse graph topology ONCE ----

build_sparse_graph <- function(id_order, rook_neighbors_unique) {
  # id_order: vector of cell IDs in the order matching the nb object
  # rook_neighbors_unique: spdep nb object (list of integer neighbor index vectors)
  
  n <- length(id_order)
  stopifnot(length(rook_neighbors_unique) == n)
  
  # Build COO edge list (from_idx -> to_idx in 1..n space of id_order)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep.int(i, length(nb_i))
      to_list[[i]]   <- nb_i
    }
  }
  
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)
  
  # Sparse adjacency matrix: A[i,j] = 1 means j is a neighbor of i
  # So row i contains the neighbors of node i
  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n, n),
    repr = "C"  # CSC, will convert to CSR via transpose trick
  )
  
  # Row degree vector (number of neighbors per node)
  degree <- diff(A@p)  # only valid for CSC if we transpose; let's compute directly
  # Actually for dgCMatrix (CSC), row sums:
  degree <- as.numeric(rowSums(A))
  
  # Edge list as data.table for min/max aggregation
  edge_dt <- data.table(
    from = from_idx,
    to   = to_idx
  )
  setkey(edge_dt, from)
  
  # Map from cell ID to positional index
  id_to_pos <- setNames(seq_len(n), as.character(id_order))
  
  list(
    A        = A,
    degree   = degree,
    edge_dt  = edge_dt,
    n        = n,
    id_order = id_order,
    id_to_pos = id_to_pos
  )
}


# ---- Step 2: Compute neighbor stats for one variable, all years ----

compute_neighbor_features_fast <- function(cell_data_dt, var_name, graph, years) {
  # cell_data_dt: data.table with columns id, year, and <var_name>
  # graph: output of build_sparse_graph
  # years: sorted unique years
  #
  # Returns: data.table with columns id, year, nb_max_<var>, nb_min_<var>, nb_mean_<var>
  
  A        <- graph$A
  degree   <- graph$degree
  edge_dt  <- graph$edge_dt
  n        <- graph$n
  id_order <- graph$id_order
  
  max_col_name  <- paste0("nb_max_", var_name)
  min_col_name  <- paste0("nb_min_", var_name)
  mean_col_name <- paste0("nb_mean_", var_name)
  
  # Pre-allocate output matrix: n_cells x n_years x 3 stats
  n_years <- length(years)
  out_max  <- matrix(NA_real_, nrow = n, ncol = n_years)
  out_min  <- matrix(NA_real_, nrow = n, ncol = n_years)
  out_mean <- matrix(NA_real_, nrow = n, ncol = n_years)
  
  # Index cell_data_dt by (id, year) for fast extraction
  # We need to extract the variable vector in id_order order for each year
  setkey(cell_data_dt, year, id)
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Extract this year's data, ordered by id_order
    year_slice <- cell_data_dt[.(yr), .(id, val = get(var_name)), nomatch = NULL]
    
    # Map to positional index
    pos <- graph$id_to_pos[as.character(year_slice$id)]
    
    # Build the attribute vector in graph-node order
    x <- rep(NA_real_, n)
    x[pos] <- year_slice$val
    
    # ---- MEAN via sparse matrix multiply ----
    # A %*% x gives sum of neighbor values for each node
    # Divide by degree to get mean
    # Nodes with degree 0 or all-NA neighbors -> NA
    
    # Replace NA with 0 for matrix multiply, but track NA count
    x_zero <- x
    x_zero[is.na(x_zero)] <- 0
    x_notna <- as.numeric(!is.na(x))
    
    neighbor_sum   <- as.numeric(A %*% x_zero)
    neighbor_count <- as.numeric(A %*% x_notna)
    
    yr_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    # Nodes with no neighbors at all (degree == 0) -> NA
    yr_mean[degree == 0] <- NA_real_
    
    out_mean[, yi] <- yr_mean
    
    # ---- MIN and MAX via edge-list aggregation ----
    # Attach neighbor values to edge list and aggregate
    # edge_dt$to are the neighbor indices; x[to] is the neighbor's value
    
    neighbor_vals <- x[edge_dt$to]
    
    # Use data.table for grouped min/max (compiled C, very fast)
    agg_dt <- data.table(
      from = edge_dt$from,
      val  = neighbor_vals
    )
    
    # Remove NA neighbor values before aggregation
    agg_dt <- agg_dt[!is.na(val)]
    
    if (nrow(agg_dt) > 0L) {
      stats <- agg_dt[, .(nb_max = max(val), nb_min = min(val)), by = from]
      out_max[stats$from, yi] <- stats$nb_max
      out_min[stats$from, yi] <- stats$nb_min
    }
  }
  
  # Reshape to long format matching cell_data structure
  result_list <- vector("list", n_years)
  for (yi in seq_along(years)) {
    result_list[[yi]] <- data.table(
      id   = id_order,
      year = years[yi],
      V1   = out_max[, yi],
      V2   = out_min[, yi],
      V3   = out_mean[, yi]
    )
  }
  result <- rbindlist(result_list, use.names = FALSE)
  setnames(result, c("V1", "V2", "V3"), c(max_col_name, min_col_name, mean_col_name))
  
  return(result)
}


# ---- Step 3: Main pipeline ----

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  # cell_data: data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
  # id_order: vector of cell IDs matching nb object order

  # rook_neighbors_unique: spdep nb object
  # rf_model: pre-trained Random Forest model (not retrained)
  
  cat("Converting to data.table...\n")
  cell_data_dt <- as.data.table(cell_data)
  
  cat("Building sparse graph topology...\n")
  graph <- build_sparse_graph(id_order, rook_neighbors_unique)
  
  years <- sort(unique(cell_data_dt$year))
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  cat("Computing neighbor features for", length(neighbor_source_vars), "variables x",
      length(years), "years...\n")
  
  setkey(cell_data_dt, id, year)
  
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    
    feat_dt <- compute_neighbor_features_fast(cell_data_dt, var_name, graph, years)
    
    # Merge back into cell_data_dt
    setkey(feat_dt, id, year)
    
    # Get the new column names
    new_cols <- setdiff(names(feat_dt), c("id", "year"))
    
    # Join
    cell_data_dt <- feat_dt[cell_data_dt, on = .(id, year)]
  }
  
  cat("Generating predictions with pre-trained Random Forest...\n")
  # The RF model is used as-is; no retraining
  predictions <- predict(rf_model, newdata = cell_data_dt)
  cell_data_dt[, prediction := predictions]
  
  cat("Done.\n")
  return(cell_data_dt)
}


# ---- Usage ----
# result <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique, rf_model)
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M string-key lookups against named vector | Built once as sparse matrix + edge `data.table` (~1.37M edges) |
| **Mean computation** | R-level loop over 6.46M list elements | 28 sparse matrixâ€“vector multiplies (BLAS-backed) |
| **Min/Max computation** | R-level loop with per-element subsetting | 28 `data.table` grouped aggregations on 1.37M-row edge list |
| **Memory** | ~6.46M-element list of integer vectors + 6.46M-element named character vector | ~20 MB sparse matrix + ~22 MB edge `data.table` + year-slice vectors |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Numerical equivalence** | Baseline | âœ… Identical: same `max`, `min`, `mean` over same non-NA neighbor values |
| **RF model** | â€” | âœ… Preserved, predict-only |

### Why numerical equivalence holds

- **Mean**: `A %*% x_zero` with `x_zero[is.na] = 0` computes `sum(non-NA neighbor values)` because the NA positions contribute 0. Dividing by `A %*% x_notna` (count of non-NA neighbors) gives exactly `mean(neighbor_vals[!is.na(neighbor_vals)])` â€” identical to the original.
- **Min/Max**: The edge-list aggregation filters `!is.na(val)` then computes `max`/`min` grouped by source node â€” identical to the original's `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]; c(max(...), min(...))`.
- Nodes with zero neighbors or all-NA neighbors produce `NA` in both implementations.
END_RESPONSE>>>
