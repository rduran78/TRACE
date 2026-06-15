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
run_id: Anthropic_F10_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting variable values and computing `max/min/mean` per node. This is repeated 5 times (once per source variable), totaling ~32.3 million R-level function calls.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt as if each cell-year has a unique neighbor set. The 344,208 cells have ~1.37M directed edges â€” this topology is reused identically across all 28 years, yet the current code embeds year into the lookup, inflating it 28Ã—.

**Root cause:** The implementation treats the problem as a 6.46M-node graph instead of a 344,208-node graph with 28 attribute snapshots. The string-key approach and per-row R-level iteration make it O(rows Ã— neighbors) with enormous constant factors.

## Optimization Strategy

1. **Separate topology from time.** Build the sparse adjacency structure once over 344,208 cells. Represent it as a CSR (Compressed Sparse Row) structure using integer vectors â€” no strings, no named lookups.

2. **Vectorize aggregation using sparse matrix multiplication.** Construct a sparse `dgCMatrix` (from the `Matrix` package) where each row `i` has non-zero entries in columns corresponding to cell `i`'s rook neighbors, with values `1/degree(i)` for mean, and binary `1` for max/min. For **mean**, a single sparse matrixâ€“vector multiply (`A %*% x`) gives all neighbor means in one shot. For **max** and **min**, use grouped operations via `data.table` on the edge list.

3. **Process year-by-year in a loop over 28 years** (not 6.46M rows), applying the sparse aggregation to each year's column vector of length 344,208.

4. **Use `data.table` for the grouped max/min** over the edge list, which is highly optimized in C.

This reduces the problem from ~6.46M R-level iterations to 28 iterations Ã— 5 variables Ã— 3 stats = 420 vectorized operations, each over ~344K cells.

**Expected speedup:** From 86+ hours to roughly **2â€“10 minutes**.

## Optimized R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build CSR-style topology ONCE from the spdep nb object
# ==============================================================================
# rook_neighbors_unique: list of length 344,208; each element is an integer
#   vector of neighbor indices (1-based, referencing positions in id_order).
# id_order: integer vector of length 344,208 giving cell IDs in the order
#   matching rook_neighbors_unique.

build_sparse_topology <- function(id_order, nb_obj) {
  n <- length(id_order)
  stopifnot(length(nb_obj) == n)
  
  # Build edge list: from -> to (in terms of positional index 1..n)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Degree of each node (number of neighbors)
  degree <- tabulate(from, nbins = n)
  
  # Sparse matrix for MEAN: entry (i, j) = 1/degree(i) if j is neighbor of i
  # So row i sums to 1.0 (or 0 if no neighbors).
  weights_mean <- 1.0 / degree[from]
  weights_mean[!is.finite(weights_mean)] <- 0  # handle degree-0 nodes
  
  A_mean <- sparseMatrix(
    i = from, j = to, x = weights_mean,
    dims = c(n, n), repr = "C"  # CSC but will transpose if needed
  )
  
  # Edge data.table for grouped max/min
  edge_dt <- data.table(from = from, to = to)
  setkey(edge_dt, from)
  
  # Map cell ID -> positional index
  id_to_pos <- setNames(seq_len(n), as.character(id_order))
  
  list(
    n        = n,
    id_order = id_order,
    id_to_pos = id_to_pos,
    A_mean   = A_mean,
    edge_dt  = edge_dt,
    degree   = degree
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for one variable across all cell-years
# ==============================================================================
# cell_dt: data.table with columns id, year, and the variable columns.
#          Must be keyed or orderable by (id, year).
# topo: output of build_sparse_topology
# var_name: character, name of the variable

compute_neighbor_features <- function(cell_dt, topo, var_name) {
  n        <- topo$n
  id_order <- topo$id_order
  A_mean   <- topo$A_mean
  edge_dt  <- topo$edge_dt
  id_to_pos <- topo$id_to_pos
  
  years <- sort(unique(cell_dt$year))
  
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # We need a fast way to go from (id, year) -> row index in cell_dt.
  # Create a positional index: for each year, map pos (1..n) -> row in cell_dt.
  # Ensure cell_dt has a "pos" column = positional index of the cell ID.
  
  # Add positional index if not present
  if (!"pos_" %in% names(cell_dt)) {
    cell_dt[, pos_ := id_to_pos[as.character(id)]]
  }
  
  for (yr in years) {
    # Extract rows for this year
    yr_idx <- which(cell_dt$year == yr)
    
    # Build a vector of length n: vals[pos] = variable value for that cell in yr
    # Some cells may be missing for some years; they stay NA.
    yr_sub <- cell_dt[yr_idx, .(pos_, val = get(var_name))]
    
    vals <- rep(NA_real_, n)
    vals[yr_sub$pos_] <- yr_sub$val
    
    # --- MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for multiplication, but adjust for missing neighbors.
    vals_zero <- vals
    vals_zero[is.na(vals_zero)] <- 0
    
    # We need a corrected mean: only average over non-NA neighbors.
    # Indicator of non-NA
    ind <- as.double(!is.na(vals))
    
    # Sum of neighbor values (treating NA as 0)
    neighbor_sum <- as.numeric(A_mean %*% vals_zero) * topo$degree
    # Count of non-NA neighbors
    neighbor_count <- as.numeric(A_mean %*% ind) * topo$degree
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN via data.table grouped operations ---
    # Look up neighbor values
    neighbor_vals_vec <- vals[edge_dt$to]
    
    # Grouped max and min
    agg_dt <- data.table(
      from = edge_dt$from,
      nval = neighbor_vals_vec
    )
    # Remove NA neighbor values before aggregation
    agg_dt <- agg_dt[!is.na(nval)]
    
    if (nrow(agg_dt) > 0) {
      agg <- agg_dt[, .(nmax = max(nval), nmin = min(nval)), by = from]
      
      neighbor_max <- rep(NA_real_, n)
      neighbor_min <- rep(NA_real_, n)
      neighbor_max[agg$from] <- agg$nmax
      neighbor_min[agg$from] <- agg$nmin
    } else {
      neighbor_max <- rep(NA_real_, n)
      neighbor_min <- rep(NA_real_, n)
    }
    
    # Write results back into cell_dt for this year's rows
    # Map from pos_ back to yr_idx
    pos_this_year <- cell_dt$pos_[yr_idx]
    
    set(cell_dt, i = yr_idx, j = max_col,  value = neighbor_max[pos_this_year])
    set(cell_dt, i = yr_idx, j = min_col,  value = neighbor_min[pos_this_year])
    set(cell_dt, i = yr_idx, j = mean_col, value = neighbor_mean[pos_this_year])
  }
  
  invisible(cell_dt)
}

# ==============================================================================
# STEP 3: Main pipeline
# ==============================================================================
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                  rf_model) {
  
  # Convert to data.table for performance (in-place, no copy if already DT)
  cell_dt <- as.data.table(cell_data)
  
  # Build topology once
  message("Building sparse graph topology...")
  topo <- build_sparse_topology(id_order, rook_neighbors_unique)
  
  # Add positional index once
  cell_dt[, pos_ := topo$id_to_pos[as.character(id)]]
  
  # Neighbor source variables

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor features for: %s", var_name))
    compute_neighbor_features(cell_dt, topo, var_name)
  }
  
  # Remove helper column
  cell_dt[, pos_ := NULL]
  
  # Apply pre-trained Random Forest (no retraining)
  message("Generating predictions with pre-trained Random Forest...")
  preds <- predict(rf_model, newdata = cell_dt)
  cell_dt[, prediction := preds]
  
  message("Done.")
  return(cell_dt)
}

# ==============================================================================
# Usage (assuming objects are already in the environment):
# ==============================================================================
# result <- run_neighbor_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

## Summary of Guarantees

| Requirement | How it's met |
|---|---|
| **Numerical equivalence** | Mean is computed as `sum(non-NA values) / count(non-NA values)`, identical to `mean(x[!is.na(x)])`. Max and min use `data.table` grouped aggregation with NAs removed, identical to `max(x, na.rm=TRUE)` / `min(x, na.rm=TRUE)`. |
| **RF model preserved** | `predict(rf_model, ...)` is called on the enriched dataset; the model object is never modified. |
| **Topology built once** | `build_sparse_topology` runs once; the sparse matrix `A_mean` and edge `data.table` are reused across all 28 years Ã— 5 variables. |
| **Memory safe for 16 GB** | Peak memory: ~344KÃ—344K sparse matrix has ~1.37M non-zeros (~33 MB). The edge `data.table` is ~1.37M rows (~22 MB). Year-sliced vectors are 344K doubles (~2.6 MB each). The full `cell_dt` at ~6.46M rows Ã— 125 columns â‰ˆ ~6.5 GB. Total well under 16 GB. |
| **Speed** | 28 years Ã— 5 variables = 140 iterations. Each iteration: one sparse mat-vec (~0.01s), one `data.table` grouped aggregation over ~1.37M edges (~0.05s). Total compute: **~2â€“10 minutes** vs. 86+ hours. |
END_RESPONSE>>>
