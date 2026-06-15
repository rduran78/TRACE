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
run_id: Anthropic_F06_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) effective cost
The function iterates over **~6.46 million rows** with `lapply`, performing per-row string pasting, named-vector lookups, and `NA` filtering. Named vector lookups via character keys (`idx_lookup[neighbor_keys]`) are hash-table operations, but doing ~6.46M of themâ€”each involving `paste()`, character coercion, and subsettingâ€”is extremely slow in interpreted R. The lookup is also **redundant across years**: every cell has the same neighbors in every year, so the neighbor *row indices* could be computed once per cell and then offset by year, rather than recomputed for every cell-year row.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows
For each of the 5 variables, `lapply` iterates over 6.46M entries, subsetting a numeric vector and computing `max`, `min`, `mean`. This is called 5 times. The R-level loop overhead on 6.46M iterations Ã— 5 variables â‰ˆ 32.3M interpreted iterations is enormous.

### Why raster focal/kernel operations are not directly applicable
Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. The panel data here is indexed by `(id, year)` with an irregular `spdep::nb` neighbor structure (rook contiguity on spatial polygons/cells that may have varying numbers of neighbors). Focal operations would require reshaping into a 3D raster (x, y, time), which may not preserve the exact neighbor topology from the `spdep::nb` object (boundary cells, irregular grids, missing cells). **To preserve the original numerical estimand exactly**, we must use the precomputed `rook_neighbors_unique` neighbor list. However, we can borrow the *spirit* of focal operations: vectorized sparse-matrix multiplication replaces the element-wise loop.

## Optimization Strategy

1. **Exploit the panel structure**: The data has `T=28` years and `N=344,208` cells. Sort data by `(id, year)` so that cell `i` occupies rows `(i-1)*T + 1` through `i*T`. Then neighbor row indices for any year can be computed by simple arithmeticâ€”no string hashing needed.

2. **Replace the lookup with a sparse matrix**: Build a single `NÃ—N` sparse adjacency matrix `W` from `rook_neighbors_unique`. To compute neighbor stats across all cell-years, expand `W` to a `(N*T) Ã— (N*T)` block-diagonal matrix (one block per year), or equivalently, operate year-by-year on the `NÃ—N` matrix.

3. **Vectorized sparse matrix operations for mean**: `W %*% x / W %*% 1` gives the neighbor mean in one shot. For max and min, use a grouped operation via the sparse matrix structure.

4. **Compute max/min via row-wise sparse iteration in C++ (Rcpp)** or via `data.table` grouped operations, avoiding 6.46M R-level loop iterations.

The approach below uses **`data.table`** + **`Matrix`** (sparse) for the mean, and an efficient **`data.table` join + grouped aggregation** for max/min. This avoids all `lapply` loops over millions of rows and should run in **minutes, not days**.

## Working R Code

```r
# ============================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Preserves the exact numerical estimand (max, min, mean of
# rook-neighbor values per cell-year) and the trained RF model.
# ============================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ----------------------------------------------------------
  # 0. Convert to data.table and ensure sorted by (id, year)
  # ----------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Create a contiguous integer cell index aligned with id_order
  # id_order[k] is the cell id for index k
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Verify all cells matched
  stopifnot(!anyNA(dt$cell_idx))
  
  # Sort by cell_idx, year for efficient processing
  setkey(dt, cell_idx, year)
  
  # Get unique sorted years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_int <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_int[as.character(year)]]
  
  N <- length(id_order)  # 344,208 cells
  
  # ----------------------------------------------------------
  # 1. Build edge list from spdep::nb object (once)
  # ----------------------------------------------------------
  # rook_neighbors_unique[[k]] gives integer vector of neighbor
  # indices (into id_order) for cell index k.
  # Build a data.table edge list: (from_idx, to_idx)
  
  edge_from <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  edge_to   <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0L for "no neighbors" in

  # some representations)
  valid <- edge_to > 0L
  edge_from <- edge_from[valid]
  edge_to   <- edge_to[valid]
  
  edges <- data.table(from_idx = edge_from, to_idx = edge_to)
  
  cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edges)))
  
  # ----------------------------------------------------------
  # 2. Build sparse adjacency matrix W (N x N) for mean calc
  # ----------------------------------------------------------
  W <- sparseMatrix(
    i = edges$from_idx,
    j = edges$to_idx,
    x = 1,
    dims = c(N, N)
  )
  
  # ----------------------------------------------------------
  # 3. For each variable, compute neighbor max, min, mean
  #    Strategy: process year-by-year using the N x N matrix
  #    - mean: sparse matrix multiply
  #    - max, min: data.table join + grouped aggregation
  # ----------------------------------------------------------
  
  # Pre-build a lookup: for each (cell_idx, year_idx), the row

  # index in dt. Since dt is keyed on (cell_idx, year), we can
  # build this efficiently.
  dt[, row_id := .I]
  
  # For the join approach, we need neighbor values.
  # Build a table: for each row in dt, find all neighbor rows.
  # This is: dt joined with edges on cell_idx == from_idx,
  # then joined back to dt on (to_idx, year_idx).
  
  # Step A: Create a compact (cell_idx, year_idx) -> row_id map
  cell_year_map <- dt[, .(cell_idx, year_idx, row_id)]
  setkey(cell_year_map, cell_idx, year_idx)
  
  # Step B: Expand edges Ã— years to get all (focal_row, neighbor_row) pairs
  # But 1.37M edges Ã— 28 years = ~38.5M rows â€” manageable in RAM
  
  cat("Building expanded neighbor-row mapping...\n")
  
  # For each edge (from_idx -> to_idx), for each year, look up
  # the row_id of from and to in dt.
  # Efficient approach: merge edges with cell_year_map twice.
  
  # First, get all (from_idx, year_idx, from_row_id)
  edges_expanded <- CJ(edge_id = seq_len(nrow(edges)),
                        year_idx = seq_len(n_years))
  edges_expanded[, from_idx := edges$from_idx[edge_id]]
  edges_expanded[, to_idx   := edges$to_idx[edge_id]]
  
  # Join to get focal row_id
  setkey(edges_expanded, from_idx, year_idx)
  edges_expanded <- cell_year_map[edges_expanded,
                                   .(from_idx, to_idx, year_idx,
                                     focal_row = row_id),
                                   on = .(cell_idx = from_idx,
                                          year_idx = year_idx)]
  
  # Join to get neighbor row_id
  setkey(edges_expanded, to_idx, year_idx)
  edges_expanded <- cell_year_map[edges_expanded,
                                   .(from_idx, to_idx, year_idx,
                                     focal_row,
                                     neighbor_row = row_id),
                                   on = .(cell_idx = to_idx,
                                          year_idx = year_idx)]
  
  # Drop rows where either focal or neighbor is missing
  edges_expanded <- edges_expanded[!is.na(focal_row) & !is.na(neighbor_row)]
  
  cat(sprintf("Expanded edge table: %d focal-neighbor-year pairs\n",
              nrow(edges_expanded)))
  
  # Now for each variable, extract neighbor values and aggregate
  setkey(edges_expanded, focal_row)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)
    
    # Extract the variable values for neighbor rows
    vals <- dt[[var_name]]
    edges_expanded[, nval := vals[neighbor_row]]
    
    # Grouped aggregation: max, min, mean by focal_row
    agg <- edges_expanded[!is.na(nval),
                           .(n_max  = max(nval),
                             n_min  = min(nval),
                             n_mean = mean(nval)),
                           by = focal_row]
    
    # Initialize columns with NA
    set(dt, j = max_col,  value = NA_real_)
    set(dt, j = min_col,  value = NA_real_)
    set(dt, j = mean_col, value = NA_real_)
    
    # Assign aggregated values
    set(dt, i = agg$focal_row, j = max_col,  value = agg$n_max)
    set(dt, i = agg$focal_row, j = min_col,  value = agg$n_min)
    set(dt, i = agg$focal_row, j = mean_col, value = agg$n_mean)
    
    cat(sprintf("  Done: %s â€” %d rows with neighbor stats\n",
                var_name, nrow(agg)))
  }
  
  # ----------------------------------------------------------
  # 4. Clean up helper columns and return
  # ----------------------------------------------------------
  dt[, c("cell_idx", "year_idx", "row_id") := NULL]
  edges_expanded[, nval := NULL]
  
  return(as.data.frame(dt))
}

# ============================================================
# USAGE
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density",
                           "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Alternative: Even More Memory-Efficient Year-by-Year Approach

If the ~38.5M-row `edges_expanded` table strains the 16 GB RAM budget (alongside the 6.46M-row dataset with ~110 columns), process one year at a time:

```r
optimize_neighbor_features_lowmem <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  library(data.table)
  
  dt <- as.data.table(cell_data)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_to_idx[as.character(id)]]
  setkey(dt, cell_idx, year)
  
  years <- sort(unique(dt$year))
  N <- length(id_order)
  
  # Build edge list once
  edge_from <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  edge_to   <- unlist(rook_neighbors_unique, use.names = FALSE)
  valid <- edge_to > 0L
  edges <- data.table(from_idx = edge_from[valid],
                      to_idx   = edge_to[valid])
  
  # Initialize output columns
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0("n_max_",  var_name), value = NA_real_)
    set(dt, j = paste0("n_min_",  var_name), value = NA_real_)
    set(dt, j = paste0("n_mean_", var_name), value = NA_real_)
  }
  
  dt[, row_id := .I]
  
  for (yr in years) {
    cat(sprintf("Year %d ...\n", yr))
    
    # Subset rows for this year
    yr_rows <- dt[year == yr, .(cell_idx, row_id)]
    setkey(yr_rows, cell_idx)
    
    # Map edges to row_ids for this year
    yr_edges <- edges[yr_rows, on = .(from_idx = cell_idx),
                      nomatch = NULL,
                      .(from_idx, to_idx, focal_row = row_id)]
    yr_edges <- yr_rows[yr_edges, on = .(cell_idx = to_idx),
                        nomatch = NULL,
                        .(focal_row, neighbor_row = row_id)]
    
    for (var_name in neighbor_source_vars) {
      vals <- dt[[var_name]]
      yr_edges[, nval := vals[neighbor_row]]
      
      agg <- yr_edges[!is.na(nval),
                       .(n_max = max(nval),
                         n_min = min(nval),
                         n_mean = mean(nval)),
                       by = focal_row]
      
      set(dt, i = agg$focal_row,
          j = paste0("n_max_", var_name),  value = agg$n_max)
      set(dt, i = agg$focal_row,
          j = paste0("n_min_", var_name),  value = agg$n_min)
      set(dt, i = agg$focal_row,
          j = paste0("n_mean_", var_name), value = agg$n_mean)
    }
  }
  
  dt[, c("cell_idx", "row_id") := NULL]
  return(as.data.frame(dt))
}
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste`/hash | Edge list built once from `nb` object (vectorized) |
| **Stats computation** | 6.46M `lapply` Ã— 5 vars = 32.3M R-loop iters | `data.table` grouped aggregation (C-level) |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Peak RAM** | Moderate (list of 6.46M vectors) | ~2â€“4 GB for edge table + data |
| **Numerical results** | Baseline | **Identical** (same max/min/mean over same neighbor sets) |
| **RF model** | Unchanged | **Unchanged** |

The key insight is replacing millions of interpreted R-level loop iterations with `data.table`'s C-optimized grouped joins and aggregations, while exploiting the panel structure (same spatial neighbors across all years) to avoid redundant string-based lookups.
END_RESPONSE>>>
