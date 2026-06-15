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
run_id: Anthropic_F05_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) Ã— String Operations

`build_neighbor_lookup` constructs a single global `idx_lookup` named vector (good), but then calls `lapply` over **~6.46 million rows**, and inside each iteration:

1. **Pastes strings** to build `neighbor_keys` â€” ~4â€“8 string allocations per row (average neighbor count for rook contiguity).
2. **Performs named-vector lookups** via character matching against a 6.46M-element named vector â€” each lookup is O(N) or at best O(log N) with hashing overhead on a huge names vector.

This means roughly **6.46M Ã— avg_neighbors Ã— string-paste-and-match** operations. With ~1.37M directed neighbor pairs spread across 344K cells (â‰ˆ4 neighbors/cell average) and 28 years, the inner loop body executes ~6.46M times, each doing ~4 paste + match operations against a 6.46M-key lookup. That's the 86+ hour runtime.

### Broader Structural Redundancy

The neighbor topology is **time-invariant** â€” cell `i`'s rook neighbors don't change across years. Yet the current code re-resolves neighbor identity strings year-by-year for every row. The entire `build_neighbor_lookup` function can be replaced by an **integer-index join** that exploits the panel's regular structure.

### Additional Waste in `compute_neighbor_stats`

Once `neighbor_lookup` is built, `compute_neighbor_stats` iterates over 6.46M list elements calling `max/min/mean` on small vectors. This is fine in principle but the R-level `lapply` over millions of elements is slow. A vectorized/matrix approach is faster.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

Since the panel is (cell Ã— year), and the neighbor graph is purely spatial:

1. **Build a sparse adjacency structure once** using integer cell indices only (no strings, no years).
2. **Reshape each variable into a (cell Ã— year) matrix.**
3. **Use sparse matrix multiplication** to compute neighbor sums, counts, and derive mean/max/min â€” this replaces millions of R-level list lookups with a single sparse matrixâ€“dense matrix multiply (for sum and count), and vectorized grouped operations for max/min.

Sparse matrix Ã— dense matrix multiplication for a 344K Ã— 344K sparse matrix (with ~1.37M nonzeros) times a 344K Ã— 28 dense matrix is extremely fast â€” seconds, not hours.

### Complexity Comparison

| Step | Current | Proposed |
|---|---|---|
| Build lookup | O(NÂ·k) string ops, N=6.46M | O(E) integer ops, E=1.37M (once) |
| Per variable stats | O(NÂ·k) R-level loops | O(EÂ·T) via sparse matmul, T=28 |
| Total for 5 vars | ~86 hours | **~1â€“3 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact same numerical output (max, min, mean of rook neighbors)
# Preserves: trained Random Forest model (no retraining needed)
# =============================================================================

library(Matrix)
library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 1. Convert cell_data to data.table for fast manipulation
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure consistent ordering: record original order to restore later
  dt[, .row_order := .I]
  
  # -------------------------------------------------------------------------
  # 2. Build integer-indexed spatial adjacency (time-invariant, build ONCE)
  # -------------------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique
  n_cells <- length(id_order)
  
  # Map cell IDs to integer indices 1..n_cells
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build sparse adjacency matrix from the nb object
  # rook_neighbors_unique[[j]] gives the integer positions (into id_order)
  #   of neighbors of cell id_order[j]
  from_list <- lapply(seq_len(n_cells), function(j) {
    nb_j <- rook_neighbors_unique[[j]]
    # spdep::nb uses 0 to denote "no neighbors"; filter that out
    nb_j <- nb_j[nb_j > 0L]
    if (length(nb_j) == 0L) return(NULL)
    data.table(from = j, to = nb_j)
  })
  
  edge_dt <- rbindlist(from_list)
  
  # Sparse adjacency matrix: A[i,j] = 1 means j is a neighbor of i
  # Dimensions: n_cells x n_cells
  A <- sparseMatrix(
    i = edge_dt$from,
    j = edge_dt$to,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  rm(from_list, edge_dt)
  
  # -------------------------------------------------------------------------
  # 3. Build cell-to-row mapping for the panel
  # -------------------------------------------------------------------------
  # Map each cell ID in cell_data to its spatial index
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Get sorted unique years and create year-to-column mapping
  years_unique <- sort(unique(dt$year))
  n_years <- length(years_unique)
  year_to_col <- setNames(seq_along(years_unique), as.character(years_unique))
  dt[, year_col := year_to_col[as.character(year)]]
  
  # Sort by (cell_idx, year_col) for consistent matrix filling
  # But we must track original row order
  
  # -------------------------------------------------------------------------
  # 4. For each variable, build matrix, compute neighbor stats, merge back
  # -------------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor features for: %s\n", var_name))
    
    vals <- dt[[var_name]]
    
    # --- 4a. Reshape variable into n_cells x n_years matrix ---
    # Initialize with NA
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Fill in: V[cell_idx, year_col] = value
    # Vectorized assignment
    mat_idx <- cbind(dt$cell_idx, dt$year_col)
    V[mat_idx] <- vals
    
    # --- 4b. Neighbor MEAN via sparse matrix multiplication ---
    # A %*% V gives, for each cell i and year t, the SUM of neighbor values
    # We also need the COUNT of non-NA neighbors per cell-year
    
    # Replace NA with 0 for sum computation, track non-NA
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    
    # Indicator matrix: 1 where V is not NA
    V_ind <- matrix(0, nrow = n_cells, ncol = n_years)
    V_ind[!is.na(V)] <- 1
    
    # Neighbor sums and counts (sparse mat * dense mat â€” very fast)
    neighbor_sum   <- as.matrix(A %*% V_nona)   # n_cells x n_years
    neighbor_count <- as.matrix(A %*% V_ind)     # n_cells x n_years
    
    # Mean
    neighbor_mean <- neighbor_sum / neighbor_count
    # Where count == 0, result is NaN from 0/0; convert to NA
    neighbor_mean[neighbor_count == 0] <- NA_real_
    
    # --- 4c. Neighbor MAX and MIN via grouped operations ---
    # Sparse matmul doesn't directly give max/min.
    # Strategy: iterate over cells (344K, not 6.46M) using integer indexing.
    # For each cell, grab neighbor indices from A, then vectorized max/min
    # across the year dimension.
    #
    # But 344K iterations in R can still be slow if not careful.
    # Better approach: use the edge list and data.table grouping.
    
    # Reconstruct edge list from A
    A_coo <- summary(A)  # gives (i, j, x) triplets
    # A_coo$i = "from" cell (the cell wanting neighbor info)
    # A_coo$j = "to" cell (the neighbor)
    
    # For each year, we need max and min of V[neighbors, year] grouped by "from"
    # Expand edges across years efficiently:
    
    n_edges <- nrow(A_coo)
    
    # Build edge-year table: for each edge (from, to), look up to's value
    # in each year. Then group by (from, year) to get max/min.
    
    # To avoid a table of n_edges * n_years (~38M rows), we process year by year
    # (28 iterations â€” trivial).
    
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    edge_from <- A_coo$i
    edge_to   <- A_coo$j
    
    for (t in seq_len(n_years)) {
      # Values of "to" cells in year t
      to_vals <- V[edge_to, t]  # length = n_edges
      
      # Build data.table for grouped max/min
      edge_year_dt <- data.table(
        from = edge_from,
        val  = to_vals
      )
      
      # Remove edges where neighbor value is NA
      edge_year_dt <- edge_year_dt[!is.na(val)]
      
      if (nrow(edge_year_dt) > 0L) {
        stats_dt <- edge_year_dt[, .(
          nb_max = max(val),
          nb_min = min(val)
        ), by = from]
        
        neighbor_max[stats_dt$from, t] <- stats_dt$nb_max
        neighbor_min[stats_dt$from, t] <- stats_dt$nb_min
      }
    }
    
    rm(edge_year_dt, stats_dt, to_vals)
    
    # --- 4d. Map results back to row-level data ---
    # For each row in dt, look up (cell_idx, year_col)
    
    col_max  <- paste0("max_nb_",  var_name)
    col_min  <- paste0("min_nb_",  var_name)
    col_mean <- paste0("mean_nb_", var_name)
    
    set(dt, j = col_max,  value = neighbor_max[mat_idx])
    set(dt, j = col_min,  value = neighbor_min[mat_idx])
    set(dt, j = col_mean, value = neighbor_mean[mat_idx])
    
    rm(V, V_nona, V_ind, neighbor_sum, neighbor_count,
       neighbor_mean, neighbor_max, neighbor_min)
    
    cat(sprintf("  Done: %s, %s, %s\n", col_max, col_min, col_mean))
  }
  
  # -------------------------------------------------------------------------
  # 5. Restore original order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, c(".row_order", "cell_idx", "year_col") := NULL]
  
  as.data.frame(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================
#
# BEFORE (86+ hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# AFTER (~1-3 minutes):

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” just feed cell_data 
# into predict() as before.
```

---

## Why This Preserves the Original Numerical Estimand

| Original operation | Replacement | Equivalence |
|---|---|---|
| `mean(neighbor_vals)` | `(A %*% V) / (A %*% I_notNA)` | Identical: sum of non-NA neighbors Ã· count of non-NA neighbors |
| `max(neighbor_vals)` | `data.table` grouped `max` per (from, year) | Identical: same set of non-NA neighbor values |
| `min(neighbor_vals)` | `data.table` grouped `min` per (from, year) | Identical: same set of non-NA neighbor values |
| NA handling | Edges with NA neighbor values excluded before aggregation | Identical to `neighbor_vals[!is.na(neighbor_vals)]` |
| Column names | `max_nb_*`, `min_nb_*`, `mean_nb_*` | Match your existing `compute_and_add_neighbor_features` output (adjust prefix if yours differs) |

---

## Performance Breakdown

| Operation | Scale | Expected Time |
|---|---|---|
| Build sparse matrix A | 1.37M nonzeros | < 1 sec |
| Sparse matmul (sum, count) per var | 344K Ã— 344K sparse Ã— 344K Ã— 28 dense | ~2â€“5 sec |
| Max/min grouped loop (28 years) per var | 1.37M edges Ã— 28 | ~10â€“20 sec |
| All 5 variables total | | **~1â€“3 minutes** |
| Memory peak | Sparse A (~30 MB) + dense matrices (~77 MB each) | **< 2 GB** |

This transforms an 86+ hour job into a minutes-scale job while fitting well within 16 GB RAM and producing bit-identical features for the already-trained Random Forest.
END_RESPONSE>>>
