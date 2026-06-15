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
run_id: Anthropic_F05_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is built once, but then `paste()` is called **per row** inside the `lapply` over 6.46M rows. Each call constructs character keys and performs named-vector lookups (which are hash-table lookups on character strings). That's ~6.46M Ã— avg_neighbors string constructions and hash probes â€” billions of character operations.

### Broader Structural Problem
The **real** architectural issue is that the entire approach is row-wise and string-based when it should be **vectorized and integer-based**:

1. **String keys are unnecessary.** Since every cell appears in every year (balanced panel), the neighbor relationship is purely spatial. A cell's neighbors in year `t` are at deterministic integer offsets from that cell's row in year `t`. If data is sorted by `(year, id)`, then all cells within a year are in a contiguous block, and neighbor row indices can be computed by simple integer arithmetic.

2. **`lapply` over 6.46M rows is inherently slow in R.** The neighbor lookup can be replaced by a single vectorized join/index operation.

3. **`compute_neighbor_stats` re-traverses the full lookup per variable.** With 5 variables, that's 5 Ã— 6.46M list element accesses. A matrix-based approach can compute all variables in one pass or use vectorized column operations.

### Complexity Comparison

| Aspect | Current | Proposed |
|---|---|---|
| Key construction | O(N Ã— k) string pastes | Zero string work |
| Lookup method | Character hash probe | Integer offset arithmetic |
| Neighbor resolution | Per-row `lapply` | Fully vectorized |
| Stats computation | Per-row list traversal Ã— 5 vars | Vectorized matrix column ops |

Where N = 6.46M rows and k = average neighbor count (~4 for rook).

---

## Optimization Strategy

1. **Sort data by `(year, id)`** and build a simple integer position map.
2. **Convert the `nb` object to a flat edge list** (cell_i â†’ cell_j) once â€” ~1.37M directed edges.
3. **Expand edges across years by integer arithmetic**: if there are `C = 344,208` cells per year and years are in contiguous blocks, then cell `i`'s row in year-block `t` is at position `(t-1)*C + rank_of_i`. Neighbor rows are computed identically.
4. **Use vectorized grouped aggregation** (via `data.table`) on the edge list to compute max/min/mean for all neighbor source variables simultaneously.
5. **Join results back** to the main data.

This replaces billions of string operations and millions of R-level loop iterations with a single vectorized `data.table` grouped aggregation.

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars) {
  # -----------------------------------------------------------
  # 1. Convert to data.table, sort by (year, id), add row index
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, orig_row := .I]                 # preserve original order
  setkey(dt, year, id)
  dt[, row_idx := .I]                  # new row index in sorted order
  
  n_total <- nrow(dt)
  
  # -----------------------------------------------------------
  # 2. Build integer mapping: for each (id), its rank in the
  #    sorted-within-year block. Because the panel is balanced
  #    and we sorted by (year, id), every year-block has the
  #    same id ordering.
  # -----------------------------------------------------------
  unique_years <- sort(unique(dt$year))
  n_years      <- length(unique_years)
  
  # Grab the id order from the first year block
  first_block <- dt[year == unique_years[1]]
  id_order_sorted <- first_block$id          # ids in sorted order
  C <- length(id_order_sorted)               # 344,208
  
  stopifnot(C * n_years == n_total)          # verify balanced panel
  
  # Map from original id to its 1-based rank in the sorted block
  id_to_rank <- setNames(seq_len(C), as.character(id_order_sorted))
  
  # -----------------------------------------------------------
  # 3. Convert nb object to flat directed edge list (rank_i -> rank_j)
  #    rook_neighbors_unique is indexed by some id_order; we need

  #    to map it to our sorted ranks.
  # -----------------------------------------------------------
  # The nb object is a list of length C, where element [[k]] gives
  # the neighbor indices (into id_order) of id_order[k].
  # We need to figure out what id_order was used when nb was built.
  # We'll accept it as a parameter or reconstruct from the nb attr.
  #
  # IMPORTANT: The caller must pass the id_order that was used with
  # the nb object (same as in the original code).
  # We'll accept it as a parameter.
  
  # -- This function needs id_order from the original pipeline --
  # We'll handle this by making it a parameter (see wrapper below).
  
  NULL
}

# ================================================================
# MAIN FUNCTION â€” drop-in replacement
# ================================================================
build_all_neighbor_features <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  
  library(data.table)
  
  # ------------------------------------------------------------------
  # 1. Convert to data.table; sort by (year, id); record original order
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, orig_row := .I]
  setkey(dt, year, id)
  dt[, row_idx := .I]
  
  unique_years <- sort(unique(dt$year))
  n_years      <- length(unique_years)
  year_to_block <- setNames(seq_along(unique_years) - 1L, as.character(unique_years))
  
  # Sorted unique ids within any year block (panel is balanced)
  first_block     <- dt[year == unique_years[1]]
  id_sorted       <- first_block$id
  C               <- length(id_sorted)
  stopifnot(C * n_years == nrow(dt))
  
  # Map: id -> rank (1-based position in sorted block)
  id_to_rank <- setNames(seq_len(C), as.character(id_sorted))
  
  # ------------------------------------------------------------------
  # 2. Build flat edge list from nb object: (from_rank, to_rank)
  #    id_order[k] is the cell id for the k-th element of the nb list.
  #    neighbors[[k]] gives indices into id_order.
  # ------------------------------------------------------------------
  from_ranks <- integer(0)
  to_ranks   <- integer(0)
  
  # Pre-map id_order to ranks
  id_order_ranks <- id_to_rank[as.character(id_order)]
  
  # Vectorized construction of edge list
  edge_lengths <- lengths(rook_neighbors_unique)
  n_edges      <- sum(edge_lengths)
  
  from_nb_idx <- rep(seq_along(rook_neighbors_unique), times = edge_lengths)
  to_nb_idx   <- unlist(rook_neighbors_unique)
  
  # Remove 0-neighbor entries (spdep uses 0L for no-neighbor sentinel)
  valid <- to_nb_idx > 0L
  from_nb_idx <- from_nb_idx[valid]
  to_nb_idx   <- to_nb_idx[valid]
  
  from_ranks <- id_order_ranks[from_nb_idx]
  to_ranks   <- id_order_ranks[to_nb_idx]
  
  edges <- data.table(from_rank = as.integer(from_ranks),
                      to_rank   = as.integer(to_ranks))
  
  # Remove any edges where mapping failed
  edges <- edges[!is.na(from_rank) & !is.na(to_rank)]
  
  cat(sprintf("Edge list: %d directed edges\n", nrow(edges)))
  
  # ------------------------------------------------------------------
  # 3. Expand edges across all years by integer arithmetic.
  #    In the sorted dt, the row for (rank r, year-block b) is:
  #        row_idx = b * C + r
  #    where b = 0, 1, ..., n_years-1
  # ------------------------------------------------------------------
  # We'll do this year by year to control memory (28 years is fine).
  
  # Pre-extract the variable columns we need
  var_cols <- neighbor_source_vars
  
  # We'll accumulate results into pre-allocated matrices
  n_total <- nrow(dt)
  
  # Result columns: for each var, we need max, min, mean â†’ 3 columns
  result_names <- character(0)
  for (v in var_cols) {
    result_names <- c(result_names,
                      paste0("neighbor_max_", v),
                      paste0("neighbor_min_", v),
                      paste0("neighbor_mean_", v))
  }
  
  # Initialize result matrix
  res_mat <- matrix(NA_real_, nrow = n_total, ncol = length(result_names))
  colnames(res_mat) <- result_names
  
  # ------------------------------------------------------------------
  # 4. For each year, compute neighbor stats vectorized
  # ------------------------------------------------------------------
  cat("Computing neighbor features by year...\n")
  
  for (yr in unique_years) {
    b <- year_to_block[as.character(yr)]
    offset <- as.integer(b) * C
    
    # Row indices in dt for this year's "from" and "to" cells
    from_rows <- offset + edges$from_rank
    to_rows   <- offset + edges$to_rank
    
    # Build a data.table of (from_row, to_row) for aggregation
    yr_edges <- data.table(from_row = from_rows, to_row = to_rows)
    
    # For each variable, pull neighbor values and aggregate
    col_offset <- 0L
    for (v in var_cols) {
      vals_vec <- dt[[v]]
      yr_edges[, nval := vals_vec[to_row]]
      
      # Grouped aggregation â€” this is the core vectorized operation
      agg <- yr_edges[!is.na(nval), 
                       .(nmax  = max(nval),
                         nmin  = min(nval),
                         nmean = mean(nval)),
                       by = from_row]
      
      # Write into result matrix
      res_mat[agg$from_row, col_offset + 1L] <- agg$nmax
      res_mat[agg$from_row, col_offset + 2L] <- agg$nmin
      res_mat[agg$from_row, col_offset + 3L] <- agg$nmean
      
      col_offset <- col_offset + 3L
    }
    
    cat(sprintf("  Year %d done\n", yr))
  }
  
  # ------------------------------------------------------------------
  # 5. Bind results back and restore original row order
  # ------------------------------------------------------------------
  for (j in seq_len(ncol(res_mat))) {
    set(dt, j = result_names[j], value = res_mat[, j])
  }
  
  # Restore original order
  setorder(dt, orig_row)
  dt[, c("orig_row", "row_idx") := NULL]
  
  # Convert back to data.frame if the input was one
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ================================================================
# DROP-IN REPLACEMENT for the original outer loop
# ================================================================
# BEFORE (86+ hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# AFTER (estimated ~2-10 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The output columns (neighbor_max_*, neighbor_min_*, neighbor_mean_*)
# contain identical numerical values as the original implementation.
```

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| **Runtime** | ~86+ hours | **~2â€“10 minutes** |
| **Peak RAM** | ~8â€“12 GB (string vectors) | **~4â€“6 GB** (integer edge list + data.table) |
| **String operations** | ~25 billion chars | **Zero** |
| **R-level loop iterations** | 6.46M Ã— `lapply` | **28 year iterations** (vectorized inside) |

### Why it's correct
- The edge list is a faithful translation of the `nb` object.
- The integer offset `b * C + rank` exactly replicates the string-key lookup `paste(id, year)` â†’ row index, because in a balanced panel sorted by `(year, id)`, every cell's position is deterministic.
- `data.table` grouped `max/min/mean` with `NA` removal produces identical results to the original `compute_neighbor_stats`.
- The trained Random Forest model is never touched â€” only the input feature columns are recomputed with identical values.
END_RESPONSE>>>
