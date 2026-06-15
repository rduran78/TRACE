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
run_id: Anthropic_F05_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `paste(..., sep="_")` call to build `idx_lookup` creates ~6.46 million string keys, and then for *each* of the 6.46 million rows, it does another `paste()` to form `neighbor_keys` and performs named-vector lookup. Named-vector lookup in R is O(n) hash-probe per call, but the real cost is the ~6.46M Ã— (avg neighbors â‰ˆ 4) string allocations and lookups â€” roughly **26 million `paste()` + match operations**.

### Broader Algorithmic Problem
More critically, the architecture is:

1. **`build_neighbor_lookup`** iterates over all 6.46M rows, doing string work to resolve which rows are the spatial neighbors of each row *in the same year*. This produces a list of integer row-indices.
2. **`compute_neighbor_stats`** then iterates over all 6.46M rows *again*, once per variable (Ã—5 variables), extracting neighbor values and computing max/min/mean.

The fundamental insight is that **the neighbor relationships are time-invariant** â€” cell `i`'s spatial neighbors are the same in every year. The current code re-discovers this structure via string keys. A proper reformulation should:

- Exploit the fact that within each year, the cell ordering is identical (or can be made identical), so neighbor row-offsets are constant across years.
- Replace all string-key operations with **integer arithmetic**.
- Vectorize the neighbor aggregation using **data.table** grouped operations or matrix arithmetic instead of row-wise `lapply`.

## Optimization Strategy

1. **Sort data by (year, id)** so that within each year-block, cells appear in the same order as `id_order`. Then a cell's position within its year-block is a fixed integer index, and its neighbors' positions are also fixed integer indices (from the `nb` object). Row index = `(year_block_offset) + (within-block position)`.

2. **Build the neighbor lookup once using pure integer arithmetic** â€” no strings, no named vectors. For each cell position `j` in `1:N_cells`, store the neighbor positions. Then for any year-block starting at offset `o`, the neighbor rows are simply `o + neighbor_positions`.

3. **Vectorize aggregation** using `data.table` with a pre-expanded edge list (cell-row â†’ neighbor-row), performing grouped `max`, `min`, `mean` in one shot per variable. This replaces 6.46M Ã— 5 R-level `lapply` iterations with 5 vectorized `data.table` joins + group-bys.

4. **Estimated speedup**: from ~86 hours to **minutes** (the dominant cost becomes a ~26M-row grouped aggregation, which `data.table` handles in seconds per variable).

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, 
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 1: Convert to data.table and establish a canonical ordering
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  N_cells <- length(id_order)
  years <- sort(unique(dt$year))
  N_years <- length(years)
  
  # Create a mapping from cell id -> position in id_order (1-based)
  id_to_pos <- integer(0)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add canonical position within year-block
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Sort by year, then cell_pos so within each year-block the order is identical
  setkey(dt, year, cell_pos)
  
  # Add a master row index after sorting
  dt[, row_idx := .I]
  
  # Compute year-block offsets: for each year, the starting row minus 1
  year_offsets <- dt[, .(offset = min(row_idx) - 1L), by = year]
  setkey(year_offsets, year)
  
  # ---------------------------------------------------------------
  # STEP 2: Build the directed edge list using integer arithmetic
  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length N_cells
  # where element j contains the neighbor indices (positions in id_order)
  # We build: from_pos -> to_pos pairs (within-block positions)
  
  # Pre-compute edge list (time-invariant)
  from_pos_list <- vector("list", N_cells)
  to_pos_list   <- vector("list", N_cells)
  
  for (j in seq_len(N_cells)) {
    nb_j <- rook_neighbors_unique[[j]]
    # spdep::nb encodes "no neighbors" as 0L in a length-1 vector
    if (length(nb_j) == 1L && nb_j[1] == 0L) next
    n_nb <- length(nb_j)
    from_pos_list[[j]] <- rep.int(j, n_nb)
    to_pos_list[[j]]   <- as.integer(nb_j)
  }
  
  edges <- data.table(
    from_pos = unlist(from_pos_list, use.names = FALSE),
    to_pos   = unlist(to_pos_list,   use.names = FALSE)
  )
  rm(from_pos_list, to_pos_list)
  
  n_edges <- nrow(edges)  # ~1,373,394 directed edges
  cat("Edge list built:", n_edges, "directed edges\n")
  
  # ---------------------------------------------------------------
  # STEP 3: Expand edge list across all years -> full row-index map
  # ---------------------------------------------------------------
  # For each year, from_row = offset + from_pos, to_row = offset + to_pos
  # Total expanded edges: n_edges * N_years â‰ˆ 1.37M * 28 â‰ˆ 38.5M rows
  # At ~16 bytes/row (two int columns) â‰ˆ 616 MB â€” fits in 16 GB RAM
  
  offsets_vec <- year_offsets$offset  # length = N_years, ordered by year
  
  # Vectorized expansion using rep + outer addition
  expanded <- data.table(
    from_row = rep(edges$from_pos, times = N_years) + 
               rep(offsets_vec, each = n_edges),
    to_row   = rep(edges$to_pos,   times = N_years) + 
               rep(offsets_vec, each = n_edges)
  )
  rm(edges)
  gc()
  
  cat("Expanded edge list:", nrow(expanded), "rows\n")
  
  # ---------------------------------------------------------------
  # STEP 4: For each variable, compute neighbor max/min/mean
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor stats for:", var_name, "\n")
    
    # Extract the variable values aligned with row_idx
    vals <- dt[[var_name]]
    
    # Look up neighbor values via integer indexing (vectorized)
    expanded[, nb_val := vals[to_row]]
    
    # Group by from_row, compute stats (dropping NAs)
    stats <- expanded[!is.na(nb_val), 
                      .(nb_max  = max(nb_val),
                        nb_min  = min(nb_val),
                        nb_mean = mean(nb_val)),
                      by = from_row]
    
    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed stats back by row index
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
    
    cat("  Done:", var_name, "\n")
  }
  
  # Clean up helper column
  expanded[, nb_val := NULL]
  
  # ---------------------------------------------------------------
  # STEP 5: Restore original row order and return as data.frame
  # ---------------------------------------------------------------
  # Restore original order: we need to undo the sort.
  # We saved row_idx after sorting, but we need original order.
  # Strategy: before sorting, save original order.
  
  # Actually, we need to be more careful. Let's use a pre-sort index.
  # We'll adjust the approach: add orig_order before sorting.
  
  dt[, cell_pos := NULL]
  dt[, row_idx  := NULL]
  
  # Return as data.frame to preserve compatibility with existing RF model
  as.data.frame(dt)
}
```

However, the above has an ordering subtlety. Here is the **corrected, complete version** that properly tracks original row order:

```r
library(data.table)

build_neighbor_features_optimized <- function(cell_data, id_order, 
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]  # preserve original row order
  
  N_cells <- length(id_order)
  years <- sort(unique(dt$year))
  N_years <- length(years)
  
  # Map cell id -> canonical position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  dt[, .cell_pos := id_to_pos[as.character(id)]]
  
  # Sort by (year, cell_pos) so within-year blocks have consistent ordering
  setkey(dt, year, .cell_pos)
  dt[, .sorted_row := .I]
  
  # Year-block offsets
  year_off <- dt[, .(off = min(.sorted_row) - 1L), by = year]
  setkey(year_off, year)
  offsets_vec <- year_off$off  # one per year, in sorted year order
  
  # ------------------------------------------------------------------
  # Build time-invariant edge list from nb object
  # ------------------------------------------------------------------
  from_list <- vector("list", N_cells)
  to_list   <- vector("list", N_cells)
  
  for (j in seq_len(N_cells)) {
    nb_j <- rook_neighbors_unique[[j]]
    if (length(nb_j) == 0L || (length(nb_j) == 1L && nb_j[1] == 0L)) next
    from_list[[j]] <- rep.int(as.integer(j), length(nb_j))
    to_list[[j]]   <- as.integer(nb_j)
  }
  
  edge_from <- unlist(from_list, use.names = FALSE)
  edge_to   <- unlist(to_list,   use.names = FALSE)
  n_edges   <- length(edge_from)
  rm(from_list, to_list)
  
  cat(sprintf("Edges: %d directed | Cells: %d | Years: %d | Rows: %d\n",
              n_edges, N_cells, N_years, nrow(dt)))
  
  # ------------------------------------------------------------------
  # Expand across years: ~38.5M rows (fits in RAM)
  # from_sorted_row = offset[y] + from_pos
  # to_sorted_row   = offset[y] + to_pos
  # ------------------------------------------------------------------
  exp_from <- rep(edge_from, times = N_years) + 
              rep(offsets_vec, each = n_edges)
  exp_to   <- rep(edge_to,   times = N_years) + 
              rep(offsets_vec, each = n_edges)
  
  # Validate: all indices must be in [1, nrow(dt)]
  # (cells at grid boundary may have neighbors outside the dataset;
  #  those would have NA cell_pos and won't appear, but let's be safe)
  valid <- exp_from >= 1L & exp_from <= nrow(dt) & 
           exp_to   >= 1L & exp_to   <= nrow(dt)
  exp_from <- exp_from[valid]
  exp_to   <- exp_to[valid]
  
  cat(sprintf("Expanded edge list: %d rows\n", length(exp_from)))
  
  # ------------------------------------------------------------------
  # Compute neighbor stats per variable (vectorized)
  # ------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    t0 <- proc.time()
    cat(sprintf("  %s ... ", var_name))
    
    vals <- dt[[var_name]]
    nb_vals <- vals[exp_to]
    
    # Build a temporary data.table for grouped aggregation
    tmp <- data.table(from = exp_from, v = nb_vals)
    tmp <- tmp[!is.na(v)]
    
    stats <- tmp[, .(vmax = max(v), vmin = min(v), vmean = mean(v)), 
                 by = from]
    
    # Column names matching original code's convention
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Initialize with NA, then fill
    set(dt, j = max_col,  value = NA_real_)
    set(dt, j = min_col,  value = NA_real_)
    set(dt, j = mean_col, value = NA_real_)
    
    set(dt, i = stats$from, j = max_col,  value = stats$vmax)
    set(dt, i = stats$from, j = min_col,  value = stats$vmin)
    set(dt, i = stats$from, j = mean_col, value = stats$vmean)
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("done (%.1f sec)\n", elapsed))
  }
  
  rm(exp_from, exp_to)
  gc()
  
  # ------------------------------------------------------------------
  # Restore original row order
  # ------------------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c(".orig_row", ".cell_pos", ".sorted_row") := NULL]
  
  as.data.frame(dt)
}
```

### Usage (drop-in replacement for the original outer loop):

```r
# BEFORE (original â€” ~86 hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# AFTER (optimized â€” estimated ~2-5 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# The existing trained Random Forest model is unchanged.
# Column names and numerical values are identical to the original.
# Proceed directly to predict():
# preds <- predict(rf_model, newdata = cell_data)
```

## Summary of Complexity Improvement

| Aspect | Original | Optimized |
|---|---|---|
| **Key construction** | 6.46M `paste()` for lookup + 6.46M Ã— ~4 `paste()` per row | **Zero** string operations |
| **Lookup mechanism** | Named character vector (hash per query) | Integer arithmetic: `offset + position` |
| **Neighbor resolution** | R-level `lapply` over 6.46M rows | Vectorized edge-list expansion |
| **Aggregation** | R-level `lapply` Ã— 5 vars Ã— 6.46M rows | 5 Ã— `data.table` grouped agg on ~38.5M-row edge table |
| **Estimated time** | ~86 hours | **~2â€“5 minutes** |
| **Peak RAM** | Moderate (list of 6.46M vectors) | ~1.5 GB for expanded edge list (fits 16 GB) |
| **Numerical output** | Baseline | **Identical** (same max/min/mean over same neighbor sets) |
| **RF model impact** | â€” | **None** (features are identical; no retraining needed) |
END_RESPONSE>>>
