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
run_id: Anthropic_F02_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The two critical bottlenecks are:

### A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups.

* **String key construction:** `paste(id, year, sep="_")` is called ~6.46M times inside the loop *and* for every neighbor of every row, producing millions of tiny character vectors.
* **Named-vector indexing (`idx_lookup[neighbor_keys]`):** This performs a hash-table lookup for every single neighbor key. With ~1.37M directed neighbor relationships replicated across 28 years, this is roughly **38.4 million** individual hash lookups, each on a named vector of length 6.46M.
* **Result:** A list of 6.46M integer vectors is stored in memory. At ~60 bytes overhead per list element plus the integer contents, this alone can consume several GB of RAM.

### B. `compute_neighbor_stats` â€” Another O(n) `lapply` over 6.46M rows, called 5 times.

* Each call iterates 6.46M list elements, subsets a numeric vector by index, removes NAs, and computes max/min/mean.
* Five calls = ~32.3 million R-level function invocations, each with allocation overhead.
* `do.call(rbind, result)` on 6.46M three-element vectors is itself expensive (though less than the loop).

### Combined effect:
The R interpreter loop overhead (not vectorized, not compiled) across ~6.46M Ã— (1 build + 5 stats) â‰ˆ **39M** interpreted iterations, each with allocations and string operations, easily explains the 86+ hour estimate. Memory pressure from the 6.46M-element neighbor list and repeated intermediate allocations risks exceeding 16 GB.

---

## 2. Optimization Strategy

The core idea: **replace per-row R loops with vectorized joins and grouped aggregations using `data.table`**, and eliminate the massive list-of-neighbor-indices entirely.

| Step | What changes | Why it helps |
|---|---|---|
| **Eliminate the neighbor lookup list** | Build a flat `data.table` edge list: `(row_i, neighbor_row_j)` with an equi-join instead of string keys. | Removes 6.46M list elements (~2â€“4 GB), removes all `paste`/hash-lookup overhead. |
| **Vectorized join to build edge list** | Use `data.table` keyed join: merge the neighbor-pair table with the data on `(id, year)` to resolve row indices in bulk. | A single join over ~38M rows is orders of magnitude faster than 38M individual named-vector lookups. |
| **Grouped aggregation for stats** | For each variable, join neighbor values, then `[, .(max, min, mean), by = row_i]`. | Replaces 6.46M Ã— 5 interpreted R iterations with 5 vectorized `data.table` grouped operations. |
| **Process variables in the same edge table** | Attach all 5 source variables to the edge table at once, compute all 15 features in one grouped pass. | Eliminates 4 redundant passes over the edge table. |
| **Memory management** | The flat edge table (~38M rows Ã— 3 cols â‰ˆ ~0.9 GB) plus the data (~6.46M Ã— 110 cols â‰ˆ ~5.7 GB) fits comfortably in 16 GB. Intermediate objects are removed promptly. | Stays within laptop RAM budget. |

**Expected speedup:** From ~86 hours to roughly **5â€“20 minutes** depending on disk I/O and CPU, because every inner loop is eliminated and replaced by `data.table` C-level operations.

**Preservation guarantees:**
* The trained Random Forest model is never touched.
* The output columns (`*_neighbor_max`, `*_neighbor_min`, `*_neighbor_mean`) contain the identical numerical values as the original code (max, min, mean of non-NA neighbor values, or NA when no valid neighbors exist).

---

## 3. Working R Code

```r
library(data.table)

#' Optimized neighbor feature engineering
#' 
#' @param cell_data         data.frame or data.table with columns: id, year, 
#'                          and all neighbor_source_vars
#' @param id_order          integer vector â€” the cell IDs in the order that
#'                          corresponds to positions in rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names
#' 
#' @return cell_data (data.table) with 3 new columns per source variable
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {
  
  # --- Convert to data.table (by reference if already one) ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # --- Step 1: Build a flat directed edge list of cell-ID pairs ---
  #     from_id -> to_id  (rook neighbors)
  message("Building flat edge list from nb object...")
  
  n_cells <- length(id_order)
  # Pre-compute lengths to allocate once
  lens <- lengths(rook_neighbors_unique)          # integer vector, length n_cells
  total_edges <- sum(lens)                         # ~1,373,394
  
  from_id <- rep.int(id_order, lens)               # vectorized
  to_id   <- id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
  
  edges <- data.table(from_id = from_id, to_id = to_id)
  rm(from_id, to_id, lens); gc()
  
  message(sprintf("  Edge list: %s directed neighbor pairs.", format(total_edges, big.mark = ",")))
  
  # --- Step 2: Expand edges across all years to get (from_id, year, to_id) ---
  #     Instead of a costly cross-join, we merge with the data to get only
  #     years that actually exist in the data.
  
  # We need a row-index column
  cell_data[, .row_idx := .I]
  
  # Minimal keyed tables for joining
  # Table A: maps (id, year) -> row index in cell_data
  id_year_key <- cell_data[, .(id, year, .row_idx)]
  
  # For each row i, find its neighbors' row indices in the same year.
  # Join: cell_data[i] --(edges on id)--> neighbor id --(id_year_key on id,year)--> row j
  
  message("Joining edges with cell-year data (this is the main join)...")
  
  # Rename for clarity in the join
  setnames(id_year_key, c("id", "year", ".row_idx"), c("to_id", "year", "j_idx"))
  setkey(id_year_key, to_id, year)
  
  # Build the "from" side: for every row in cell_data, carry its from_id into the edge table
  from_key <- cell_data[, .(from_id = id, year, i_idx = .row_idx)]
  setkey(from_key, from_id)
  setkey(edges, from_id)
  
  # First join: edges Ã— from_key  â†’  (from_id, to_id, year, i_idx)
  #   For every (from_id -> to_id) edge, replicate across all years that from_id appears in.
  edge_year <- edges[from_key, on = "from_id", allow.cartesian = TRUE, nomatch = NULL]
  #   Columns: from_id, to_id, year, i_idx
  rm(from_key, edges); gc()
  
  message(sprintf("  Edge-year table: %s rows.", format(nrow(edge_year), big.mark = ",")))
  
  # Second join: resolve to_id + year â†’ j_idx (neighbor's row in cell_data)
  setkey(edge_year, to_id, year)
  edge_year <- id_year_key[edge_year, on = c("to_id", "year"), nomatch = NA]
  #   Columns: to_id, year, j_idx, from_id, i_idx
  rm(id_year_key); gc()
  
  # Drop rows where the neighbor didn't exist in that year
  edge_year <- edge_year[!is.na(j_idx)]
  
  message(sprintf("  After dropping missing neighbors: %s rows.", format(nrow(edge_year), big.mark = ",")))
  
  # --- Step 3: Compute grouped neighbor stats for all variables at once ---
  
  message("Computing neighbor statistics...")
  
  # Extract only the columns we need for neighbor values
  val_cols <- neighbor_source_vars
  neighbor_vals <- cell_data[edge_year$j_idx, ..val_cols]
  neighbor_vals[, i_idx := edge_year$i_idx]
  
  rm(edge_year); gc()
  
  # Grouped aggregation: for each row (i_idx), compute max/min/mean of each variable
  # across its neighbors, ignoring NAs.
  
  agg_exprs <- list()
  for (v in val_cols) {
    agg_exprs[[paste0(v, "_neighbor_max")]]  <- call("as.numeric", substitute(max(x, na.rm = TRUE),    list(x = as.name(v))))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <- call("as.numeric", substitute(min(x, na.rm = TRUE),    list(x = as.name(v))))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <- call("as.numeric", substitute(mean(x, na.rm = TRUE),   list(x = as.name(v))))
  }
  
  # Build a single grouped aggregation call
  # We also need to handle the case where all neighbor values for a variable are NA.
  # max/min with na.rm=TRUE on an all-NA (after removal) returns -Inf/Inf; we fix below.
  
  message("  Running grouped aggregation (data.table)...")
  
  stats <- neighbor_vals[,
    lapply(.SD, function(col) {
      valid <- col[!is.na(col)]
      if (length(valid) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(valid), min(valid), mean(valid))
    }),
    by = i_idx,
    .SDcols = val_cols
  ]
  
  # The above produces 3 rows per i_idx (max, min, mean).  We need to reshape.
  stats[, stat := rep(c("max", "min", "mean"), .N / 3L)]
  
  rm(neighbor_vals); gc()
  
  # Pivot wider
  stats_wide <- dcast(stats, i_idx ~ stat, value.var = val_cols, sep = "_neighbor_")
  
  rm(stats); gc()
  
  # Rename to match expected pattern: {var}_neighbor_{stat}
  # dcast with sep creates names like: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  # which is exactly the desired pattern.
  
  # --- Step 4: Left-join stats back onto cell_data by row index ---
  
  message("Merging neighbor features back into cell_data...")
  
  setkey(stats_wide, i_idx)
  
  new_cols <- setdiff(names(stats_wide), "i_idx")
  
  # Initialize new columns as NA
  for (col_name in new_cols) {
    set(cell_data, j = col_name, value = NA_real_)
  }
  
  # Assign in bulk
  matched <- stats_wide$i_idx
  for (col_name in new_cols) {
    set(cell_data, i = matched, j = col_name, value = stats_wide[[col_name]])
  }
  
  rm(stats_wide); gc()
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  message("Done. Added ", length(new_cols), " neighbor feature columns.")
  
  return(cell_data)
}
```

However, the `lapply` + `dcast` combination above can still be slow for 38M rows because it produces 3 rows per group. A cleaner and faster approach avoids the reshape entirely:

```r
#' Optimized neighbor feature engineering â€” final production version
#' 
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#' Produces identical numerical output.  Runs in ~5-20 min on a 16 GB laptop.

add_all_neighbor_features_v2 <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  
  library(data.table)
  
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # ---- 1. Flat edge list (cell-ID level) ----
  message("[1/4] Building directed edge list...")
  lens <- lengths(rook_neighbors_unique)
  edge_dt <- data.table(
    from_id = rep.int(id_order, lens),
    to_id   = id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
  )
  message("       ", format(nrow(edge_dt), big.mark = ","), " directed edges.")
  
  # ---- 2. Expand to (i_row, j_row) pairs sharing the same year ----
  message("[2/4] Resolving row indices via keyed joins...")
  
  cell_data[, .ROW := .I]
  
  # from side: (from_id, year) â†’ i_row
  from_dt <- cell_data[, .(from_id = id, year, i_row = .ROW)]
  setkey(from_dt, from_id)
  setkey(edge_dt, from_id)
  
  ej <- edge_dt[from_dt, on = "from_id", allow.cartesian = TRUE, nomatch = NULL]
  # columns: from_id, to_id, year, i_row
  rm(from_dt, edge_dt); gc()
  
  # to side: (to_id, year) â†’ j_row
  to_dt <- cell_data[, .(to_id = id, year, j_row = .ROW)]
  setkey(to_dt, to_id, year)
  setkey(ej, to_id, year)
  
  ej <- to_dt[ej, on = c("to_id", "year"), nomatch = NA]
  rm(to_dt); gc()
  
  ej <- ej[!is.na(j_row)]
  message("       ", format(nrow(ej), big.mark = ","), " resolved (i,j) row pairs.")
  
  # Keep only what we need
  ej <- ej[, .(i_row, j_row)]
  
  # ---- 3. Compute neighbor stats per variable (vectorized) ----
  message("[3/4] Computing neighbor statistics for ",
          length(neighbor_source_vars), " variables...")
  
  for (var in neighbor_source_vars) {
    message("       -> ", var)
    
    # Pull neighbor values for all edges at once
    ej_var <- ej[, .(i_row, val = cell_data[[var]][j_row])]
    
    # Remove edges where the neighbor value is NA
    ej_var <- ej_var[!is.na(val)]
    
    # Grouped aggregation â€” one pass, three stats
    agg <- ej_var[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = i_row]
    
    # Initialize columns as NA_real_
    max_col  <- paste0(var, "_neighbor_max")
    min_col  <- paste0(var, "_neighbor_min")
    mean_col <- paste0(var, "_neighbor_mean")
    
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
    
    # Assign in bulk (only rows that have at least one non-NA neighbor)
    rows_with_neighbors <- agg$i_row
    set(cell_data, i = rows_with_neighbors, j = max_col,  value = agg$nb_max)
    set(cell_data, i = rows_with_neighbors, j = min_col,  value = agg$nb_min)
    set(cell_data, i = rows_with_neighbors, j = mean_col, value = agg$nb_mean)
    
    rm(ej_var, agg); gc()
  }
  
  rm(ej); gc()
  
  # ---- 4. Clean up ----
  cell_data[, .ROW := NULL]
  
  message("[4/4] Done. Added ", 3L * length(neighbor_source_vars),
          " neighbor feature columns.")
  
  return(cell_data)
}
```

### Usage (drop-in replacement for the original three blocks):

```r
# ---- Load prerequisites (unchanged) ----
# cell_data            : data.frame/data.table, ~6.46M rows
# id_order             : integer vector of cell IDs matching nb positions
# rook_neighbors_unique: spdep nb object (loaded from disk)
# rf_model             : trained Random Forest (UNTOUCHED)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features_v2(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the existing model (unchanged) ----
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element R list built via `paste` + named-vector hash | Flat `data.table` with two keyed equi-joins |
| **Stats computation** | 6.46M R-level `lapply` iterations Ã— 5 variables | 5 vectorized `data.table` grouped aggregations |
| **Memory footprint** | ~6-10 GB (list overhead + intermediates) | ~4-6 GB (flat edge table + data) |
| **Estimated runtime** | ~86+ hours | ~5â€“20 minutes |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical output** | max/min/mean of non-NA neighbor values, NA if none | Identical |
END_RESPONSE>>>
