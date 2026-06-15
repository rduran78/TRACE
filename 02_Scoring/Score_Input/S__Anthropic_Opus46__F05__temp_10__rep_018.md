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
run_id: Anthropic_F05_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of redundancy:

### 1. **O(N) string-key construction inside an O(N) loop = O(NÂ²) string operations**
`build_neighbor_lookup` creates `idx_lookup` once (good), but then calls `paste()` inside `lapply` over every row (~6.46M iterations). Each iteration constructs `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” a string allocation for every neighbor of every cell-year. With ~1.37M neighbor pairs Ã— 28 years, this is tens of millions of `paste` calls plus hash lookups against a 6.46M-entry named vector.

### 2. **The string-keyed lookup itself is unnecessary**
The entire purpose is to answer: *"Given cell `i` in year `t`, which rows belong to cell `i`'s spatial neighbors in the same year `t`?"* This is a join on `(neighbor_id, year)` â€” a structured, integer-indexable operation that doesn't need string hashing at all.

### 3. **The neighbor topology is time-invariant**
Rook neighbors don't change across years. The per-row `lapply` rediscovers the same spatial neighbor set 28 times (once per year for each cell). The lookup should be factored into: (a) a spatial neighbor map (cell â†’ neighbor cells), and (b) a year-to-row-offset map.

### 4. **`compute_neighbor_stats` is efficient given the lookup, but the lookup is the bottleneck**
Once `neighbor_lookup` is a list of integer row indices, the stats computation is vectorized and fast. The entire cost is in building that lookup.

---

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Key type | String paste + named-vector hash | Integer arithmetic |
| Complexity | O(N Ã— avg_neighbors) string ops, N=6.46M | O(C Ã— avg_neighbors) integer ops, C=344K, then O(1) offset per year |
| Core idea | Per cell-year: find neighbor rows via string match | Pre-build a cell-index-to-row-offset table; expand spatial neighbors to row indices via integer math |
| RAM | 6.46M-element named character vector | Integer matrix (344K Ã— 28) â‰ˆ 77 MB |
| Expected speedup | ~86+ hours | Minutes |

**Key insight**: If data is sorted by `(id, year)` â€” or we build a `(id, year) â†’ row` integer matrix â€” then for any cell's spatial neighbors, we can look up all 28 year-rows instantly with integer indexing. We compute stats for one year at a time, or all at once.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement â€” preserves numerical output exactly.
# =============================================================================

library(data.table)

build_neighbor_features_fast <- function(cell_data, 
                                         id_order, 
                                         rook_neighbors_unique, 
                                         neighbor_source_vars) {
  
  # ---------- Step 0: Convert to data.table for speed, keep original order ----
  dt <- as.data.table(cell_data)
  dt[, ..orig_row := .I]
  
  # ---------- Step 1: Build integer cell-index and year-index -----------------
  # Map each unique id to a contiguous integer index
  unique_ids   <- as.character(id_order)
  id_to_cidx   <- setNames(seq_along(unique_ids), unique_ids)
  
  unique_years <- sort(unique(dt$year))
  year_to_yidx <- setNames(seq_along(unique_years), as.character(unique_years))
  
  n_cells <- length(unique_ids)
  n_years <- length(unique_years)
  
  # ---------- Step 2: Build (cell_idx, year_idx) â†’ row number matrix ----------
  # cell_year_row[c, y] = row number in dt for cell c, year y (or NA)
  # This replaces the entire string-keyed lookup.
  
  cidx_vec <- id_to_cidx[as.character(dt$id)]
  yidx_vec <- year_to_yidx[as.character(dt$year)]
  
  cell_year_row <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  cell_year_row[cbind(cidx_vec, yidx_vec)] <- seq_len(nrow(dt))
  
  cat(sprintf("Cell-year matrix: %d cells Ã— %d years (%.1f MB)\n",
              n_cells, n_years, object.size(cell_year_row) / 1e6))
  
  # ---------- Step 3: Expand spatial neighbor list to cell-index space --------
  # rook_neighbors_unique[[i]] contains neighbor indices into id_order,
  # which is the same as cidx. So we can use it directly.
  # Validate: rook_neighbors_unique is a list of length n_cells.
  
  stopifnot(length(rook_neighbors_unique) == n_cells)
  
  # Pre-flatten the neighbor structure into a CSR-like representation
  # for fully vectorized computation.
  
  nb_lengths <- lengths(rook_neighbors_unique)       # integer vector, length n_cells
  nb_flat    <- unlist(rook_neighbors_unique)         # all neighbor cidx values
  nb_offsets <- c(0L, cumsum(nb_lengths))             # CSR offsets
  
  total_directed_pairs <- length(nb_flat)
  cat(sprintf("Total directed neighbor pairs: %d\n", total_directed_pairs))
  
  # ---------- Step 4: Compute neighbor stats per variable, fully vectorized ---
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    vals <- dt[[var_name]]
    
    # Preallocate output columns
    col_max  <- rep(NA_real_, nrow(dt))
    col_min  <- rep(NA_real_, nrow(dt))
    col_mean <- rep(NA_real_, nrow(dt))
    
    # Process year-by-year: for each year, all cells share the same year index,
    # so neighbor row lookups are a single column slice of cell_year_row.
    
    for (yi in seq_along(unique_years)) {
      
      # Which rows in dt belong to this year?
      rows_this_year <- which(yidx_vec == yi)   # row numbers in dt
      if (length(rows_this_year) == 0) next
      
      # Which cells are present this year?
      cells_this_year <- cidx_vec[rows_this_year]  # their cell indices
      
      # For each cell present this year, get its neighbor cell indices,
      # then look up their row numbers in this same year.
      # Vectorized via the CSR structure.
      
      # Neighbor rows for this year column
      nb_row_this_year <- cell_year_row[, yi]  # length n_cells; maps cidx â†’ row
      
      # For each cell in cells_this_year, gather neighbor values
      # Using a fast C-level loop via vapply over the CSR structure
      
      n_this <- length(cells_this_year)
      
      stats <- vapply(seq_len(n_this), function(j) {
        c_idx <- cells_this_year[j]
        n_nb  <- nb_lengths[c_idx]
        if (n_nb == 0L) return(c(NA_real_, NA_real_, NA_real_))
        
        # Indices into nb_flat
        start <- nb_offsets[c_idx] + 1L
        end   <- nb_offsets[c_idx + 1L]
        
        # Neighbor cell indices â†’ row numbers in this year
        nb_rows <- nb_row_this_year[nb_flat[start:end]]
        nb_rows <- nb_rows[!is.na(nb_rows)]
        
        if (length(nb_rows) == 0L) return(c(NA_real_, NA_real_, NA_real_))
        
        nv <- vals[nb_rows]
        nv <- nv[!is.na(nv)]
        
        if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
        
        c(max(nv), min(nv), mean(nv))
      }, numeric(3))
      
      # stats is 3 Ã— n_this
      col_max[rows_this_year]  <- stats[1, ]
      col_min[rows_this_year]  <- stats[2, ]
      col_mean[rows_this_year] <- stats[3, ]
    }
    
    # Assign to dt with standard naming convention
    set(dt, j = paste0("neighbor_max_",  var_name), value = col_max)
    set(dt, j = paste0("neighbor_min_",  var_name), value = col_min)
    set(dt, j = paste0("neighbor_mean_", var_name), value = col_mean)
  }
  
  # ---------- Step 5: Return as data.frame in original row order --------------
  dt[, ..orig_row := NULL]
  as.data.frame(dt)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# The trained Random Forest model is untouched â€” only the feature columns
# are regenerated with identical numerical values.
```

---

## Even Faster: Fully Vectorized (No `vapply` Inner Loop)

If the `vapply` per-cell-per-year is still too slow (~344K iterations Ã— 28 years Ã— 5 vars), here is a **fully vectorized** version that eliminates all R-level loops over cells:

```r
build_neighbor_features_vectorized <- function(cell_data, 
                                                id_order, 
                                                rook_neighbors_unique, 
                                                neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  unique_ids <- as.character(id_order)
  id_to_cidx <- setNames(seq_along(unique_ids), unique_ids)
  
  unique_years <- sort(unique(dt$year))
  year_to_yidx <- setNames(seq_along(unique_years), as.character(unique_years))
  
  n_cells <- length(unique_ids)
  n_years <- length(unique_years)
  
  # (cell_idx, year_idx) for every row
  cidx_vec <- id_to_cidx[as.character(dt$id)]
  yidx_vec <- year_to_yidx[as.character(dt$year)]
  
  # Cell-year â†’ row matrix
  cell_year_row <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  cell_year_row[cbind(cidx_vec, yidx_vec)] <- seq_len(nrow(dt))
  
  # Build edge list: (from_cidx, to_cidx) for all directed neighbor pairs
  from_cidx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_cidx   <- unlist(rook_neighbors_unique)
  n_edges   <- length(from_cidx)
  
  cat(sprintf("Edge list: %d directed pairs\n", n_edges))
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing: %s ...\n", var_name))
    
    vals <- dt[[var_name]]
    
    # Preallocate result columns
    res_max  <- rep(NA_real_, nrow(dt))
    res_min  <- rep(NA_real_, nrow(dt))
    res_mean <- rep(NA_real_, nrow(dt))
    
    for (yi in seq_along(unique_years)) {
      
      # Row indices for source (from) and target (to) cells in this year
      from_rows <- cell_year_row[from_cidx, yi]  # length n_edges
      to_rows   <- cell_year_row[to_cidx, yi]    # length n_edges
      
      # Keep only edges where both from and to exist this year
      valid <- !is.na(from_rows) & !is.na(to_rows)
      
      e_from_rows <- from_rows[valid]
      e_to_rows   <- to_rows[valid]
      
      # Get neighbor values
      neighbor_vals <- vals[e_to_rows]
      
      # Further filter out NA values in the variable
      not_na <- !is.na(neighbor_vals)
      e_from_rows   <- e_from_rows[not_na]
      neighbor_vals  <- neighbor_vals[not_na]
      
      if (length(e_from_rows) == 0L) next
      
      # Now aggregate: group neighbor_vals by e_from_rows (the source row)
      # Using data.table for vectorized grouped aggregation
      agg_dt <- data.table(from_row = e_from_rows, nval = neighbor_vals)
      agg <- agg_dt[, .(nmax = max(nval), 
                         nmin = min(nval), 
                         nmean = mean(nval)), 
                     by = from_row]
      
      res_max[agg$from_row]  <- agg$nmax
      res_min[agg$from_row]  <- agg$nmin
      res_mean[agg$from_row] <- agg$nmean
    }
    
    set(dt, j = paste0("neighbor_max_",  var_name), value = res_max)
    set(dt, j = paste0("neighbor_min_",  var_name), value = res_min)
    set(dt, j = paste0("neighbor_mean_", var_name), value = res_mean)
  }
  
  as.data.frame(dt)
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_vectorized(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)
```

---

## Performance Analysis

| Component | Original | Optimized (vapply) | Fully Vectorized |
|---|---|---|---|
| Key construction | ~6.46M `paste()` inside loop | 0 string ops | 0 string ops |
| Lookup structure | Named char vector (6.46M entries) | Integer matrix (344K Ã— 28 â‰ˆ 77MB) | Same matrix + edge list |
| Per-row work | `paste` + hash probe per neighbor | Integer index per neighbor | None â€” fully vectorized |
| Inner aggregation | `lapply` over 6.46M rows | `vapply` over 344K cells Ã— 28 years | `data.table` grouped agg per year |
| Estimated time | 86+ hours | ~30-60 min | **~2-10 min** |
| RAM | High (string duplication) | ~77 MB matrix + edge list | ~77 MB + ~44 MB edge DT per year (transient) |

The fully vectorized version processes **~1.37M edges per year** in a single `data.table` grouped aggregation â€” no R-level loop over cells at all. With 28 years Ã— 5 variables = 140 `data.table` aggregations, each over ~1.37M rows, this should complete in minutes on a standard laptop.

**The trained Random Forest model requires no changes** â€” the output columns are numerically identical to the originals.
END_RESPONSE>>>
