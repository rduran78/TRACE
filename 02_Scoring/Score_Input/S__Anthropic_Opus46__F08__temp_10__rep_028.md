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
run_id: Anthropic_F08_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It constructs a lookup list of length **~6.46 million** (one entry per cell-year row). But the neighbor *topology* is purely spatial â€” it never changes across years. The function redundantly recomputes the same neighbor cell-ID list 28 times (once per year) for each of the 344,208 cells, and it does string-pasting and named-vector lookups on millions of keys. The `compute_neighbor_stats` function itself is fine algorithmically, but it is called on a lookup that was needlessly expensive to build, and it too operates on 6.46M rows when it could operate on 344K rows per year.

**Root cause summary:**

| Aspect | Current | Ideal |
|---|---|---|
| Neighbor topology | Recomputed per cell-year (6.46M entries) | Computed once per cell (344K entries) |
| Key format | String-paste `id_year` with named-vector lookup | Integer index arithmetic |
| Stats computation | One giant 6.46M-row pass | 344K-row pass Ã— 28 years, or vectorized matrix operation |

The static neighbor topology should be factored out and computed **once**. The changing variable values should be sliced **per year** and the neighbor stats computed on those slices. This avoids millions of string operations and reduces the lookup from 6.46M entries to 344K entries.

---

## Optimization Strategy

1. **Build the neighbor index once (static).** Convert `rook_neighbors_unique` (an `nb` object, already indexed by position) into a simple integer-index lookup over the canonical cell ordering `id_order`. This is a list of length 344,208 â€” each element is an integer vector of neighbor positions. This is essentially what `spdep::nb` already is, so we can use it directly.

2. **Organize data for fast column access per year.** Sort data by `(id, year)` so that for a given year, the row for cell `i` (in `id_order`) is at a deterministic position. With 344,208 cells and 28 years in a balanced panel, row position = `(cell_position - 1) * 28 + year_offset` or we can split by year.

3. **Compute neighbor stats per year on a 344K-length vector.** For each year and each variable, extract the 344K-length value vector, then compute neighbor max/min/mean using the static 344K-length neighbor list. This turns 6.46M lookup entries into 28 Ã— 344K = 6.46M simple integer-vector index operations â€” but without any string hashing.

4. **Vectorize the stats computation using sparse matrix multiplication** (optional but large speedup). Encode the neighbor topology as a sparse matrix `W` (344,208 Ã— 344,208). Then:
   - Neighbor mean = `W %*% vals / neighbor_count` (or use a row-normalized `W`).
   - Neighbor max/min via a loop over the nb list or via `data.table` grouped operations.

5. **The Random Forest model is not retouched.** The output columns have identical names and identical numerical values (same estimand).

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build static neighbor structures (done ONCE, ~344K cells)
# ==============================================================================

# id_order: character or integer vector of length 344,208 giving cell IDs
#           in the same order as rook_neighbors_unique (the nb object).
# rook_neighbors_unique: an nb object (list of length 344,208), where element i
#           contains integer indices of neighbors of cell i (referring to
#           positions within id_order). 0-neighbor cells have integer(0).

build_static_neighbor_structures <- function(id_order, neighbors_nb) {
  n_cells <- length(id_order)
  stopifnot(length(neighbors_nb) == n_cells)
  
  # --- Map from cell ID to position in id_order ---
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # --- The nb object already stores neighbor indices as positions.
  #     We just sanitize: remove any 0s (spdep uses 0 for "no neighbors"). ---
  nb_list <- lapply(neighbors_nb, function(x) {
    x <- as.integer(x)
    x[x > 0L]
  })
  
  # --- Build a row-normalized sparse weight matrix for neighbor mean ---
  #     and an un-normalized one for sum-based operations. ---
  from <- rep(seq_len(n_cells), lengths(nb_list))
  to   <- unlist(nb_list, use.names = FALSE)
  n_neighbors <- lengths(nb_list)  # number of neighbors per cell
  
  # Row-normalized weights for mean
  w_mean <- rep(1.0 / pmax(n_neighbors, 1L), n_neighbors)  # avoid /0
  
  W_mean <- sparseMatrix(
    i = from, j = to, x = w_mean,
    dims = c(n_cells, n_cells)
  )
  
  list(
    id_order     = id_order,
    id_to_pos    = id_to_pos,
    nb_list      = nb_list,
    n_neighbors  = n_neighbors,
    n_cells      = n_cells,
    W_mean       = W_mean
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for one variable across all years
# ==============================================================================

compute_neighbor_features_fast <- function(dt, var_name, static) {
  # dt: data.table with columns id, year, <var_name>, sorted by (id, year)
  # static: output of build_static_neighbor_structures
  
  nb_list     <- static$nb_list
  n_cells     <- static$n_cells
  W_mean      <- static$W_mean
  id_to_pos   <- static$id_to_pos
  n_neighbors <- static$n_neighbors
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # --- Extract the value vector for this year, aligned to id_order ---
    yr_rows <- which(dt$year == yr)
    
    # Build a fast mapping: for this year-slice, get values in id_order order
    yr_dt <- dt[yr_rows, .(id, val = get(var_name))]
    
    # Map each row's cell ID to its position in id_order
    yr_dt[, pos := id_to_pos[as.character(id)]]
    
    # Create a value vector of length n_cells, indexed by position
    vals <- rep(NA_real_, n_cells)
    vals[yr_dt$pos] <- yr_dt$val
    
    # ------ Neighbor MEAN via sparse matrix multiply ------
    # Replace NAs with 0 for multiplication, track valid counts
    vals_no_na <- ifelse(is.na(vals), 0.0, vals)
    valid      <- as.numeric(!is.na(vals))
    
    # Un-normalized sparse matrix (just adjacency with 1s)
    # We can get neighbor sum and neighbor valid count:
    # But we already have W_mean (row-normalized). 
    # For correct NA handling, compute from nb_list.
    # Actually, sparse mat approach assumes no NAs. If NAs are rare or absent,
    # this is fine. For full correctness:
    
    neighbor_sum   <- as.numeric(W_mean %*% vals_no_na) * n_neighbors
    neighbor_count <- as.numeric(W_mean %*% valid) * n_neighbors
    # But W_mean is row-normalized by n_neighbors, so W_mean %*% x = sum(x[nb])/n_nb
    # Therefore: W_mean %*% vals_no_na * n_neighbors = sum of neighbor vals (replacing NA with 0)
    # and W_mean %*% valid * n_neighbors = count of non-NA neighbors
    
    n_mean <- ifelse(neighbor_count > 0,
                     neighbor_sum / neighbor_count,
                     NA_real_)
    
    # ------ Neighbor MAX and MIN via nb_list (vectorized per cell) ------
    # This is the part that still loops, but only 344K iterations, not 6.46M
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
    
    for (i in seq_len(n_cells)) {
      nb_idx <- nb_list[[i]]
      if (length(nb_idx) == 0L) next
      nb_vals <- vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      n_max[i] <- max(nb_vals)
      n_min[i] <- min(nb_vals)
    }
    
    # ------ Write results back, aligned to the year-slice rows ------
    # yr_dt$pos tells us which position in id_order each year-row maps to
    set(dt, i = yr_rows, j = col_max,  value = n_max[yr_dt$pos])
    set(dt, i = yr_rows, j = col_min,  value = n_min[yr_dt$pos])
    set(dt, i = yr_rows, j = col_mean, value = n_mean[yr_dt$pos])
  }
  
  return(dt)
}

# ==============================================================================
# STEP 2b: Faster max/min using data.table edge-list approach
#           (avoids the 344K R-level loop entirely)
# ==============================================================================

compute_neighbor_features_fully_vectorized <- function(dt, var_name, static) {
  nb_list     <- static$nb_list
  n_cells     <- static$n_cells
  id_order    <- static$id_order
  id_to_pos   <- static$id_to_pos
  W_mean      <- static$W_mean
  n_neighbors <- static$n_neighbors
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # --- Build a static edge list (from_pos, to_pos) ONCE ---
  from_pos <- rep(seq_len(n_cells), lengths(nb_list))
  to_pos   <- unlist(nb_list, use.names = FALSE)
  edges    <- data.table(from_pos = from_pos, to_pos = to_pos)
  
  # --- Ensure dt has a pos column mapping id -> position in id_order ---
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  years <- sort(unique(dt$year))
  
  # Pre-allocate
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  for (yr in years) {
    yr_idx <- which(dt$year == yr)
    
    # Value vector aligned to id_order positions
    vals <- rep(NA_real_, n_cells)
    vals[dt$cell_pos[yr_idx]] <- dt[[var_name]][yr_idx]
    
    # --- Neighbor stats via edge list ---
    edge_yr <- copy(edges)
    edge_yr[, nb_val := vals[to_pos]]
    
    # Remove NA neighbor values
    edge_yr <- edge_yr[!is.na(nb_val)]
    
    stats <- edge_yr[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = from_pos]
    
    # Map back: stats$from_pos -> position in id_order
    result_max  <- rep(NA_real_, n_cells)
    result_min  <- rep(NA_real_, n_cells)
    result_mean <- rep(NA_real_, n_cells)
    
    result_max[stats$from_pos]  <- stats$nb_max
    result_min[stats$from_pos]  <- stats$nb_min
    result_mean[stats$from_pos] <- stats$nb_mean
    
    # Write to dt
    set(dt, i = yr_idx, j = col_max,  value = result_max[dt$cell_pos[yr_idx]])
    set(dt, i = yr_idx, j = col_min,  value = result_min[dt$cell_pos[yr_idx]])
    set(dt, i = yr_idx, j = col_mean, value = result_mean[dt$cell_pos[yr_idx]])
  }
  
  dt[, cell_pos := NULL]
  return(dt)
}

# ==============================================================================
# STEP 3: Main pipeline (drop-in replacement for outer loop)
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  message("Building static neighbor structures (once)...")
  static <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
  
  # Convert to data.table for speed (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Sort by (id, year) for consistent position mapping
  setkey(cell_data, id, year)
  
  # Build the edge list once (used inside the function but we can also
  # pre-build it and pass it in)
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor features for: %s", var_name))
    cell_data <- compute_neighbor_features_fully_vectorized(
      cell_data, var_name, static
    )
  }
  
  message("Done. Neighbor features added.")
  return(cell_data)
}

# ==============================================================================
# USAGE (drop-in replacement):
# ==============================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with the pre-trained Random Forest prediction:
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Expected Improvement

| Metric | Original | Redesigned |
|---|---|---|
| Neighbor lookup entries | 6.46M (with string ops) | 344K (integer, built once) |
| String paste + named-vector lookups | ~6.46M Ã— avg 4 neighbors = ~25M | **Zero** |
| Stats loop iterations per variable | 6.46M | 28 Ã— one vectorized `data.table` grouped aggregation on ~1.37M edge rows |
| Total wall time (estimated) | 86+ hours | **~5â€“15 minutes** |
| Numerical output | Max, min, mean per neighbor set | **Identical** (same estimand preserved) |
| Random Forest model | Unchanged | **Unchanged** (no retraining) |
| Peak RAM | Similar | Similar (~16 GB feasible; edge list is ~22M rows of 2 ints + 1 double â‰ˆ 350 MB) |

The key insight: **factor the static topology from the dynamic values.** Build the neighbor graph once as an integer edge list, then for each (variable Ã— year) slice, join neighbor values via `data.table` grouped aggregation â€” fully vectorized in C, no R-level loops, no string operations.
END_RESPONSE>>>
