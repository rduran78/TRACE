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
run_id: Anthropic_F08_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) relationships.** It creates a list of 6.46 million entries, one per cell-year row. But rook-neighbor relationships are purely spatial â€” they don't change across years. There are only 344,208 unique cells. The function is doing 28Ã— redundant work by re-resolving the same neighbor topology for every year.

2. **String-key hashing is expensive.** The function constructs `paste(id, year)` keys for all 6.46M rows, builds a named lookup vector, and then for each row, constructs neighbor keys by pasting neighbor IDs with the current year. This involves ~6.46M string concatenations for the index, plus ~6.46M Ã— avg_neighbors more concatenations during lookup â€” hundreds of millions of string operations.

3. **`compute_neighbor_stats` iterates row-by-row over 6.46M rows** using `lapply`, calling `max`, `min`, `mean` individually per row. This is done 5 times (once per variable), totaling ~32.3 million R-level function calls with per-element subsetting.

4. **The neighbor lookup list itself consumes massive memory** â€” 6.46M list entries of integer vectors, with overhead per element.

### Key Insight

The neighbor graph is **static** (cell-to-cell, year-invariant). The variable values are **dynamic** (change by year). The correct design is:

- Build the neighbor lookup **once** over 344,208 cells (not 6.46M cell-years).
- For each variable, extract values **per year**, apply the cell-level neighbor lookup to compute stats, then write results back.

This reduces the lookup construction from O(6.46M) to O(344K) and makes the stats computation naturally vectorizable.

---

## Optimization Strategy

### 1. Separate Static Topology from Dynamic Data

Build a **cell-level** neighbor index once (344K entries instead of 6.46M). This is just a direct reformatting of `rook_neighbors_unique` â€” it's already an `nb` object indexed by cell position.

### 2. Vectorized Year-Sliced Computation

For each year:
- Extract the column vector of values for that variable (344K values, one per cell, in cell-order).
- Use the cell-level neighbor index to compute max/min/mean via vectorized operations.

### 3. Use Matrix Operations or data.table for Speed

Instead of `lapply` over 344K cells per year, use a **sparse adjacency approach** or a pre-flattened index with `vapply`/C-level grouping. A sparse matrix multiply can compute neighbor means in one shot; max and min require a grouped approach but can be done efficiently with pre-built flat indices.

### 4. Estimated Speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Lookup construction | 6.46M string-key entries | 344K integer entries (reuse `nb` directly) | ~19Ã— |
| Stats computation | 6.46M Ã— 5 vars = 32.3M R calls | 344K Ã— 28 years Ã— 5 vars, vectorized | ~50-200Ã— |
| Memory | ~6.46M list elements + string keys | ~344K list + dense year-vectors | ~10Ã— less |

**Expected total runtime: 1â€“5 minutes** instead of 86+ hours.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE: Separate static topology from dynamic (yearly) variables
# ==============================================================================
#
# Assumptions carried forward:
#   - cell_data is a data.frame/data.table with columns: id, year, ntl, ec,
#     pop_density, def, usd_est_n2, and ~110 predictor columns.
#   - id_order is a vector of cell IDs in the order matching rook_neighbors_unique.
#   - rook_neighbors_unique is an spdep::nb object (list of length 344,208),
#     where each element is an integer vector of neighbor *positions* into id_order.
#   - cell_data is sorted by (id, year) or at minimum has consistent ordering.
#   - The pre-trained Random Forest model (rf_model) is loaded and untouched.
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table if not already (for performance)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --------------------------------------------------------------------------
# STEP 1: Build the STATIC cell-level neighbor index (done ONCE)
#
# rook_neighbors_unique is already an nb object indexed by position in id_order.
# Each element rook_neighbors_unique[[i]] gives the positional indices of
# neighbors of id_order[i]. We just need to map cell IDs to positions.
#
# We also pre-build "flat" index vectors for fast grouped operations.
# --------------------------------------------------------------------------

build_cell_neighbor_flat_index <- function(nb_obj) {
  # nb_obj: list of length N_cells, each element is integer vector of neighbor

  # positions (0 means no neighbors in spdep convention; we handle that).
  
  n_cells <- length(nb_obj)
  
  # Count neighbors per cell
  n_neighbors <- vapply(nb_obj, function(x) {
    x <- x[x > 0L]  # spdep uses 0 for "no neighbors"
    length(x)
  }, integer(1))
  
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate flat vectors
  # cell_idx: which cell "owns" this neighbor entry (repeated for each neighbor)
  # neighbor_idx: the positional index of the neighbor cell
  cell_idx     <- integer(total_edges)
  neighbor_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    k <- length(nbrs)
    if (k > 0L) {
      cell_idx[pos:(pos + k - 1L)]     <- i
      neighbor_idx[pos:(pos + k - 1L)] <- nbrs
      pos <- pos + k
    }
  }
  
  list(
    cell_idx     = cell_idx,
    neighbor_idx = neighbor_idx,
    n_neighbors  = n_neighbors,
    n_cells      = n_cells
  )
}

cat("Building static cell-level neighbor flat index...\n")
flat_nb <- build_cell_neighbor_flat_index(rook_neighbors_unique)
cat(sprintf("  %d cells, %d directed neighbor edges\n",
            flat_nb$n_cells, length(flat_nb$cell_idx)))

# --------------------------------------------------------------------------
# STEP 2: Ensure cell_data has a cell-position column for fast indexing
# --------------------------------------------------------------------------

# Map each cell ID to its position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify all cells are mapped
stopifnot(!anyNA(cell_data$cell_pos))

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_cells <- flat_nb$n_cells

cat(sprintf("Processing %d years Ã— %d cells = %d cell-years\n",
            length(years), n_cells, nrow(cell_data)))

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor stats efficiently (year-by-year, vectorized)
#
# For each variable and each year:
#   1. Extract a dense vector of values indexed by cell_pos (length = n_cells).
#   2. Look up neighbor values using the flat index.
#   3. Compute grouped max, min, mean using data.table's fast grouping.
#   4. Write results back into cell_data.
# --------------------------------------------------------------------------

compute_neighbor_stats_optimized <- function(cell_data, flat_nb, var_name, years, n_cells) {
  
  cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns with NA
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
  
  # Key cell_data by year and cell_pos for fast subsetting
  # We'll iterate by year
  
  cell_idx     <- flat_nb$cell_idx
  neighbor_idx <- flat_nb$neighbor_idx
  n_neighbors  <- flat_nb$n_neighbors
  
  for (yr in years) {
    # Get row indices in cell_data for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Build a dense vector: values_by_pos[cell_pos] = value
    # This assumes each cell appears exactly once per year
    yr_subset <- cell_data[yr_rows, .(cell_pos, val = get(var_name))]
    
    values_by_pos <- rep(NA_real_, n_cells)
    values_by_pos[yr_subset$cell_pos] <- yr_subset$val
    
    # Look up neighbor values using flat index
    neighbor_vals <- values_by_pos[neighbor_idx]
    
    # Compute grouped stats using data.table (very fast C-level grouping)
    # cell_idx tells us which cell each neighbor_val belongs to
    edge_dt <- data.table(
      cell = cell_idx,
      val  = neighbor_vals
    )
    
    # Remove NA values before aggregation
    edge_dt <- edge_dt[!is.na(val)]
    
    # Compute max, min, mean per cell in one pass
    stats <- edge_dt[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = cell]
    
    # Map results back: stats$cell is cell_pos
    # We need to find the row in cell_data for this year and this cell_pos
    # Build a mapping from cell_pos to yr_rows index
    yr_cell_pos <- cell_data$cell_pos[yr_rows]
    
    # Create a pos-to-row-index lookup (dense, since cell_pos âˆˆ 1:n_cells)
    pos_to_yr_row <- rep(NA_integer_, n_cells)
    pos_to_yr_row[yr_cell_pos] <- yr_rows
    
    # Write results
    matched_rows <- pos_to_yr_row[stats$cell]
    valid <- !is.na(matched_rows)
    
    if (any(valid)) {
      set(cell_data, i = matched_rows[valid], j = col_max,  value = stats$nb_max[valid])
      set(cell_data, i = matched_rows[valid], j = col_min,  value = stats$nb_min[valid])
      set(cell_data, i = matched_rows[valid], j = col_mean, value = stats$nb_mean[valid])
    }
  }
  
  return(cell_data)
}

# --------------------------------------------------------------------------
# STEP 4: Run the optimized pipeline
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (optimized)...\n")
t_start <- Sys.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_optimized(
    cell_data, flat_nb, var_name, years, n_cells
  )
}

t_end <- Sys.time()
cat(sprintf("Neighbor feature computation completed in %.1f minutes\n",
            as.numeric(difftime(t_end, t_start, units = "mins"))))

# Clean up helper column
cell_data[, cell_pos := NULL]

# --------------------------------------------------------------------------
# STEP 5: Predict using the pre-trained Random Forest (UNCHANGED)
#
# The RF model is already trained. We only feed it the updated cell_data
# with the same feature columns it expects. The neighbor_max_*, neighbor_min_*,
# neighbor_mean_* columns are numerically identical to the original pipeline's
# output, preserving the original numerical estimand.
# --------------------------------------------------------------------------

cat("Running pre-trained Random Forest predictions...\n")

# Example (adjust to your actual model object and prediction call):
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data[, prediction := predictions]

cat("Pipeline complete.\n")
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup scope** | 6.46M cell-year rows | 344K cells (static, built once) |
| **Lookup data structure** | Named character vector + paste keys | Integer flat index (cell_idx, neighbor_idx) |
| **Stats computation** | `lapply` over 6.46M rows per variable | `data.table` grouped aggregation over ~1.37M edges per year |
| **Total iterations** | ~32.3M R-level function calls | 28 years Ã— 5 vars = 140 vectorized passes |
| **Estimated runtime** | 86+ hours | **1â€“5 minutes** |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** â€” same max, min, mean of same neighbor sets |
| **RF model** | Pre-trained, not retrained | **Preserved exactly** â€” only `predict()` is called |
| **Memory** | ~6.46M list entries + string keys | ~1.37M Ã— 2 integer vectors + 344K dense vector per year-pass |

The optimized code produces **exactly the same `neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*` columns** as the original, ensuring the pre-trained Random Forest receives identical input features and the original numerical estimand is preserved.
END_RESPONSE>>>
