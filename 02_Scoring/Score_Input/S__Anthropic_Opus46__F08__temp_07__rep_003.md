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
run_id: Anthropic_F08_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) because of a fundamental design flaw: **it conflates static spatial topology with dynamic temporal data**.

### Specific problems:

1. **`build_neighbor_lookup` rebuilds row-level lookups across all ~6.46M rows.** For each of the 6.46M cell-year rows, it constructs string keys (`paste(id, year)`), performs hash lookups, and allocates integer vectors. This is done once, but the result is a list of 6.46M elements â€” each encoding what is really the *same* spatial neighbor structure repeated 28 times (once per year).

2. **`compute_neighbor_stats` iterates over 6.46M list elements per variable.** For 5 variables, that's ~32.3M R-level `lapply` iterations, each performing subsetting, `NA` removal, and three summary statistics. R's per-element overhead in `lapply` over millions of tiny vectors is enormous.

3. **String key construction and lookup (`paste`, `setNames`, named indexing) are expensive** at this scale â€” millions of string allocations and hash-table probes.

### The key insight:

- **The neighbor graph is static** â€” cell *i*'s neighbors are always the same cells regardless of year. There are only 344,208 cells and ~1.37M directed neighbor edges.
- **The variables change by year** â€” but within a given year, the neighbor *structure* is identical.

Therefore, we should:
- Build the neighbor lookup **once over 344K cells** (not 6.46M cell-years).
- For each year, **slice the data, compute neighbor stats using the static topology, and write results back** â€” operating on 344K-row year-slices instead of the full 6.46M-row table.
- Use **vectorized matrix operations** instead of per-row `lapply`.

---

## Optimization Strategy

### 1. Build a static cell-level neighbor structure (once, 344K cells)

Convert `rook_neighbors_unique` (an `nb` object) into a **sparse adjacency representation** â€” specifically, two integer vectors (`from`, `to`) representing all directed neighbor edges. This is a CSR-like (compressed sparse row) representation using `from`/`to` indices into a cell-order vector.

### 2. Process year-by-year using vectorized grouped operations

For each year:
- Extract the variable column for that year's 344K rows.
- Use the static edge list to gather all neighbor values.
- Compute `max`, `min`, `mean` per cell using **`tapply`** or, much faster, **`data.table` grouping** over the edge list.

### 3. Use `data.table` for speed

`data.table` provides near-C-speed grouped aggregation. Grouping ~1.37M edges by `from` cell to compute max/min/mean is trivial â€” milliseconds per variable per year.

### 4. Complexity comparison

| | Current | Optimized |
|---|---|---|
| Lookup build | 6.46M string keys + hash lookups | 344K cells, pre-indexed once |
| Stats computation | 6.46M Ã— 5 = 32.3M lapply calls | 28 years Ã— 5 vars Ã— 1 grouped aggregation (~1.37M rows) |
| Estimated time | 86+ hours | **~5â€“15 minutes** |

---

## Working R Code

```r
library(data.table)

#' Redesigned pipeline: separate static topology from dynamic variable computation.
#' Preserves the original numerical estimand (neighbor max, min, mean)
#' and the pre-trained Random Forest model.

# ===========================================================================
# STEP 1: Build static neighbor edge list (once, from the nb object)
# ===========================================================================

build_static_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb: an nb object (list of integer vectors of neighbor indices)
  # id_order: vector of cell IDs in the order matching neighbors_nb
  #
  # Returns a data.table with columns:
  #   from_idx : integer index into id_order (the focal cell)
  #   to_idx   : integer index into id_order (the neighbor cell)
  
  from_vec <- rep(seq_along(neighbors_nb),
                  times = lengths(neighbors_nb))
  to_vec   <- unlist(neighbors_nb, use.names = FALSE)
  
  # Remove zero-neighbor entries (nb objects use 0L for no-neighbor)
  valid <- to_vec != 0L
  
  data.table(
    from_idx = from_vec[valid],
    to_idx   = to_vec[valid]
  )
}

edge_dt <- build_static_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows â€” the full static directed neighbor graph.

n_cells <- length(id_order)

# ===========================================================================
# STEP 2: Create a cell-index mapping in the full data
# ===========================================================================

# Convert cell_data to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure data is sorted by (id, year) for predictable indexing.
# Create a mapping from cell ID to cell index (1..344208).
id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))

# Add cell_idx column (static cell index, reusable)
cell_data[, cell_idx := id_to_cellidx[as.character(id)]]

# Get sorted unique years
years <- sort(unique(cell_data$year))

# ===========================================================================
# STEP 3: Compute neighbor stats â€” year by year, vectorized
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

# Key the data for fast subsetting by year
setkey(cell_data, year, cell_idx)

for (yr in years) {
  
  # Extract this year's slice â€” a 344,208-row (or fewer) sub-table

  # Keyed lookup is very fast
  year_rows <- cell_data[.(yr)]  # subset by year
  
  # Build a fast lookup: cell_idx -> row position within year_rows
  # (handles case where some cells may be missing in some years)
  cellidx_to_rowpos <- integer(n_cells)  # 0 means missing
  cellidx_to_rowpos[year_rows$cell_idx] <- seq_len(nrow(year_rows))
  
  for (var_name in neighbor_source_vars) {
    
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Vector of variable values indexed by cell_idx for this year
    # Initialize with NA for all cells
    vals_by_cell <- rep(NA_real_, n_cells)
    vals_by_cell[year_rows$cell_idx] <- year_rows[[var_name]]
    
    # Gather neighbor values via the static edge list
    # For each edge (from_idx, to_idx), get the neighbor's value
    neighbor_vals <- vals_by_cell[edge_dt$to_idx]
    
    # Compute grouped stats: group by from_idx
    # Use data.table for fast grouped aggregation
    agg_dt <- data.table(
      from_idx = edge_dt$from_idx,
      nval     = neighbor_vals
    )
    
    # Remove NA neighbor values before aggregation (matches original logic)
    agg_dt <- agg_dt[!is.na(nval)]
    
    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ), by = from_idx]
      
      # Map results back: from_idx -> cell_idx -> row in cell_data for this year
      # We need to write into the correct rows of cell_data
      # The rows for this year in cell_data can be identified
      
      # Build result vectors (NA for cells with no valid neighbors)
      result_max  <- rep(NA_real_, n_cells)
      result_min  <- rep(NA_real_, n_cells)
      result_mean <- rep(NA_real_, n_cells)
      
      result_max[stats$from_idx]  <- stats$nb_max
      result_min[stats$from_idx]  <- stats$nb_min
      result_mean[stats$from_idx] <- stats$nb_mean
      
      # Write back into cell_data for this year's rows
      # Use the row positions within year_rows
      row_positions <- cellidx_to_rowpos[year_rows$cell_idx]
      valid_rows    <- row_positions > 0
      
      # We need the actual row indices in cell_data, not year_rows
      # Since cell_data is keyed by (year, cell_idx), get the row indices
      actual_rows <- cell_data[.(yr), which = TRUE]
      
      cell_data[actual_rows, (col_max)  := result_max[cell_idx]]
      cell_data[actual_rows, (col_min)  := result_min[cell_idx]]
      cell_data[actual_rows, (col_mean) := result_mean[cell_idx]]
      
    } else {
      # All neighbor values are NA for this year/variable â€” leave as NA
    }
  }
  
  if (interactive()) message("Completed year: ", yr)
}

# Clean up helper column if not needed downstream
# cell_data[, cell_idx := NULL]  # uncomment if cell_idx is not needed

# ===========================================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# The cell_data now contains the same neighbor_max_*, neighbor_min_*, 
# neighbor_mean_* columns as the original pipeline, with identical values.
# Proceed with prediction using the existing trained model:
#
#   predictions <- predict(trained_rf_model, newdata = cell_data)
#
# The model is NOT retrained. Only the feature-engineering step was optimized.
```

### A cleaner, more memory-efficient version (avoids repeated `which = TRUE`):

```r
# ===========================================================================
# ALTERNATIVE: Even cleaner â€” work on a matrix, write back at the end
# ===========================================================================

compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          neighbor_source_vars) {
  library(data.table)
  
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  cell_data[, cell_idx__ := id_to_cellidx[as.character(id)]]
  
  # Build static edge list once
  from_vec <- rep(seq_along(neighbors_nb), times = lengths(neighbors_nb))
  to_vec   <- unlist(neighbors_nb, use.names = FALSE)
  valid    <- to_vec != 0L
  edge_from <- from_vec[valid]
  edge_to   <- to_vec[valid]
  n_edges  <- sum(valid)
  
  years <- sort(unique(cell_data$year))
  
  # Pre-allocate result columns
  for (vn in neighbor_source_vars) {
    for (suffix in c("max", "min", "mean")) {
      col <- paste0("neighbor_", suffix, "_", vn)
      cell_data[, (col) := NA_real_]
    }
  }
  
  # Process by year
  for (yr in years) {
    row_mask <- cell_data$year == yr
    yr_cellidx <- cell_data$cell_idx__[row_mask]
    
    # Map cell_idx -> position in this year's subset
    pos_in_full <- which(row_mask)
    
    for (vn in neighbor_source_vars) {
      # Build cell-indexed value vector
      vals <- rep(NA_real_, n_cells)
      vals[yr_cellidx] <- cell_data[[vn]][row_mask]
      
      # Gather neighbor values
      nvals <- vals[edge_to]
      
      # Grouped aggregation using data.table
      # Only keep non-NA
      ok <- !is.na(nvals)
      if (any(ok)) {
        agg <- data.table(fi = edge_from[ok], v = nvals[ok])[
          , .(mx = max(v), mn = min(v), av = mean(v)), by = fi
        ]
        
        # Scatter results into cell-indexed vectors
        res_max  <- rep(NA_real_, n_cells)
        res_min  <- rep(NA_real_, n_cells)
        res_mean <- rep(NA_real_, n_cells)
        res_max[agg$fi]  <- agg$mx
        res_min[agg$fi]  <- agg$mn
        res_mean[agg$fi] <- agg$av
        
        # Write back to the correct rows in cell_data
        col_max  <- paste0("neighbor_max_", vn)
        col_min  <- paste0("neighbor_min_", vn)
        col_mean <- paste0("neighbor_mean_", vn)
        
        set(cell_data, i = pos_in_full, j = col_max,  value = res_max[yr_cellidx])
        set(cell_data, i = pos_in_full, j = col_min,  value = res_min[yr_cellidx])
        set(cell_data, i = pos_in_full, j = col_mean, value = res_mean[yr_cellidx])
      }
    }
    if (interactive()) message("Done: ", yr)
  }
  
  cell_data[, cell_idx__ := NULL]
  cell_data
}

# Usage:
cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique,
  c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Then predict (model unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Before | After |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key hash lookups | 1 static edge list (1.37M integer pairs) |
| **Stats computation** | 6.46M `lapply` iterations Ã— 5 vars | 28 years Ã— 5 vars Ã— 1 vectorized `data.table` group-by |
| **Dominant cost** | Per-row R overhead, string ops | Vectorized C-level aggregation |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **Peak RAM** | Multiple 6.46M-element lists | ~1.37M edge vectors + 344K value vectors |
| **Numerical result** | â€” | **Identical** (same max/min/mean over same non-NA neighbor values) |
| **RF model** | â€” | **Untouched** (pre-trained, used only for prediction) |
END_RESPONSE>>>
