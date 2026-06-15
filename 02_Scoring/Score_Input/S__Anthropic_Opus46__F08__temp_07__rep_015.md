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
run_id: Anthropic_F08_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup (one entry per cell-year row, ~6.46 million entries), even though the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs string matching via `paste()` and named-vector lookups millions of times, and produces a list of ~6.46 million integer vectors. This is astronomically wasteful because:

1. **Redundant topology replication**: The rook-neighbor graph has ~344K cells. The neighbor relationships are the same every year. Yet the lookup is built for every cellÃ—year combination (~6.46M entries), repeating the same neighbor-cell-ID resolution 28 times per cell.

2. **String-key bottleneck**: `paste(id, year, sep="_")` and named-vector indexing on ~6.46M string keys is extremely slow in R â€” both in memory (storing millions of character strings) and CPU (hashing/matching).

3. **`compute_neighbor_stats` iterates row-by-row**: An `lapply` over 6.46M rows, each subsetting a numeric vector and computing `max/min/mean`, is slow in interpreted R.

4. **The outer loop repeats this for 5 variables**, so the total cost is 5 Ã— 6.46M row-level operations on top of the already-expensive lookup construction.

**Estimated cost breakdown**: Building the 6.46M-entry lookup list with string operations dominates (~hours). Then 5 Ã— 6.46M stat computations add more hours. Total: 86+ hours.

## Optimization Strategy

**Key insight**: Separate the *static topology* (which cells are neighbors of which) from the *dynamic values* (year-varying variable columns).

### Step 1: Build a cell-level neighbor index once (344K entries, not 6.46M)

Build a simple list of length 344,208 where entry `i` contains the integer positions (in the cell-ID ordering) of cell `i`'s rook neighbors. This is done once and reused for all years and all variables. No string operations needed.

### Step 2: Vectorized year-sliced computation

For each year, extract the rows for that year (ordered by cell ID), then for each variable compute neighbor stats using the cell-level neighbor index. Because all cells share the same ordering within each year-slice, the cell-level neighbor index directly maps to row positions within the slice.

### Step 3: Use matrix operations or data.table for speed

Instead of `lapply` over millions of rows, use a **sparse-matrix multiplication** approach or a **pre-allocated matrix + vectorized C-level operations** via `data.table` or direct indexing.

### Complexity reduction

| Aspect | Before | After |
|---|---|---|
| Lookup entries | 6.46M | 344K |
| String key ops | ~12.9M paste + match | 0 |
| Stat computations | 5 Ã— 6.46M | 5 Ã— 28 Ã— 344K (same total rows, but vectorized) |
| Expected runtime | 86+ hours | **~5â€“15 minutes** |

The numerical results are **identical** â€” same max, min, mean of the same neighbor values â€” just computed via a different indexing path. The trained Random Forest model is untouched.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2 (and ~110 other predictors)
#   - id_order: vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique: spdep::nb object (list of length = length(id_order))
#   - rf_model: pre-trained Random Forest model (unchanged)
# =============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 1: Build a CELL-LEVEL neighbor lookup (done ONCE, topology is static)
# --------------------------------------------------------------------------
# rook_neighbors_unique[[i]] gives the neighbor indices (into id_order) for 
# cell id_order[i]. We keep this as-is â€” it's already what we need.
# We just ensure it's a clean integer list.

build_cell_neighbor_lookup <- function(neighbors) {
  # neighbors is an spdep::nb object: list of integer vectors

  # Each element i contains indices (into the same list) of neighbors of cell i
  # Remove the nb class attributes for clean processing
  n <- length(neighbors)
  cell_neighbors <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[0] == 0L) {
      cell_neighbors[[i]] <- integer(0)
    } else {
      cell_neighbors[[i]] <- as.integer(nb_i)
    }
  }
  cell_neighbors
}

cell_neighbors <- build_cell_neighbor_lookup(rook_neighbors_unique)
n_cells <- length(id_order)

# --------------------------------------------------------------------------
# STEP 2: Pre-build a "flat" neighbor structure for vectorized computation
# --------------------------------------------------------------------------
# For each cell i, we know its neighbors. We flatten this into two vectors:
#   - cell_idx: the cell index (repeated for each neighbor it has)
#   - neighbor_idx: the neighbor's cell index
# This lets us do fully vectorized lookups.

flat_cell_idx <- integer(0)
flat_nbr_idx  <- integer(0)

# Pre-allocate by counting total edges
total_edges <- sum(vapply(cell_neighbors, length, integer(1)))
flat_cell_idx <- integer(total_edges)
flat_nbr_idx  <- integer(total_edges)

pos <- 1L
for (i in seq_len(n_cells)) {
  nb_i <- cell_neighbors[[i]]
  len_i <- length(nb_i)
  if (len_i > 0L) {
    flat_cell_idx[pos:(pos + len_i - 1L)] <- i
    flat_nbr_idx[pos:(pos + len_i - 1L)]  <- nb_i
    pos <- pos + len_i
  }
}

# Number of neighbors per cell (for computing means)
n_neighbors <- tabulate(flat_cell_idx, nbins = n_cells)

cat(sprintf("Static topology: %d cells, %d directed edges\n", n_cells, total_edges))

# --------------------------------------------------------------------------
# STEP 3: Convert cell_data to data.table and ensure consistent ordering
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Create a mapping from cell id to cell index (position in id_order)
id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))

# Add cell index column
cell_data[, cell_idx := id_to_cellidx[as.character(id)]]

# Sort by year and cell_idx so that within each year, rows are in cell_idx order
setkey(cell_data, year, cell_idx)

# Verify all cells are present in each year (panel is balanced)
years <- sort(unique(cell_data$year))
n_years <- length(years)
stopifnot(nrow(cell_data) == n_cells * n_years)

# --------------------------------------------------------------------------
# STEP 4: Compute neighbor stats for all variables â€” vectorized by year
# --------------------------------------------------------------------------
# Within each year-slice (after sorting by cell_idx), row j corresponds to
# cell_idx j. So we can directly use flat_cell_idx and flat_nbr_idx as 
# row indices within the year-slice.

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

# Process year by year
for (yr in years) {
  # Row range for this year (data is keyed by year, cell_idx)
  # Since panel is balanced and sorted, rows for year yr are:
  yr_idx_in_dt <- which(cell_data$year == yr)
  
  # Sanity check: should be exactly n_cells rows, in cell_idx order 1..n_cells
  stopifnot(length(yr_idx_in_dt) == n_cells)
  
  # The offset: first row of this year in cell_data
  offset <- yr_idx_in_dt[1L] - 1L
  
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Extract the variable values for this year, ordered by cell_idx
    vals <- cell_data[[var_name]][yr_idx_in_dt]  # length = n_cells
    
    # Get neighbor values using the flat index
    # vals[flat_nbr_idx] gives the value of the neighbor for each edge
    nbr_vals <- vals[flat_nbr_idx]  # length = total_edges
    
    # Handle NAs: we need na.rm behavior
    # For max and min, we use a grouping approach
    # For mean, sum(non-NA) / count(non-NA)
    
    not_na <- !is.na(nbr_vals)
    
    # Initialize result vectors
    res_max  <- rep(NA_real_, n_cells)
    res_min  <- rep(NA_real_, n_cells)
    res_sum  <- rep(0, n_cells)
    res_cnt  <- rep(0L, n_cells)
    
    # Only process non-NA edges
    if (any(not_na)) {
      valid_cell <- flat_cell_idx[not_na]
      valid_vals <- nbr_vals[not_na]
      
      # Use data.table for fast grouped aggregation
      edge_dt <- data.table(cell = valid_cell, val = valid_vals)
      agg <- edge_dt[, .(
        vmax = max(val),
        vmin = min(val),
        vsum = sum(val),
        vcnt = .N
      ), by = cell]
      
      res_max[agg$cell]  <- agg$vmax
      res_min[agg$cell]  <- agg$vmin
      res_sum[agg$cell]  <- agg$vsum
      res_cnt[agg$cell]  <- agg$vcnt
    }
    
    res_mean <- ifelse(res_cnt > 0L, res_sum / res_cnt, NA_real_)
    
    # Write back into cell_data at the correct rows
    set(cell_data, i = yr_idx_in_dt, j = col_max,  value = res_max)
    set(cell_data, i = yr_idx_in_dt, j = col_min,  value = res_min)
    set(cell_data, i = yr_idx_in_dt, j = col_mean, value = res_mean)
  }
  
  cat(sprintf("  Year %d done\n", yr))
}

# Clean up temporary column
cell_data[, cell_idx := NULL]

cat("Neighbor feature computation complete.\n")

# --------------------------------------------------------------------------
# STEP 5: Prediction with the pre-trained Random Forest (UNCHANGED)
# --------------------------------------------------------------------------
# The cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original pipeline, with identical numerical values.
# Proceed with prediction exactly as before:

# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Property | Original | Optimized |
|---|---|---|
| Neighbor lookup granularity | Per cell-year (6.46M) | Per cell (344K) â€” static topology |
| Key mechanism | String paste + named vector match | Integer index into sorted year-slices |
| Stats computation | `lapply` over 6.46M rows | Vectorized `data.table` grouped aggregation per year |
| Topology rebuilt? | Implicitly, every run | Once, reused across all years and variables |
| Output columns | `neighbor_{max,min,mean}_{var}` | Identical column names and values |
| Numerical equivalence | Baseline | Preserved â€” same `max`, `min`, `mean` with `na.rm` semantics |
| Random Forest model | Untouched | Untouched |
| Estimated runtime | 86+ hours | ~5â€“15 minutes |
| Peak memory | ~6.46M-entry list of int vectors + string keys | ~2 Ã— 1.37M integer vectors (flat edges) + year-slice vectors |

The fundamental redesign principle: **build the topology once at the cell level (static graph), then sweep the year-varying values through it**. This eliminates all redundant string operations and converts millions of interpreted-R list iterations into vectorized grouped aggregations.
END_RESPONSE>>>
