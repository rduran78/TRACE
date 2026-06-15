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
run_id: Anthropic_F05_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

Inside `build_neighbor_lookup`, for **each** of the ~6.46 million rows, the code:

1. **Converts an id to a character and looks it up** in `id_to_ref` â€” minor cost per row, but 6.46M times.
2. **Pastes neighbor cell IDs with the current year** to form string keys â€” this is the dominant per-row cost. Each row has ~8 rook neighbors on average (4 cardinal directions, but directed relationships give ~4 per cell). That's ~6.46M Ã— 4 `paste()` calls = ~25.8M string concatenations.
3. **Looks up those string keys in a named vector** (`idx_lookup`) â€” named-vector lookup in R is hash-based but still involves per-call overhead across 25.8M lookups.

But the deeper structural inefficiency is this: **the neighbor topology is time-invariant**. Every cell has the same neighbors in every year. The `build_neighbor_lookup` function rebuilds what is essentially a spatial relationship for every cell-year combination, when it only needs to be computed once per cell and then broadcast across years.

Furthermore, `compute_neighbor_stats` is called **5 separate times** (once per variable), each time iterating over all 6.46M rows. With vectorized operations, all 5 variables can be processed simultaneously.

### Quantifying the Waste

| Operation | Current Cost | Necessary Cost |
|---|---|---|
| String key construction | ~6.46M `paste` calls for `idx_lookup` + ~25.8M for neighbor keys | **Zero** (use integer indexing) |
| Neighbor resolution | ~6.46M `lapply` iterations with hash lookups | **344,208** cell-level lookups (time-invariant) |
| Stat computation | 5 Ã— 6.46M `lapply` iterations | **One** vectorized pass |

Estimated speedup: **~500Ã—â€“1000Ã—**, bringing the runtime from 86+ hours to **~5â€“15 minutes**.

---

## Optimization Strategy

### 1. Separate Space from Time
The neighbor structure is purely spatial. Build a mapping from each **cell** (not cell-year) to its neighbor **cells** once.

### 2. Use Integer Indexing Instead of String Keys
Create a 2D index: `(cell_position, year_position) â†’ row_number`. This is an integer matrix lookup â€” orders of magnitude faster than string hashing.

### 3. Vectorize Neighbor Stat Computation
Instead of `lapply` over 6.46M rows, use a sparse-matrix or long-table approach:
- Expand the neighbor list into an edge table `(row_i, neighbor_row_j)`.
- Extract all neighbor values at once via vectorized subsetting.
- Compute grouped `max`, `min`, `mean` via `data.table` grouping.

### 4. Process All Variables in One Pass
The edge table is the same for all variables. Gather all 5 variables into the grouped computation simultaneously.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a fast integer row-index matrix
#         Rows = cells (in id_order), Cols = years
#         Cell (c, y) -> row number in cell_data
# ==============================================================

build_row_index_matrix <- function(data, id_order, years) {
  # data must have columns: id, year
  # Returns a matrix: n_cells x n_years, containing row indices into data
  
  dt <- as.data.table(data)[, .(id, year, row_idx = .I)]
  
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Map cell id -> cell position (1..n_cells)
  cell_pos <- setNames(seq_along(id_order), as.character(id_order))
  # Map year -> year position (1..n_years)
  year_pos <- setNames(seq_along(years), as.character(years))
  
  dt[, cell_p := cell_pos[as.character(id)]]
  dt[, year_p := year_pos[as.character(year)]]
  
  mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_p, dt$year_p)] <- dt$row_idx
  
  list(matrix = mat, cell_pos = cell_pos, year_pos = year_pos)
}

# ==============================================================
# STEP 2: Build the edge table (row_i, neighbor_row_j) for ALL
#         cell-year rows, using only integer arithmetic.
# ==============================================================

build_edge_table <- function(row_index_mat, neighbors, years) {
  # neighbors: spdep nb object, indexed by cell position in id_order
  # row_index_mat: matrix from Step 1
  
  n_cells <- nrow(row_index_mat)
  n_years <- ncol(row_index_mat)
  
  # Pre-compute total edges for memory pre-allocation
  n_neighbors_per_cell <- vapply(neighbors, length, integer(1))
  total_edges <- sum(as.numeric(n_neighbors_per_cell)) * n_years
  
  # Build cell-level edge list: (focal_cell_pos, neighbor_cell_pos)
  focal_cell <- rep(seq_len(n_cells), times = n_neighbors_per_cell)
  neighbor_cell <- unlist(neighbors)
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(neighbor_cell) & neighbor_cell > 0
  focal_cell <- focal_cell[valid]
  neighbor_cell <- neighbor_cell[valid]
  n_spatial_edges <- length(focal_cell)
  
  # Expand across all years
  # For each year y, focal_row = row_index_mat[focal_cell, y]
  #                  neighbor_row = row_index_mat[neighbor_cell, y]
  
  focal_rows <- integer(n_spatial_edges * n_years)
  neighbor_rows <- integer(n_spatial_edges * n_years)
  
  for (y in seq_len(n_years)) {
    offset <- (y - 1L) * n_spatial_edges
    idx_range <- (offset + 1L):(offset + n_spatial_edges)
    focal_rows[idx_range] <- row_index_mat[focal_cell, y]
    neighbor_rows[idx_range] <- row_index_mat[neighbor_cell, y]
  }
  
  # Remove any NA pairs (cells not present in certain years)
  valid2 <- !is.na(focal_rows) & !is.na(neighbor_rows)
  
  data.table(
    focal_row = focal_rows[valid2],
    neighbor_row = neighbor_rows[valid2]
  )
}

# ==============================================================
# STEP 3: Compute neighbor stats for all variables at once
#         using vectorized data.table grouped operations.
# ==============================================================

compute_all_neighbor_stats <- function(data, edge_dt, var_names) {
  # edge_dt: data.table with (focal_row, neighbor_row)
  # var_names: character vector of column names
  # Returns a data.table with columns:
  #   {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
  #   for each var in var_names, with nrow = nrow(data)
  
  dt <- as.data.table(data)
  n <- nrow(dt)
  
  # Extract neighbor values for all variables at once
  # Build a sub-table of neighbor values keyed by focal_row
  neighbor_vals <- dt[edge_dt$neighbor_row, ..var_names]
  neighbor_vals[, focal_row := edge_dt$focal_row]
  
  # Compute grouped stats
  stats <- neighbor_vals[,
    lapply(.SD, function(x) {
      x <- x[!is.na(x)]
      if (length(x) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(x), min(x), mean(x))
    }),
    by = focal_row,
    .SDcols = var_names
  ]
  
  # The above returns 3 rows per focal_row (max, min, mean stacked).
  # We need a different approach for proper column separation.
  # Use explicit aggregation instead:
  
  agg_exprs <- list()
  for (v in var_names) {
    agg_exprs[[paste0(v, "_neighbor_max")]]  <- 
      substitute(max_narm(x), list(x = as.name(v)))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <- 
      substitute(min_narm(x), list(x = as.name(v)))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <- 
      substitute(mean_narm(x), list(x = as.name(v)))
  }
  
  # Helper functions that return NA for empty/all-NA inputs
  max_narm  <- function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x) }
  min_narm  <- function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x) }
  mean_narm <- function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x) }
  
  # Cleaner approach: melt, aggregate, dcast
  # But for 5 variables this direct approach is efficient:
  
  result_dt <- data.table(row_idx = seq_len(n))
  
  for (v in var_names) {
    message("  Computing neighbor stats for: ", v)
    # Attach neighbor values to edge table
    edge_v <- data.table(
      focal_row = edge_dt$focal_row,
      val = dt[[v]][edge_dt$neighbor_row]
    )
    # Remove NA neighbor values
    edge_v <- edge_v[!is.na(val)]
    
    # Aggregate
    if (nrow(edge_v) > 0) {
      agg <- edge_v[, .(
        v_max  = max(val),
        v_min  = min(val),
        v_mean = mean(val)
      ), by = focal_row]
      
      # Allocate NA columns, then fill
      result_dt[, (paste0(v, "_neighbor_max"))  := NA_real_]
      result_dt[, (paste0(v, "_neighbor_min"))  := NA_real_]
      result_dt[, (paste0(v, "_neighbor_mean")) := NA_real_]
      
      result_dt[agg$focal_row, (paste0(v, "_neighbor_max"))  := agg$v_max]
      result_dt[agg$focal_row, (paste0(v, "_neighbor_min"))  := agg$v_min]
      result_dt[agg$focal_row, (paste0(v, "_neighbor_mean")) := agg$v_mean]
    }
  }
  
  result_dt[, row_idx := NULL]
  result_dt
}

# ==============================================================
# MAIN: Drop-in replacement for the outer loop
# ==============================================================

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  message("Step 1/3: Building row-index matrix...")
  years <- sort(unique(cell_data$year))
  ri <- build_row_index_matrix(cell_data, id_order, years)
  
  message("Step 2/3: Building edge table (~",
          length(id_order), " cells x ", length(years), " years)...")
  edge_dt <- build_edge_table(ri$matrix, rook_neighbors_unique, years)
  message("  Edge table: ", format(nrow(edge_dt), big.mark = ","), " directed edges")
  
  message("Step 3/3: Computing neighbor statistics for all variables...")
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  stats_dt <- compute_all_neighbor_stats(cell_data, edge_dt, neighbor_source_vars)
  
  # Bind new columns to cell_data
  # (preserves original column order; new columns appended)
  for (col_name in names(stats_dt)) {
    cell_data[[col_name]] <- stats_dt[[col_name]]
  }
  
  message("Done. Added ", ncol(stats_dt), " neighbor feature columns.")
  cell_data
}

# ==============================================================
# USAGE (replaces the entire original outer loop):
# ==============================================================

# cell_data <- optimize_neighbor_features(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The trained Random Forest model is untouched.
# # Predictions proceed as before:
# # preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` computed over the identical set of neighbor values per cell-year row. The edge table encodes exactly the same `(focal, neighbor, year)` triples as the original string-key approach. |
| **Trained RF model untouched** | No model retraining. The code only reconstructs the same feature columns with the same names (`{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean`). |
| **Column names** | Adjust the `paste0` patterns if your original `compute_and_add_neighbor_features` used different naming conventions (e.g., `ntl_max_neighbor`). |
| **NA handling** | Identical: neighbors not present in a year â†’ excluded; all-NA neighbor sets â†’ `NA` output. |

## Expected Performance

| Phase | Estimated Time | Peak RAM |
|---|---|---|
| Row-index matrix | ~5 seconds | ~50 MB (344K Ã— 28 int matrix) |
| Edge table construction | ~30 seconds | ~600 MB (two int vectors, ~38.5M edges) |
| Neighbor stats (5 vars) | ~3â€“8 minutes | ~1.5 GB peak (edge table + one value column at a time) |
| **Total** | **~5â€“10 minutes** | **< 4 GB** |

This is a **~500â€“1000Ã— speedup** over the original 86+ hour estimate, achieved by eliminating string operations entirely, exploiting the time-invariance of spatial topology, and replacing row-level `lapply` with vectorized `data.table` grouped aggregation.
END_RESPONSE>>>
