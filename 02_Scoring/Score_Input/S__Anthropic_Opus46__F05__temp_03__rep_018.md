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
run_id: Anthropic_F05_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” repeated string allocation and concatenation.
2. **Looks up those keys** in `idx_lookup`, a named character vector of length 6.46M â€” each lookup is O(N) in the worst case because R named vector lookup uses linear hashing that degrades at this scale.

This means `build_neighbor_lookup` alone performs ~6.46M Ã— (avg ~8 neighbors) â‰ˆ **51.7 million string constructions and hash lookups against a 6.46M-entry table**. That's the 86+ hour bottleneck.

### Why It's a Broader Architectural Issue

The string-keyed approach **conflates two independent indexing dimensions** (cell ID and year) into a single flat string key. But the data has clean structure:

- Each cell appears exactly once per year.
- Neighbor relationships are **time-invariant** (rook contiguity doesn't change across years).
- The neighbor stats computation is **per-variable, per-row**, but the *neighbor row indices* are the same for all variables.

This means the entire lookup can be reformulated as an **integer-indexed matrix operation** that eliminates all string work entirely.

## Optimization Strategy

1. **Build a 2D integer index matrix** `row_index[cell_position, year_position]` â†’ row number in `data`. This is O(N) to build, O(1) to query.
2. **Resolve neighbor cell indices once** (time-invariant), producing an integer list of cell-position neighbors per cell.
3. **Vectorize the neighbor stats** using `data.table` or matrix operations â€” for each year, gather all neighbor values via integer indexing and compute stats in bulk.

This reduces the complexity from **O(N Ã— K Ã— string_hash_cost)** to **O(N Ã— K)** with pure integer indexing, where K is the average neighbor count.

**Expected speedup**: from 86+ hours to **minutes** (roughly 2â€“10 minutes depending on RAM pressure).

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build integer-indexed lookup structures
# ==============================================================

build_fast_neighbor_lookup <- function(data, id_order, neighbors) {
  # Convert data to data.table for fast operations (non-destructive)
  dt <- as.data.table(data)
  dt[, orig_row := .I]  # preserve original row index

  # --- Map cell IDs to contiguous integer positions ---
  # id_order is the canonical ordering matching the nb object
  n_cells <- length(id_order)
  id_to_pos <- integer(max(id_order))  # direct-address table

id_to_pos[id_order] <- seq_len(n_cells)
  # If IDs are not guaranteed to be small integers, use a hash:
  # id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  # and index with as.character(). But direct-address is faster if feasible.

  # --- Map years to contiguous integer positions ---
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_pos <- setNames(seq_along(years_sorted), as.character(years_sorted))

  # --- Build 2D index: row_index[cell_pos, year_pos] = row in data ---
  row_index <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  cell_positions <- id_to_pos[dt$id]
  year_positions <- year_to_pos[as.character(dt$year)]
  row_index[cbind(cell_positions, year_positions)] <- dt$orig_row

  # --- Convert nb neighbor list from cell IDs to cell positions ---
  # neighbors (nb object) stores neighbors as indices into id_order,

  # so neighbors[[i]] already gives positions into id_order.
  # We just need to confirm this and keep as-is.
  # Each element neighbors[[cell_pos]] = integer vector of neighbor cell positions.

  list(
    row_index    = row_index,
    neighbors    = neighbors,
    id_to_pos    = id_to_pos,
    year_to_pos  = year_to_pos,
    years_sorted = years_sorted,
    n_cells      = n_cells,
    n_years      = n_years
  )
}

# ==============================================================
# STEP 2: Compute neighbor stats for one variable (vectorized)
# ==============================================================

compute_neighbor_stats_fast <- function(data, var_name, lookup) {
  row_index  <- lookup$row_index
  neighbors  <- lookup$neighbors
  n_cells    <- lookup$n_cells
  n_years    <- lookup$n_years
  n_rows     <- nrow(data)

  vals <- data[[var_name]]

  # Pre-allocate output columns
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)

  # Process year-by-year to keep memory bounded

  for (yr_pos in seq_len(n_years)) {

    for (cell_pos in seq_len(n_cells)) {
      target_row <- row_index[cell_pos, yr_pos]
      if (is.na(target_row)) next

      nb_positions <- neighbors[[cell_pos]]
      if (length(nb_positions) == 0L) next

      # Integer-index directly into row_index to get neighbor rows
      nb_rows <- row_index[nb_positions, yr_pos]
      nb_rows <- nb_rows[!is.na(nb_rows)]
      if (length(nb_rows) == 0L) next

      nb_vals <- vals[nb_rows]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next

      out_max[target_row]  <- max(nb_vals)
      out_min[target_row]  <- min(nb_vals)
      out_mean[target_row] <- mean(nb_vals)
    }
  }

  cbind(out_max, out_min, out_mean)
}

# ==============================================================
# STEP 2b: FASTER fully-vectorized version (recommended)
# ==============================================================

compute_neighbor_stats_vectorized <- function(data, var_name, lookup) {
  row_index <- lookup$row_index   # [n_cells, n_years]
  neighbors <- lookup$neighbors   # list of length n_cells
  n_cells   <- lookup$n_cells
  n_years   <- lookup$n_years
  n_rows    <- nrow(data)

  vals <- data[[var_name]]

  # Pre-allocate
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)

  # --- Build a flat edge list (cell_pos -> neighbor_cell_pos) ---
  # This is time-invariant, so we expand over years below.
  from_cell <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_cell   <- unlist(neighbors, use.names = FALSE)

  # --- For each year, resolve rows and compute stats ---
  for (yr_pos in seq_len(n_years)) {
    # Row indices for all cells this year
    yr_rows <- row_index[, yr_pos]  # length n_cells; NA if cell absent

    # Target rows (the cell whose neighbors we're summarizing)
    target_rows <- yr_rows[from_cell]
    # Neighbor rows
    nb_rows     <- yr_rows[to_cell]

    # Filter to edges where both target and neighbor exist this year
    valid <- !is.na(target_rows) & !is.na(nb_rows)
    if (!any(valid)) next

    t_rows_v  <- target_rows[valid]
    nb_rows_v <- nb_rows[valid]
    nb_vals_v <- vals[nb_rows_v]

    # Further filter out NA variable values
    valid2 <- !is.na(nb_vals_v)
    if (!any(valid2)) next

    t_rows_v  <- t_rows_v[valid2]
    nb_vals_v <- nb_vals_v[valid2]

    # Use data.table for grouped aggregation (very fast)
    edge_dt <- data.table(target_row = t_rows_v, nb_val = nb_vals_v)
    agg <- edge_dt[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = target_row]

    out_max[agg$target_row]  <- agg$nb_max
    out_min[agg$target_row]  <- agg$nb_min
    out_mean[agg$target_row] <- agg$nb_mean
  }

  cbind(out_max, out_min, out_mean)
}

# ==============================================================
# STEP 3: Wrapper to add features to data (preserves column naming)
# ==============================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, lookup) {
  stats <- compute_neighbor_stats_vectorized(data, var_name, lookup)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]

  data
}

# ==============================================================
# STEP 4: Main pipeline (drop-in replacement)
# ==============================================================

# Build lookup once (seconds, not hours)
lookup <- build_fast_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, lookup)
}

# The trained Random Forest model is untouched.
# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Property | Original | Optimized |
|---|---|---|
| Neighbor resolution | String key `"id_year"` â†’ row | Integer matrix `[cell_pos, year_pos]` â†’ row |
| Neighbor relationships | Same `rook_neighbors_unique` | Same, indices used directly |
| Stats computed | `max`, `min`, `mean` of non-NA neighbor values | Identical `max`, `min`, `mean` of non-NA neighbor values |
| Column names | `{var}_neighbor_max/min/mean` | Identical |
| RF model | Untouched | Untouched |

The numerical estimand is preserved exactly â€” same neighbor sets, same aggregation functions, same NA handling.

## Performance Analysis

| Component | Original | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string pastes + named vector build | One integer matrix fill: O(N) |
| Per-row neighbor resolution | `paste` + named vector lookup per row (~51.7M string ops) | Integer matrix index: O(1) per edge |
| Stats computation | `lapply` over 6.46M rows | `data.table` grouped aggregation per year (28 iterations) |
| **Estimated total time** | **86+ hours** | **2â€“10 minutes** |
| Memory | ~6.46M-entry named character vector (~500MB+) | Integer matrix 344K Ã— 28 (~38MB) + edge list (~22MB) |

The key insight: **neighbor topology is time-invariant, so separate the spatial dimension (which neighbor cells?) from the temporal dimension (which year?) and index both with integers.**
END_RESPONSE>>>
