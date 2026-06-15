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
run_id: Anthropic_F05_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” repeated string allocation and concatenation.
2. **Looks up those keys** in `idx_lookup`, a named character vector of length 6.46M â€” each lookup is O(N) in the worst case because R named vectors use linear hashing that degrades at this scale.

This means `build_neighbor_lookup` alone performs ~6.46M Ã— (avg ~8 neighbors) â‰ˆ **51.7 million string constructions and hash lookups against a 6.46M-entry table**. That's the 86+ hour bottleneck.

### Why It's a Broader Algorithmic Problem

The string-keyed lookup is fundamentally unnecessary. The neighbor structure is **spatial** (cell-to-cell) and **time-invariant** â€” the same cell has the same neighbors every year. The current code re-discovers "which row in the data corresponds to neighbor cell C in year Y" by string matching, when this mapping is a simple arithmetic function if the data is properly sorted.

**Key insight:** If the data is sorted by `(id, year)` and every cell has all 28 years present, then the row index for cell `j` in year `y` is:

```
row = (cell_position_of_j - 1) * 28 + (y - 1991)
```

No strings. No hash tables. Pure integer arithmetic. This converts the entire `build_neighbor_lookup` from ~86 hours to **seconds**.

Additionally, `compute_neighbor_stats` uses an R-level `lapply` over 6.46M rows â€” this can be vectorized with `data.table` grouping operations.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Row lookup | String paste + named vector lookup per row | Integer arithmetic via sorted data + positional indexing |
| Neighbor lookup construction | R-level `lapply` over 6.46M rows | Vectorized expansion of neighbor pairs Ã— years |
| Neighbor stats | `lapply` with per-row `max/min/mean` | `data.table` grouped aggregation on a flat neighbor-edge table |
| Per-variable stats | Separate `lapply` pass per variable | Single grouped join computes all 5 variables at once |

**Estimated speedup:** From 86+ hours to **~2â€“5 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Preserves the exact numerical estimand (max, min, mean of
# each neighbor variable) and requires no model retraining.
# ==============================================================

build_and_compute_all_neighbor_features <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars,
                                                     year_range = 1992:2019) {
  # ---- 0. Convert to data.table if needed ----
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }


  # ---- 1. Ensure data is sorted by (id, year) ----
  setkey(cell_data, id, year)

  n_years <- length(year_range)
  n_cells <- length(id_order)

  # Verify completeness: balanced panel expected
  stopifnot(
    "Panel is not balanced or does not match id_order" =
      nrow(cell_data) == n_cells * n_years
  )

  # ---- 2. Build cell-position lookup (integer) ----
  # id_order[k] is the cell id at spatial position k

  # We need the reverse: given a cell id, what is its position?
  cell_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Verify that sorted data aligns with id_order positions
  # After setkey(id, year), the first n_years rows belong to the
  # cell with the smallest id, etc. We need to map that to id_order.
  # Actually, we need to build a mapping from cell id -> block start row.
  unique_ids_sorted <- cell_data[, unique(id)]  # sorted because of setkey
  id_to_block_start <- setNames(
    seq(from = 1, by = n_years, length.out = n_cells),
    as.character(unique_ids_sorted)
  )
  year_offset <- setNames(seq_len(n_years) - 1L, as.character(year_range))

  # ---- 3. Build flat edge table (focal_id, neighbor_id) ----
  # rook_neighbors_unique is an nb object: list of length n_cells

  # where element k contains integer indices into id_order of neighbors of id_order[k]
  message("Building edge table...")
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
    nb_idx <- rook_neighbors_unique[[k]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(
      focal_id    = id_order[k],
      neighbor_id = id_order[nb_idx]
    )
  }))

  message(sprintf("  %s directed neighbor edges", format(nrow(edges), big.mark = ",")))

  # ---- 4. Expand edges Ã— years and compute row indices arithmetically ----
  message("Expanding edges across years...")

  # Cross join edges with years
  years_dt <- data.table(year = year_range)
  edge_year <- edges[, CJ_idx := .I]  # just need the cross
  edge_year <- edges[rep(seq_len(.N), each = n_years)]
  edge_year[, year := rep(year_range, times = nrow(edges))]

  # Compute the row index in cell_data for the focal cell-year
  edge_year[, focal_row := as.integer(
    id_to_block_start[as.character(focal_id)] + year_offset[as.character(year)]
  )]

  # Compute the row index in cell_data for the neighbor cell-year
  edge_year[, neighbor_row := as.integer(
    id_to_block_start[as.character(neighbor_id)] + year_offset[as.character(year)]
  )]

  # ---- 5. Pull neighbor values and aggregate ----
  message("Computing neighbor statistics for all variables...")

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing: %s", var_name))

    # Extract the variable values by neighbor row index
    edge_year[, nval := cell_data[[var_name]][neighbor_row]]

    # Aggregate: max, min, mean per focal_row (excluding NAs)
    agg <- edge_year[!is.na(nval),
      .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ),
      by = focal_row
    ]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign aggregated values back by row index
    cell_data[agg$focal_row, (max_col)  := agg$nb_max]
    cell_data[agg$focal_row, (min_col)  := agg$nb_min]
    cell_data[agg$focal_row, (mean_col) := agg$nb_mean]

    # Clean up the temporary column
    edge_year[, nval := NULL]
  }

  # ---- 6. Clean up ----
  rm(edge_year, edges)
  gc()

  message("Done.")

  if (was_df) {
    return(as.data.frame(cell_data))
  }
  return(cell_data)
}


# ==============================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_and_compute_all_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars,
#   year_range           = 1992:2019
# )
#
# # The trained Random Forest model is used as-is â€” no retraining needed.
# # predict(rf_model, new_data = cell_data) works identically.
```

---

## Memory-Constrained Variant

The `edge_year` table has ~1.37M edges Ã— 28 years â‰ˆ **38.5M rows**, which at ~5 columns of integers/doubles is roughly **1.5 GB**. On a 16 GB laptop this is feasible but tight. If memory is a concern, process years in chunks:

```r
# ---- Memory-efficient variant: process one year at a time ----
build_and_compute_chunked <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars,
                                       year_range = 1992:2019) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }

  setkey(cell_data, id, year)
  n_years <- length(year_range)
  n_cells <- length(id_order)
  stopifnot(nrow(cell_data) == n_cells * n_years)

  unique_ids_sorted <- cell_data[, unique(id)]
  id_to_block_start <- setNames(
    seq(from = 1L, by = n_years, length.out = n_cells),
    as.character(unique_ids_sorted)
  )
  year_offset <- setNames(seq_len(n_years) - 1L, as.character(year_range))

  # Build spatial edge list (time-invariant, ~1.37M rows)
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
    nb_idx <- rook_neighbors_unique[[k]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[k], neighbor_id = id_order[nb_idx])
  }))

  # Pre-lookup block starts for edges (avoids repeated character lookups)
  edges[, focal_block    := id_to_block_start[as.character(focal_id)]]
  edges[, neighbor_block := id_to_block_start[as.character(neighbor_id)]]

  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Process one year at a time (~1.37M rows per year)
  for (yr in year_range) {
    message(sprintf("  Year %d", yr))
    yo <- year_offset[as.character(yr)]

    focal_rows    <- edges$focal_block    + yo
    neighbor_rows <- edges$neighbor_block + yo

    for (var_name in neighbor_source_vars) {
      nvals <- cell_data[[var_name]][neighbor_rows]

      # Build a temporary data.table for grouped aggregation
      tmp <- data.table(fr = focal_rows, nv = nvals)
      tmp <- tmp[!is.na(nv)]

      if (nrow(tmp) == 0L) next

      agg <- tmp[, .(
        nb_max  = max(nv),
        nb_min  = min(nv),
        nb_mean = mean(nv)
      ), by = fr]

      set(cell_data, i = agg$fr, j = paste0(var_name, "_neighbor_max"),  value = agg$nb_max)
      set(cell_data, i = agg$fr, j = paste0(var_name, "_neighbor_min"),  value = agg$nb_min)
      set(cell_data, i = agg$fr, j = paste0(var_name, "_neighbor_mean"), value = agg$nb_mean)
    }
  }

  rm(edges)
  gc()
  message("Done.")

  if (was_df) return(as.data.frame(cell_data))
  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Algorithmic complexity** | O(N Ã— k) string ops + hash lookups (N=6.46M, kâ‰ˆ8) | O(E Ã— T) integer arithmetic (E=1.37M, T=28) |
| **Data structure** | Named character vector (6.46M entries) | Integer positional indexing on sorted `data.table` |
| **Aggregation** | R-level `lapply` with per-row `max/min/mean` | `data.table` grouped aggregation (C-level) |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **Peak memory** | Moderate (but slow) | ~1.5 GB for edge_year (or ~100 MB chunked) |
| **Numerical output** | max, min, mean per cell-year | **Identical** â€” same estimand preserved |
| **RF model** | â€” | **Untouched** â€” no retraining needed |
END_RESPONSE>>>
