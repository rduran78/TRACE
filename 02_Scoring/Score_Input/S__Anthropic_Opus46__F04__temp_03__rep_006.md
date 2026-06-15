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
run_id: Anthropic_F04_T03_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks up those keys in `idx_lookup` (a named character vector).

String construction (`paste`) and named-vector lookups over millions of iterations in R are extremely slow. `compute_neighbor_stats` then iterates over 6.46M entries again per variable (Ã—5 variables), doing subsetting and summary stats in pure R. Combined, this yields the estimated 86+ hour runtime.

**Root causes:**
- **Row-level `lapply` with string operations** over 6.46M rows is the primary bottleneck.
- **Redundant per-variable looping** in `compute_neighbor_stats` with R-level list operations is the secondary bottleneck.
- The neighbor topology is **time-invariant** (same grid, same rook neighbors every year), but the code rebuilds string keys for every cell-year pair as if topology changes per year.

## Optimization Strategy

1. **Exploit time-invariance of topology.** The neighbor graph is defined over 344,208 cells, not 6.46M cell-years. Build the lookup at the cell level (344K entries), then broadcast across years using vectorized integer indexing.

2. **Replace string key lookups with integer arithmetic.** If data is sorted by `(id, year)`, each cell's rows occupy a contiguous block of 28 rows. The row index for cell `c` in year `y` is `(c-1)*28 + (y - 1991)`. No `paste`, no named vector lookup needed.

3. **Vectorize `compute_neighbor_stats` using `data.table` grouping** or a single pre-allocated matrix operation instead of `lapply` over 6.46M elements.

4. **Pre-allocate output columns** and fill via vectorized assignment rather than `do.call(rbind, ...)` on a 6.46M-element list.

These changes reduce the problem from ~6.46M string-manipulation iterations to ~344K integer-index iterations (for the lookup) and fully vectorized column computation (for the stats), cutting runtime from 86+ hours to minutes.

## Optimized Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0 â€” Convert to data.table and sort deterministically
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Create integer cell index (1-based) aligned with id_order
cell_dt[, cell_idx := match(id, id_order)]

# Ensure year is integer
cell_dt[, year := as.integer(year)]

# Sort by (cell_idx, year) â€” this is critical for the arithmetic trick
setkey(cell_dt, cell_idx, year)

# Confirm contiguous year panel
years      <- sort(unique(cell_dt$year))
n_years    <- length(years)           # 28
n_cells    <- length(id_order)        # 344,208
year_min   <- min(years)              # 1992
stopifnot(nrow(cell_dt) == n_cells * n_years)

# Year offset: 1-based position of each year in the panel
cell_dt[, year_off := year - year_min + 1L]

# ---------------------------------------------------------------
# STEP 1 â€” Build neighbor lookup at the CELL level (344K, not 6.46M)
#           rook_neighbors_unique[[c]] gives neighbor cell indices
#           into id_order (already 1-based integer vectors).
# ---------------------------------------------------------------
# Nothing to change: rook_neighbors_unique is already a list of
# integer vectors indexed by cell position in id_order.

# ---------------------------------------------------------------
# STEP 2 â€” Expand cell-level neighbors to row-level indices
#           using integer arithmetic (fully vectorized).
#
#   Row index of cell c (1-based) in year_off t:
#       row = (c - 1) * n_years + t
# ---------------------------------------------------------------

# For every cell c, its neighbors are rook_neighbors_unique[[c]].
# For a given year_off t, the row indices of those neighbors are:
#   (neighbor_cell_idx - 1) * n_years + t
#
# We build three long vectors: (source_row, neighbor_row) pairs.

message("Building vectorized neighbor edge list...")

# Number of neighbors per cell
n_nbrs <- lengths(rook_neighbors_unique)  # length = n_cells

# Cell indices repeated by their neighbor count
source_cells <- rep(seq_len(n_cells), times = n_nbrs)
# Corresponding neighbor cell indices (unlisted)
target_cells <- unlist(rook_neighbors_unique, use.names = FALSE)

# Now expand across all years: each (source_cell, target_cell) pair
# appears once per year.
n_edges_per_year <- length(source_cells)  # ~1.37M

# Replicate for each year offset (1..28)
year_offsets <- rep(seq_len(n_years), each = n_edges_per_year)
source_rows  <- rep((source_cells - 1L) * n_years, times = n_years) + year_offsets
target_rows  <- rep((target_cells - 1L) * n_years, times = n_years) + year_offsets

# edge_dt: every row is one directed (source_row -> neighbor_row) edge
edge_dt <- data.table(src = source_rows, tgt = target_rows)
rm(source_rows, target_rows, year_offsets); gc()

message(sprintf("Edge list: %s edges", format(nrow(edge_dt), big.mark = ",")))

# ---------------------------------------------------------------
# STEP 3 â€” Compute neighbor stats for each variable (vectorized)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))

  # Attach the neighbor's value to each edge
  edge_dt[, val := cell_dt[[var_name]][tgt]]

  # Compute grouped stats: max, min, mean per source row
  stats <- edge_dt[!is.na(val),
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   keyby = src]

  # Initialize columns with NA
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Fill in computed values by integer index
  cell_dt[stats$src, (max_col)  := stats$nb_max]
  cell_dt[stats$src, (min_col)  := stats$nb_min]
  cell_dt[stats$src, (mean_col) := stats$nb_mean]
}

# ---------------------------------------------------------------
# STEP 4 â€” Restore original row order and convert back
# ---------------------------------------------------------------
# If the original cell_data had a specific row order, restore it.
# We kept cell_idx and year, so we can match back.
setkey(cell_dt, cell_idx, year)

# Drop helper columns
cell_dt[, c("cell_idx", "year_off") := NULL]

# Convert back to data.frame for downstream Random Forest predict()
cell_data <- as.data.frame(cell_dt)
rm(cell_dt, edge_dt, stats); gc()

message("Neighbor feature construction complete.")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values as the original code. No approximation. |
| **Trained RF model** | No model retraining. The code only constructs predictor columns; `predict()` is called downstream unchanged. |
| **Column names** | Output columns follow the same `nb_max_<var>`, `nb_min_<var>`, `nb_mean_<var>` naming convention (adjust prefixes to match your `compute_and_add_neighbor_features` wrapper if they differ). |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M `paste` + named-vector lookups | Integer arithmetic, no strings |
| Stats computation | 5 Ã— `lapply` over 6.46M R lists | 5 Ã— `data.table` grouped aggregation on ~38.4M-row edge table |
| Memory peak | Millions of small R list elements (high overhead) | ~38.4M-row `data.table` with 3 columns (~1.2 GB, fits in 16 GB) |
| **Estimated runtime** | **86+ hours** | **~5â€“15 minutes** |

The key insight is that the rook-neighbor topology is **time-invariant over the 28-year panel**, so the 6.46M-row problem decomposes into a 344K-cell spatial problem broadcast across 28 years via pure integer arithmeticâ€”eliminating all string operations entirely.
END_RESPONSE>>>
