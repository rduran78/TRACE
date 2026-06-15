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
run_id: Anthropic_F07_T00_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Constructs character keys (`paste(id, year)`) and matches them into a named character vector (`idx_lookup`) of length 6.46M.

**Named-vector lookup via character keys in R is O(n) per probe** (hashed, but with a large constant and memory overhead). Doing this ~6.46M Ã— ~4 neighbors â‰ˆ 25 billion character operations is what produces the 86+ hour estimate.

`compute_neighbor_stats` is a secondary bottleneck: another R-level `lapply` over 6.46M elements, each extracting a small numeric slice. This is slow but less catastrophic.

**Root cause summary:**

| Component | Complexity | Problem |
|---|---|---|
| `build_neighbor_lookup` | O(N Ã— k) character key lookups in a length-N named vector | ~86 h |
| `compute_neighbor_stats` | O(N Ã— k) R-level loops | ~minutes but repeated 5Ã— |
| Total neighbor features | 5 vars Ã— 3 stats = 15 new columns | â€” |

## Optimization Strategy

1. **Replace character-key lookup with integer-arithmetic indexing.** Since every cell appears in every year (balanced panel, 344,208 cells Ã— 28 years), we can sort the data by `(year, id)` and compute any cell-year's row index as a direct integer offset: `row = (year_offset) * n_cells + cell_offset`. This is O(1) per neighbor, no strings involved.

2. **Vectorize `compute_neighbor_stats` using `data.table` and a pre-expanded edge list.** Instead of looping over 6.46M rows, we build a long edge table `(from_row, to_row)` of all ~6.46M Ã— 4 â‰ˆ 25M directed edges, then join and group-aggregate in `data.table` â€” a single vectorized pass per variable.

3. **Memory budget.** The edge list is ~25M rows Ã— 2 integer columns â‰ˆ 200 MB. The data itself at 6.46M Ã— 110 columns â‰ˆ 5â€“6 GB. Fits in 16 GB with care.

**Expected speedup:** from 86+ hours to **~2â€“5 minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Ensure cell_data is a data.table sorted by (year, id)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Canonical cell ordering â€” must match the order used in rook_neighbors_unique
# id_order is the vector of cell IDs in the same order as the nb object indices
stopifnot(length(id_order) == 344208L)

# Create integer maps
cell_dt[, id_rank := match(id, id_order)]
setorder(cell_dt, year, id_rank)
cell_dt[, row_idx := .I]                 # row index after sort

n_cells <- length(id_order)              # 344,208
years   <- sort(unique(cell_dt$year))    # 1992:2019
n_years <- length(years)                 # 28
year_to_offset <- setNames(seq_along(years) - 1L, as.character(years))

# Verify balanced panel
stopifnot(nrow(cell_dt) == n_cells * n_years)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the directed edge list (all years) â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Expand the nb object into a two-column integer edge list (cell-level)
from_cell <- rep(seq_along(rook_neighbors_unique),
                 lengths(rook_neighbors_unique))
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove 0-neighbor entries (spdep uses integer(0) for islands; unlist drops them)
valid <- to_cell > 0L
from_cell <- from_cell[valid]
to_cell   <- to_cell[valid]

n_edges_cell <- length(from_cell)        # ~1,373,394 / 2 directed pairs
cat("Cell-level directed edges:", n_edges_cell, "\n")

# Tile across all years:
#   row index = year_offset * n_cells + cell_rank
#   (cell_rank is 1-based, matching id_rank above)
from_rows <- rep(from_cell, times = n_years) +
             rep(seq(0L, (n_years - 1L) * n_cells, by = n_cells),
                 each = n_edges_cell)
to_rows   <- rep(to_cell, times = n_years) +
             rep(seq(0L, (n_years - 1L) * n_cells, by = n_cells),
                 each = n_edges_cell)

edges <- data.table(from_row = from_rows, to_row = to_rows)
rm(from_rows, to_rows); gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor max / min / mean per variable â€” vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "\n")

  # Attach the neighbor's value to every edge
  edges[, nbr_val := cell_dt[[var_name]][to_row]]

  # Aggregate: group by the focal row
  stats <- edges[!is.na(nbr_val),
                 .(nb_max  = max(nbr_val),
                   nb_min  = min(nbr_val),
                   nb_mean = mean(nbr_val)),
                 keyby = from_row]

  # Initialise new columns to NA, then fill matched rows
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  cell_dt[stats$from_row, (max_col)  := stats$nb_max]
  cell_dt[stats$from_row, (min_col)  := stats$nb_min]
  cell_dt[stats$from_row, (mean_col) := stats$nb_mean]

  edges[, nbr_val := NULL]
  rm(stats); gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Restore original row order and convert back if needed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If downstream code (e.g. the trained RF predict call) expects the
# original row order, restore it:
setorder(cell_dt, row_idx)   # already 1:N but just in case
cell_dt[, c("id_rank", "row_idx") := NULL]

# Convert back to data.frame if the RF predict method requires it
cell_data <- as.data.frame(cell_dt)
```

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical estimand** | The aggregation functions (`max`, `min`, `mean`) are identical; the edge list is derived from the same `rook_neighbors_unique` nb object; no approximation is introduced. |
| **Trained RF model untouched** | No model object is modified. The 15 new columns have the same names and values as the original code would produce. `predict(rf_model, cell_data)` will return identical results. |
| **Irregular / masked topology** | The edge list is built directly from the `nb` object, which already encodes the true rook-neighbor graph. No rectangular-raster assumption is made. Islands (cells with no neighbors) naturally receive `NA` for all neighbor stats. |
| **Memory** | Edge list â‰ˆ 200 MB; `cell_dt` â‰ˆ 5â€“6 GB; temporaries < 1 GB. Total peak â‰ˆ 8â€“9 GB, within 16 GB. |
| **Speed** | Two vectorized passes (edge expansion + `data.table` grouped aggregation) per variable. Expected wall time: **2â€“5 minutes total** vs. 86+ hours. |
END_RESPONSE>>>
