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
run_id: Anthropic_F02_T03_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` and a named-vector lookup (`id_to_ref`).
- `paste()` to build neighbor keys for that row's year.
- Named-vector lookup into `idx_lookup` (a 6.46M-length named character vector â€” each lookup is O(n) in the worst case for R's internal hashing).

This produces ~6.46M list elements, each containing an integer vector. The total work is proportional to the sum of all neighbor counts across all cell-years (~1.37M neighbor pairs Ã— 28 years â‰ˆ 38.5M key lookups), all done in interpreted R with character-key hashing.

### 2. `compute_neighbor_stats` â€” another O(n) `lapply` over 6.46M rows

For each of the 5 variables, this iterates over every row, subsets a numeric vector by the neighbor indices, removes NAs, and computes max/min/mean. That's 5 Ã— 6.46M â‰ˆ 32.3M interpreted iterations.

### 3. Memory pressure

The `neighbor_lookup` list alone stores ~6.46M integer vectors. Combined with the 6.46M Ã— 110 data frame and intermediate copies, this easily exceeds 16 GB.

**Estimated wall-clock**: The two `lapply` calls dominate. With ~6.46M iterations each (and 5 variable passes), the interpreted overhead on a laptop yields the reported 86+ hour estimate.

---

## Optimization Strategy

The key insight is: **eliminate the row-level R loop entirely by converting the problem to vectorized joins and grouped aggregations using `data.table`.**

| Step | Current approach | Optimized approach |
|---|---|---|
| Neighbor lookup | 6.46M-iteration `lapply` with `paste`/named-vector lookup | A single `data.table` equi-join of an edge table on `(neighbor_id, year)` â€” fully vectorized in C |
| Neighbor stats | 5 Ã— 6.46M-iteration `lapply` with per-element subsetting | A single grouped `data.table` aggregation: `[, .(max, min, mean), by = .(id, year)]` over all 5 vars at once |
| Memory | 6.46M-element list of integer vectors (~2â€“4 GB) | Edge table (~38.5M rows Ã— 3 int columns â‰ˆ 0.9 GB); no list overhead |
| Parallelism | None | `data.table` uses OpenMP threads for grouping/sorting automatically |

**Expected speedup**: The join + grouped aggregation should complete in **minutes**, not hours. Memory peak drops well below 16 GB.

**Preservation guarantees**:
- The trained Random Forest model is never touched.
- The numerical outputs (max, min, mean of neighbor values) are identical to the original code.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a tidy edge table from the spdep nb object (one-time cost)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  data.table(
    id       = id_order[from_idx],   # the focal cell
    neighbor_id = id_order[to_idx]   # the neighbor cell
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for all variables in one pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # cell_dt  : data.table with columns id, year, and all var_names
  # edge_dt  : data.table with columns id, neighbor_id
  # var_names: character vector of source variable names

  # --- a) Join edges with cell data to get neighbor values -----------
  #
  #   For every (focal id, year) we need the values of each var_name
  #   at (neighbor_id, year).  We achieve this with a single keyed join.
  #
  #   Left side : edge_dt merged with the year dimension from cell_dt
  #               (one row per focal-cell-year-neighbor triple)
  #   Right side: cell_dt keyed on (id, year)

  # Minimal subset of cell_dt: only the columns we need
  keep_cols <- c("id", "year", var_names)
  neighbor_vals <- cell_dt[, ..keep_cols]

  # Key the neighbor value table for the join
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)

  # Expand edges Ã— years:
  #   Instead of a full cross-join (expensive), we join edges onto the
  #   focal cell's (id, year) pairs first, then look up neighbor values.

  focal_keys <- cell_dt[, .(id, year)]
  setkey(edge_dt, id)
  setkey(focal_keys, id)

  # Each focal (id, year) gets its neighbor list  â†’  ~38.5 M rows
  expanded <- edge_dt[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, neighbor_id, year

  # Look up neighbor variable values
  expanded <- neighbor_vals[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has columns: neighbor_id, year, <var_names>, id

  # --- b) Grouped aggregation ----------------------------------------
  #   For each (id, year), compute max / min / mean of each variable
  #   across all neighbors.

  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression programmatically
  j_expr <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  stats <- expanded[, eval(j_expr), by = .(id, year)]

  # Replace -Inf / Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Main driver â€” drop-in replacement for the original outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table (in-place, no copy if already data.table)
setDT(cell_data)

# Build edge table once
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all 15 neighbor features (3 stats Ã— 5 vars) in one pass
neighbor_features <- compute_all_neighbor_features(
  cell_dt  = cell_data,
  edge_dt  = edge_dt,
  var_names = neighbor_source_vars
)

# Merge back onto the main table
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | The code only adds feature columns to `cell_data`. The model object is never loaded, modified, or retrained. |
| **Numerical estimand** | `max`, `min`, and `mean` are computed over exactly the same neighbor sets (rook contiguity, same year, NA-excluded) as the original code. The only difference is the computation path (vectorized C vs. interpreted R). Floating-point results are identical to machine precision. |
| **Column naming** | Output columns follow the pattern `neighbor_{max,min,mean}_{var}`, matching what `compute_and_add_neighbor_features` would produce. Adjust the naming template if your downstream pipeline expects a different convention. |

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| Wall-clock | ~86+ hours | **~5â€“15 minutes** (depending on disk I/O and `data.table` thread count) |
| Peak RAM | >16 GB (fails or swaps) | **~8â€“12 GB** (edge expansion is the peak; fits in 16 GB) |
| Lines of hot-path R loops | ~12.9 M iterations | **0** (fully vectorized) |

If memory is still tight, the `compute_all_neighbor_features` function can be trivially split to process one `var_name` at a time (loop over the 5 variables, each time carrying only one extra column in the join), reducing peak memory by roughly 40% at the cost of 5 passes instead of 1.
END_RESPONSE>>>
