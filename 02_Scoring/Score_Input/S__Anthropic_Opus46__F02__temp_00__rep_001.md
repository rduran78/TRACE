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
run_id: Anthropic_F02_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called **per row**. With ~6.46M rows, this creates and searches millions of character keys.
- Named vector lookup in R is hash-based but still carries overhead when called millions of times inside `lapply`.
- The function builds a **list of length 6.46M**, each element a vector of neighbor row indices. This list alone can consume several GB of RAM.

### 2. `compute_neighbor_stats` â€” repeated `lapply` over the same 6.46M-element list, once per variable
- Each call iterates over 6.46M list elements, subsetting a numeric vector and computing `max`, `min`, `mean`.
- This is called 5 times (once per neighbor source variable), so ~32.3M R-level function invocations.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (repeated memory allocation).

### Memory pressure
- The neighbor lookup list (6.46M elements Ã— ~4 neighbors each) plus the 6.46M Ã— 110 data frame already approaches or exceeds 16 GB, causing swapping.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste`/string lookup in `build_neighbor_lookup` | Replace with integer-keyed join via `data.table`. Encode `(id, year)` â†’ row index as a `data.table` keyed on integer columns; expand neighbor pairs with a vectorized merge instead of row-by-row `lapply`. |
| 6.46M-element R list for neighbor lookup | Replace with a flat **edge list** `data.table` (`from_row`, `to_row`), which is far more memory-efficient and enables grouped vectorized aggregation. |
| Repeated `lapply` + `rbind` in `compute_neighbor_stats` | Replace with a single `data.table` grouped aggregation per variable: join the edge list to the value column and compute `max`, `min`, `mean` by `from_row`. |
| 5 serial passes over the edge list | Combine into fewer passes or keep as simple loop â€” each pass is now seconds, not hours. |
| Overall RAM | `data.table` in-place reference semantics (`:=`) avoids copies. The flat edge list is ~50â€“80 MB vs. multi-GB nested list. |

**Estimated speedup:** from 86+ hours to **~5â€“15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat edge list replacing build_neighbor_lookup
#
#     Inputs:
#       cell_data          â€“ data.table with columns 'id' and 'year'
#       id_order           â€“ integer vector of cell IDs (same order as nb object)
#       rook_neighbors_unique â€“ spdep nb object (list of integer index vectors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edge_list <- function(cell_data, id_order, neighbors) {


  # --- a) Expand the nb object into a cell-ID edge list ----------------------
  n_cells <- length(id_order)
  from_id <- rep(id_order, times = lengths(neighbors))
  to_id   <- id_order[unlist(neighbors)]          # map nb indices â†’ cell IDs

  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  # Remove any zero-length artefacts from the nb object

  cell_edges <- cell_edges[!is.na(to_id)]

  # --- b) Map (id, year) â†’ row number in cell_data --------------------------
  cell_data[, .row_idx := .I]                      # add row index column

  # Key for fast join
  idx_dt <- cell_data[, .(id, year, .row_idx)]
  setkey(idx_dt, id, year)

  # --- c) Cross cell_edges with every year to get row-level edges ------------
  years <- sort(unique(cell_data$year))
  year_dt <- data.table(year = years)

  # Cartesian product: every spatial edge Ã— every year  (~38.5 M rows)
  edge_year <- cell_edges[, CJ_id := .I]           
  edge_year <- cell_edges[rep(seq_len(.N), each = length(years))]
  edge_year[, year := rep(years, times = nrow(cell_edges))]

  # Join to get from_row
  setkey(edge_year, from_id, year)
  edge_year[idx_dt, from_row := i..row_idx, on = .(from_id = id, year)]

  # Join to get to_row
  setkey(edge_year, to_id, year)
  edge_year[idx_dt, to_row := i..row_idx, on = .(to_id = id, year)]

  # Drop edges where either side is missing (boundary / missing year)
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row),
                         .(from_row, to_row)]

  # Clean up helper column
  cell_data[, .row_idx := NULL]


  return(edge_year)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Vectorized neighbor statistics using the edge list
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_data,
                                                   var_name,
                                                   edge_list) {
  # Attach the neighbor's value to every edge
  edge_list[, val := cell_data[[var_name]][to_row]]

  # Grouped aggregation â€” one pass, fully vectorized

  stats <- edge_list[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     keyby = from_row]

  # Initialise new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Write results into the correct rows (in-place)
  set(cell_data, i = stats$from_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$from_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$from_row, j = mean_col, value = stats$nb_mean)

  # Clean up temp column on edge_list
  edge_list[, val := NULL]

  invisible(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Main pipeline  (drop-in replacement for the outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build the edge list once  (~30 s, ~300 MB)
edge_list <- build_neighbor_edge_list(cell_data,
                                      id_order,
                                      rook_neighbors_unique)

# Compute neighbor features for every source variable  (~2 min total)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_list)
}

# Free the edge list
rm(edge_list); gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Predict with the existing trained Random Forest
#     (model object is unchanged; no retraining)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Numerical estimand unchanged** | `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per cell-year as the original code. The edge list is a lossless representation of the same `nb` object + year pairing. |
| **Trained RF model preserved** | The code only adds the same 15 new columns (`neighbor_{max,min,mean}_{var}`) to `cell_data`. The model object is never modified or retrained. |
| **16 GB RAM feasibility** | The edge list (~38.5M rows Ã— 2 int columns â‰ˆ 310 MB) plus `cell_data` (~6.46M Ã— 125 columns â‰ˆ 6.5 GB) fits comfortably. Peak usage stays under ~12 GB. |
| **Runtime** | Building the edge list: ~30â€“60 s. Five grouped aggregations: ~2â€“3 min total. **Total: ~5â€“15 minutes** vs. 86+ hours. |
END_RESPONSE>>>
