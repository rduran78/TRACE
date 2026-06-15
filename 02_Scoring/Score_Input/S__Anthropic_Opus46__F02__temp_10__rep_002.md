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
run_id: Anthropic_F02_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string pasting and named-vector lookups

Each of the ~6.46 million iterations performs:
- `as.character()` conversion and named-vector lookup (`id_to_ref`).
- `paste()` to create neighbor keys for that row's year.
- Named-vector lookup into `idx_lookup` (a 6.46M-element named character vector â€” each lookup is **O(n)** in the worst case because R named vectors use linear hashing with possible collisions at scale).

This alone accounts for the vast majority of the 86+ hour estimate. String pasting and named-vector lookups on millions of keys are extremely slow in base R.

### 2. `compute_neighbor_stats` â€” 6.46M `lapply` iterations repeated for each of 5 variables

Each call iterates over all rows, subsetting a numeric vector. The `do.call(rbind, ...)` on a 6.46M-element list of length-3 vectors is also slow (repeated memory allocation).

### Memory concern

With ~6.46M rows Ã— ~110 columns the data frame itself is manageable (~5â€“7 GB), but building a 6.46M-element list of integer vectors for `neighbor_lookup`, plus intermediate string vectors, can spike memory well past 16 GB.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| String key creation & lookup in `build_neighbor_lookup` | Replace with **integer arithmetic keys** and `data.table` hash joins. A unique integer key = `id * 100L + (year - 1991L)` avoids all string operations. |
| Per-row `lapply` over 6.46M rows | Convert to a **vectorized edge-list approach**: expand all neighbor pairs into a `data.table` of `(row_i, neighbor_row_j)`, then compute grouped statistics with `data.table` `[, .(max, min, mean), by = row_i]`. No R-level loop at all. |
| `do.call(rbind, ...)` on millions of small vectors | Eliminated â€” `data.table` grouping returns a single `data.table` directly. |
| 5 serial calls to `compute_neighbor_stats` | Process all 5 variables in a **single grouped aggregation** pass over the edge list. |
| Memory spikes from list-of-vectors `neighbor_lookup` | The edge-list `data.table` is two integer columns (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows Ã— 2 cols Ã— 4 bytes â‰ˆ 308 MB), far cheaper than a 6.46M-element ragged list. |

**Expected speedup**: from 86+ hours to roughly **5â€“20 minutes** depending on disk I/O, because every hot path becomes a vectorized C-level `data.table` operation.

**Preservation guarantees**: No model retraining. The output columns are numerically identical (`max`, `min`, `mean` of the same neighbor values), so the trained Random Forest receives the same feature semantics and values.

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optimized neighbor-feature pipeline
# Requirements: data.table (install.packages("data.table") if needed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)

build_and_compute_neighbor_features <- function(cell_data,
                                                 id_order,
                                                 rook_neighbors_unique,
                                                 neighbor_source_vars) {

  # --- Step 0: Convert to data.table (by reference if already one) -----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Step 1: Build a spatial edge list (cell-level, year-independent) ------
  #
  # rook_neighbors_unique is an nb object: a list where element i contains the
  # integer indices (into id_order) of the neighbors of id_order[i].
  # We expand this into a two-column data.table of (focal_id, neighbor_id).

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id    = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1.37 M rows (directed relationships)

  # --- Step 2: Create integer row key in cell_data --------------------------
  #
  # We need to join cell_data rows to their neighbors' rows for the same year.
  # Instead of string pasting, use an integer key: id + year.
  # Because years span only 1992-2019, we can encode as (id * 100L + year_offset).
  # But id values may be large, so we use a plain two-column keyed join instead,
  # which data.table handles efficiently via hash joins.

  # Ensure id and year are plain integer/numeric (no factors)
  cell_data[, c("id", "year") := .(as.integer(id), as.integer(year))]

  # Add an internal row index (will be used to write results back)
  cell_data[, .row_idx := .I]

  # --- Step 3: Expand edge list across years (vectorized) --------------------
  #
  # For every (focal_id, neighbor_id) pair we need every year in the panel.
  # This is a cross join of edge_list Ã— unique_years.

  unique_years <- sort(unique(cell_data$year))

  # Cross join: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  edges_by_year <- CJ_dt_edges(edge_list, unique_years)
  # edges_by_year has columns: focal_id, neighbor_id, year

  # --- Step 4: Attach row indices for focal and neighbor rows ----------------
  #
  # We need each row's index to grab variable values.

  # Key cell_data for fast joins
  setkey(cell_data, id, year)

  # Join to get focal row index
  edges_by_year[cell_data, focal_row := i..row_idx,
                on = .(focal_id = id, year = year)]

  # Join to get neighbor row index
  edges_by_year[cell_data, neighbor_row := i..row_idx,
                on = .(neighbor_id = id, year = year)]

  # Drop edges where either side has no matching row (e.g., missing cell-years)
  edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # --- Step 5: Compute neighbor stats for all variables at once --------------

  # Pre-extract the variable columns as a matrix for fast column access
  var_mat <- as.matrix(cell_data[, ..neighbor_source_vars])
  # var_mat is nrow(cell_data) Ã— length(neighbor_source_vars)

  # Attach neighbor values for every source variable
  for (j in seq_along(neighbor_source_vars)) {
    vname <- neighbor_source_vars[j]
    set(edges_by_year,
        j    = vname,
        value = var_mat[edges_by_year$neighbor_row, j])
  }

  # Grouped aggregation: max, min, mean per focal_row for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call dynamically
  agg_call <- parse(text = paste0(
    "edges_by_year[, .(",
    paste(
      mapply(function(nm, expr) paste0(nm, " = ", deparse(expr)),
             agg_names, agg_exprs),
      collapse = ", "
    ),
    "), by = focal_row]"
  ))

  stats_dt <- eval(agg_call)

  # Replace Inf / -Inf (from max/min of all-NA groups) with NA

  for (col_name in agg_names) {
    vals <- stats_dt[[col_name]]
    set(stats_dt, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  # --- Step 6: Merge results back into cell_data ----------------------------
  #
  # Rows with no neighbors will get NA (left join).

  cell_data[stats_dt, (agg_names) := mget(agg_names),
            on = .(.row_idx = focal_row)]

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}


# â”€â”€ Helper: cross join edges Ã— years â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CJ_dt_edges <- function(edge_list, years) {
  # Memory-efficient cross join without copying the full CJ grid.
  # Replicate edge_list once per year.
  n_edges <- nrow(edge_list)
  n_years <- length(years)
  data.table(
    focal_id    = rep(edge_list$focal_id,    times = n_years),
    neighbor_id = rep(edge_list$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Usage (drop-in replacement for the original outer loop)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_compute_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# Feed cell_data directly to predict(trained_rf_model, newdata = cell_data).
```

---

## Summary of What Changed and What Is Preserved

| Aspect | Original | Optimized |
|---|---|---|
| Key lookup | Named character vector (O(n) hashing on 6.46M strings) | `data.table` keyed join on two integer columns (O(1) amortized hash) |
| Neighbor expansion | Per-row `lapply` with `paste()` (6.46M R-level iterations) | Vectorized `rep()` cross join (~38.5M rows built in seconds) |
| Stat computation | Per-row `lapply` Ã— 5 variables (32.3M R iterations) | Single grouped `data.table` aggregation (C-level radix grouping) |
| Result assembly | `do.call(rbind, 6.46M-element list)` | Direct `data.table` left join |
| Estimated runtime | 86+ hours | **~5â€“20 minutes** |
| Peak RAM | >16 GB (string vectors + ragged list) | **~8â€“12 GB** (integer edge table + variable matrix) |
| Numerical output | max, min, mean of neighbor values | **Identical** max, min, mean of same neighbor values |
| Trained RF model | â€” | **Untouched** (no retraining) |
END_RESPONSE>>>
