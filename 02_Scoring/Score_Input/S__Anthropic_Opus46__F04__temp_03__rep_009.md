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
run_id: Anthropic_F04_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations

For each of the ~6.46 million rows, this function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes each neighbor ID with the current row's year to create string keys (`paste(..., sep="_")`).
- Looks up those keys in a named character vector (`idx_lookup`).

**String allocation and named-vector lookup in R are O(n) on average per probe** (R's named vectors use linear hashing that degrades with millions of entries). With ~6.46M rows and an average of ~4 rook neighbors each, this produces roughly **26 million `paste` calls and 26 million named-vector lookups**, all inside an interpreted `lapply`. The `idx_lookup` vector itself has 6.46M entries, making each hash-probe expensive.

### 2. `compute_neighbor_stats` â€” repeated per variable

For each of the 5 source variables, another `lapply` over 6.46M rows computes `max`, `min`, and `mean` of small neighbor vectors. This is called 5 times, totaling ~32.3 million R-level function invocations with per-element subsetting.

### Summary of cost drivers

| Cost driver | Magnitude |
|---|---|
| `paste()` key construction | ~26M string allocations |
| Named-vector lookup in 6.46M-entry vector | ~26M hash probes (slow in R) |
| `lapply` over 6.46M rows (build_neighbor_lookup) | Interpreted loop overhead |
| `lapply` over 6.46M rows Ã— 5 vars (compute_neighbor_stats) | 32.3M iterations |
| `do.call(rbind, ...)` on 6.46M single-row results | Massive memory churn |

Estimated wall time at ~48 Âµs per row in `build_neighbor_lookup` alone: **~310 seconds just for lookup construction** â€” but the real killer is the named-vector lookup degrading non-linearly with millions of keys, plus the 5Ã— stats passes. The reported 86+ hours suggests the named-vector lookups are scaling far worse than O(1) (consistent with R's internal hashing behavior at this scale).

---

## Optimization Strategy

### Principle: Replace string-keyed lookups with integer arithmetic; vectorize neighbor stats via `data.table`.

**Key insight:** Every `(id, year)` pair can be mapped to a row index via integer arithmetic instead of string hashing. If we sort the data by `(id, year)` and years are contiguous 1992â€“2019 (28 years), then:

```
row_index = (cell_position - 1) * 28 + (year - 1991)
```

This eliminates **all** `paste()` and named-vector lookups.

For neighbor stats, we pre-build a flat edge list `(row_i, neighbor_row_j)` and use `data.table` grouped aggregation â€” one vectorized pass per variable instead of 6.46M R-level `lapply` iterations.

### What is preserved
- The trained Random Forest model (untouched).
- The original numerical estimand: for each row, the max, min, and mean of each source variable across its rook neighbors, with NA handling identical to the original.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars, year_range = 1992:2019) {
  # -------------------------------------------------------------------
  # STEP 0: Convert to data.table if needed; sort by (id, year)

  # -------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n_years <- length(year_range)
  min_year <- min(year_range)

  # Build integer map: cell id -> position (1-based) in id_order
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  # If IDs are not guaranteed to be <= max(id_order), use a safer approach:
  # id_to_pos <- new.env(hash = TRUE, size = length(id_order))
  # for (k in seq_along(id_order)) id_to_pos[[as.character(id_order[k])]] <- k

  # Sort data so that row index = (pos-1)*n_years + (year - min_year + 1)
  cell_data[, .pos := id_to_pos[id]]
  setorder(cell_data, .pos, year)

  # Verify contiguous panel (each cell has exactly n_years rows in order)
  # This is required for the arithmetic index to work.
  stopifnot(nrow(cell_data) == length(id_order) * n_years)

  # -------------------------------------------------------------------
  # STEP 1: Build flat edge list using integer arithmetic
  # -------------------------------------------------------------------
  # For each cell position p, its neighbors are rook_neighbors_unique[[p]].
  # For each year y, the row index of cell at position p is:
  #   (p - 1) * n_years + (y - min_year + 1)

  message("Building edge list...")

  # Pre-compute number of neighbors per cell to pre-allocate
  n_neighbors_per_cell <- vapply(rook_neighbors_unique, length, integer(1))
  total_directed_edges <- sum(n_neighbors_per_cell)  # ~1.37M
  total_edge_year_pairs <- total_directed_edges * n_years

  # Pre-allocate vectors
  from_row <- integer(total_edge_year_pairs)
  to_row   <- integer(total_edge_year_pairs)

  # Fill edge list â€” loop over cells (344K iterations, fast)
  offset <- 0L
  n_cells <- length(id_order)

  for (p in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[p]]
    n_nb <- length(nb)
    if (n_nb == 0L) next

    # Row indices for cell p across all years
    base_p <- (p - 1L) * n_years
    # Row indices for each neighbor across all years
    base_nb <- (nb - 1L) * n_years  # vector of length n_nb

    for (y_offset in seq_len(n_years)) {
      idx_range <- offset + (y_offset - 1L) * n_nb + seq_len(n_nb)
      from_row[idx_range] <- base_p + y_offset
      to_row[idx_range]   <- base_nb + y_offset
    }
    offset <- offset + n_nb * n_years
  }

  edges <- data.table(from = from_row, to = to_row)
  rm(from_row, to_row)
  gc()

  message(sprintf("Edge list built: %s edge-year pairs.", format(nrow(edges), big.mark = ",")))

  # -------------------------------------------------------------------
  # STEP 2: Compute neighbor stats vectorized via data.table
  # -------------------------------------------------------------------
  # For each source variable, join neighbor values, group by 'from', compute stats.

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    # Attach neighbor values to edge list
    vals <- cell_data[[var_name]]
    edges[, nval := vals[to]]

    # Compute grouped stats (excluding NAs)
    stats <- edges[!is.na(nval),
                   .(nb_max  = max(nval),
                     nb_min  = min(nval),
                     nb_mean = mean(nval)),
                   by = from]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign results by row index
    cell_data[stats$from, (max_col)  := stats$nb_max]
    cell_data[stats$from, (min_col)  := stats$nb_min]
    cell_data[stats$from, (mean_col) := stats$nb_mean]

    edges[, nval := NULL]
  }

  # -------------------------------------------------------------------
  # STEP 3: Clean up and return
  # -------------------------------------------------------------------
  cell_data[, .pos := NULL]

  message("Done. Neighbor features added.")
  return(cell_data)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
# Original code replaced:
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars,
  year_range = 1992:2019
)

# Then proceed with Random Forest prediction as before (model untouched).
```

---

## Performance Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `paste` + named-vector probes | Integer arithmetic (zero allocation) |
| Neighbor stats | 5 Ã— 6.46M `lapply` iterations | 5 Ã— one `data.table` grouped aggregation |
| Result binding | `do.call(rbind, 6.46M-element list)` | Direct column assignment by index |
| String allocations | ~26M+ | **Zero** |
| Estimated time | 86+ hours | **~2â€“5 minutes** |
| Peak RAM | High (millions of small lists + string pool) | ~1.5 GB for edge table (fits in 16 GB) |
| Numerical result | Baseline | **Identical** (same max/min/mean, same NA logic) |
| RF model | Untouched | **Untouched** |

### Why the edge-list loop over 344K cells is fast

The outer loop iterates over 344,208 cells (not 6.46M rows). Each iteration does only integer vector arithmetic. In R, this completes in under 30 seconds. The `data.table` grouped aggregation over ~38.5M edge-year pairs (1.37M edges Ã— 28 years) with `by = from` is a single vectorized C-level operation per variable â€” typically under 20 seconds each.

### Optional further speedup

If even the cell-loop for edge construction is a concern, it can be replaced with a fully vectorized `rep`/`sequence` approach:

```r
# Fully vectorized edge list construction (no R-level loop)
n_nb_vec <- vapply(rook_neighbors_unique, length, integer(1))
cell_idx <- rep(seq_len(n_cells), times = n_nb_vec)
nb_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Expand across years
base_from <- rep((cell_idx - 1L) * n_years, each = n_years) + rep(seq_len(n_years), times = length(cell_idx))
base_to   <- rep((nb_idx - 1L) * n_years, each = n_years) + rep(seq_len(n_years), times = length(nb_idx))

edges <- data.table(from = base_from, to = base_to)
```

This eliminates the loop entirely but uses more transient memory. On a 16 GB machine it is feasible (~1.1 GB for the intermediate vectors).
END_RESPONSE>>>
