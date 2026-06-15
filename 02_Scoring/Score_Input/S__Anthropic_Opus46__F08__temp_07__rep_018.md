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
run_id: Anthropic_F08_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. It constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor graph is static.** Cell A's neighbors are the same in 1992 as in 2019. The `rook_neighbors_unique` object encodes this once, yet `build_neighbor_lookup` re-derives neighbor relationships for every cell-year combination â€” effectively duplicating the spatial graph 28 times.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M string keys (`paste(id, year, sep="_")`), builds a named lookup vector, and then for each of the 6.46M rows, pastes neighbor keys and looks them up. This is O(NÃ—K) string operations where N â‰ˆ 6.46M and K â‰ˆ average neighbor count (~4 for rook), totaling ~25â€“30 billion character operations.

3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the 6.46M-element list with `lapply`. The per-element overhead of R's `lapply` on a list this long is substantial.

4. **Memory pressure.** A 6.46M-element list of integer vectors, plus intermediate string vectors, can easily consume several GB on a 16 GB machine, causing GC thrashing.

**In summary:** The code treats a static spatial problem as if it were a dynamic one, paying a 28Ã— penalty in both time and memory.

---

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors) from the *dynamic attributes* (variable values that change by year).

1. **Build a cell-level neighbor index once** â€” a list of length 344,208 mapping each cell's position to the positions of its neighbors. This is just `rook_neighbors_unique` itself (an `nb` object already provides this).

2. **Process year-by-year.** For each year, extract the subset of rows, pull the variable column as a vector, and use the static cell-level neighbor list to compute max/min/mean via fast vectorized operations. Each year-slice has exactly 344,208 rows (one per cell), so indexing is direct â€” no string keys needed.

3. **Vectorize the neighbor aggregation** using `vapply` on the 344K-cell list (not the 6.46M-row list), once per year per variable. This reduces the inner-loop iterations from 6.46M to 344K â€” an 18.75Ã— reduction â€” and eliminates all string operations.

4. **Pre-sort data** by `(id, year)` or `(year, id)` to guarantee positional alignment with the cell index, enabling direct integer indexing.

**Expected speedup:** From ~86 hours to roughly **5â€“15 minutes** (conservative estimate), depending on I/O and GC behavior.

**Numerical equivalence:** The same `max`, `min`, `mean` of the same neighbor values are computed â€” only the indexing strategy changes. The trained Random Forest model is untouched.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec,
#                pop_density, def, usd_est_n2, ... (~6.46M rows)
#   - id_order: vector of 344,208 cell IDs defining the positional index
#               that corresponds to rook_neighbors_unique
#   - rook_neighbors_unique: an nb object (list of length 344,208), where
#               element i is an integer vector of neighbor positions in id_order
#   - rf_model: the pre-trained Random Forest (unchanged)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {
  # ---- Step 0: Convert to data.table for speed ----
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  # ---- Step 1: Build static cell-position lookup (once) ----
  # Map each cell ID to its position in id_order (1-based).
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # ---- Step 2: Ensure consistent row ordering ----
  # Sort by (id, year) so we can reliably extract year-slices.
  # Add a column for cell position to enable direct indexing.
  cell_data[, cell_pos := id_to_pos[as.character(id)]]

  # Sort by year then cell_pos so that within each year,

  # rows are in id_order position order (1..344208).
  setkey(cell_data, year, cell_pos)

  # ---- Step 3: Pre-allocate output columns ----
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }

  # ---- Step 4: Process year-by-year, variable-by-variable ----
  for (yr in years) {
    # Get row indices for this year (contiguous block due to setkey)
    yr_rows <- which(cell_data$year == yr)

    # Verify alignment: yr_rows should be exactly n_cells long
    # and cell_pos should run 1..n_cells
    if (length(yr_rows) != n_cells) {
      # Handle case where some cells are missing in a year:
      # Build a full-length vector with NAs for missing cells.
      yr_cell_pos <- cell_data$cell_pos[yr_rows]
      pos_to_row  <- rep(NA_integer_, n_cells)
      pos_to_row[yr_cell_pos] <- yr_rows
      aligned <- FALSE
    } else {
      # Fast path: all cells present, rows are in cell_pos order
      aligned <- TRUE
    }

    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)

      if (aligned) {
        # Extract the variable values in cell_pos order (1..n_cells)
        vals <- cell_data[[var_name]][yr_rows]

        # Compute neighbor stats for each cell using static neighbor list
        stats <- vapply(seq_len(n_cells), function(i) {
          nb_idx <- neighbors[[i]]
          # nb objects use 0 to indicate no neighbors
          if (length(nb_idx) == 1L && nb_idx[1L] == 0L) {
            return(c(NA_real_, NA_real_, NA_real_))
          }
          nv <- vals[nb_idx]
          nv <- nv[!is.na(nv)]
          if (length(nv) == 0L) {
            return(c(NA_real_, NA_real_, NA_real_))
          }
          c(max(nv), min(nv), mean(nv))
        }, numeric(3))
        # stats is 3 x n_cells matrix

        set(cell_data, i = yr_rows, j = col_max,  value = stats[1L, ])
        set(cell_data, i = yr_rows, j = col_min,  value = stats[2L, ])
        set(cell_data, i = yr_rows, j = col_mean, value = stats[3L, ])

      } else {
        # Slow path for incomplete years
        vals_full <- rep(NA_real_, n_cells)
        vals_full[cell_data$cell_pos[yr_rows]] <-
          cell_data[[var_name]][yr_rows]

        stats <- vapply(seq_len(n_cells), function(i) {
          nb_idx <- neighbors[[i]]
          if (length(nb_idx) == 1L && nb_idx[1L] == 0L) {
            return(c(NA_real_, NA_real_, NA_real_))
          }
          nv <- vals_full[nb_idx]
          nv <- nv[!is.na(nv)]
          if (length(nv) == 0L) {
            return(c(NA_real_, NA_real_, NA_real_))
          }
          c(max(nv), min(nv), mean(nv))
        }, numeric(3))

        # Map back only to rows that exist
        yr_cell_pos <- cell_data$cell_pos[yr_rows]
        set(cell_data, i = yr_rows, j = col_max,  value = stats[1L, yr_cell_pos])
        set(cell_data, i = yr_rows, j = col_min,  value = stats[2L, yr_cell_pos])
        set(cell_data, i = yr_rows, j = col_mean, value = stats[3L, yr_cell_pos])
      }
    }

    # Progress reporting
    message(sprintf("Year %d complete (%d of %d)", yr,
                    which(years == yr), n_years))
  }

  # ---- Step 5: Clean up helper column ----
  cell_data[, cell_pos := NULL]

  return(cell_data)
}

# =============================================================================
# USAGE
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# ---- Prediction with the pre-trained Random Forest (unchanged) ----
# cell_data now has the same 15 neighbor feature columns
# (neighbor_max_*, neighbor_min_*, neighbor_mean_* for each of 5 vars)
# with numerically identical values to the original implementation.

predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element list built via string keys | Static `nb` object used directly (344K elements, built once) |
| **Inner loop iterations** | 6.46M per variable (32.3M total) | 344K per year Ã— 28 years Ã— 5 vars = 48.2M, but each iteration is pure integer indexing (no string ops) |
| **String operations** | ~50M `paste()` calls + named-vector lookups | **Zero** |
| **Memory** | ~6.46M-element list + string vectors (~4â€“8 GB) | One numeric vector of length 344K per year-variable pass (~2.6 MB) |
| **Estimated runtime** | ~86+ hours | ~5â€“15 minutes |
| **Numerical result** | max/min/mean of neighbor values | **Identical** â€” same values, same neighbor sets |
| **RF model** | Pre-trained | **Untouched** â€” no retraining |
END_RESPONSE>>>
