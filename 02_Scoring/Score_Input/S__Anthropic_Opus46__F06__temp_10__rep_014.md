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
run_id: Anthropic_F06_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string hashing

`build_neighbor_lookup` creates a lookup list with **~6.46 million entries**. For each row, it:
1. Converts the cell `id` to a reference index via a named character vector lookup.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes `id_year` keys as strings.
4. Looks them up in a named character vector (`idx_lookup`) of length 6.46M.

Named vector lookups in R use **linear hashing on strings** that degrades badly at millions of keys. Building the 6.46M-element list of neighbor row-indices involves ~6.46M Ã— ~4 neighbors Ã— string paste + match operations. This alone can take **tens of hours**.

### Bottleneck B: `compute_neighbor_stats` â€” Repeated `lapply` over 6.46M elements

`compute_neighbor_stats` iterates over the full 6.46M-row lookup **once per variable** (Ã—5 variables), computing `max`, `min`, `mean` for each row's neighbor values using `lapply` and a per-element anonymous function with subsetting and NA removal. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also slow.

### Summary of cost drivers

| Step | Operations | Estimated time |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named-vector lookups in 6.46M-key vector | ~40â€“60 hrs |
| `compute_neighbor_stats` Ã— 5 vars | 5 Ã— 6.46M `lapply` iterations + `do.call(rbind, 6.46M)` | ~25â€“30 hrs |
| **Total** | | **~65â€“90 hrs** |

---

## 2. Optimization Strategy

### Key insight: Eliminate all string operations; work entirely with integer indices and vectorized/matrix operations.

**Step 1: Replace the string-keyed lookup with integer arithmetic.**

Each row can be located by `(cell_index, year_index)`. If the data is sorted by `(id, year)` and every cell has all 28 years, then the row index for cell `c` (1-indexed in `id_order`) in year `y` (1-indexed) is simply `(c - 1) * 28 + y`. No string pasting, no hash lookups.

If the panel is unbalanced (some cell-years missing), we build a small integer matrix mapping `(cell_index, year_index) â†’ row_index` using direct integer indexing â€” still O(1) per lookup and trivially fast.

**Step 2: Build a sparse directed-edge representation (from_row, to_row) as two integer vectors.**

Expand the `nb` object into a directed edge list of ~1.37M cell-pairs, then replicate across 28 years to get ~38.5M `(from_row, to_row)` pairs. This is a one-time vectorized operation.

**Step 3: Compute neighbor stats using `rowsum()` or grouping on the edge list.**

For each variable, extract `vals[to_row]`, group by `from_row`, and compute max/min/mean using fast vectorized grouped operations (via `data.table` or direct C-level `rowsum` equivalent). This avoids all per-row `lapply`.

**Expected speedup: from ~86 hours to ~2â€“5 minutes.**

---

## 3. Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation for cell-year panel data
#' with rook contiguity.
#'
#' @param cell_data     data.frame/data.table with columns: id, year, and all
#'                      neighbor_source_vars. Rows need not be sorted but every
#'                      cell-year combination must have a unique row.
#' @param id_order       integer/character vector: the cell IDs in the order
#'                      matching rook_neighbors_unique (i.e., id_order[i] is
#'                      the cell whose neighbors are rook_neighbors_unique[[i]]).
#' @param rook_neighbors_unique  spdep nb object (list of integer vectors of
#'                      neighbor indices into id_order).
#' @param neighbor_source_vars  character vector of variable names to summarise.
#' @return cell_data with new columns: {var}_max, {var}_min, {var}_mean for
#'         each var in neighbor_source_vars.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ---- Convert to data.table (by reference if already one) ----
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  # ---- Step 1: Build integer cell-index and year-index mappings ----
  # Map cell id -> sequential cell index (1-based, matching id_order)
  cell_id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map year -> sequential year index
  all_years  <- sort(unique(dt$year))
  n_years    <- length(all_years)
  year_to_idx <- setNames(seq_along(all_years), as.character(all_years))

  # Add integer indices to dt
  dt[, cell_idx := cell_id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_idx[as.character(year)]]

  # ---- Step 2: Build (cell_idx, year_idx) -> row_index lookup matrix ----
  n_cells <- length(id_order)
  # Integer matrix: rows = cells, cols = years. 0 means missing.
  cell_year_to_row <- matrix(0L, nrow = n_cells, ncol = n_years)
  cell_year_to_row[cbind(dt$cell_idx, dt$year_idx)] <- seq_len(nrow(dt))

  # ---- Step 3: Build directed cell-level edge list from nb object ----
  # Each edge: (from_cell_idx, to_cell_idx)
  from_cell <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_cell <- unlist(rook_neighbors_unique, use.names = FALSE)
  # Remove the 0-neighbor sentinel that spdep uses for islands
  valid <- to_cell != 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]
  n_edges_cell <- length(from_cell)

  # ---- Step 4: Expand to (from_row, to_row) across all years ----
  # For each cell-edge Ã— year, look up the actual row indices.
  # Replicate each cell-edge n_years times:
  from_cell_exp <- rep(from_cell, each = n_years)
  to_cell_exp   <- rep(to_cell,   each = n_years)
  year_idx_exp  <- rep(seq_len(n_years), times = n_edges_cell)

  # Look up row indices via the matrix (vectorised)
  from_row <- cell_year_to_row[cbind(from_cell_exp, year_idx_exp)]
  to_row   <- cell_year_to_row[cbind(to_cell_exp,   year_idx_exp)]

  # Keep only edges where both the focal and neighbor row exist
  valid_edge <- (from_row > 0L) & (to_row > 0L)
  from_row <- from_row[valid_edge]
  to_row   <- to_row[valid_edge]

  # Free intermediate large vectors
  rm(from_cell_exp, to_cell_exp, year_idx_exp, valid_edge)

  n_rows <- nrow(dt)

  # ---- Step 5: For each variable, compute grouped max, min, mean ----
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]

    # Get neighbor values aligned to each directed edge
    neighbor_vals <- vals[to_row]

    # Mask NAs: we need to exclude them from aggregation
    not_na <- !is.na(neighbor_vals)
    fr <- from_row[not_na]
    nv <- neighbor_vals[not_na]

    if (length(fr) == 0L) {
      # All NA â€” set output columns to NA
      set(dt, j = paste0(var_name, "_max"),  value = rep(NA_real_, n_rows))
      set(dt, j = paste0(var_name, "_min"),  value = rep(NA_real_, n_rows))
      set(dt, j = paste0(var_name, "_mean"), value = rep(NA_real_, n_rows))
      next
    }

    # --- Grouped max ---
    # data.table approach for fast grouped aggregation
    edge_dt <- data.table(fr = fr, nv = nv)

    agg <- edge_dt[, .(
      nb_max  = max(nv),
      nb_min  = min(nv),
      nb_mean = mean(nv)
    ), by = fr]

    # Allocate output columns (NA for rows with no valid neighbors)
    col_max  <- rep(NA_real_, n_rows)
    col_min  <- rep(NA_real_, n_rows)
    col_mean <- rep(NA_real_, n_rows)

    col_max[agg$fr]  <- agg$nb_max
    col_min[agg$fr]  <- agg$nb_min
    col_mean[agg$fr] <- agg$nb_mean

    set(dt, j = paste0(var_name, "_max"),  value = col_max)
    set(dt, j = paste0(var_name, "_min"),  value = col_min)
    set(dt, j = paste0(var_name, "_mean"), value = col_mean)

    rm(edge_dt, agg, col_max, col_min, col_mean, neighbor_vals, not_na, fr, nv)
  }

  # ---- Cleanup helper columns ----
  dt[, c("cell_idx", "year_idx") := NULL]

  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt[])
}
```

### Usage â€” drop-in replacement for the original outer loop:

```r
# ---- Load data and pre-trained model (unchanged) ----
# cell_data               <- readRDS("cell_data.rds")
# id_order                <- readRDS("id_order.rds")
# rook_neighbors_unique   <- readRDS("rook_neighbors_unique.rds")
# rf_model                <- readRDS("rf_model.rds")       # preserved as-is

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Compute all neighbor features (replaces the 86-hour loop) ----
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the pre-trained Random Forest (unchanged) ----
# preds <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why this preserves the original numerical estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor definition** | We expand exactly the same `rook_neighbors_unique` nb object â€” same directed pairs. |
| **Same aggregation functions** | `max`, `min`, `mean` computed on the identical set of non-NA neighbor values per focal row. |
| **Same NA handling** | Edges where the neighbor value is `NA` are excluded before aggregation; rows with zero valid neighbors get `NA` â€” identical to the original. |
| **Same column names** | `{var}_max`, `{var}_min`, `{var}_mean` (assuming `compute_and_add_neighbor_features` uses that convention; adjust the suffix pattern if yours differs). |
| **Random Forest untouched** | The trained model object is loaded and used for `predict()` only â€” no retraining. |

## 5. Performance estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~40â€“60 hrs | ~30 seconds (matrix fill + vectorized edge expansion) |
| 5 Ã— neighbor stat computation | ~25â€“30 hrs | ~1â€“3 minutes (`data.table` grouped aggregation on ~38.5M edges) |
| Peak RAM | ~8â€“12 GB (6.46M-element list of lists + string keys) | ~4â€“6 GB (integer vectors + data.table) |
| **Total** | **~86 hours** | **~2â€“5 minutes** |

The ~1,700Ã— speedup comes from eliminating all string operations, replacing named-vector lookups with direct integer matrix indexing, and replacing 6.46 million `lapply` iterations with vectorized `data.table` grouped aggregation.
END_RESPONSE>>>
