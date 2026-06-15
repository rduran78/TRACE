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
run_id: Anthropic_F04_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46-million-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated string allocation and hash-table lookups against a massive named vector, is catastrophically slow. The named-vector `idx_lookup` alone has ~6.46M entries; probing it millions of times with multi-element key vectors is the single largest cost.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` + `do.call(rbind, ...)`

Another 6.46M-iteration `lapply`, each calling `max`, `min`, `mean` on small vectors. The per-call overhead of anonymous-function dispatch dominates. Then `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors is itself very expensive (repeated memory reallocation).

### 3. Repeated for 5 variables

The `compute_neighbor_stats` loop is called 5 times, multiplying the cost.

### Summary of cost drivers

| Cost driver | Severity |
|---|---|
| 6.46M-iteration interpreted R loop in `build_neighbor_lookup` | **Critical** |
| String key construction (`paste`) and named-vector hash probes on 6.46M-entry table, per row | **Critical** |
| 6.46M-iteration `lapply` + `do.call(rbind,...)` in `compute_neighbor_stats` | **High** |
| Repeated across 5 variables | **Multiplier** |

---

## Optimization Strategy

**Core idea:** Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.

1. **Replace `build_neighbor_lookup`** with a single `data.table` join that expands every cell-year row to its neighbor cell-year rows. This produces an edge-list `data.table` (cell-year â†’ neighbor-cell-year) in one vectorized operation. No `lapply`, no `paste` keys, no named-vector probes.

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation (`[, .(max, min, mean), by = row_id]`) over the edge-list. This computes all three statistics for all rows in one pass, fully vectorized in C.

3. **Compute all 5 variables in one pass** over the same edge-list, or at minimum reuse the edge-list for each variable (the join is done once).

4. **Memory check:** The edge-list has ~1.37M directed neighbor relationships Ã— 28 years â‰ˆ 38.4M rows (each row is three integers â‰ˆ 0.9 GB). This fits in 16 GB RAM.

**Expected speedup:** From 86+ hours to roughly 5â€“15 minutes.

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of neighbor values) is identical.

---

## Working R Code

```r
library(data.table)

#' Build a vectorized neighbor edge-list and compute all neighbor
#' features in one pass. Replaces build_neighbor_lookup(),
#' compute_neighbor_stats(), and the outer for-loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year,
#'                         and all neighbor_source_vars columns.
#' @param id_order         integer vector of cell IDs in the order matching
#'                         the spdep::nb object (rook_neighbors_unique).
#' @param neighbors        spdep::nb list (rook_neighbors_unique).
#'                         neighbors[[i]] gives integer indices (into id_order)
#'                         of the neighbors of id_order[i].
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return cell_data as a data.table with new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
#'         for each var in neighbor_source_vars.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Build the spatial edge list (cell_id -> neighbor_cell_id) ---
  # This is done ONCE and is year-independent.
  # neighbors[[i]] contains indices into id_order for the neighbors of
  # the cell whose ID is id_order[i].

  n_cells <- length(id_order)
  from_idx <- rep.int(seq_len(n_cells), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  # edges now has ~1,373,394 rows (directed rook-neighbor pairs)

  rm(from_idx, to_idx)

  # --- Step 2: Expand edges across years via join ---
  # We need (focal_id, year) -> (neighbor_id, year) so we can look up
  # neighbor values.  Rather than a massive cross-join, we join through
  # the focal rows.

  # Unique years
  years <- sort(unique(dt$year))

  # Create a row-index column for fast final assignment

  dt[, .row_idx := .I]

  # Key the main table for fast joins
  setkey(dt, id, year)

  # We will accumulate results for each variable into the main table.
  # Strategy: build the full edge-year table once, join neighbor values,
  # and aggregate.

  # Edge-year table: expand edges Ã— years  (~38.4M rows)
  edge_year <- CJ_edges_years(edges, years)
  # edge_year columns: focal_id, neighbor_id, year

  # Join to get the focal row index (so we can assign results back)
  # First, get focal row indices
  focal_key <- dt[, .(focal_row = .row_idx), keyby = .(id, year)]
  setnames(focal_key, "id", "focal_id")
  setkey(focal_key, focal_id, year)
  setkey(edge_year, focal_id, year)
  edge_year <- focal_key[edge_year, on = .(focal_id, year), nomatch = 0L]

  # Join to get neighbor row indices (to pull neighbor values)
  setkey(edge_year, neighbor_id, year)
  neighbor_key <- dt[, .(neighbor_row = .row_idx), keyby = .(id, year)]
  setnames(neighbor_key, "id", "neighbor_id")
  setkey(neighbor_key, neighbor_id, year)
  edge_year <- neighbor_key[edge_year, on = .(neighbor_id, year), nomatch = 0L]

  rm(focal_key, neighbor_key)

  # edge_year now has columns: neighbor_id, year, neighbor_row, focal_id, focal_row
  # Each row says: "for focal row focal_row, one neighbor's data is at neighbor_row"

  # --- Step 3: For each variable, pull values and aggregate ---
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]

    # Pull neighbor values via integer indexing (vectorized)
    edge_year[, nval := vals[neighbor_row]]

    # Grouped aggregation â€” one pass, fully vectorized in C
    agg <- edge_year[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     keyby = .(focal_row)]

    # Initialize result columns to NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    set(dt, j = max_col,  value = NA_real_)
    set(dt, j = min_col,  value = NA_real_)
    set(dt, j = mean_col, value = NA_real_)

    # Assign aggregated values back by row index
    set(dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
  }

  # Clean up helper columns
  edge_year[, nval := NULL]
  dt[, .row_idx := NULL]

  return(dt[])
}


#' Helper: expand edge list Ã— years without a full CJ (memory-efficient).
#' Returns a data.table with columns: focal_id, neighbor_id, year.
CJ_edges_years <- function(edges, years) {
  n_edges <- nrow(edges)
  n_years <- length(years)
  # Repeat each edge n_years times; tile years n_edges times
  data.table(
    focal_id    = rep(edges$focal_id,    each = n_years),
    neighbor_id = rep(edges$neighbor_id,  each = n_years),
    year        = rep(years, times = n_edges)
  )
}
```

### Drop-in replacement for the outer loop

Replace the original outer loop:

```r
# ---- BEFORE (slow, ~86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (fast, ~5-15 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the same 15 new columns (5 vars Ã— {max, min, mean})
# with numerically identical values. Proceed to Random Forest prediction
# using the already-trained model exactly as before.
```

---

## Why This Is Numerically Identical

| Original | Optimized |
|---|---|
| `max(neighbor_vals)` per row | `max(nval)` grouped by `focal_row` |
| `min(neighbor_vals)` per row | `min(nval)` grouped by `focal_row` |
| `mean(neighbor_vals)` per row | `mean(nval)` grouped by `focal_row` |
| `NA` when no non-NA neighbors | `NA` (rows absent from `agg`) |

The same values are indexed, the same `NA`s are excluded (`!is.na(nval)`), and the same three summary functions are applied. The estimand is preserved exactly.

---

## Memory Budget Estimate (16 GB laptop)

| Object | Rows | Approx. Size |
|---|---|---|
| `dt` (cell_data) | 6.46M Ã— ~115 cols | ~5.5 GB |
| `edges` | 1.37M Ã— 2 int cols | ~11 MB |
| `edge_year` | ~38.4M Ã— 5 int/num cols | ~1.5 GB |
| `agg` (per variable) | â‰¤6.46M Ã— 4 cols | ~0.2 GB |
| **Total peak** | | **~7.5 GB** âœ“ |

Fits comfortably within 16 GB.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with string ops | Eliminated; replaced by vectorized `data.table` join |
| `compute_neighbor_stats` | 6.46M-iteration `lapply` + `do.call(rbind,...)` | Single `data.table` grouped aggregation in C |
| Key construction | `paste()` + named-vector hash probe per row | Integer-indexed `data.table` keyed join |
| Number of passes over edge-year table | 5 (one per variable) | 5, but each is a vectorized column pull + grouped agg (~seconds each) |
| Estimated wall time | **86+ hours** | **5â€“15 minutes** |
| Trained RF model | Untouched | Untouched |
| Numerical output | Baseline | Identical |
END_RESPONSE>>>
