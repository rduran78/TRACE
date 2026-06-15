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
run_id: Anthropic_F02_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction**: `paste(id, year, sep="_")` is called millions of times, both during setup and inside the per-row lambda.
- **Named vector lookup** (`idx_lookup[neighbor_keys]`): Named vector indexing in R is backed by hashing, but doing it row-by-row in an `lapply` over 6.46M rows means millions of small hash lookups with intermediate character vector allocations.
- **Result**: This single function likely accounts for the majority of the ~86-hour runtime. The `lapply` creates ~6.46M small integer vectors, and each iteration does string concatenation, subsetting, and `NA` filtering.

### 2. `compute_neighbor_stats` â€” Another O(n) `lapply` with per-row subsetting and aggregation
- For each of the 6.46M rows, it subsets a numeric vector by index, drops NAs, and computes max/min/mean.
- Called 5 times (once per neighbor source variable), so ~32.3M iterations total.
- The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also expensive.

### 3. Memory pressure
- 6.46M rows Ã— 110 columns already occupies several GB. Intermediate list objects (the neighbor lookup list of 6.46M elements, plus temporary character vectors) push memory toward or past the 16 GB limit, triggering GC thrashing.

---

## Optimization Strategy

The strategy rests on three principles: **vectorize**, **use data.table for grouped operations**, and **eliminate the per-row R-level loop entirely**.

| Step | What changes | Speedup mechanism |
|---|---|---|
| A | Replace the row-level `lapply` in `build_neighbor_lookup` with a `data.table` equi-join | Eliminates millions of string ops and named-vector lookups; C-level join |
| B | Replace the row-level `lapply` in `compute_neighbor_stats` with a grouped `data.table` aggregation | Vectorized grouped max/min/mean in C |
| C | Process all 5 variables in one grouped aggregation pass | Avoids 5 separate full-table scans |
| D | Avoid materializing a 6.46M-element R list for the neighbor lookup | Cuts memory by several GB |

**Estimated runtime**: Under 10 minutes on a 16 GB laptop (data.table joins and grouped aggregations on ~50M-row expanded neighbor tables are routine).

**Preservation guarantees**:
- The trained Random Forest model is untouched; we only modify feature engineering.
- The output columns are numerically identical (same max, min, mean of the same neighbor sets).

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all neighbor_source_vars columns.
#' @param id_order         integer vector of cell IDs in the order that matches
#'                         the index positions in rook_neighbors_unique.
#' @param rook_neighbors   spdep::nb list (rook_neighbors_unique).  Element i
#'                         is an integer vector of positional indices into
#'                         id_order giving the neighbors of id_order[i].
#' @param neighbor_source_vars  character vector of variable names
#'                              (e.g. c("ntl","ec","pop_density","def","usd_est_n2"))
#'
#' @return cell_data as a data.table, with new columns
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
#'         for every var in neighbor_source_vars.

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # ------------------------------------------------------------------
  # Step 1: Build an edge table (cell_id -> neighbor_cell_id) from the
  #         nb object.  This is done once and is very fast.
  # ------------------------------------------------------------------
  # id_order[i] is the cell_id for position i.
  # rook_neighbors[[i]] gives the neighbor *positions* for position i.

  edge_from <- integer(0)
  edge_to   <- integer(0)

  # Pre-compute total edges for pre-allocation
  n_edges <- sum(lengths(rook_neighbors))
  edge_from <- integer(n_edges)
  edge_to   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(rook_neighbors)) {
    nb_i <- rook_neighbors[[i]]
    # spdep uses 0L to denote "no neighbors"
    if (length(nb_i) == 1L && nb_i[1L] == 0L) next
    n_i <- length(nb_i)
    idx <- pos:(pos + n_i - 1L)
    edge_from[idx] <- id_order[i]
    edge_to[idx]   <- id_order[nb_i]
    pos <- pos + n_i
  }

  # Trim if there were zero-neighbor entries
  if (pos - 1L < n_edges) {
    edge_from <- edge_from[1:(pos - 1L)]
    edge_to   <- edge_to[1:(pos - 1L)]
  }

  edges_dt <- data.table(id = edge_from, neighbor_id = edge_to)

  # ------------------------------------------------------------------
  # Step 2: Convert cell_data to data.table (no-copy if already one).
  # ------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Ensure keyed for fast join
  # We need: for every (id, year), look up (neighbor_id, year).
  # Strategy:

  #   expanded = edges_dt[cell_data, on = "id"]          -- gives (id, year, neighbor_id, own vars)
  #   then join cell_data again on .(id = neighbor_id, year) to get neighbor vars
  #   then group by (id, year) and aggregate.

  # ------------------------------------------------------------------
  # Step 3: Expand â€” attach year (and row identity) to every edge.
  # ------------------------------------------------------------------
  # We only need id and year from cell_data for the first join, plus a row key.
  cell_data[, .row_id := .I]

  # Minimal table for the first join: just id and year and row_id
  id_year <- cell_data[, .(id, year, .row_id)]
  setkey(id_year, id)
  setkey(edges_dt, id)

  # For each (id, year) row, expand to all its neighbors
  # Result columns: id, year, .row_id, neighbor_id
  expanded <- edges_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year, .row_id

  # ------------------------------------------------------------------
  # Step 4: Join neighbor values.
  # ------------------------------------------------------------------
  # We need the neighbor_source_vars values for (neighbor_id, year).
  # Build a lookup keyed on (id, year) containing only the vars we need.
  keep_cols <- c("id", "year", neighbor_source_vars)
  nb_vals <- cell_data[, ..keep_cols]
  setnames(nb_vals, "id", "neighbor_id")
  setkey(nb_vals, neighbor_id, year)

  setkey(expanded, neighbor_id, year)
  expanded <- nb_vals[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, <source_vars>, id, .row_id

  # ------------------------------------------------------------------
  # Step 5: Grouped aggregation â€” compute max, min, mean per (.row_id)
  #         for all source vars at once.
  # ------------------------------------------------------------------
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0(v, "_neighbor_max")]]  <-
      bquote(if (all(is.na(.(v_sym)))) NA_real_ else max(.(v_sym), na.rm = TRUE),
             list(v_sym = v_sym))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <-
      bquote(if (all(is.na(.(v_sym)))) NA_real_ else min(.(v_sym), na.rm = TRUE),
             list(v_sym = v_sym))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <-
      bquote(mean(.(v_sym), na.rm = TRUE),
             list(v_sym = v_sym))
  }

  agg_result <- expanded[, eval(as.call(c(as.name("list"),
                                           agg_exprs))),
                          by = .row_id]

  # ------------------------------------------------------------------
  # Step 6: Left-join aggregated stats back onto cell_data.
  # ------------------------------------------------------------------
  setkey(agg_result, .row_id)
  setkey(cell_data, .row_id)

  # Rows that had no neighbors won't appear in agg_result -> NA after join.
  new_cols <- setdiff(names(agg_result), ".row_id")
  cell_data[agg_result, (new_cols) := mget(paste0("i.", new_cols)),
            on = ".row_id"]

  cell_data[, .row_id := NULL]

  return(cell_data)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
library(data.table)

# ---- Load your existing objects ----
# cell_data              : the panel data.frame  (~6.46M rows)
# id_order               : integer vector mapping position -> cell_id
# rook_neighbors_unique  : spdep::nb object (precomputed, loaded from disk)
# rf_model               : the trained Random Forest (untouched)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Replace the slow loop with one call ----
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains the 15 new columns:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ...
# identical numerical values to the original pipeline.

# ---- Predict with the existing RF model (unchanged) ----
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Identical

| Original | Optimized | Equivalence |
|---|---|---|
| For row `i`, find all rows sharing `(neighbor_id, year[i])` | `data.table` equi-join on `(neighbor_id, year)` | Same row set selected |
| `max(vals)`, `min(vals)`, `mean(vals)` after dropping NAs | `max(..., na.rm=TRUE)`, `min(..., na.rm=TRUE)`, `mean(..., na.rm=TRUE)` with `all(is.na(...))` guard | Identical IEEE 754 results |
| Rows with no neighbors get `NA` | Left-join leaves unmatched `.row_id` as `NA` | Identical |

---

## Memory Estimate

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 numeric cols) | ~5.4 GB |
| `edges_dt` (~1.37M rows Ã— 2 int cols) | ~11 MB |
| `expanded` (â‰ˆ6.46M Ã— 4 rook neighbors Ã— 7 cols) | ~4â€“6 GB peak, freed after aggregation |
| `agg_result` (6.46M Ã— 16 cols) | ~0.8 GB |

Peak is ~12 GB, feasible on a 16 GB machine. If memory is still tight, process the 5 variables in two batches (3 + 2) instead of all at once â€” replace the single `agg_exprs` call with two calls, each computing stats for a subset of variables, and join both results back.

### Optional: batched variant for tighter memory

```r
# Split into batches to reduce peak memory of 'expanded'
batch1 <- c("ntl", "ec", "pop_density")
batch2 <- c("def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, id_order,
                                           rook_neighbors_unique, batch1)
gc()
cell_data <- compute_all_neighbor_features(cell_data, id_order,
                                           rook_neighbors_unique, batch2)
gc()
```

This cuts the peak width of `expanded` and keeps memory comfortably under 16 GB.
END_RESPONSE>>>
