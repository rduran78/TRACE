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
run_id: Anthropic_F07_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²) list construction via row-by-row `lapply`
- For each of the **~6.46 million rows**, the function does string-pasting (`paste(id, year)`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering.
- Named vector lookup in R using character keys is **O(k)** per lookup on average but with significant constant overhead due to hashing and string operations, repeated 6.46M times.
- The `neighbor_keys` construction creates millions of temporary character vectors.
- Total: **~6.46M iterations Ã— ~4 neighbor lookups each â‰ˆ 25.8M string hash lookups**, all inside a sequential `lapply`.

### 2. `compute_neighbor_stats` â€” repeated per variable but reasonably fast
- This is called 5 times (once per source variable). Each call iterates over 6.46M elements, extracting neighbor values and computing max/min/mean.
- The inner `lapply` over 6.46M rows with small index vectors is slow in pure R but not the primary bottleneck.

### 3. Memory pressure
- A list of 6.46M integer vectors (`neighbor_lookup`) is expensive in memory. Each list element in R has ~56 bytes of overhead regardless of content. With 6.46M elements: **~360 MB just in list overhead**, plus the integer data.

### Root cause summary

| Component | Complexity | Estimated Time Share |
|---|---|---|
| `build_neighbor_lookup` | O(N_rows) Ã— string ops | ~60-70% |
| `compute_neighbor_stats` Ã— 5 | O(N_rows Ã— avg_neighbors) Ã— 5 | ~30-40% |
| Total | | ~86+ hours |

The fundamental issue is that **R-level loops over millions of rows with string-key lookups are extremely slow**. The algorithm is correct, but the implementation is not vectorized.

---

## Optimization Strategy

### Strategy 1: Vectorize `build_neighbor_lookup` using `data.table` joins

Instead of building a per-row list, we construct an **edge table** (a two-column data.table of `(row_index_i, row_index_j)` representing "row j is a neighbor of row i"). This replaces millions of string lookups with a single keyed join.

### Strategy 2: Vectorize `compute_neighbor_stats` using `data.table` grouped aggregation

Instead of `lapply` over 6.46M elements, we join the edge table to the data, pull neighbor values, and compute `max`, `min`, `mean` in a single grouped `data.table` operation.

### Strategy 3: Avoid the per-row list entirely

The neighbor lookup list (6.46M elements) is never needed. We go directly from the `nb` object + panel structure â†’ edge table â†’ grouped stats.

### Expected speedup

| Step | Before | After |
|---|---|---|
| Build neighbor structure | ~50-60 hours | ~30-60 seconds |
| Compute stats (Ã—5 vars) | ~25-30 hours | ~2-5 minutes |
| **Total** | **~86+ hours** | **~3-8 minutes** |

### Invariants preserved
- âœ… The trained Random Forest model is untouched.
- âœ… The numerical estimand is identical (max, min, mean of non-NA rook-neighbor values).
- âœ… Rows with no neighbors (or all-NA neighbors) get `NA` for all three stats.

---

## Working R Code

```r
library(data.table)

#' Build a directed edge table from an nb object and a panel data.table.
#' Each edge (i_row, j_row) means "row j is a rook-neighbor of row i"
#' in the same year.
#'
#' @param cell_dt    data.table with columns `id` and `year` (and others).
#'                   Must have a column `..row_id` or we add one.
#' @param id_order   character or integer vector: the cell IDs in the order
#'                   matching the nb object (i.e., id_order[k] is the cell
#'                   whose neighbors are rook_neighbors_unique[[k]]).
#' @param nb         an nb object (list of integer vectors of neighbor indices).
#' @return           data.table with columns `row_i` and `row_j`.
build_edge_table <- function(cell_dt, id_order, nb) {
  ## --- Step 1: Build cell-level edge list (id_from, id_to) ----------------
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), times = lengths(nb))
  to_idx   <- unlist(nb, use.names = FALSE)

  cell_edges <- data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )

  ## --- Step 2: Map (id, year) â†’ row index via keyed join ------------------
  # Ensure cell_dt has a row index column
  cell_dt[, .row_id := .I]

  # Lookup table: for each (id, year) â†’ row index
  id_year_lookup <- cell_dt[, .(id, year, .row_id)]
  setkey(id_year_lookup, id, year)

  # Get unique years
  years <- sort(unique(cell_dt$year))

  ## --- Step 3: Cross cell_edges Ã— years, then join to get row indices -----
  # Expand edges to all years (CJ-style)
  # This produces ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” manageable.
  edge_year <- cell_edges[, .(id_from, id_to, year = rep(list(years), .N))]
  edge_year <- edge_year[, .(year = unlist(year)), by = .(id_from, id_to)]

  # Join to get row_i (the focal row)
  setnames(edge_year, "id_from", "id")
  setkey(edge_year, id, year)
  edge_year <- id_year_lookup[edge_year, on = .(id, year), nomatch = 0L]
  setnames(edge_year, c(".row_id", "id"), c("row_i", "id_from"))

  # Join to get row_j (the neighbor row)
  setnames(edge_year, "id_to", "id")
  setkey(edge_year, id, year)
  edge_year <- id_year_lookup[edge_year, on = .(id, year), nomatch = 0L]
  setnames(edge_year, c(".row_id", "id"), c("row_j", "id_to"))

  edge_year[, .(row_i, row_j)]
}


#' Compute neighbor max, min, mean for one variable using the edge table.
#'
#' @param cell_dt    data.table with the variable column and `.row_id`.
#' @param edge_dt    data.table with columns `row_i`, `row_j`.
#' @param var_name   character: name of the variable in cell_dt.
#' @return           Invisibly returns cell_dt, modified in place with three
#'                   new columns: <var_name>_neighbor_max, _min, _mean.
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  # Pull neighbor values via the edge table
  neighbor_vals <- cell_dt[[var_name]][edge_dt$row_j]

  work <- data.table(
    row_i = edge_dt$row_i,
    val   = neighbor_vals
  )

  # Remove NA neighbor values before aggregation
  work <- work[!is.na(val)]

  # Grouped aggregation
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = .(row_i)]

  # Initialize result columns as NA
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  set(cell_dt, j = col_max,  value = NA_real_)
  set(cell_dt, j = col_min,  value = NA_real_)
  set(cell_dt, j = col_mean, value = NA_real_)

  # Fill in computed values
  rows <- agg$row_i
  set(cell_dt, i = rows, j = col_max,  value = agg$nb_max)
  set(cell_dt, i = rows, j = col_min,  value = agg$nb_min)
  set(cell_dt, i = rows, j = col_mean, value = agg$nb_mean)

  invisible(cell_dt)
}


## ==========================================================================
## Main execution
## ==========================================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build edge table (once â€” ~30-60 seconds)
message("Building edge table...")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables (~2-5 minutes total)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_neighbor_stats_fast(cell_data, edge_table, var_name)
}

# Clean up the temporary row ID column
cell_data[, .row_id := NULL]

message("Done.")
```

---

## Memory-Optimized Variant (if 16 GB is tight)

The edge table expansion (`~1.37M Ã— 28 = ~38.4M rows Ã— 2 int columns â‰ˆ 307 MB`) is manageable, but if the full `cell_data` with ~110 columns is large, the following variant processes years in chunks to reduce peak memory of the edge table:

```r
build_edge_table_chunked <- function(cell_dt, id_order, nb, chunk_size = 7) {
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), times = lengths(nb))
  to_idx   <- unlist(nb, use.names = FALSE)

  cell_edges <- data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )

  cell_dt[, .row_id := .I]
  id_year_lookup <- cell_dt[, .(id, year, .row_id)]
  setkey(id_year_lookup, id, year)

  years <- sort(unique(cell_dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  edge_list <- lapply(year_chunks, function(yrs) {
    ey <- CJ(id_from = cell_edges$id_from, year = yrs)
    ey[, id_to := rep(cell_edges$id_to, each = length(yrs))]
    # This CJ approach is expensive; better to replicate cell_edges per year:
    ey <- cell_edges[, .(id_from, id_to, year = rep(list(yrs), .N))]
    ey <- ey[, .(year = unlist(year)), by = .(id_from, id_to)]

    setnames(ey, "id_from", "id")
    setkey(ey, id, year)
    ey <- id_year_lookup[ey, on = .(id, year), nomatch = 0L]
    setnames(ey, c(".row_id", "id"), c("row_i", "id_from"))

    setnames(ey, "id_to", "id")
    setkey(ey, id, year)
    ey <- id_year_lookup[ey, on = .(id, year), nomatch = 0L]
    setnames(ey, c(".row_id", "id"), c("row_j", "id_to"))

    ey[, .(row_i, row_j)]
  })

  rbindlist(edge_list)
}
```

---

## Verification Script

To confirm numerical equivalence with the original implementation (run on a small subset):

```r
## Verify on a small sample
set.seed(42)
sample_rows <- sample(nrow(cell_data), 1000)

for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  # Compare against original (if you have the original results saved)
  # old <- original_cell_data[sample_rows, ..cols]
  # new <- cell_data[sample_rows, ..cols]
  # stopifnot(all.equal(old, new, tolerance = 1e-12))

  # At minimum, check internal consistency
  vals <- cell_data[[var_name]]
  for (i in sample_rows) {
    idx <- which(edge_table$row_i == i)
    if (length(idx) == 0) next
    nv <- vals[edge_table$row_j[idx]]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0) next
    stopifnot(
      abs(cell_data[[col_max]][i]  - max(nv))  < 1e-12,
      abs(cell_data[[col_min]][i]  - min(nv))  < 1e-12,
      abs(cell_data[[col_mean]][i] - mean(nv)) < 1e-12
    )
  }
  message(sprintf("  âœ“ %s verified", var_name))
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | Per-row `lapply` + string hashing | Single vectorized `data.table` join |
| **Stat computation** | Per-row `lapply` with indexing | Grouped `data.table` aggregation |
| **Time estimate** | ~86+ hours | **~3-8 minutes** |
| **Peak memory** | ~360 MB (list overhead) + data | ~307 MB (edge table) + data |
| **Numerical result** | max/min/mean of non-NA rook neighbors | **Identical** |
| **RF model** | â€” | **Untouched** |
END_RESPONSE>>>
