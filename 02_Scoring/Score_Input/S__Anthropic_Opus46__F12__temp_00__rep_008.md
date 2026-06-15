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
run_id: Anthropic_F12_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on a list of 6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses `lapply` to build a list in one pass, then binds once. This is standard and efficient R.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup` (a named character vector of length 6.46 million) via character matching. Named vector lookup by character key in R is O(n) per probe in the worst case (hash collisions aside, the overhead of repeated hashing and matching against a 6.46M-entry names vector is enormous).

2. **The `lapply` over 6.46 million rows** each performing: one `as.character` coercion, one named-vector lookup into `id_to_ref`, one subset of `id_order` by a variable-length index vector, one `paste` call generating multiple keys, one named-vector lookup into `idx_lookup` (6.46M entries), and one `is.na` filter. With ~1,373,394 directed neighbor relationships spread across 344,208 cells Ã— 28 years, the average row touches ~4 neighbors, meaning roughly **25.8 million** string constructions and hash lookups into a 6.46M-entry table â€” all inside an interpreted R loop.

3. **This function is called once and produces the lookup used by all 5 variables.** But that single call dominates total runtime. `compute_neighbor_stats()` is called 5 times and is comparatively cheap: it does only integer indexing into a numeric vector (vectorized, cache-friendly) plus simple `max`/`min`/`mean` on small neighbor sets.

**Estimated time breakdown (approximate):**
- `build_neighbor_lookup()`: ~80+ hours (character key construction and lookup, 6.46M interpreted iterations)
- `compute_neighbor_stats()` Ã— 5: ~1â€“3 hours total
- `do.call(rbind, ...)` Ã— 5: seconds each

## Optimization Strategy

1. **Eliminate all string key construction and character-based lookup.** Replace the `paste(..., sep="_")` keying scheme with direct integer arithmetic. Since each `(id, year)` pair maps to a row, build a fast integer-indexed lookup matrix or use a hash table (`data.table` or environment) keyed on integer pairs.

2. **Vectorize `build_neighbor_lookup()` entirely** using `data.table` joins. Expand the neighbor relationships into an edge table `(id, neighbor_id)`, join with the data's `(id, year)` â†’ `row_index` mapping on `(neighbor_id, year)` to get all neighbor row indices in one vectorized merge, then split by source row index.

3. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped aggregation on the edge table rather than `lapply` over 6.46M elements.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup
# ============================================================
# Instead of returning a list of length nrow(data), we return
# an edge data.table: (source_row, neighbor_row) which is far
# more efficient to construct and to aggregate over.

build_neighbor_edges <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Step 1: Build edge list of (cell_id -> neighbor_cell_id) from the nb object
  # This is only ~1.37M edges (or up to 344,208 * avg_neighbors)
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(cell_id = id_order[i], neighbor_cell_id = id_order[nb])
  }))

  # Step 2: Create a keyed lookup: (id, year) -> row_idx
  # We join edge_list with dt on year to expand across all years
  # For each (cell_id, year) row in dt, find all (neighbor_cell_id, year) rows

  # Get unique cell-year to row mapping
  cell_year <- dt[, .(cell_id = id, year, row_idx)]
  setkey(cell_year, cell_id, year)

  # For source rows: get (cell_id, year, source_row_idx)
  source <- cell_year[, .(cell_id, year, source_row = row_idx)]

  # Join source with edge_list to get (source_row, neighbor_cell_id, year)
  setkey(source, cell_id)
  setkey(edge_list, cell_id)

  # This is the key vectorized join: expand edges across years
  expanded <- edge_list[source, on = "cell_id", allow.cartesian = TRUE,
                        nomatch = NULL]
  # expanded now has: cell_id, neighbor_cell_id, year, source_row

  # Step 3: Join to find neighbor row indices
  setkey(cell_year, cell_id, year)
  setnames(cell_year, "cell_id", "neighbor_cell_id")
  setnames(cell_year, "row_idx", "neighbor_row")

  result <- cell_year[expanded, on = c("neighbor_cell_id", "year"),
                      nomatch = NULL]
  # result has: neighbor_cell_id, year, neighbor_row, cell_id, source_row

  result[, .(source_row, neighbor_row)]
}

# ============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ============================================================
compute_neighbor_stats_fast <- function(data, edges, var_name) {
  # edges: data.table with (source_row, neighbor_row)
  # Returns a data.table with columns:
  #   neighbor_max_{var_name}, neighbor_min_{var_name}, neighbor_mean_{var_name}
  # aligned to row order of data

  n <- nrow(data)
  vals <- data[[var_name]]

  # Attach neighbor values
  edge_vals <- edges[, .(source_row, nval = vals[neighbor_row])]
  edge_vals <- edge_vals[!is.na(nval)]

  # Grouped aggregation â€” single pass, vectorized
  agg <- edge_vals[, .(
    nmax  = max(nval),
    nmin  = min(nval),
    nmean = mean(nval)
  ), by = source_row]

  # Allocate output aligned to all rows (NA for rows with no neighbors)
  out <- data.table(
    nmax  = rep(NA_real_, n),
    nmin  = rep(NA_real_, n),
    nmean = rep(NA_real_, n)
  )
  out[agg$source_row, `:=`(nmax = agg$nmax,
                            nmin = agg$nmin,
                            nmean = agg$nmean)]

  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(out, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  out
}

# ============================================================
# MAIN PIPELINE (replaces the outer loop)
# ============================================================
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cat("Building neighbor edge table (vectorized)...\n")
  t0 <- Sys.time()
  edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  cat("  Edge table built:", nrow(edges), "edges in",
      round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  is_dt <- is.data.table(cell_data)
  if (!is_dt) cell_data <- as.data.table(cell_data)

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    t1 <- Sys.time()

    stats <- compute_neighbor_stats_fast(cell_data, edges, var_name)

    # Add columns to cell_data (in place for data.table)
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_data[, (max_col)  := stats[[max_col]]]
    cell_data[, (min_col)  := stats[[min_col]]]
    cell_data[, (mean_col) := stats[[mean_col]]]

    cat("  Done in", round(difftime(Sys.time(), t1, units = "secs"), 1), "sec\n")
  }

  if (!is_dt) cell_data <- as.data.frame(cell_data)

  cat("Neighbor pipeline complete.\n")
  cell_data
}

# ============================================================
# USAGE (drop-in replacement for the original outer loop)
# ============================================================
# cell_data <- run_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original.
# Expected runtime: ~5-15 minutes instead of 86+ hours.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M interpreted iterations with string key construction and character hash lookups | Eliminated entirely |
| **Lookup construction** | `lapply` + `paste` + named vector probe per row | Single `data.table` equi-join (vectorized, C-level) |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | `data.table` grouped aggregation (`[, .(), by=]`) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Numerical output** | `max`, `min`, `mean` per neighbor set | Identical `max`, `min`, `mean` per neighbor set |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |

The colleague's diagnosis was wrong: `do.call(rbind, result)` is a one-shot bind of pre-allocated list elements and is not the bottleneck. The real cost is the **6.46 million character-key constructions and named-vector lookups** inside `build_neighbor_lookup()`, which is solved by replacing the entire approach with vectorized `data.table` joins.
END_RESPONSE>>>
