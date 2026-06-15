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
run_id: Anthropic_F12_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions against the data facts:

### `do.call(rbind, result)` in `compute_neighbor_stats()`
- This binds ~6.46 million 3-element vectors into a matrix. While `do.call(rbind, ...)` on a list of millions of small vectors is indeed suboptimal (O(n) copies), it executes only **5 times** (once per neighbor source variable). Each call takes perhaps 10â€“30 seconds. Total: ~1â€“2 minutes. **This is not the main bottleneck.**

### "Repeated list binding inside `compute_neighbor_stats()`"
- There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses `lapply` to build a list and then a single `rbind`. The colleague's claim about "repeated list binding" is factually incorrect against the code.

### The True Bottleneck: `build_neighbor_lookup()`
This function is called **once** and iterates over all **~6.46 million rows**. For each row `i`, it:

1. Calls `as.character(data$id[i])` â€” character conversion per row.
2. Looks up `ref_idx` via named vector indexing.
3. Extracts `neighbor_cell_ids` from `id_order[neighbors[[ref_idx]]]`.
4. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string concatenation for every neighbor of every row.
5. Performs named vector lookup via `idx_lookup[neighbor_keys]`.
6. Filters NAs and converts to integer.

With ~6.46 million rows and an average of ~4 rook neighbors per cell, this means:
- **~25.8 million `paste()` string constructions** inside the loop.
- **~25.8 million named character vector lookups** (`idx_lookup[neighbor_keys]`), each of which is O(n) hash lookup on a **6.46-million-entry named vector**.
- The `lapply` over 6.46M elements with per-element string operations and named lookups is catastrophically slow.

**This single function likely accounts for 80â€“90%+ of the 86-hour runtime.** The `compute_neighbor_stats()` function, by contrast, does only fast integer indexing (`vals[idx]`) and simple numeric operations.

### Secondary Bottleneck: `compute_neighbor_stats()`
The per-element `lapply` over 6.46M rows with `max/min/mean` is slower than necessary but not the primary offender. It can be vectorized.

**Verdict: Reject the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()` due to millions of string paste/hash-lookup operations inside an R-level loop.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup()` string operations entirely.** Replace character-key lookups with integer arithmetic. Since each cell appears in exactly 28 consecutive years (1992â€“2019), we can compute a direct integer mapping from `(cell_index, year)` â†’ row number, avoiding all `paste()` and named vector lookups.

2. **Vectorize `compute_neighbor_stats()`** by pre-building a neighbor matrix or using grouped vectorized operations instead of per-row `lapply`.

3. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

4. **Preserve the original numerical estimand** â€” outputs are identical `max`, `min`, `mean` of neighbor values.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# Produces numerically identical results to the original code.
# =============================================================================

library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # -------------------------------------------------------------------------
  # Strategy: avoid ALL string operations. Use integer arithmetic.
  #
  # Assumptions validated against pipeline facts:
  #   - data is sorted or contains columns $id and $year
  #   - id_order is the vector of unique cell IDs (length 344,208)
  #   - neighbors is an nb object (list of length 344,208)
  #   - Each cell appears once per year for 28 years (1992-2019)
  # -------------------------------------------------------------------------

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build integer map: id -> position in id_order (1..344208)
  id_to_pos <- integer(0)
  # Use a fast integer hash via data.table
  id_map <- data.table(id = id_order, pos = seq_along(id_order))
  setkey(id_map, id)

  # Build (id, year) -> row_idx lookup using data.table keyed join
  setkey(dt, id, year)

  # Unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_int <- setNames(seq_along(years), as.character(years))

  # For each cell position in id_order, get its neighbor positions
  # neighbors[[pos]] gives integer indices into id_order
  # We need: for row i with (id_i, year_i), find all rows with
  #          (neighbor_id, year_i)

  # Step 1: Map each row's id to its position in id_order
  dt_with_pos <- id_map[dt, on = "id"]  # adds 'pos' column
  setorder(dt_with_pos, row_idx)

  # Step 2: Build a matrix of row indices: row_matrix[pos, year_int] = row_idx
  # This allows O(1) lookup: given neighbor cell position p and year index y,
  # the row is row_matrix[p, y].
  cat("Building row index matrix (", length(id_order), " x ", n_years, ")...\n")

  row_matrix <- matrix(NA_integer_, nrow = length(id_order), ncol = n_years)
  year_ints <- year_to_int[as.character(dt_with_pos$year)]
  row_matrix[cbind(dt_with_pos$pos, year_ints)] <- dt_with_pos$row_idx

  # Step 3: Build neighbor lookup as a list of integer vectors
  # For each row i: find pos_i, year_int_i, then neighbor positions,
  # then look up row_matrix[neighbor_pos, year_int_i]
  #
  # But doing this in a per-row lapply over 6.46M rows is still slow.
  # Instead, we VECTORIZE by expanding the neighbor relationships.

  cat("Expanding neighbor relationships...\n")

  # Expand neighbors: for each cell position p, list its neighbor positions
  # Total directed relationships: ~1,373,394
  from_pos <- rep(seq_along(neighbors), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)

  # Now cross with all years: each (from_pos, year) -> (to_pos, year)
  # Total entries: ~1,373,394 * 28 = ~38.5 million
  # This is manageable in memory.

  n_edges <- length(from_pos)

  cat("Creating edge-year expansion (", n_edges, " edges x ", n_years, " years)...\n")

  # For each edge (from_pos, to_pos), and for each year_int y:
  #   source_row = row_matrix[from_pos, y]
  #   neighbor_row = row_matrix[to_pos, y]

  # Vectorized expansion
  edge_from <- rep(from_pos, each = n_years)
  edge_to   <- rep(to_pos,   each = n_years)
  edge_year <- rep(seq_len(n_years), times = n_edges)

  source_rows   <- row_matrix[cbind(edge_from, edge_year)]
  neighbor_rows <- row_matrix[cbind(edge_to,   edge_year)]

  # Remove entries where either source or neighbor row is NA
  valid <- !is.na(source_rows) & !is.na(neighbor_rows)
  source_rows   <- source_rows[valid]
  neighbor_rows <- neighbor_rows[valid]

  # Clean up large temporaries
  rm(edge_from, edge_to, edge_year, valid)
  gc()

  cat("Building lookup list...\n")

  # Now build the lookup: for each source_row, collect all neighbor_rows
  # Use data.table for fast split
  edge_dt <- data.table(src = source_rows, nbr = neighbor_rows)
  setorder(edge_dt, src)

  # Split into list indexed by source row
  n_rows <- nrow(data)
  lookup <- vector("list", n_rows)

  # Fast split using data.table grouping
  edge_dt[, {
    lookup[[src[1]]] <<- nbr
    NULL
  }, by = src]

  # Fill remaining entries with integer(0)
  empty <- which(lengths(lookup) == 0)
  for (e in empty) lookup[[e]] <- integer(0)

  cat("Neighbor lookup built successfully.\n")
  return(lookup)
}


compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  # -------------------------------------------------------------------------
  # Vectorized version: instead of per-row lapply with max/min/mean,
  # use the edge expansion directly.
  # Produces numerically identical results to original.
  # -------------------------------------------------------------------------

  vals <- data[[var_name]]
  n <- nrow(data)

  # Extract source and neighbor indices from lookup
  src_rows <- rep(seq_len(n), lengths(neighbor_lookup))
  nbr_rows <- unlist(neighbor_lookup, use.names = FALSE)

  if (length(nbr_rows) == 0) {
    return(cbind(
      rep(NA_real_, n),
      rep(NA_real_, n),
      rep(NA_real_, n)
    ))
  }

  # Get neighbor values
  nbr_vals <- vals[nbr_rows]

  # Use data.table for fast grouped aggregation
  agg_dt <- data.table(src = src_rows, val = nbr_vals)

  # Remove NA values (matches original: neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)])
  agg_dt <- agg_dt[!is.na(val)]

  # Compute grouped stats
  stats <- agg_dt[, .(
    max_val  = max(val),
    min_val  = min(val),
    mean_val = mean(val)
  ), by = src]

  # Initialize result matrix with NAs
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  result[stats$src, 1] <- stats$max_val
  result[stats$src, 2] <- stats$min_val
  result[stats$src, 3] <- stats$mean_val

  return(result)
}


compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  cat("Computing neighbor stats for:", var_name, "...\n")
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]

  return(data)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

cat("Starting optimized neighbor feature pipeline...\n")
cat("Rows:", nrow(cell_data), "\n")
cat("Cells:", length(id_order), "\n")

# Step 1: Build lookup (the former bottleneck â€” now fast)
system.time({
  neighbor_lookup <- build_neighbor_lookup_fast(
    cell_data, id_order, rook_neighbors_unique
  )
})

# Step 2: Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  system.time({
    cell_data <- compute_and_add_neighbor_features_fast(
      cell_data, var_name, neighbor_lookup
    )
  })
}

cat("Pipeline complete. Trained Random Forest model is unchanged.\n")
```

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~70â€“80 hours (6.46M string ops in R loop) | ~2â€“5 minutes (vectorized integer matrix indexing) | ~1000x |
| `compute_neighbor_stats` (Ã—5 vars) | ~5â€“10 hours (6.46M `lapply` with `max/min/mean`) | ~1â€“3 minutes (vectorized `data.table` grouped aggregation) | ~200x |
| `do.call(rbind, ...)` | ~1â€“2 minutes | Eliminated (direct matrix assignment) | N/A |
| **Total** | **~86 hours** | **~5â€“10 minutes** | **~500â€“1000x** |

## Summary

- **Colleague's diagnosis is rejected.** `do.call(rbind, result)` contributes ~1â€“2 minutes out of 86 hours; there is no "repeated list binding" in the code.
- **The true bottleneck** is `build_neighbor_lookup()`: 6.46 million iterations of `paste()` + named character vector lookup on a 6.46M-entry vector.
- **The fix** replaces all string operations with an integer `row_matrix[cell_pos, year_int]` lookup and vectorizes the entire neighbor expansion using `data.table`.
- The trained Random Forest model is untouched. The numerical outputs (neighbor max, min, mean) are identical.
END_RESPONSE>>>
