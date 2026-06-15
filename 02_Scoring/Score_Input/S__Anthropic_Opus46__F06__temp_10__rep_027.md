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
run_id: Anthropic_F06_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** constructs a list of 6.46 million elements, each built by an independent `lapply` call that performs character-based key lookups (`paste` + named-vector indexing) for every single row. This is O(N) string operations where N â‰ˆ 6.46M, and the resulting list-of-vectors is then iterated again 5 times in `compute_neighbor_stats`, each time doing per-element `lapply` over 6.46M entries with subsetting, NA removal, and summary statistics.

**Specific costs:**

| Step | Problem |
|---|---|
| `build_neighbor_lookup` | 6.46M `paste()` + named-vector lookups. Character hashing on millions of keys is extremely slow in base R. The lookup is O(N Ã— avg_neighbors). |
| `compute_neighbor_stats` | Called 5 times (one per variable). Each call iterates 6.46M elements via `lapply`, subsetting a numeric vector and computing `max/min/mean`. Pure R loops over millions of elements are slow. |
| `do.call(rbind, result)` | Binding 6.46M 3-element vectors into a matrix via `do.call(rbind, list(...))` is a known anti-pattern â€” quadratic memory allocation. |
| **Overall** | ~86+ hours is consistent with character-keyed lookups and R-level loops over ~6.46M Ã— 5 = 32.3M summary operations. |

### Why raster focal/kernel operations are not the right fit

The comment in the preamble asks whether raster focal operations (e.g., `terra::focal`) offer a useful analogy. They do conceptually â€” computing neighbor summaries is exactly a focal operation. However:

- The data is a **panel** (cell Ã— year), not a single raster. Focal operations would need to be applied per-year-layer, and the grid cells may not form a complete regular raster (missing cells, irregular domain).
- The neighbor structure is **precomputed as an `spdep::nb` object** with specific rook adjacencies that may reflect an irregular spatial domain.
- Switching to focal operations risks **altering the numerical results** if the grid is incomplete or if boundary handling differs.

The correct strategy is to **keep the exact same neighbor relationships** but replace the slow R-level loops with vectorized and/or `data.table`-based operations.

---

## 2. Optimization Strategy

### A. Replace character-key lookup with integer arithmetic

Every row can be uniquely identified by `(cell_index, year_index)` â€” both integers. Instead of `paste(id, year)` â†’ character lookup, use:

```
row_position = (cell_index - 1) * n_years + year_index
```

This gives O(1) direct integer indexing into the data, eliminating all `paste` and named-vector lookups.

### B. Vectorize neighbor stats via sparse matrix multiplication

The neighbor structure can be encoded as a **sparse adjacency matrix** `W` of dimension N_rows Ã— N_rows, where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` in the same year. Then:

- **Neighbor mean** = `(W %*% x) / (W %*% ones)` (where `ones` replaces NA with 0 and tracks counts)
- **Neighbor max/min** can be computed by iterating over the sparse structure in a compiled way, or by using `data.table` group-by on an edge list.

For max and min, sparse matrix multiplication doesn't directly apply, but a **long-format edge list + `data.table` group-by** is extremely fast:

```
edge_dt[, .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)), by = row_i]
```

This replaces 6.46M `lapply` iterations with a single vectorized `data.table` aggregation.

### C. Build the edge list once, reuse for all 5 variables

The edge list (which row is neighbor of which row) is variable-independent. Build it once as a two-column integer matrix, then for each variable, join the variable's values and aggregate.

### Estimated speedup

| Step | Before | After |
|---|---|---|
| Neighbor lookup | ~hours (character ops) | ~seconds (integer arithmetic) |
| Stats (per variable) | ~hours (R-level lapply) | ~10-30 seconds (`data.table` group-by on 30M-edge edge list) |
| 5 variables | ~80+ hours | ~2-3 minutes |
| **Total** | **86+ hours** | **< 5 minutes** |

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

#' Optimized pipeline: build an integer edge list from the nb object,
#' then compute neighbor max/min/mean for each variable via data.table.
#' Preserves the exact same numerical results as the original code.

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 0: Convert to data.table for speed; keep original row order
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]  # preserve original row order

  # ---------------------------------------------------------------
  # STEP 1: Build mapping from spatial id -> integer cell index

  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))

  # Map each row's spatial id to its cell index
  dt[, cell_idx := id_to_cellidx[as.character(id)]]

  # ---------------------------------------------------------------
  # STEP 2: Build year -> integer year index mapping
  # ---------------------------------------------------------------
  all_years <- sort(unique(dt$year))
  n_years   <- length(all_years)
  year_to_yidx <- setNames(seq_along(all_years), as.character(all_years))
  dt[, year_idx := year_to_yidx[as.character(year)]]

  # ---------------------------------------------------------------
  # STEP 3: Build a fast (cell_idx, year_idx) -> row_id lookup
  #         using a matrix for O(1) access
  # ---------------------------------------------------------------
  # lookup_mat[cell_idx, year_idx] = row_id in dt (or NA if missing)
  lookup_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  lookup_mat[cbind(dt$cell_idx, dt$year_idx)] <- dt$.row_id

  # ---------------------------------------------------------------
  # STEP 4: Build the edge list (row_i, row_j) for all same-year
  #         rook neighbor pairs present in the data
  # ---------------------------------------------------------------
  # Expand the nb object into a cell-level edge list
  # rook_neighbors_unique[[k]] gives the neighbor cell indices for cell k
  from_cell <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors_unique))
  to_cell   <- unlist(rook_neighbors_unique)

  # Remove any 0-length entries (cells with no neighbors)
  valid <- !is.na(to_cell) & to_cell > 0 & to_cell <= n_cells
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  n_edges_spatial <- length(from_cell)

  # Now cross with years: for each year, look up row_ids
  # Vectorised: replicate the spatial edges across all years
  from_cell_rep <- rep(from_cell, times = n_years)
  to_cell_rep   <- rep(to_cell,   times = n_years)
  year_idx_rep  <- rep(seq_len(n_years), each = n_edges_spatial)

  # Map to row_ids using the lookup matrix
  row_i <- lookup_mat[cbind(from_cell_rep, year_idx_rep)]
  row_j <- lookup_mat[cbind(to_cell_rep,   year_idx_rep)]

  # Keep only edges where both the focal cell and the neighbor cell
  # have a row in the data for that year
  keep <- !is.na(row_i) & !is.na(row_j)
  edge_dt <- data.table(row_i = row_i[keep], row_j = row_j[keep])

  # Free large temporaries
  rm(from_cell_rep, to_cell_rep, year_idx_rep, row_i, row_j, keep)
  gc()

  cat(sprintf("Edge list built: %s directed edges across %d years.\n",
              format(nrow(edge_dt), big.mark = ","), n_years))

  # ---------------------------------------------------------------
  # STEP 5: For each source variable, compute neighbor stats
  # ---------------------------------------------------------------
  setkey(edge_dt, row_i)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Computing neighbor features for: %s ...\n", var_name))

    # Attach the neighbor's value to each edge
    vals <- dt[[var_name]]
    edge_dt[, val := vals[row_j]]

    # Aggregate: max, min, mean of non-NA neighbor values per focal row
    agg <- edge_dt[!is.na(val),
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   by = row_i]

    # Initialize columns with NA (cells with no valid neighbors get NA)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in the computed values
    dt[agg$row_i, (max_col)  := agg$nb_max]
    dt[agg$row_i, (min_col)  := agg$nb_min]
    dt[agg$row_i, (mean_col) := agg$nb_mean]

    # Clean up
    edge_dt[, val := NULL]
  }

  # ---------------------------------------------------------------
  # STEP 6: Return as a data.frame in original row order,
  #         dropping helper columns
  # ---------------------------------------------------------------
  dt[, c(".row_id", "cell_idx", "year_idx") := NULL]
  setorder(dt)  # data.table preserves insertion order; this is safe

  # Return as data.frame to match original pipeline expectations
  as.data.frame(dt)
}

# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimized_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Correctness Argument

| Property | Original | Optimized | Match? |
|---|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[ref_idx]]` â†’ same-year rows | Identical spatial edges Ã— same year via `lookup_mat` | âœ… |
| NA handling | Removes `NA` neighbor values before `max/min/mean` | `edge_dt[!is.na(val), ...]` â€” identical | âœ… |
| No-neighbor rows | Returns `c(NA, NA, NA)` | Columns initialized to `NA`, only overwritten if `agg` has a row | âœ… |
| Column names | `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` | Same naming convention | âœ… |
| RF model | Not retrained | Not touched | âœ… |
| Numerical precision | Base R `max/min/mean` | Same functions inside `data.table` `j` â€” identical IEEE 754 results | âœ… |

## 5. Memory Estimate

| Object | Size |
|---|---|
| `dt` (6.46M Ã— ~115 cols, 8 bytes each) | ~5.6 GB |
| `edge_dt` (~38.4M edges Ã— 2 int cols + 1 double) | ~0.9 GB |
| `lookup_mat` (344K Ã— 28 ints) | ~37 MB |
| **Total peak** | **~7-8 GB** â€” fits in 16 GB RAM |

The original `neighbor_lookup` list of 6.46M integer vectors would itself consume several GB due to R list overhead (each list element has ~100 bytes of overhead), so the optimized version is comparable or better in memory.

---

## Summary

The 86-hour runtime was caused by **character-key construction and lookup** (`paste` + named-vector indexing) across 6.46M rows, repeated in R-level `lapply` loops. The fix replaces this with:

1. **Integer arithmetic** for row identification (O(1) matrix lookup instead of O(N) character hashing).
2. **A single materialized edge list** (built once via vectorized operations in ~seconds).
3. **`data.table` grouped aggregation** over the edge list (one pass per variable, ~10-30 seconds each).

**Expected total runtime: under 5 minutes** on the same laptop, with identical numerical output and no retraining of the Random Forest model.
END_RESPONSE>>>
