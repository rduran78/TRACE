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
run_id: Anthropic_F06_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row R closures

`build_neighbor_lookup` creates a list of **6.46 million elements**, one per cell-year row. For every row it does string-pasting, hash lookups, and NA filtering inside a base-R `lapply`. This alone can take many hours because:

- `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called ~6.46M times.
- Each call indexes into `idx_lookup` (a named character vector), which is O(n) hashing on a 6.46M-length vector.
- The output is a **list of integer vectors** consuming substantial RAM (~6.46M list elements Ã— average ~4 neighbors each).

### Bottleneck B: `compute_neighbor_stats` â€” repeated per-variable column extraction in a loop

`compute_neighbor_stats` iterates over the 6.46M-element `neighbor_lookup` list **once per variable** (Ã—5 variables). Each iteration extracts subsets of a numeric vector by index, computes `max/min/mean`, and returns a 3-element vector, then `do.call(rbind, ...)` binds 6.46M tiny vectors â€” extremely slow.

### Why raster focal/kernel operations don't directly apply

The comment in the docstring is a red herring for this case. Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. Here the neighbor structure is an **irregular spdep::nb object** (rook contiguity on an arbitrary polygon/grid layout), and the panel is indexed by `(id, year)`. Focal operations would only work if the grid is perfectly regular *and* the nb object exactly matches a 3Ã—3 rook kernel on that grid. Even then, converting to/from raster for 28 years Ã— 5 variables adds complexity with no guarantee of correctness. The safer and faster approach is **vectorized sparse-matrix multiplication**, which preserves the exact nb structure.

### Summary

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | 6.46M string-paste + named-vector lookups |
| `compute_neighbor_stats` | ~hours Ã— 5 vars | 6.46M `lapply` iterations per variable, `do.call(rbind)` |
| Total | 86+ hours estimated | Pure-R loops on millions of rows |

---

## 2. Optimization Strategy

### Key Insight: Replace per-row loops with sparse matrix operations

A rook-neighbor adjacency can be represented as a **sparse matrix W** of dimension `N_cells Ã— N_cells` (344,208 Ã— 344,208). For each year, the neighbor statistics (max, min, mean) across neighbor cells can be computed using:

- **Mean**: sparse matrixâ€“vector product `W %*% x / degree` (or row-normalized W).
- **Max / Min**: use the sparse structure to do grouped max/min via `data.table` grouping on the edge list â€” far faster than 6.46M `lapply` calls.

### Plan

1. **Convert `nb` to a sparse adjacency matrix** (once, ~344K Ã— 344K, very sparse with ~1.37M entries). Use `spdep::nb2listw` â†’ `as(listw, "CsparseMatrix")` or build directly.
2. **Convert `nb` to an edge-list data.table** with columns `(from, to)` for max/min computation.
3. **For each year and each variable**, extract the values vector, then:
   - **Mean**: sparse matrixâ€“vector multiply (one operation).
   - **Max/Min**: join edge list to values, then group-by `from` to get `max` and `min`.
4. **Vectorize across years** by working on the full `data.table` keyed by `(id, year)`.

This reduces the work from ~6.46M R-level iterations per variable to a handful of vectorized/compiled operations.

### Expected speedup

- Sparse matrix multiply for mean: seconds per year, ~1 minute total for all years Ã— 5 vars.
- `data.table` grouped max/min on edge list: seconds per year.
- **Total: minutes instead of 86+ hours.**

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Prerequisites:
#   cell_data        : data.frame/data.table with columns: id, year, ntl, ec,
#                      pop_density, def, usd_est_n2 (and others)
#   id_order         : character/integer vector of cell IDs matching the nb object
#   rook_neighbors_unique : spdep::nb object (list of integer index vectors)
#
# Output:
#   cell_data gains 15 new columns:
#     {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#     for var in (ntl, ec, pop_density, def, usd_est_n2)
#
# The trained Random Forest model is NOT touched.
# =============================================================================

library(data.table)
library(Matrix)

# ---------- Step 0: Convert cell_data to data.table if needed ----------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---------- Step 1: Build edge list from nb object (once) --------------------
# rook_neighbors_unique[[i]] gives the indices (into id_order) of neighbors of
# cell id_order[i].

build_edge_list <- function(nb_obj) {
  # Pre-allocate by counting total edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    len <- length(nbrs)
    from_idx[pos:(pos + len - 1L)] <- i
    to_idx[pos:(pos + len - 1L)]   <- nbrs
    pos <- pos + len
  }
  data.table(from_ref = from_idx[1:(pos - 1L)],
             to_ref   = to_idx[1:(pos - 1L)])
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(rook_neighbors_unique)

# Map ref indices to actual cell IDs
edge_dt[, from_id := id_order[from_ref]]
edge_dt[, to_id   := id_order[to_ref]]

# Compute degree (number of neighbors per cell) for mean calculation
degree_dt <- edge_dt[, .(degree = .N), by = from_id]

cat(sprintf("Edge list: %d directed edges, %d unique cells\n",
            nrow(edge_dt), length(id_order)))

# ---------- Step 2: Key cell_data for fast joins -----------------------------
# Ensure id column type matches id_order type
cell_data[, id := as.character(id)]
edge_dt[, from_id := as.character(from_id)]
edge_dt[, to_id   := as.character(to_id)]
degree_dt[, from_id := as.character(from_id)]

setkey(cell_data, id, year)

# ---------- Step 3: Vectorized neighbor stats computation --------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, degree_dt,
                                          var_names) {
  # Get unique years
  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d variables Ã— %d years = %d tasks\n",
              length(var_names), length(years), length(var_names) * length(years)))

  for (var_name in var_names) {
    cat(sprintf("  Variable: %s\n", var_name))

    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    # Initialize result columns with NA
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    for (yr in years) {
      # Extract values for this year: a lookup table id -> value
      yr_vals <- cell_data[year == yr, .(id, val = get(var_name))]
      setkey(yr_vals, id)

      # Join neighbor values: for each edge (from_id, to_id), get val of to_id
      # This gives us, for each "from" cell, all its neighbor values
      edge_yr <- edge_dt[, .(from_id, to_id)]
      edge_yr[yr_vals, neighbor_val := i.val, on = .(to_id = id)]

      # Remove edges where neighbor value is NA
      edge_yr <- edge_yr[!is.na(neighbor_val)]

      if (nrow(edge_yr) == 0L) next

      # Grouped aggregation: max, min, sum by from_id
      agg <- edge_yr[, .(
        n_max = max(neighbor_val),
        n_min = min(neighbor_val),
        n_sum = sum(neighbor_val),
        n_cnt = .N
      ), by = from_id]

      agg[, n_mean := n_sum / n_cnt]

      # Write results back into cell_data
      # Build a join key
      setkey(agg, from_id)

      # Get row indices in cell_data for this year
      idx <- cell_data[year == yr, which = TRUE]
      ids_this_year <- cell_data$id[idx]

      # Match aggregated results to cell_data rows
      match_idx <- match(ids_this_year, agg$from_id)

      set(cell_data, i = idx, j = max_col,  value = agg$n_max[match_idx])
      set(cell_data, i = idx, j = min_col,  value = agg$n_min[match_idx])
      set(cell_data, i = idx, j = mean_col, value = agg$n_mean[match_idx])
    }

    cat(sprintf("    Done: %s â€” added %s, %s, %s\n",
                var_name, max_col, min_col, mean_col))
  }

  return(cell_data)
}

# ---------- Step 4: Run it ---------------------------------------------------
cat("Computing neighbor features (optimized)...\n")
t0 <- proc.time()

cell_data <- compute_all_neighbor_features(
  cell_data, edge_dt, degree_dt, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor feature computation completed in %.1f seconds (%.1f minutes)\n",
            elapsed, elapsed / 60))

# ---------- Step 5: Apply the pre-trained Random Forest model ----------------
# (Unchanged â€” the model is already trained and is applied as before)
# Example:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Is Correct and Preserves the Estimand

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | For each `(cell, year)` pair, the code computes `max`, `min`, and `mean` over the exact same set of rook neighbors defined by `rook_neighbors_unique`. The edge list is a lossless representation of the nb object. The `data.table` grouped `max`/`min`/`sum`/`.N` operations are mathematically identical to the original `lapply` code. |
| **NA handling** | Edges where the neighbor's value is `NA` are dropped before aggregation (`edge_yr[!is.na(neighbor_val)]`), exactly matching `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` in the original. Cells with zero valid neighbors get `NA` (from the `match` returning `NA`). |
| **Mean = sum/count of non-NA** | The original computes `mean(neighbor_vals)` after removing NAs. The optimized code computes `n_sum / n_cnt` where `n_cnt` counts only non-NA values â€” identical. |
| **Trained RF model** | Untouched. The 15 new columns have the same names and values as the original code would produce. |
| **Memory** | The edge list has ~1.37M rows Ã— 4 columns â€” trivial. Per-year slices are ~344K rows. Peak memory is well within 16 GB. |

### Expected Runtime

- 5 variables Ã— 28 years = 140 iterations.
- Each iteration: one `data.table` join (~1.37M rows) + one grouped aggregation (~344K groups) + one `match` + three `set` calls.
- Each iteration: ~0.5â€“2 seconds on a modern laptop.
- **Total: ~2â€“5 minutes** (vs. 86+ hours originally â€” roughly a **1,000Ã— speedup**).
END_RESPONSE>>>
