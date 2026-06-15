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
run_id: Anthropic_F12_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M rows Ã— 3 columns â€” this takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** The `lapply` inside `build_neighbor_lookup` iterates over every row and, for each row, performs:
   - `as.character(data$id[i])` â€” character coercion per row.
   - `id_to_ref[as.character(...)]` â€” named vector lookup (hashed, but still per-row overhead).
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string construction for every neighbor of every row.
   - `idx_lookup[neighbor_keys]` â€” named vector lookup for each neighbor key.

2. **Scale of the problem:** With ~6.46 million rows and an average of ~4 rook neighbors per cell, this inner function performs roughly **25.8 million string paste operations** and **25.8 million named-vector lookups**, all inside an R-level `lapply` with no vectorization across rows. The `paste` and named-lookup pattern is extremely slow in R at this scale.

3. **The lookup is constructed once but costs the vast majority of the 86+ hours.** `compute_neighbor_stats` is comparatively cheap: it's just integer indexing into a numeric vector and computing `max/min/mean` on small subsets â€” all fast operations. The `do.call(rbind, ...)` on a list of length 6.46M of 3-element vectors takes on the order of seconds to a minute.

**Conclusion:** The deep bottleneck is the row-level string-key construction and lookup strategy in `build_neighbor_lookup()`. The entire approach of mapping spatial neighbors to panel rows via string keys inside a per-row `lapply` is the architectural problem.

---

## Optimization Strategy

1. **Replace string-key lookups with integer arithmetic.** If every cell appears in every year (a balanced panel), we can compute the row index of any (cell, year) combination directly via integer math: `row = (cell_position - 1) * n_years + year_position`. This eliminates all `paste()` and named-vector lookups entirely.

2. **Vectorize the neighbor lookup construction.** Instead of iterating row-by-row, expand the neighbor list once at the cell level (344,208 cells), then broadcast across all 28 years using vectorized operations.

3. **Replace `do.call(rbind, ...)` with pre-allocated matrix output** in `compute_neighbor_stats` for a minor additional speedup.

4. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline, producing numerically identical columns.

5. **Preserve the original numerical estimand** â€” the optimized code computes the same `max`, `min`, `mean` of neighbor values, yielding identical results.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE
# Preserves: trained RF model, original numerical estimand (neighbor max/min/mean)
# ==============================================================================

# --- Step 0: Ensure the panel is sorted by (id, year) -----------------------
# This is CRITICAL: the integer-arithmetic indexing assumes a balanced panel
# sorted by cell id, then by year within each cell.

id_order_chr <- as.character(id_order)
cell_data <- cell_data[order(match(as.character(cell_data$id), id_order_chr),
                             cell_data$year), ]

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

# --- Step 1: Build neighbor lookup via integer arithmetic --------------------
# For a balanced panel sorted by (id_order, year):
#   row index of cell c (1-indexed into id_order), year t (1-indexed) =
#       (c - 1) * n_years + t
#
# We expand cell-level neighbors to row-level neighbors in a fully vectorized way.

build_neighbor_lookup_fast <- function(n_cells, n_years, neighbors) {
  # neighbors is an nb object: neighbors[[c]] gives integer indices into id_order
  # of the neighbors of cell c.

  # Number of neighbors per cell
  n_nbrs <- lengths(neighbors)  # integer vector, length n_cells

  # Total directed neighbor relationships
  total_nbr <- sum(n_nbrs)  # ~1,373,394

  # Expand: for each cell c, repeat c once per neighbor
  cell_idx <- rep(seq_len(n_cells), times = n_nbrs)  # which cell
  nbr_idx  <- unlist(neighbors, use.names = FALSE)    # which neighbor cell

  # For each (cell, neighbor_cell) pair, we need to create entries for all n_years.
  # Row of cell c, year t:       (c - 1) * n_years + t
  # Row of neighbor n, year t:   (n - 1) * n_years + t

  # Expand across years: repeat each pair n_years times
  cell_idx_exp <- rep(cell_idx, each = n_years)
  nbr_idx_exp  <- rep(nbr_idx,  each = n_years)
  year_idx_exp <- rep(seq_len(n_years), times = total_nbr)

  # Compute row indices
  row_from <- (cell_idx_exp - 1L) * n_years + year_idx_exp
  row_to   <- (nbr_idx_exp  - 1L) * n_years + year_idx_exp

  # Return as a two-column matrix or split into per-row neighbor lists
  # For compute_neighbor_stats, we need: for each row, the set of neighbor rows.
  # split() is efficient here.
  split(row_to, row_from)
}

message("Building fast neighbor lookup...")
t0 <- proc.time()
neighbor_lookup_fast <- build_neighbor_lookup_fast(n_cells, n_years,
                                                   rook_neighbors_unique)
message(sprintf("Neighbor lookup built in %.1f seconds.",
                (proc.time() - t0)[3]))

# The result of split() is keyed by character versions of row_from.
# Convert to a simple list indexed 1..n_rows for O(1) access.
n_rows <- nrow(cell_data)
neighbor_lookup <- vector("list", n_rows)
idx_keys <- as.integer(names(neighbor_lookup_fast))
neighbor_lookup[idx_keys] <- neighbor_lookup_fast
# Rows with no neighbors remain NULL; handle in stats function.
rm(neighbor_lookup_fast); gc()


# --- Step 2: Optimized compute_neighbor_stats --------------------------------
# Pre-allocate output matrix; avoid do.call(rbind, ...).

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals   <- data[[var_name]]
  n      <- length(neighbor_lookup)
  out    <- matrix(NA_real_, nrow = n, ncol = 3)

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (is.null(idx) || length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1L] <- max(nv)
    out[i, 2L] <- min(nv)
    out[i, 3L] <- mean(nv)
  }
  out
}

# --- Step 3: Compute and attach neighbor features ----------------------------
# This mirrors compute_and_add_neighbor_features() but uses the fast versions.

compute_and_add_neighbor_features_fast <- function(data, var_name,
                                                    neighbor_lookup) {
  message(sprintf("  Computing neighbor stats for '%s'...", var_name))
  t0  <- proc.time()
  mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  elapsed <- (proc.time() - t0)[3]
  message(sprintf("    Done in %.1f seconds.", elapsed))

  data[[paste0(var_name, "_neighbor_max")]]  <- mat[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- mat[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- mat[, 3L]
  data
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features...")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name,
                                                       neighbor_lookup)
}
message("All neighbor features computed.")

# --- Step 4: Predict with the existing trained Random Forest -----------------
# The trained RF model object is unchanged. The feature columns are numerically
# identical to the original pipeline's output, so predictions are identical.
#
# Example (uncomment and adapt to your actual model object name):
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `paste()` + named-vector lookups in R-level loop | Vectorized integer arithmetic, one `split()` call |
| **`compute_neighbor_stats`** | `lapply` + `do.call(rbind, ...)` | Pre-allocated matrix, simple `for` loop |
| **Estimated runtime** | 86+ hours | **Minutes** (lookup: ~30â€“120 s; stats: ~1â€“3 min per variable) |
| **Numerical output** | max/min/mean of neighbor values | **Identical** |
| **Trained RF model** | â€” | **Preserved, untouched** |

The true bottleneck was never `do.call(rbind, ...)`. It was the **O(n_rows Ã— avg_neighbors)** string construction and hash-table lookup pattern inside `build_neighbor_lookup()`, executed 6.46 million times in an interpreted R loop. Replacing string keys with integer-arithmetic row indexing and vectorized expansion eliminates this entirely.
END_RESPONSE>>>
