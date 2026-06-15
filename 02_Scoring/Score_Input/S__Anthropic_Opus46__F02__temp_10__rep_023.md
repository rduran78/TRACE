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
run_id: Anthropic_F02_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string operations
- Each of the ~6.46M rows calls `paste()` to build neighbor keys, then performs named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but has significant per-call overhead at this scale.
- The `lapply` returns a **list of 6.46M integer vectors** â€” this is a massive, fragmented memory structure (each list element is a separate R vector allocation). On a 16 GB laptop this alone can consume several GB and cause severe GC pressure.

### 2. `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows per variable
- For each of the 5 neighbor source variables, another `lapply` loop traverses 6.46M list elements, extracting subsets of a numeric vector and computing `max`, `min`, `mean`.
- This is called 5 times, so ~32.3 million R function calls total.
- `do.call(rbind, result)` on 6.46M small 3-element vectors is extremely slow â€” it creates millions of temporary row objects.

### Summary of cost
| Step | Calls | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M Ã— `paste` + hash lookup | String allocation, hash lookup overhead, list memory |
| `compute_neighbor_stats` | 5 Ã— 6.46M Ã— subset + aggregation | R-level loop overhead, `do.call(rbind, ...)` on millions of rows |

The 86+ hour estimate is consistent with pure-R loop overhead at this scale.

---

## Optimization Strategy

The strategy rests on three principles: **(a)** replace list-of-vectors neighbor lookup with a flat CSR (Compressed Sparse Row) representation, **(b)** replace per-row R `lapply` with vectorized `data.table` grouped operations, and **(c)** perform all 5 variables' aggregation in a single pass. The trained Random Forest model is never touched; we only produce the same numerical feature columns faster.

### Key changes

| Original | Optimized | Why |
|---|---|---|
| Named character vector lookup (`idx_lookup[paste(...)]`) | Integer join via `data.table` on `(id, year)` | Eliminates millions of `paste` and hash lookups |
| List of 6.46M integer vectors | Flat edge table `(row_i, row_j)` | ~1.37M Ã— 28 â‰ˆ 38M rows, contiguous memory, GC-friendly |
| `lapply` + per-element `max/min/mean` | `data.table` grouped `.(max, min, mean)` by `row_i` | Vectorized C-level aggregation |
| 5 separate passes over lookup | Single edge table reused; 5 grouped aggregations | Edge table built once |
| `do.call(rbind, 6.46M vectors)` | `data.table` returns a single matrix directly | No millions of tiny allocations |

### Memory estimate
- Edge table: ~38.4M rows Ã— 2 integer columns â‰ˆ 0.6 GB.
- One numeric join column at a time â‰ˆ 0.3 GB.
- `cell_data` with 110 columns â‰ˆ 5.7 GB (already present).
- Comfortable within 16 GB.

### Complexity
- Build: O(E Ã— Y) where E â‰ˆ 1.37M edges, Y = 28 years â†’ ~38M rows, via a data.table cross-join + integer join.
- Aggregate: O(38M) per variable via `data.table` grouped ops.
- Total wall-clock: estimated **5â€“15 minutes** (vs. 86+ hours).

---

## Working R Code

```r
# ==============================================================================
# Optimized neighbor feature computation
# Preserves the trained RF model and produces identical numerical output.
# Requirements: data.table (install.packages("data.table") if needed)
# ==============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ------------------------------------------------------------------
  # 0. Convert to data.table (by reference if already one, copy if not)
  # ------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order for deterministic output
  cell_data[, .row_order := .I]

  # ------------------------------------------------------------------
  # 1. Build a flat directed edge list from the nb object

  #    Each element rook_neighbors_unique[[k]] is an integer vector of

  #    neighbor positions into id_order.
  # ------------------------------------------------------------------
  message("Building edge list from nb object ...")

  # Pre-allocate vectors for speed
  n_cells   <- length(id_order)
  n_lengths <- vapply(rook_neighbors_unique, length, integer(1))
  total_edges <- sum(n_lengths)             # ~1.37M directed edges

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (k in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[k]]
    # spdep::nb encodes "no neighbors" as a single 0; skip those
    if (length(nb) == 1L && nb[1L] == 0L) next
    len <- length(nb)
    idx <- pos:(pos + len - 1L)
    from_id[idx] <- id_order[k]
    to_id[idx]   <- id_order[nb]
    pos <- pos + len
  }

  edges <- data.table(from_id = from_id[1:(pos - 1L)],
                      to_id   = to_id[1:(pos - 1L)])

  message(sprintf("  %s directed cell-level edges.", format(nrow(edges), big.mark = ",")))

  # ------------------------------------------------------------------
  # 2. Cross-join edges with years to get (from_id, year, to_id)
  #    Then integer-join to row indices in cell_data.
  # ------------------------------------------------------------------
  message("Expanding edges across years ...")

  years_dt <- data.table(year = sort(unique(cell_data$year)))
  # CJ-like expansion: every edge Ã— every year
  edge_year <- edges[rep(seq_len(.N), each = nrow(years_dt))]
  edge_year[, year := rep(years_dt$year, times = nrow(edges))]

  rm(edges); gc(verbose = FALSE)

  message(sprintf("  %s edge-year rows.", format(nrow(edge_year), big.mark = ",")))

  # ------------------------------------------------------------------
  # 3. Map (from_id, year) -> row_i  and  (to_id, year) -> row_j
  #    using keyed joins on cell_data.
  # ------------------------------------------------------------------
  message("Joining edge-year table to row indices ...")

  # Thin lookup: (id, year) -> row index
  lookup <- cell_data[, .(id, year, .row_order)]
  setkey(lookup, id, year)

  # row_i: the focal cell-year row
  setnames(edge_year, "from_id", "id")
  edge_year <- lookup[edge_year, on = .(id, year), nomatch = 0L]
  setnames(edge_year, ".row_order", "row_i")

  # row_j: the neighbor cell-year row
  setnames(edge_year, "to_id", "id2")
  setnames(edge_year, "id", "from_id")         # park from_id
  setnames(edge_year, "id2", "id")
  edge_year <- lookup[edge_year, on = .(id, year), nomatch = 0L]
  setnames(edge_year, ".row_order", "row_j")

  # Keep only what we need
  edge_year <- edge_year[, .(row_i, row_j)]
  setkey(edge_year, row_i)

  rm(lookup); gc(verbose = FALSE)
  message(sprintf("  %s matched edge-year pairs.", format(nrow(edge_year), big.mark = ",")))

  # ------------------------------------------------------------------
  # 4. For each source variable, compute grouped max / min / mean
  #    and attach the three new columns to cell_data.
  # ------------------------------------------------------------------
  message("Computing neighbor statistics ...")

  n_rows <- nrow(cell_data)

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Variable: %s", var_name))

    # Pull the numeric vector once
    vals <- cell_data[[var_name]]

    # Attach neighbor values
    edge_year[, nval := vals[row_j]]

    # Grouped aggregation (NA-aware)
    agg <- edge_year[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     keyby = .(row_i)]

    # Initialize result columns as NA
    col_max  <- paste0("nb_max_",  var_name)
    col_min  <- paste0("nb_min_",  var_name)
    col_mean <- paste0("nb_mean_", var_name)

    set(cell_data, j = col_max,  value = rep(NA_real_, n_rows))
    set(cell_data, j = col_min,  value = rep(NA_real_, n_rows))
    set(cell_data, j = col_mean, value = rep(NA_real_, n_rows))

    # Fill matched rows
    matched <- agg$row_i
    set(cell_data, i = matched, j = col_max,  value = agg$nb_max)
    set(cell_data, i = matched, j = col_min,  value = agg$nb_min)
    set(cell_data, i = matched, j = col_mean, value = agg$nb_mean)
  }

  # ------------------------------------------------------------------
  # 5. Clean up helper columns and return
  # ------------------------------------------------------------------
  edge_year[, nval := NULL]
  cell_data[, .row_order := NULL]

  message("Done â€” neighbor features added.")
  return(cell_data)
}


# ======================================================================
# USAGE (drop-in replacement for the original outer loop)
# ======================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the 15 new columns:
#   nb_max_ntl, nb_min_ntl, nb_mean_ntl,
#   nb_max_ec,  nb_min_ec,  nb_mean_ec,
#   ... etc.
#
# Feed cell_data directly to predict(trained_rf_model, newdata = cell_data)
# The trained Random Forest is unchanged.
```

---

## Why This Produces Identical Results

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[k]]` indexes into `id_order` | Same `nb` object, same edges |
| Year matching | `paste(id, year)` key match | `data.table` equi-join on `(id, year)` â€” identical logic |
| Aggregation | `max`, `min`, `mean` of non-NA neighbor values | Identical functions, `na.rm` via pre-filter `!is.na(nval)` |
| NA handling | Returns `NA` when no valid neighbors | `set(..., value = NA_real_)` default; only matched rows overwritten |
| Output columns | `nb_max_<var>`, `nb_min_<var>`, `nb_mean_<var>` | Identical column names and semantics |
| RF model | Not involved | Not involved â€” only feature columns are produced |

The numerical estimand is preserved exactly; no approximation or sampling is introduced.

---

## Expected Performance

| Step | Estimated time | Estimated peak memory |
|---|---|---|
| Edge list construction | ~2 seconds | ~22 MB |
| Cross-join with 28 years | ~5 seconds | ~0.6 GB |
| Two keyed joins (row_i, row_j) | ~30 seconds | ~1.2 GB transient |
| 5 Ã— grouped aggregation | ~3 minutes total | ~0.3 GB per variable |
| **Total** | **~5â€“10 minutes** | **~8â€“9 GB peak** |

This fits comfortably within a 16 GB RAM laptop and replaces the estimated 86+ hour runtime.
END_RESPONSE>>>
