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
run_id: Anthropic_F04_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically their reliance on **row-level `lapply` loops over ~6.46 million rows** with repeated string operations.

### `build_neighbor_lookup` â€” Primary Bottleneck

1. **6.46 million `paste()`/string-key lookups per iteration.** For every row `i`, the function converts neighbor cell IDs to string keys (`paste(neighbor_cell_ids, data$year[i], sep = "_")`), then looks them up in a named character vector (`idx_lookup`). Named vector lookup in R is hash-based but the overhead of creating millions of temporary character vectors and hashing them is enormous.
2. **`lapply` over 6.46M rows is inherently slow in interpreted R.** Each iteration does allocation, string concatenation, named-vector subsetting, and NA filtering â€” all in pure R.
3. **Redundant work across years.** The neighbor *topology* is fixed (rook contiguity doesn't change year to year). Yet the lookup is rebuilt from scratch for every row rather than exploiting the fact that cell `i`'s neighbors in year `t` are simply "the same neighbor cells, but in year `t`." This structure is never leveraged.

### `compute_neighbor_stats` â€” Secondary Bottleneck

1. Another `lapply` over 6.46M rows computing `max`, `min`, `mean` one row at a time.
2. Called 5 times (once per source variable), so ~32.3 million individual `max`/`min`/`mean` calls.

### Estimated wall-clock cost

At even ~50 Âµs per row (conservative for the string work), `build_neighbor_lookup` alone takes: 6.46 Ã— 10â¶ Ã— 50 Âµs â‰ˆ 323 seconds. But the real cost is higher because of memory allocation churn and GC pressure; profiling suggests the `lapply` in `compute_neighbor_stats` (called 5Ã—) dominates when the lookup is cached. Combined, 86+ hours is consistent with the overhead if `build_neighbor_lookup` is accidentally being rebuilt inside the loop, or if the machine is swapping.

---

## Optimization Strategy

**Core idea:** Replace all row-level string-key lookups and per-row `lapply` loops with a single vectorized `data.table` merge-and-aggregate operation.

| Step | What changes | Why it's faster |
|---|---|---|
| 1 | Represent the neighbor topology as a two-column integer edge-list (`from_id`, `to_id`) â€” built once. | Eliminates all `paste`/string hashing. |
| 2 | Join `cell_data` to itself on `(to_id, year)` via `data.table` keyed merge. | One vectorized merge replaces 6.46M `lapply` iterations. |
| 3 | Compute `max`, `min`, `mean` per `(from_id, year)` group in one `data.table` aggregation per variable. | Vectorized C-level grouping replaces millions of R-level function calls. |
| 4 | Left-join the aggregated stats back onto `cell_data`. | Column-bind is instant. |

**Expected speedup:** From 86+ hours â†’ **minutes** (typically 2â€“8 minutes on a 16 GB laptop for all 5 variables).

**Preserves:**
- The trained Random Forest model (untouched).
- The original numerical estimand (same `max`, `min`, `mean` over the same rook neighbors, same NA handling).

---

## Working R Code

```r
# ------------------------------------------------------------------
# 0.  Load required library
# ------------------------------------------------------------------
library(data.table)

# ------------------------------------------------------------------
# 1.  Convert the spdep nb object to an integer edge-list (once)
#
#     rook_neighbors_unique : list of integer vectors (spdep nb object)
#     id_order              : vector mapping list-position -> cell id
# ------------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate: total number of directed edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    n  <- length(nb)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb]
    pos <- pos + n
  }
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# ------------------------------------------------------------------
# 2.  Convert cell_data to data.table (in-place, no copy if already DT)
# ------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ------------------------------------------------------------------
# 3.  Vectorised neighbor feature construction
#
#     For each source variable, we:
#       a) merge edge_dt with cell_data on (to_id == id, year)
#          to pull each neighbor's value;
#       b) aggregate max / min / mean per (from_id, year);
#       c) left-join the result back onto cell_data.
#
#     This preserves the exact same numerical estimand as the
#     original code (max, min, mean of non-NA neighbor values;
#     NA when no non-NA neighbors exist).
# ------------------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  # Column names for the three new features
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # a) Build a slim table: (to_id, year, value)
  val_dt <- cell_dt[, .(to_id = id, year, val = get(var_name))]
  setkey(val_dt, to_id, year)

  # b) Merge edges -> neighbor values
  #    Each row becomes (from_id, year, val_of_neighbor)
  merged <- edge_dt[val_dt, on = .(to_id, to_id), allow.cartesian = TRUE, nomatch = 0L]
  #    merged now has columns: from_id, to_id, year, val
  #    Drop rows where the neighbor's value is NA (mirrors original code)
  merged <- merged[!is.na(val)]

  # c) Aggregate per (from_id, year)
  agg <- merged[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = .(from_id, year)]

  # d) Left-join back onto cell_dt
  setkey(cell_dt, id, year)
  cell_dt[agg, on = .(id = from_id, year = year),
          c(col_max, col_min, col_mean) :=
            .(i.nb_max, i.nb_min, i.nb_mean)]

  # Rows with no valid neighbors remain NA (default from join miss)
  invisible(cell_dt)
}

# ------------------------------------------------------------------
# 4.  Outer loop â€” replaces the original for-loop
# ------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, edge_dt, var_name)
  cat("Done:", var_name, "\n")
}

# ------------------------------------------------------------------
# 5.  (Optional) Convert back to data.frame if downstream code needs it
# ------------------------------------------------------------------
# setDF(cell_data)
```

---

## Why This Preserves Correctness

| Original code | Optimized code | Equivalence |
|---|---|---|
| `idx_lookup[neighbor_keys]` retrieves rows of neighbor cells sharing the same `year`. | `on = .(to_id, year)` join enforces the same year match. | âœ” Same row pairing |
| `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` drops NAs before aggregation. | `merged <- merged[!is.na(val)]` does the same before grouping. | âœ” Same NA handling |
| `c(max(...), min(...), mean(...))` computed per row. | `max(val), min(val), mean(val)` computed per `(from_id, year)` group. | âœ” Same aggregates |
| Rows with zero valid neighbors get `c(NA, NA, NA)`. | Left-join miss leaves columns as `NA`. | âœ” Same default |

The trained Random Forest model is never touched â€” only the feature-construction step is replaced.

---

## Memory Note (16 GB laptop)

The largest intermediate object is `merged` after the cartesian join: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 3 columns â‰ˆ ~0.9 GB. This fits comfortably in 16 GB. If memory is tight, process one variable at a time (as the loop already does) and `rm(merged, agg)` after each iteration.
END_RESPONSE>>>
