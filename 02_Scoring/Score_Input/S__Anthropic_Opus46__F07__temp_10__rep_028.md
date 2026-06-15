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
run_id: Anthropic_F07_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each built via string-pasting, named-vector lookups, and NA filtering inside an `lapply` over every row.** This is O(n) named-vector lookups (which are themselves O(n) in base R) nested inside an O(n) loop â€” effectively O(nÂ²) behavior on ~6.46M rows. The `compute_neighbor_stats` loop is lighter but still iterates 6.46M times in pure R, and this is repeated for 5 variables.

**Specific problems:**

1. **`id_to_ref` and `idx_lookup` are named character vectors.** Lookup in a named vector is linear scan, not hashed. With ~6.46M keys in `idx_lookup`, each lookup is extremely slow. This alone explains the 86+ hour estimate.

2. **String pasting (`paste(id, year, sep="_")`)** is done 6.46M times to build keys, and again inside the inner loop for every neighbor of every row.

3. **The neighbor lookup is row-level but the topology is cell-level.** There are only 344,208 cells. The neighbor graph doesn't change across years. Yet the code rebuilds neighbor index vectors for every cell-year row (6.46M times instead of 344K times).

4. **`compute_neighbor_stats` uses an R-level `lapply` over 6.46M elements** with per-element allocation â€” slow due to interpreter overhead.

5. **The loop runs 5 times** (once per variable), each time reiterating over 6.46M rows.

---

## Optimization Strategy

### Principle: Vectorize via merge/join on the sparse adjacency structure.

The neighbor relationships are a **sparse directed edge list** (~1.37M edges). For each edge `(cell_i, cell_j)` in a given year, we want the value of `var` at cell_j. Then we group by `(cell_i, year)` and compute `max`, `min`, `mean`.

This is a **join + grouped aggregation** â€” exactly what `data.table` excels at.

**Steps:**

1. Convert the `nb` object to an **edge list** (once, ~1.37M rows).
2. Convert `cell_data` to a `data.table`, keyed on `(id, year)`.
3. For each variable, join the edge list against the data to retrieve neighbor values, then aggregate by `(id, year)`.
4. Left-join the aggregated stats back onto `cell_data`.

**Complexity:** O(E Ã— T) for the join, where E â‰ˆ 1.37M and T = 28, so ~38.4M join-lookups â€” trivial for `data.table` with binary-search keys. Total runtime: **minutes, not days.**

**Memory:** The edge list Ã— years is ~38.4M rows Ã— a few columns â€” well within 16 GB.

The trained Random Forest model is untouched. The numerical output (max, min, mean of non-NA neighbor values per cell-year) is identical to the original.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1.  Convert the nb object to a directed edge list (one-time)
# ---------------------------------------------------------------
nb_to_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of integer index vectors (spdep::nb format)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses

  valid    <- to_idx > 0L
  data.table(
    id_from = id_order[from_idx[valid]],
    id_to   = id_order[to_idx[valid]]
  )
}

edges <- nb_to_edge_list(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows: (id_from, id_to)

# ---------------------------------------------------------------
# 2.  Convert cell_data to data.table and set key
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# ---------------------------------------------------------------
# 3.  Function: compute neighbor max/min/mean for one variable
# ---------------------------------------------------------------
add_neighbor_features_dt <- function(dt, edges, var_name) {
  # Build a lookup table: (id, year, value)
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Get all unique years present in the data
  years <- unique(dt$year)

  # Cross-join edges Ã— years, then look up the neighbor's value

  # CJ inside edges is expensive; instead, join edges onto val_dt
  # by expanding edges per year.

  # Approach: for each year, join edges -> val_dt to get neighbor values,
  # then aggregate.  With 28 years this is a simple loop, each iteration
  # operating on ~1.37M rows â€” very fast.

  agg_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Subset values for this year
    vyr <- val_dt[.(unique(val_dt$id), yr), nomatch = 0L, on = .(id, year)]
    # Rename for join: we want val of the *neighbor* (id_to)
    setnames(vyr, c("id", "year", "val"), c("id_to", "year", "neighbor_val"))
    setkey(vyr, id_to)

    # Join: for every edge, get the neighbor's value
    joined <- vyr[edges, on = .(id_to), nomatch = NA, allow.cartesian = TRUE]
    # joined has columns: id_to, year, neighbor_val, id_from

    # Aggregate by id_from (the focal cell)
    agg <- joined[
      !is.na(neighbor_val),
      .(nmax = max(neighbor_val),
        nmin = min(neighbor_val),
        nmean = mean(neighbor_val)),
      by = .(id = id_from)
    ]
    agg[, year := yr]
    agg_list[[yi]] <- agg
  }

  agg_all <- rbindlist(agg_list)
  setkey(agg_all, id, year)

  # Name the new columns to match original convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  setnames(agg_all, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Left-join back onto dt
  # First remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(dt)) dt[, (col) := NULL]
  }

  dt <- merge(dt, agg_all, by = c("id", "year"), all.x = TRUE)
  setkey(dt, id, year)
  dt
}

# ---------------------------------------------------------------
# 4.  Outer loop over the 5 source variables
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  dt <- add_neighbor_features_dt(dt, edges, var_name)
}

# ---------------------------------------------------------------
# 5.  Convert back to data.frame if downstream code expects one
# ---------------------------------------------------------------
cell_data <- as.data.frame(dt)

# The trained Random Forest model is unchanged.
# The new columns (e.g., ntl_neighbor_max, ntl_neighbor_min,
# ntl_neighbor_mean, etc.) are numerically identical to the
# original implementation's output.
```

---

## Why This Is Correct and Equivalent

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | rook contiguity from `spdep::nb` | Same edges, converted to edge list |
| Per cell-year: which neighbors? | Cells sharing an edge, **same year** | Identical join key `(id_to, year)` |
| NA handling | `neighbor_vals[!is.na(...)]` | `joined[!is.na(neighbor_val), ...]` |
| No-neighbor / all-NA result | `NA, NA, NA` | `merge(..., all.x=TRUE)` â†’ `NA` |
| Aggregation functions | `max`, `min`, `mean` | Identical |
| RF model | Untouched | Untouched |

**Expected runtime: 5â€“15 minutes** on a 16 GB laptop instead of 86+ hours â€” roughly a **300â€“1000Ã— speedup** â€” because every operation is now vectorized over contiguous memory via `data.table`'s radix-sort joins and grouped aggregation in C.
END_RESPONSE>>>
