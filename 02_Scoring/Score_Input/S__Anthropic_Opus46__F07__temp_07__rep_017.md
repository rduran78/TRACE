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
run_id: Anthropic_F07_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that performs character coercion, `paste`, and named-vector lookups. This is O(N) in pure R with heavy per-row overhead (string allocation, hash lookups on a 6.46M-entry named vector). The subsequent `compute_neighbor_stats` is a second O(N) `lapply` loop over the same 6.46M rows, also in pure R.

**Specific costs:**

| Component | Why it's slow |
|---|---|
| `paste(data$id, data$year, sep="_")` + `setNames` on 6.46M keys | Builds a 6.46M-entry character hash map â€” expensive to construct and probe |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Pure-R loop; each iteration does `as.character`, `paste`, named-vector subset |
| `lapply` over 6.46M rows in `compute_neighbor_stats` (Ã—5 variables) | Pure-R loop; `max`/`min`/`mean` on small vectors with NA handling |
| Storing `neighbor_lookup` as a 6.46M-element list of integer vectors | ~25â€“50 GB of list overhead for millions of small vectors |

Estimated: ~86+ hours is consistent with ~48 Î¼s/row Ã— 6.46M rows Ã— 6 passes.

---

## Optimization Strategy

**Core idea:** Replace per-row R loops with vectorized operations on an **edge list** (a sparse adjacency representation), then use `data.table` grouped operations to compute neighbor max, min, and mean in one vectorized pass per variable.

### Steps

1. **Expand the `nb` object into a directed edge list once** â€” two integer columns `(from_cell_id, to_cell_id)` representing all ~1.37M rook-neighbor pairs. This is O(E) and small.

2. **Join the edge list to the panel `data.table` by `(to_cell_id, year)`** to look up neighbor values. This replaces the 6.46M-row `lapply` + `paste` + named-vector lookup with a single keyed `data.table` merge â€” O(E Ã— T) rows but executed in C.

3. **Group by `(from_cell_id, year)` and compute `max`, `min`, `mean`** in one pass per variable using `data.table`'s grouped aggregation â€” fully vectorized in C.

4. **Left-join the results back** to the main panel, filling cells with no neighbors with `NA`.

5. Repeat for each of the 5 neighbor source variables (or batch them).

**Complexity reduction:** From ~6.46M Ã— 6 R-level iterations to a handful of vectorized `data.table` joins and group-bys. Expected runtime: **minutes, not hours.**

**Numerical equivalence:** `max`, `min`, `mean` are computed on exactly the same non-NA neighbor values as before, preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert the spdep nb object to a directed edge list (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of length N; nb_obj[[i]] contains integer indices

  # of neighbors of cell i (in terms of position in id_order).
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-entries that spdep uses for cells with no neighbors

  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows: (from_id, to_id)

# ---------------------------------------------------------------
# 2. Convert panel to data.table if not already
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---------------------------------------------------------------
# 3. Compute neighbor stats for all source vars (vectorized)
# ---------------------------------------------------------------
compute_neighbor_features <- function(dt, edges, var_names) {
  # Minimal subset for the join: id, year, and the source variables
  cols_needed <- c("id", "year", var_names)
  neighbor_vals <- edges[
    dt[, ..cols_needed],
    on = .(to_id = id),       # join edges.to_id == dt.id
    allow.cartesian = TRUE,
    nomatch = NULL             # inner join: drop non-matches
  ]
  # Result columns: from_id, to_id, year, <var_names>
  # Each row = one directed neighbor observation for (from_id, year)

  # Group by (from_id, year) and compute stats for every variable
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(  .(as.name(v)), na.rm = TRUE)),
      bquote(min(  .(as.name(v)), na.rm = TRUE)),
      bquote(mean( .(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_list <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  stats <- neighbor_vals[, eval(j_list), by = .(from_id, year)]

  # Replace -Inf/Inf from max/min on zero-length groups (shouldn't

  # happen after inner join + na.rm, but be safe)
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # Left-join back to the main table
  dt <- merge(dt, stats, by.x = c("id", "year"),
              by.y = c("from_id", "year"),
              all.x = TRUE, sort = FALSE)

  return(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features(cell_data, edges, neighbor_source_vars)

# ---------------------------------------------------------------
# 4. Done â€” cell_data now has n_<var>_max, n_<var>_min, n_<var>_mean
#    for all 5 source variables (15 new columns).
#    The trained Random Forest model is unchanged.
# ---------------------------------------------------------------
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | Same `max`/`min`/`mean` over the same non-NA neighbor values; `na.rm = TRUE` replicates the original `neighbor_vals[!is.na(â€¦)]` filter. |
| **Irregular/masked topology** | The edge list is built directly from the `nb` object â€” no rectangular-raster assumption. Cells with no neighbors get `NA` via the left join. |
| **Memory** | Edge list: ~1.37M Ã— 2 ints â‰ˆ 11 MB. Expanded join table: ~1.37M Ã— 28 years Ã— 7 cols â‰ˆ 2â€“3 GB peak â€” fits in 16 GB. |
| **Trained RF model** | Untouched; we only recompute the same predictor columns. |
| **Expected runtime** | ~2â€“10 minutes on a modern laptop (one keyed join + one grouped aggregation, all in C via `data.table`). |
END_RESPONSE>>>
