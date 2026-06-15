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
run_id: Anthropic_F07_T00_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` call that performs character-based key lookups (`paste` + named-vector indexing) for every single cell-year row.** This is O(N) string operations where N â‰ˆ 6.46M, and the named-vector lookup (`idx_lookup[neighbor_keys]`) is effectively a hash-table probe per key but is called millions of times from interpreted R. Then `compute_neighbor_stats` iterates over that 6.46M-element list again in R-level `lapply`.

Specific problems:

1. **Character key construction and lookup for every row.** `paste(id, year)` and named-vector indexing is extremely slow at scale â€” ~6.46M `paste` calls in `build_neighbor_lookup`, each producing multiple keys.
2. **Row-level R `lapply` over 6.46M rows** â€” twice (once for building the lookup, once per variable for computing stats). R's interpreted loop overhead dominates.
3. **The neighbor lookup is year-invariant but rebuilt as if it were year-specific.** Every cell has the same neighbors in every year. The topology is static; only the row indices change by year. This means we can exploit a **join-based** approach rather than a per-row procedural approach.
4. **`compute_neighbor_stats` recomputes `vals[idx]` subsetting inside an R loop** for each of 5 variables Ã— 6.46M rows = ~32.3M R-level list operations.

**Estimated complexity of current approach:** ~6.46M Ã— (string ops + list indexing) for the lookup build, then 5 Ã— 6.46M Ã— (subsetting + `max/min/mean`) for stats. On a laptop, this easily reaches 86+ hours.

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.**

Key insight: The neighbor relationship is a **spatial graph** that is constant across years. We can represent it as an edge list `(from_id, to_id)`, join it to the panel data by `(to_id, year)` to get neighbor values, then group by `(from_id, year)` to compute `max`, `min`, `mean` â€” all in `data.table`, which does this in C.

Steps:

1. **Convert `rook_neighbors_unique` (an `nb` object) to an edge list `data.table`** with columns `(from_id, to_id)`.
2. **Join** the edge list to `cell_data` on `(to_id = id, year)` to attach each neighbor's variable values.
3. **Group by `(from_id, year)`** and compute `max`, `min`, `mean`.
4. **Join** the results back to `cell_data`.

This eliminates all R-level loops. Expected runtime: **minutes, not hours.**

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert cell_data to data.table (non-destructive)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# ---------------------------------------------------------------
# 1.  Build edge list from the nb object (one-time, fast)
#
#     rook_neighbors_unique is a list of integer vectors (spdep nb).
#     id_order maps position -> cell id.
# ---------------------------------------------------------------
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
    return(data.table(from_id = integer(0), to_id = integer(0)))
  }
  data.table(from_id = id_order[i], to_id = id_order[nb_i])
}))

# ---------------------------------------------------------------
# 2.  For each source variable, compute neighbor stats via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_dt, id, year)

for (var_name in neighbor_source_vars) {

  # Subset to only the columns we need for the join (saves memory)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Join: for every edge (from_id -> to_id), get the neighbor's value
  # by matching to_id = id and same year.
  # We rename columns so the join is clean.
  setnames(val_dt, "id", "to_id")
  # edges has (from_id, to_id); val_dt has (to_id, year, val)
  setkey(val_dt, to_id)
  setkey(edges, to_id)

  # Cross with years: merge edges with val_dt on to_id + year
  setkey(val_dt, to_id, year)
  joined <- edges[val_dt, on = "to_id", allow.cartesian = TRUE, nomatch = 0L]
  # joined now has: from_id, to_id, year, val
  # (only rows where the neighbor actually exists in that year)

  # Remove NA values before aggregation
  joined <- joined[!is.na(val)]

  # Aggregate by (from_id, year)
  stats <- joined[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(from_id, year)]

  # Rename columns to match expected output naming convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  setnames(stats, "from_id", "id")

  # Join stats back to cell_dt
  setkey(stats, id, year)
  setkey(cell_dt, id, year)

  # Remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- stats[cell_dt, on = .(id, year)]

  # Restore val_dt name
  setnames(val_dt, "to_id", "id")

  message(paste0("Done: ", var_name))
}

# ---------------------------------------------------------------
# 3.  Convert back to data.frame if downstream code expects it
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor sets â€” the edge list is derived from the identical `rook_neighbors_unique` nb object and `id_order` mapping. NA handling is identical (NAs excluded before aggregation; cells with zero valid neighbors get `NA` via the left join). |
| **Trained Random Forest untouched** | No model object is modified. Only the feature columns in `cell_data` are (re)computed with identical values. Prediction with the existing model proceeds unchanged. |
| **Irregular/masked topology preserved** | The edge list is built directly from the `nb` object, which already encodes the true rook-contiguity graph â€” no rectangular-raster assumption is introduced. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| Edge list construction (344K cells, ~1.37M edges) | < 5 seconds |
| Per-variable join + aggregation (~6.46M rows Ã— ~1.37M edges) | ~1â€“3 minutes each |
| **Total for 5 variables** | **~5â€“15 minutes** |
| Memory peak (16 GB laptop) | ~4â€“6 GB (well within budget) |

This reduces the runtime from **86+ hours to under 15 minutes** while producing bit-identical numerical results.
END_RESPONSE>>>
