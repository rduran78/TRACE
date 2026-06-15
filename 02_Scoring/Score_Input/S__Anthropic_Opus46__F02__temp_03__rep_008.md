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
run_id: Anthropic_F02_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing ~6.46M of them â€” each touching multiple neighbors â€” creates billions of small string allocations and hash probes. The `compute_neighbor_stats` function then iterates over the same 6.46M rows again per variable, and this is repeated for 5 variables.

**Specific problems:**

1. **String-key lookups at scale:** `paste(id, year)` keys and `idx_lookup[neighbor_keys]` create millions of temporary character vectors and perform hash-table lookups one row at a time. This is extremely slow in base R.
2. **`lapply` over 6.46M rows:** Each iteration has R-level overhead (function call, allocation, GC pressure). Returning a list of 6.46M integer vectors is also memory-heavy.
3. **Redundant work across years:** The neighbor *structure* is purely spatial (rook contiguity) and identical for every year. Yet the lookup rebuilds neighbor indices per cell-year row, effectively repeating the same spatial logic 28 times.
4. **`do.call(rbind, result)` on a 6.46M-element list:** This is a known anti-pattern â€” it is O(nÂ²) in memory copies.
5. **Serial, single-threaded execution** for all of the above.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate string hashing** | Replace character-key lookups with integer-indexed joins via `data.table`. |
| **Vectorize the neighbor join** | Explode the `nb` object into an edge-list `data.table` once, then use keyed `data.table` joins (binary search, C-level) instead of per-row `lapply`. |
| **Exploit year-invariance** | Build the spatial edge list once (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M edges). Join to panel data by `(neighbor_id, year)` â€” the join engine handles the 28-year fan-out. |
| **Vectorize aggregation** | Use `data.table`'s `by=` grouped aggregation (`max`, `min`, `mean`) instead of `lapply` + `rbind`. |
| **Minimize memory** | Operate column-by-column; never materialize a 6.46M-element list of vectors. |
| **Preserve the trained RF model** | Only the feature columns are being computed; the model object is untouched. |
| **Preserve the numerical estimand** | `max`, `min`, `mean` over the same non-NA neighbor values â€” identical results. |

**Expected improvement:** From ~86+ hours down to **minutes** (the dominant operation becomes a keyed `data.table` join on ~38M rows, which is highly optimized in C).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the spatial edge list ONCE  (small: ~1.37M rows)
#     rook_neighbors_unique is an nb object (list of integer vectors).
#     id_order is the vector of cell IDs in the same order as the nb list.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for the
  # neighbors of cell id_order[i].  Index 0 means no neighbors (spdep convention).
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid    <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id       = id_order[from_idx],   # focal cell
    nb_id    = id_order[to_idx]      # neighbor cell
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: id, nb_id   (~1.37M rows)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute neighbor features â€” one variable at a time
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_data, edge_dt, var_name) {
  # --- a) Slim table of neighbor values: (nb_id, year, value) ----------
  #     We only need the variable of interest to keep memory low.
  nb_vals <- cell_data[, .(nb_id = id, year, value = get(var_name))]
  setkey(nb_vals, nb_id, year)

  # --- b) Expand edges Ã— years: join focalâ†’neighbor, then look up value
  #     edge_dt has (id, nb_id).  We cross-join with the focal cell's year
  #     by merging edge_dt onto the focal cell's (id, year) rows,
  #     then look up the neighbor's value for that same year.

  # Focal rows: just id and year (small projection)
  focal <- cell_data[, .(id, year)]
  setkey(focal, id)

  # Merge focal rows with edge list â†’ one row per (focal cell-year, neighbor)
  # This is the "fan-out" step: ~6.46M Ã— ~4 â‰ˆ ~26-38M rows
  edges_by_year <- edge_dt[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edges_by_year columns: id, nb_id, year

  # Look up the neighbor's value for the same year
  setkey(edges_by_year, nb_id, year)
  edges_by_year[nb_vals, value := i.value, on = .(nb_id, year)]

  # --- c) Aggregate: max, min, mean per (id, year) --------------------
  stats <- edges_by_year[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Name the new columns to match the original pipeline's convention
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  # --- d) Left-join back onto cell_data --------------------------------
  #     Cells with no valid neighbors get NA (preserving original behavior).
  cell_data[stats, (new_names) := mget(paste0("i.", new_names)),
            on = .(id, year)]

  invisible(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Run for each variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_neighbor_features_dt(cell_data, edge_dt, var_name)
}

# cell_data now contains the 15 new columns (3 stats Ã— 5 variables)
# and is ready for prediction with the existing trained Random Forest.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor set** | `build_edge_list` reads the same `rook_neighbors_unique` nb object and `id_order`; the directed edges are identical. |
| **Same aggregation** | `max`, `min`, `mean` over non-NA neighbor values, grouped by `(id, year)` â€” mathematically identical to the original `lapply`. |
| **NA handling** | Rows with no valid neighbors receive `NA` for all three stats (via the left-join; unmatched rows stay `NA`). |
| **Trained RF model** | Untouched. Only feature columns are computed. Column names follow the same convention so `predict()` works as before. |

## Expected Performance

- **Edge list build:** < 1 second (vectorized `unlist` + indexing).
- **Per-variable join + aggregation:** ~30â€“90 seconds each (keyed `data.table` join on ~38M rows, C-level grouped aggregation).
- **Total for 5 variables:** **~3â€“8 minutes** on a 16 GB laptop, versus the original ~86+ hours.
- **Peak RAM:** The largest intermediate (`edges_by_year`) is ~38M rows Ã— 4 columns of integers/doubles â‰ˆ ~1.2 GB, well within 16 GB.
END_RESPONSE>>>
