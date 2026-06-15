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
run_id: Anthropic_F07_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It creates **~6.46 million individual list elements**, each constructed inside an `lapply` call that performs per-row character coercion, string pasting, and named-vector lookups. Specifically:

1. **`build_neighbor_lookup` is O(NÂ·k) with enormous constant factors.** For each of the 6.46M rows, it converts IDs to characters, pastes strings, and does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-based but carries substantial per-call overhead when done millions of times in an interpreted loop.

2. **The lookup is built row-by-row in pure R.** `lapply` over 6.46M rows with string operations inside is catastrophically slow. The estimated 86+ hours is almost entirely attributable to this function.

3. **`compute_neighbor_stats` is also row-by-row** but is cheaper per iteration (just numeric subsetting). Still, iterating 6.46M times Ã— 5 variables = 32.3M iterations of interpreted R is unnecessarily slow.

4. **Memory layout is list-of-vectors**, which has high overhead for millions of small vectors.

The core insight: the neighbor structure is **time-invariant** (the same spatial neighbors apply to every year). We should exploit this by doing a **merge/join** operation at the spatial level, then vectorizing the statistics computation across all cell-years using `data.table`.

---

## Optimization Strategy

1. **Replace the per-row lookup with a `data.table` equi-join.** Build an edge table (cell_id â†’ neighbor_id) from the `nb` object, then join `cell_data` to itself on `(neighbor_id, year)` to retrieve neighbor values. This is a single vectorized join â€” no interpreted loop at all.

2. **Compute grouped statistics with `data.table` aggregation.** After the join, group by `(id, year)` and compute `max`, `min`, `mean` in one pass per variable. `data.table`'s GForce will optimize these to C-level operations.

3. **Loop only over the 5 source variables**, not over rows. Each variable requires one join + one grouped aggregation â€” about 5 passes total.

4. **Memory management:** The edge table has ~1.37M edges. The joined table (edges Ã— years) has ~1.37M Ã— 28 â‰ˆ 38.5M rows, each with just (id, year, value). At ~3 columns of 8 bytes each, that's ~0.9 GB â€” well within 16 GB RAM. We process one variable at a time and discard intermediates.

**Expected speedup:** From 86+ hours to **~2â€“10 minutes** (roughly 500â€“2500Ã— faster).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build a spatial edge table from the nb object (once)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # Expand into a two-column data.table: (id, neighbor_id)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: one per directed rook-neighbor pair

# ---------------------------------------------------------------
# Step 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# Step 3: For each source variable, compute neighbor stats via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_data, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # cell_data must have columns: id, year, <var_name>
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join edges with values: for each (id, year), find neighbor values
  # edge_dt has (id, neighbor_id); we join neighbor_id -> val_dt$id
  # to get the neighbor's value in the same year
  joined <- merge(
    edge_dt,
    val_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE  # each neighbor appears in 28 years
  )
  # joined now has columns: neighbor_id, id, year, value
  # where 'value' is the NEIGHBOR's value of var_name in that year
  # and 'id' is the focal cell
  
  # Compute grouped stats: group by (id, year)
  stats <- joined[
    !is.na(value),
    .(
      nmax  = max(value),
      nmin  = min(value),
      nmean = mean(value)
    ),
    keyby = .(id, year)
  ]
  
  # Name the output columns
  suffix <- var_name
  setnames(stats, c("nmax", "nmin", "nmean"),
           paste0(c("nmax_", "nmin_", "nmean_"), suffix))
  
  return(stats)
}

# ---------------------------------------------------------------
# Step 4: Loop over variables, merge results back into cell_data
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  
  # Remove old columns if re-running (idempotency)
  old_cols <- paste0(c("nmax_", "nmin_", "nmean_"), var_name)
  old_cols_present <- intersect(old_cols, names(cell_data))
  if (length(old_cols_present) > 0) {
    cell_data[, (old_cols_present) := NULL]
  }
  
  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  
  # Left join back to cell_data (preserves rows with no neighbors as NA)
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
  
  rm(stats_dt)
  gc()
}

# Verify
cat("Final dimensions:", nrow(cell_data), "x", ncol(cell_data), "\n")
cat("Neighbor feature columns added:\n")
print(grep("^(nmax_|nmin_|nmean_)", names(cell_data), value = TRUE))

# ---------------------------------------------------------------
# cell_data now has the same neighbor max/min/mean columns as the
# original implementation, computed identically (same numerical
# estimand), ready for prediction with the existing trained
# Random Forest model â€” no retraining required.
# ---------------------------------------------------------------
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `edge_dt` is built directly from the same `rook_neighbors_unique` nb object and `id_order` â€” identical directed pairs. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values, grouped by `(id, year)`. |
| **Same NA behavior** | Cells with no neighbors (or all-NA neighbors) get `NA` via the `all.x = TRUE` left join, matching the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | Only the feature-engineering pipeline is replaced. Column names and values are identical, so the existing model's `predict()` call works unchanged. |

---

## Expected Performance

| Step | Time estimate (laptop, 16 GB) |
|---|---|
| `build_edge_table` | < 1 second |
| One variable: join + aggregate | ~20â€“60 seconds |
| All 5 variables | **~2â€“5 minutes total** |
| **Original implementation** | **86+ hours** |

The speedup comes from replacing ~32 million interpreted R loop iterations with 5 vectorized `data.table` merge + group-by operations executed in C.
END_RESPONSE>>>
