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
run_id: Anthropic_F02_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated 5 times, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-key lookups at scale.** `idx_lookup` is a named integer vector with ~6.46M entries. Named vector lookup in R is O(n) per query in the worst case (hashing helps, but overhead is large at this scale). This is called millions of times inside `lapply`.
2. **Per-row `paste` and character operations.** Creating `neighbor_keys` via `paste()` for every row is expensive and produces enormous transient character allocations.
3. **`lapply` over 6.46M rows.** Returns a list of 6.46M integer vectors â€” massive memory overhead from list structure alone.
4. **`do.call(rbind, result)` on a 6.46M-element list.** This is notoriously slow in R; it copies data repeatedly.
5. **No vectorization or use of data.table/matrix operations.** Everything is scalar/list-based.

---

## Optimization Strategy

**Replace the row-level list-based approach with a fully vectorized, edge-listâ€“based `data.table` join-and-aggregate strategy.**

The key insight: the neighbor lookup and aggregation can be expressed as a **join** between an edge table (cellâ†’neighbor) and the data table (keyed by cell-id and year), followed by a **grouped aggregation**. `data.table` performs this in optimized C, eliminating all per-row R overhead.

**Steps:**

1. **Build an edge table once** from the `nb` object: a two-column data.table of `(id, neighbor_id)`.
2. **Join** the edge table to the data (keyed on `id` and `year`) to get each row's neighbor values â€” this is a single indexed merge, not 6.46M sequential lookups.
3. **Aggregate** (max, min, mean) by `(id, year)` in one grouped `data.table` operation per variable.
4. **Join** the aggregated stats back to the main table.

This eliminates all `lapply`, all `paste`-based key construction, all named-vector lookups, and all `do.call(rbind, ...)` calls. Expected speedup: **~100â€“500Ã—** (minutes instead of days). Memory stays well within 16 GB because we never materialize a 6.46M-element list of variable-length integer vectors.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build the edge table ONCE from the nb object
#    id_order: vector of cell IDs (same order as rook_neighbors_unique)
#    rook_neighbors_unique: an nb object (list of integer index vectors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)


  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # nb objects use 0L to denote "no neighbors"
    if (length(nb_i) == 1L && nb_i[0 + 1] == 0L) next
    n_i <- length(nb_i)
    idx <- pos:(pos + n_i - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + n_i
  }

  # Trim if any 0-neighbor nodes caused over-allocation
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor stats for one variable via data.table join + agg
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_dt <- function(dt, edge_dt, var_name) {
  # dt must be a data.table with columns: id, year, <var_name>
  # edge_dt has columns: id, neighbor_id

  # Subset to needed columns for the join target (neighbor side)
  nb_vals <- dt[, .(neighbor_id = id, year, nb_val = get(var_name))]
  setkey(nb_vals, neighbor_id, year)

  # Expand edges by year: join edge_dt to dt to get (id, year, neighbor_id),
  # then join to nb_vals to get the neighbor's value.
  # Step A: get all (id, year) pairs with their neighbor_ids
  #   â€” cross-join edge_dt with years per id is unnecessary because every id
  #     appears for every year. We just merge dt's (id, year) with edge_dt on id.
  id_year <- dt[, .(id, year)]
  setkey(id_year, id)
  setkey(edge_dt, id)

  # This produces one row per (id, year, neighbor_id) â€” ~1.37M edges Ã— 28 years
  # â‰ˆ 38.5M rows, well within 16 GB.
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year

  # Step B: attach neighbor values
  setkey(expanded, neighbor_id, year)
  expanded[nb_vals, nb_val := i.nb_val, on = .(neighbor_id, year)]

  # Step C: aggregate â€” drop NAs, compute max/min/mean per (id, year)
  agg <- expanded[!is.na(nb_val),
                  .(nb_max  = max(nb_val),
                    nb_min  = min(nb_val),
                    nb_mean = mean(nb_val)),
                  by = .(id, year)]

  # Rename columns to match the variable
  new_names <- paste0("nb_", c("max_", "min_", "mean_"), var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)

  agg
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Main pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Build edge table once
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Loop: compute and left-join each variable's neighbor stats
for (var_name in neighbor_source_vars) {
  message("Processing neighbor stats for: ", var_name)
  agg <- compute_neighbor_stats_dt(cell_data, edge_dt, var_name)
  setkey(agg, id, year)

  # Left-join new columns onto cell_data (in-place via :=)
  new_cols <- setdiff(names(agg), c("id", "year"))
  cell_data[agg, (new_cols) := mget(paste0("i.", new_cols)), on = .(id, year)]

  rm(agg)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same neighbor feature columns as before,
# with identical numerical values, so predictions are unchanged.
#
# Example:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` are computed over exactly the same neighbor sets (defined by the same `nb` object and year). No approximation is introduced. |
| **Trained RF model** | The model object is never touched. Column names and semantics are preserved (adjust `new_names` formatting if the original code used a different naming convention). |
| **Column alignment** | The left-join via `on = .(id, year)` ensures rows without any valid neighbors receive `NA`, matching the original `c(NA, NA, NA)` fallback. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `lapply` + string ops) | ~seconds (vectorized edge table build) |
| Stats per variable | ~15+ hours (list iteration + `do.call(rbind)`) | ~1â€“3 minutes (`data.table` keyed join + grouped agg) |
| **Total (5 variables)** | **~86+ hours** | **~10â€“20 minutes** |
| Peak RAM | Unbounded list overhead | ~4â€“6 GB (expanded edge table is largest object) |
END_RESPONSE>>>
