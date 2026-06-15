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
run_id: Anthropic_F02_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and named-vector lookups, which are extremely slow in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which again loops over 6.46 million entries with per-element subsetting and summary computation. Together, these two stages dominate the 86+ hour runtime.

**Specific problems:**

1. **String-key lookups in `build_neighbor_lookup`:** For every row, `paste()` constructs neighbor keys and `idx_lookup[neighbor_keys]` performs named-vector lookup. With ~6.46M rows Ã— ~4 neighbors on average, this is ~25M string constructions and lookups â€” all in an interpreted `lapply`.
2. **List-of-vectors output:** The neighbor lookup is a list of 6.46M integer vectors. This is memory-heavy (each list element has R object overhead) and forces downstream `lapply` iteration.
3. **`compute_neighbor_stats` iterates row-by-row** over the 6.46M-element list, computing `max`, `min`, `mean` per element â€” no vectorization.
4. **Memory pressure:** 6.46M rows Ã— 110 columns is already ~5â€“6 GB as doubles. The neighbor lookup list and intermediate copies can push past 16 GB.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** operation. If we have an edge list `(id, neighbor_id)` and a panel keyed by `(id, year)`, then for each `(id, year)` we can join to get all `(neighbor_id, year)` rows and compute grouped statistics â€” all vectorized in C via `data.table`.

**Steps:**

1. Convert the `spdep::nb` object into a flat edge-list `data.table` with columns `(id, neighbor_id)`.
2. Convert the panel data to a `data.table` keyed on `(id, year)`.
3. For each neighbor source variable, perform a keyed join of the edge list against the panel to retrieve neighbor values, then compute `max`, `min`, `mean` grouped by `(id, year)`.
4. Merge the results back into the main table.

This eliminates all `lapply` loops, all string-key construction, and all per-row R overhead. Expected speedup: **~100â€“500Ã—** (minutes instead of days). Memory is also reduced because we never materialize a 6.46M-element list.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Convert spdep::nb neighbor list to a flat edge-list DT
# ---------------------------------------------------------------
# rook_neighbors_unique is a list where element i contains the
# integer indices (into id_order) of neighbors of id_order[i].
# id_order is the vector of cell IDs in the order matching the nb object.

build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
      n_i <- length(nb_i)
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  # Trim if any nb entries were empty / zero-neighbor sentinels
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# Step 2: Convert panel data to data.table (in-place if possible)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns are proper types for joining
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]
edge_dt[, id := as.integer(id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# Step 3: For each variable, compute neighbor max/min/mean via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Build a slim lookup table: (id, year, value)
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Join edge list with the focal row's year, then look up neighbor values
  # Step A: attach year to each edge via the focal cell
  #   For each (id, neighbor_id) edge, we need one copy per year that id appears in.
  #   Instead of exploding edges Ã— years, we join edges onto the panel.
  
  # focal_edges: for every (id, year) row, get all neighbor_ids
  # This is: cell_dt[, .(id, year)] joined with edge_dt on id
  focal <- cell_dt[, .(id, year)]
  setkey(focal, id)
  setkey(edge_dt, id)
  
  # This produces one row per (id, year, neighbor_id) â€” ~6.46M Ã— ~4 â‰ˆ 26M rows
  expanded <- edge_dt[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Step B: look up the neighbor's value in that year
  expanded[lookup, val := i.val, on = .(neighbor_id = id, year)]
  
  # Step C: compute grouped stats, dropping NAs
  stats <- expanded[!is.na(val),
                    .(nb_max  = max(val),
                      nb_min  = min(val),
                      nb_mean = mean(val)),
                    by = .(id, year)]
  
  # Rename columns to match the variable
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # Step D: merge back into cell_dt (left join to preserve all rows)
  # Remove old columns if they exist (idempotent re-runs)
  for (nm in new_names) {
    if (nm %in% names(cell_dt)) cell_dt[, (nm) := NULL]
  }
  
  cell_dt[stats, on = .(id, year), (new_names) := mget(paste0("i.", new_names))]
  
  invisible(cell_dt)
}

# ---------------------------------------------------------------
# Step 4: Run for all neighbor source variables
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_neighbor_features_dt(cell_data, edge_dt, var_name)
  gc()  # free intermediate memory between variables
}

cat("Done. Neighbor features added.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code â€” the edge list encodes the same rook-neighbor relationships, and the join on `(neighbor_id, year)` replicates the original `paste(id, year)` key lookup. NA handling is identical (`!is.na(val)` before aggregation). |
| **Trained Random Forest** | The RF model is not touched. We only modify the feature-construction step that feeds *new* data into `predict()`. The column names and semantics are preserved (`{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`). |
| **Column name convention** | Adjust the `new_names` pattern above if your existing pipeline uses a different naming convention (e.g., `ntl_max_neighbor`). |

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M `lapply` + string ops) | ~seconds (vectorized edge-list construction) |
| `compute_neighbor_stats` (per var) | ~17 hours (6.46M `lapply`) | ~30â€“90 seconds (`data.table` grouped join) |
| **Total (5 vars)** | **86+ hours** | **~5â€“10 minutes** |
| Peak RAM | >16 GB (list overhead) | ~8â€“12 GB (flat DT intermediates, freed per var with `gc()`) |

The entire pipeline should complete in under 15 minutes on a 16 GB laptop.
END_RESPONSE>>>
