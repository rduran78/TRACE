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
run_id: Anthropic_F02_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs an `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups via character keys in R are hash-table operations, but doing millions of `paste` + `[` lookups inside a loop is extremely slow. The secondary bottleneck is **`compute_neighbor_stats`**, which iterates over 6.46 million list elements, extracting subsets of a numeric vector each time. Together, these two stages dominate the 86+ hour runtime.

**Specific problems:**

1. **`build_neighbor_lookup`:** For each of ~6.46M rows, it converts the cell's ID to a reference index, retrieves neighbor cell IDs, pastes them with the year to form keys, and looks them up in a named vector (`idx_lookup`). The `paste()` and named-vector subsetting inside a per-row `lapply` is O(n Ã— k) with large constant factors. With ~6.46M rows and an average of ~4 rook neighbors each, this is ~25.8M string operations plus lookups.

2. **`compute_neighbor_stats`:** Iterating a 6.46M-element list and calling `max`, `min`, `mean` per element is slow due to R's per-call overhead, even though each individual call is trivial.

3. **Memory:** Storing a 6.46M-element list of integer vectors (the neighbor lookup) is memory-intensive. With ~4 neighbors per cell-year, this is manageable but adds up alongside the ~110-column data frame.

---

## Optimization Strategy

**Replace the per-row list-based lookup with a vectorized, `data.table`-based edge-list join.**

Instead of building a list of neighbor row indices for every row, we:

1. **Build an edge list** of (cell_id, neighbor_cell_id) from the `nb` object â€” done once, ~1.37M edges.
2. **Join the edge list with the data on (neighbor_cell_id, year)** using `data.table` keyed joins. This expands to ~1.37M Ã— 28 â‰ˆ ~38.5M edge-year rows (but handled efficiently in columnar memory).
3. **Group-by aggregate** (max, min, mean) over (cell_id, year) to produce neighbor stats â€” fully vectorized in C via `data.table`.
4. **Join the aggregated stats back** to the original data.

This eliminates all per-row R-level loops and string operations. Expected speedup: **~100â€“500Ã—**, bringing runtime from 86+ hours to roughly **10â€“30 minutes**.

**Memory is also improved:** a long-format edge table with integer IDs and one double-precision value column is far more cache-friendly than 6.46M list elements.

**The trained Random Forest model and original numerical estimand are fully preserved** â€” we are only changing how features are computed, not their values.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert the nb object to a flat edge list (done once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer vectors of positional indices)
  # id_order is the vector mapping position -> cell_id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows with columns: id, neighbor_id

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Convert cell_data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute and attach neighbor features for all source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Minimal subset for the join: only id, year, and the variable of interest
  # This keeps memory low â€” we never materialize all 110 columns in the join.
  sub_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(sub_dt, id, year)

  # Join edges with the data to get each neighbor's value for each year
  # edge_dt has (id, neighbor_id); we want the neighbor's value in the same year.
  # Step A: attach year by joining edge_dt with the set of (id, year) combos
  #   â€” but this would explode to edges Ã— years. Instead, merge on neighbor side.

  # For every (id, year) row in the data, find its neighbors via edge_dt,
  # then look up the neighbor's value in the same year.

  # Efficient approach: join edge_dt to sub_dt on neighbor side.
  # Result: for each edge (id -> neighbor_id) and each year the *neighbor* has data,
  # we get the neighbor's value. Then we filter/group by (id, year).

  # Actually the cleanest way: 
  #   merged = edge_dt â‹ˆ sub_dt  ON  edge_dt.neighbor_id = sub_dt.id
  # This gives (id, neighbor_id, year, val) â€” one row per edge per year.

  setkey(edge_dt, neighbor_id)
  setnames(sub_dt, "id", "neighbor_id")  # rename for join
  
  merged <- sub_dt[edge_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = 0L]
  # merged columns: neighbor_id, year, val, id  (id comes from edge_dt via join)
  # This means: for cell `id` in year `year`, neighbor `neighbor_id` has value `val`.

  # Restore name
  setnames(sub_dt, "neighbor_id", "id")

  # Aggregate: for each (id, year), compute max/min/mean of neighbor vals
  agg <- merged[!is.na(val),
    .(
      nbr_max  = max(val),
      nbr_min  = min(val),
      nbr_mean = mean(val)
    ),
    by = .(id, year)
  ]

  # Rename columns to match original feature names
  prefix <- paste0("neighbor_", var_name)
  setnames(agg, c("nbr_max", "nbr_min", "nbr_mean"),
           c(paste0(prefix, "_max"), paste0(prefix, "_min"), paste0(prefix, "_mean")))

  # Join aggregated stats back to cell_dt
  setkey(agg, id, year)
  setkey(cell_dt, id, year)
  cell_dt[agg, (names(agg)[3:5]) := mget(paste0("i.", names(agg)[3:5]))]

  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Loop over the 5 source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-key edge_dt on neighbor_id (will be re-used each iteration)
setkey(edge_dt, neighbor_id)

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_and_add_neighbor_features_fast(cell_data, edge_dt, var_name)
}

# cell_data now has 15 new columns (3 per variable Ã— 5 variables):
#   neighbor_ntl_max, neighbor_ntl_min, neighbor_ntl_mean,
#   neighbor_ec_max,  neighbor_ec_min,  neighbor_ec_mean,
#   ... etc.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | The `max`, `min`, `mean` aggregations over the exact same neighbor sets produce identical values. The join logic replicates the original neighbor-lookup semantics: same cell-year pairs, same neighbor definitions. |
| **Trained RF model** | Untouched. We only changed the feature-engineering step, not the model or its inputs. |
| **Memory** | The largest intermediate object (`merged`) has ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 3 columns (int, int, double) â‰ˆ ~920 MB, which fits in 16 GB alongside the main table (~6.46M Ã— 110 cols â‰ˆ 5.7 GB). Each variable is processed and then `merged` is freed. |
| **Speed** | All operations are vectorized C-level `data.table` joins and group-by aggregations. Expected wall-clock time: **10â€“30 minutes total** on a 16 GB laptop. |

### Optional Further Optimizations

- **Process variables in parallel** using `parallel::mclapply` or `future.apply` if memory allows (each iteration needs ~1 GB headroom).
- **Use `collapse` package** (`fmax`, `fmin`, `fmean` with grouping) for even faster grouped aggregation.
- **Chunk by year** if memory is still tight: process 4â€“5 years at a time, reducing peak `merged` size to ~140M rows â†’ ~130 MB.
END_RESPONSE>>>
