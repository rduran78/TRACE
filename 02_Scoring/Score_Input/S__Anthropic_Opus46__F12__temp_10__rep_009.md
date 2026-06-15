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
run_id: Anthropic_F12_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The colleague is **partially correct but misses the deeper bottleneck**. Let me audit each claim against the code:

### Claim 1: `do.call(rbind, result)` is a bottleneck

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not optimal, but it executes in **seconds** (it's a single call binding pre-allocated vectors). This is a **minor** bottleneck.

### Claim 2: "Repeated list binding inside `compute_neighbor_stats()`"

There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses `lapply` to produce a list in one pass and then binds once. The colleague's diagnosis here is **factually wrong** â€” the code doesn't grow a list iteratively.

### The actual deep bottleneck: `build_neighbor_lookup()`

The true bottleneck is **`build_neighbor_lookup()`**, specifically:

1. **`paste()` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`)** is called **6.46 million times** inside `lapply`. Each call constructs character keys and performs name-based lookups on a **6.46-million-element named vector**. Named vector lookup in R is **O(n)** per query because R's named vectors use linear hashing that degrades at scale. Over 6.46M iterations, each touching multiple neighbors, this produces **billions of character-match operations**.

2. **`as.character()` and `paste()` allocations** inside the per-row lambda create enormous garbage-collection pressure (~6.46M Ã— k string allocations).

3. The **total work** is approximately 6.46M rows Ã— ~4 rook neighbors Ã— O(lookup) per neighbor. With the naive named-vector approach, this is the source of the **86+ hour runtime**.

`compute_neighbor_stats()` by contrast is a simple numeric indexing operation (`vals[idx]`) â€” essentially free once the lookup exists.

**Verdict: REJECT the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()` with its per-row string construction and named-vector lookups over a 6.46M-element vector.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized approach using `data.table` hash joins instead of named-vector character lookups. This reduces lookup from effective O(n) to amortized O(1) per query.

2. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped operations, eliminating the per-row `lapply` and the `do.call(rbind, ...)` entirely.

3. **Pre-expand the neighbor-edge list once** (cellâ†’neighbor Ã— year), then do a single equi-join to resolve row indices, then group-by to compute max/min/mean. This turns the entire pipeline into a few vectorized `data.table` operations.

4. **Preserve** the trained Random Forest model (no retraining) and the original numerical outputs (max, min, mean of neighbor values per variable).

**Expected speedup**: From 86+ hours to **minutes** (roughly 3â€“10 minutes depending on disk I/O).

---

## Working R Code

```r
library(data.table)

#' Optimized pipeline: replaces build_neighbor_lookup + compute_neighbor_stats
#' Preserves the original numerical estimand (max, min, mean of neighbor values).
#' Does NOT touch the trained Random Forest model.

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # -------------------------------------------------------------------
  # STEP 1: Convert to data.table and create a unique row index
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # -------------------------------------------------------------------
  # STEP 2: Build an integer-keyed edge list from the nb object
  #
  # rook_neighbors_unique is an nb object: a list of length

  # length(id_order), where element i contains integer indices into
  # id_order of cell i's neighbors (0L means no neighbors in nb).
  # -------------------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  cat(sprintf("Edge list: %d directed edges\n", nrow(edges)))

  # -------------------------------------------------------------------
  # STEP 3: For each source variable, compute neighbor stats via
  #          vectorized data.table joins and grouped aggregation.
  #
  # Logic equivalent to the original:
  #   For each row (id, year), find all neighbors sharing the same year,

  #   then compute max, min, mean of the neighbor's variable value.
  # -------------------------------------------------------------------

  # Minimal join table: only (id, year, row_idx) + the variable columns we need
  join_cols <- c("id", "year", "row_idx", neighbor_source_vars)
  dt_join <- dt[, ..join_cols]

  # Key the join table for fast equi-joins
  setkey(dt_join, id, year)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s ...\n", var_name))

    # -- 3a: Build focal table: (focal_id, year, row_idx)
    focal <- dt_join[, .(focal_id = id, year, row_idx)]

    # -- 3b: Join focal -> edges to get (focal_row_idx, neighbor_id, year)
    #         This expands each focal row by its number of neighbors.
    focal_edges <- merge(focal, edges, by = "focal_id", allow.cartesian = TRUE)
    #   columns: focal_id, year, row_idx (of focal), neighbor_id

    # -- 3c: Join to get the neighbor's variable value in the same year
    #         Key neighbor lookup table on (id, year)
    neighbor_vals_dt <- dt_join[, .(id, year, nval = get(var_name))]
    setkey(neighbor_vals_dt, id, year)

    setnames(focal_edges, "neighbor_id", "id")
    setkey(focal_edges, id, year)
    matched <- neighbor_vals_dt[focal_edges, on = .(id, year), nomatch = NA]
    #   columns: id (=neighbor_id), year, nval, focal_id, row_idx

    # -- 3d: Aggregate: group by focal row_idx, compute stats
    stats <- matched[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = row_idx]

    # -- 3e: Merge stats back into dt by row_idx
    #         Rows with no valid neighbors get NA (preserving original behavior)
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[stats, on = "row_idx",
       c(max_col, min_col, mean_col) := .(nb_max, nb_min, nb_mean)]

    # Rows not in stats remain NA (default for new columns in data.table)

    # Clean up large intermediates
    rm(focal, focal_edges, neighbor_vals_dt, matched, stats)
    gc()

    cat(sprintf("  Done: %s\n", var_name))
  }

  # -------------------------------------------------------------------
  # STEP 4: Remove helper column and return as data.frame
  #          (to remain compatible with downstream Random Forest predict)
  # -------------------------------------------------------------------
  dt[, row_idx := NULL]
  as.data.frame(dt)
}


# =====================================================================
# USAGE â€” drop-in replacement for the original pipeline
# =====================================================================

# cell_data             : the existing data.frame (~6.46M rows, ~110 columns)
# id_order              : integer vector of cell IDs matching nb object order
# rook_neighbors_unique : the precomputed spdep::nb object loaded from disk
# The trained RF model object (e.g., rf_model) is untouched.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Now use the existing trained Random Forest model as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, result)` | "Main bottleneck" | Minor cost (~seconds) |
| "Repeated list binding in `compute_neighbor_stats`" | Claimed present | **Does not exist in the code** |
| `build_neighbor_lookup()` | Not mentioned | **True bottleneck**: 6.46M iterations of string construction + O(n) named-vector lookups |
| Fix | â€” | Replace with vectorized `data.table` join + group-by aggregation |
| Expected runtime | 86+ hours | **~3â€“10 minutes** |
| RF model | â€” | Preserved, untouched |
| Numerical output | â€” | Identical (max, min, mean of non-NA neighbor values; NA when no neighbors) |
END_RESPONSE>>>
