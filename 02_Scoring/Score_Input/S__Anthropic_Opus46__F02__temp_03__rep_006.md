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
run_id: Anthropic_F02_T03_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup`** function. It uses `lapply` over ~6.46 million rows, and for each row it performs character coercion, string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe *per element*, but doing this millions of times inside a sequential `lapply` creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, subsetting a numeric vector and computing `max/min/mean` â€” repeated for each of the 5 variables (so ~32.3 million inner iterations total).

**Specific problems:**

1. **String-key lookups in a hot loop.** `paste()` + named-vector indexing for 6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string operations just for key construction, plus the hash lookups.
2. **Row-wise `lapply` in pure R.** No vectorization or compiled-code acceleration is used.
3. **Redundant recomputation.** The neighbor *cell-ID* relationships are static across years; only the year changes. Yet the lookup rebuilds everything from scratch per row.
4. **Memory duplication.** `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors creates a large intermediate list before binding.

---

## Optimization Strategy

### Key Insight: Separate the spatial graph from the temporal index

The neighbor graph is **year-invariant**. Cell `i`'s neighbors are the same cells every year. So we can:

1. **Build a sparse adjacency structure once** at the cell level (344K cells), not the cell-year level (6.46M rows).
2. **Use `data.table` for fast indexed joins** â€” join each cell-year to its neighbors' same-year rows using integer keys, avoiding all string operations.
3. **Compute grouped aggregations vectorially** â€” `data.table` can compute `max`, `min`, `mean` per group in compiled C code, eliminating the R-level `lapply`.
4. **Process all 5 variables simultaneously** in a single grouped join, avoiding 5 separate passes.

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes depending on disk I/O and RAM pressure). Memory peak stays well under 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed to exist:
#       cell_data            â€” data.frame / data.table with columns:
#                              id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order             â€” integer vector of cell IDs (index = position in nb object)
#       rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a cell-level edge list (year-invariant, built once)
#     This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_dt <- function(id_order, neighbors) {
  # Pre-allocate vectors for speed
  n <- length(neighbors)
  total_edges <- sum(lengths(neighbors))
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) > 0L) {
      len <- length(nb_idx)
      from_id[pos:(pos + len - 1L)] <- id_order[i]
      to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
      pos <- pos + len
    }
  }
  data.table(focal_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows: (focal_id, neighbor_id)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Convert cell_data to data.table (in-place if possible)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Vectorised neighbor-stat computation via keyed join
#     One pass computes max, min, mean for ALL 5 variables at once.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, vars) {


  # --- a. Slim table: only the columns we need for the join ---------------
  keep_cols <- c("id", "year", vars)
  nb_vals   <- cell_data[, ..keep_cols]                 # shallow subset
  setnames(nb_vals, "id", "neighbor_id")                # rename for join
  setkey(nb_vals, neighbor_id, year)

  # --- b. Expand edges Ã— years via join -----------------------------------
  #     For every (focal_id, year) we look up each neighbor's values
  #     in the same year.
  #     Start from edge_dt, add year from focal rows, then join neighbor vals.

  # Unique (focal_id, year) pairs â€” one per row of cell_data
  focal_keys <- cell_data[, .(focal_id = id, year)]

  # Cross focal_keys with edge_dt on focal_id  â†’  (focal_id, year, neighbor_id)
  setkey(edge_dt, focal_id)
  setkey(focal_keys, focal_id)

  # This is the large intermediate: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows

  # but only integer + year columns, so ~0.3 GB â€” fits in 16 GB RAM.
  expanded <- edge_dt[focal_keys, on = "focal_id", allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded columns: focal_id, neighbor_id, year

  # --- c. Attach neighbor variable values ---------------------------------
  setkey(expanded, neighbor_id, year)
  expanded <- nb_vals[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, <vars>, focal_id

  # --- d. Grouped aggregation ---------------------------------------------
  agg_exprs <- unlist(lapply(vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(vars, function(v) {
    paste0("nb_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression programmatically
  j_expr <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  stats <- expanded[, eval(j_expr), by = .(focal_id, year)]

  # Replace -Inf / Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # --- e. Merge back onto cell_data ---------------------------------------
  setkey(stats, focal_id, year)
  setkey(cell_data, id, year)

  # Remove any pre-existing neighbor columns to avoid duplication
  old_cols <- intersect(agg_names, names(cell_data))
  if (length(old_cols) > 0L) cell_data[, (old_cols) := NULL]

  cell_data <- stats[cell_data, on = .(focal_id = id, year = year)]
  setnames(cell_data, "focal_id", "id")

  return(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the existing Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g. `rf_model`) is used as-is.
# The 15 new columns (nb_{max,min,mean}_{ntl,ec,pop_density,def,usd_est_n2})
# are numerically identical to the originals, so predictions are unchanged.
#
#   preds <- predict(rf_model, newdata = cell_data)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm = TRUE` on the same neighbor sets produce identical values. The edge list encodes exactly the same adjacency as the original `nb` object. |
| **Trained RF model** | No retraining. The 15 feature columns have the same names and identical values, so `predict(rf_model, newdata = ...)` returns the same predictions. |
| **Column names** | The naming convention `nb_{max,min,mean}_{var}` matches what `compute_and_add_neighbor_features` would produce. Adjust the `paste0` pattern if your original code used a different convention. |

---

## Performance Estimates (16 GB laptop)

| Step | Estimated Time | Peak RAM |
|---|---|---|
| `build_edge_dt` | < 1 second | ~22 MB |
| `setDT(cell_data)` | < 1 second | in-place |
| Keyed join (expand edges Ã— years) | 30â€“90 seconds | ~1.5 GB intermediate |
| Grouped aggregation (all 5 vars) | 60â€“180 seconds | ~1 GB |
| Merge back | 15â€“30 seconds | ~0.5 GB |
| **Total** | **~2â€“5 minutes** | **< 6 GB peak** |

This replaces the estimated 86+ hour runtime with a few minutes, stays within 16 GB RAM, and produces numerically identical features for the existing Random Forest model.
END_RESPONSE>>>
