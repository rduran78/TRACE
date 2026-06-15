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
run_id: Anthropic_F07_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookups** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called *inside a `lapply` over ~6.46 million rows*. Named character vector lookup in R is O(n) hashing per call, and doing this millions of times is catastrophic.
2. **The neighbor lookup is year-aware but recomputes the same spatial topology for every year.** The rook-neighbor graph has only 344,208 cells; the year dimension simply replicates it. Yet the code treats every cell-year as a unique lookup problem, multiplying work by 28Ã—.
3. **`compute_neighbor_stats` is fine algorithmically** (a single vectorised `vals[idx]` per row), but it is called *after* the slow lookup is built, and it also loops over 6.46M rows in R-level `lapply`.

**Summary of cost:**
- `build_neighbor_lookup`: ~6.46M iterations Ã— string operations Ã— named-vector lookups â†’ **dominant cost (~86+ hours)**.
- `compute_neighbor_stats`: ~6.46M R-level iterations Ã— 5 variables â†’ secondary cost.

## Optimization Strategy

### Key Insight
The neighbor graph is **purely spatial** â€” it does not change across years. We should:

1. **Work in integer index space, not string space.** Replace all `paste`/named-vector lookups with integer arithmetic.
2. **Vectorise across years.** Build a spatial-only neighbor lookup once (344K cells), then use vectorised operations across the full panel.
3. **Use `data.table` for grouped, vectorised neighbor aggregation** â€” join the data to its neighbors in one merge, then compute `max`, `min`, `mean` by group. This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with a single vectorised pipeline.

### Complexity Reduction
| Step | Old | New |
|---|---|---|
| Lookup construction | 6.46M Ã— string ops | 344K integer list (once) |
| Stat computation (per var) | 6.46M R-level iterations | One `data.table` equi-join + grouped aggregation |
| Total R-level iterations | ~38.7M (6 Ã— 6.46M) | ~0 (fully vectorised) |

**Expected runtime: minutes, not days.**

## Working R Code

```r
# ============================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Drop-in replacement â€” preserves the original numerical
# estimand (neighbor max, min, mean) exactly.
# Does NOT touch the trained Random Forest model.
# ============================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ----------------------------------------------------------
  # 0.  Convert to data.table (by reference if already one)
  # ----------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure a row-order key so we can put results back
  dt[, .rowid := .I]

  # ----------------------------------------------------------
  # 1.  Build a SPATIAL-ONLY edge list (integer cell ids)
  #     This is done once for 344,208 cells â€” trivial cost.
  # ----------------------------------------------------------
  # id_order[i] is the cell id whose neighbors are
  # rook_neighbors_unique[[i]].  Neighbor indices point back
  # into id_order.
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb > 0L]                       # spdep uses 0 for no-neighbor
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i],
               neighbor_id = id_order[nb])
  }))
  # edges now has ~1.37 M rows (directed rook pairs)

  # ----------------------------------------------------------
  # 2.  For each source variable, join + aggregate
  # ----------------------------------------------------------
  # We need neighbor values keyed by (neighbor_id, year).
  # Strategy:

  #   a) Create a slim table: (id, year, value)
  #   b) Join edges on focal_id â†’ get (focal_id, neighbor_id, year)
  #   c) Join neighbor values on (neighbor_id, year)
  #   d) Aggregate max/min/mean grouped by (focal_id, year)
  #   e) Merge results back onto dt

  # Pre-index for fast joins
  setkey(edges, focal_id)

  for (var_name in neighbor_source_vars) {

    message("Processing neighbor stats for: ", var_name)

    # a) slim value table
    val_dt <- dt[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # b) expand edges Ã— years via join
    #    For each focal_id in edges, we get all years from val_dt
    #    We need (focal_id, neighbor_id, year).
    #    Efficient approach: join edges with the focal's years.
    focal_years <- unique(dt[, .(id, year)])
    setkey(focal_years, id)

    # Merge edges with focal years: for every edge, replicate across
    # all years the focal cell appears in.
    # But since every cell appears in every year (balanced panel),
    # we can use a cross-join shortcut:
    years_vec <- sort(unique(dt$year))

    # Build (focal_id, neighbor_id, year) â€” ~1.37M Ã— 28 = ~38.4M rows
    # This fits in memory: 3 integer columns Ã— 38.4M â‰ˆ 0.9 GB
    expanded <- edges[, .(year = years_vec), by = .(focal_id, neighbor_id)]

    # c) join neighbor values
    setkey(expanded, neighbor_id, year)
    expanded[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]

    # d) aggregate
    agg <- expanded[!is.na(neighbor_val),
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(focal_id, year)]

    # Name the output columns to match original convention
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

    # e) merge back
    setkey(agg, focal_id, year)
    setkey(dt,  id,       year)
    dt[agg, (c(max_col, min_col, mean_col)) :=
         mget(paste0("i.", c(max_col, min_col, mean_col))),
       on = .(id = focal_id, year)]

    # Clean up to keep memory bounded
    rm(val_dt, expanded, agg)
    gc()
  }

  # ----------------------------------------------------------
  # 3.  Return as data.frame (or data.table) preserving order
  # ----------------------------------------------------------
  setorder(dt, .rowid)
  dt[, .rowid := NULL]

  # If caller expects a plain data.frame:
  setDF(dt)
  return(dt)
}

# ============================================================
# USAGE â€” replaces the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains the 15 new columns:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ...
# with EXACTLY the same numerical values as the original code.
# The trained Random Forest model is untouched.
```

## Memory-Constrained Variant

If the ~38.4M-row expanded table is too large for 16 GB RAM (unlikely, but possible if many columns are carried), process years in chunks:

```r
# Replace the single expansion with a year-chunked loop:
# Inside the for(var_name ...) loop, replace steps b-d with:

agg_list <- vector("list", length(years_vec))
for (yi in seq_along(years_vec)) {
  yr <- years_vec[yi]
  val_yr <- val_dt[year == yr]
  setkey(val_yr, id)

  chunk <- copy(edges)
  chunk[, year := yr]
  setkey(chunk, neighbor_id)
  chunk[val_yr, neighbor_val := i.val, on = .(neighbor_id = id)]

  agg_list[[yi]] <- chunk[!is.na(neighbor_val),
                           .(nb_max  = max(neighbor_val),
                             nb_min  = min(neighbor_val),
                             nb_mean = mean(neighbor_val)),
                           by = .(focal_id, year)]
}
agg <- rbindlist(agg_list)
```

This peaks at only ~1.37M rows per chunk (~33 MB), keeping total RAM well under 16 GB.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup build | 6.46M string ops in R loop | One-time 344K integer edge list |
| Stat computation | R-level `lapply` Ã— 6.46M Ã— 5 vars | Vectorised `data.table` join + group-by |
| Estimated runtime | 86+ hours | **5â€“15 minutes** |
| Peak RAM | Moderate (6.46M-element list) | ~1â€“2 GB (chunked: ~0.5 GB) |
| Numerical result | Reference | **Identical** (same max/min/mean) |
| RF model | Untouched | **Untouched** |
END_RESPONSE>>>
