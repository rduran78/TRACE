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
run_id: Anthropic_F07_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` call that performs character pasting, hash lookups, and NA filtering row-by-row in pure R.** This is an O(n) loop with expensive string operations at every iteration (~6.46M iterations). Then `compute_neighbor_stats` does another 6.46M-iteration `lapply` over the result. With 5 variables, that's ~32.3M R-level loop iterations total, plus the initial build. On a 16 GB laptop, the nested character key lookups and list allocations dominate both time and memory.

Specific problems:

1. **String-key lookups (`paste` + named vector indexing):** For every row, `paste(neighbor_cell_ids, data$year[i], sep="_")` creates strings, then looks them up in a 6.46M-element named vector. Named vector lookup in R is O(n) per query in the worst case (hash collisions) and always involves string allocation/hashing overhead.

2. **Row-by-row `lapply` over 6.46M rows:** Pure R loops/lapply over millions of rows is inherently slow â€” no vectorization, no compiled-code fast path.

3. **Redundant recomputation:** The neighbor *structure* is time-invariant (same grid every year), but the lookup is rebuilt as if it could change. The 344,208 cells have fixed rook neighbors; only the variable values change across years.

4. **Memory pressure:** A 6.46M-element list of integer vectors, plus intermediate character vectors, can consume several GB.

## Optimization Strategy

**Key insight:** Because the neighbor topology is *time-invariant*, we can separate the spatial structure from the temporal panel. We only need a 344,208-element neighbor lookup (cell-to-cell), then use vectorized operations across all years simultaneously via `data.table` joins and grouped aggregation.

**Approach:**

1. Convert `rook_neighbors_unique` (an `nb` object) into an edge list (two-column integer matrix of `(cell_id, neighbor_id)` pairs) â€” ~1.37M rows.
2. Store the panel in a `data.table` keyed on `(id, year)`.
3. For each source variable, join the edge list to the data to get all neighbor values in one vectorized merge, then aggregate (`max`, `min`, `mean`) by `(id, year)` â€” fully vectorized, no R-level row loop.

This replaces ~6.46M R-level iterations with a single `data.table` keyed join + grouped aggregation per variable â€” estimated speedup: **~500â€“1000Ã—**, bringing runtime from 86+ hours to **minutes**.

The numerical results are identical: every cell-year gets the max, min, and mean of its rook neighbors' values for each variable, with `NA` handling preserved.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert the nb object to an edge list (one-time, fast)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of length length(id_order);

# nb_obj[[i]] contains integer indices into id_order of neighbors of cell i
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbs <- nb_obj[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nbs <- nbs[nbs > 0L]
    if (length(nbs) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nbs])
  }))
  edges
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has columns: id, neighbor_id
# ~1.37M rows (directed rook-neighbor pairs)

# ---------------------------------------------------------------
# 2. Convert panel to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# ---------------------------------------------------------------
# 3. Vectorized neighbor stats computation
# ---------------------------------------------------------------
compute_and_add_all_neighbor_features <- function(cell_dt, edges, source_vars) {
  # We join edges to the data twice:
  #   - first to get the year for each cell (implicitly via the join)
  #   - then to get the neighbor's value for that year

  # Create a slim table: id, year, and all source vars
  val_cols <- source_vars
  slim <- cell_dt[, c("id", "year", val_cols), with = FALSE]

  # Key for fast join on neighbor side
  setkey(slim, id, year)

  # Expand edges Ã— years: for each edge (id, neighbor_id),

  # we need every year. But rather than a cross join (expensive),
  # we merge edges with the panel on the focal cell to get the years
  # that exist, then look up the neighbor's value.

  # Step A: Get all (id, year) pairs from the panel
  id_year <- unique(cell_dt[, .(id, year)])

  # Step B: Join id_year with edges on 'id' to get (id, year, neighbor_id)
  #         This gives us ~1.37M * 28 â‰ˆ 38.5M rows (manageable)
  setkey(id_year, id)
  setkey(edges, id)
  expanded <- edges[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year

  # Step C: Look up neighbor values by joining on (neighbor_id, year)
  setnames(slim, "id", "neighbor_id")
  setkey(slim, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  merged <- slim[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged has: neighbor_id, year, <val_cols>, id
  # where <val_cols> are the neighbor's values

  # Step D: Aggregate by (id, year) to get max, min, mean per variable
  setkey(merged, id, year)

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Evaluate aggregation
  agg_result <- merged[,
    lapply(agg_exprs, eval, envir = .SD),
    by = .(id, year),
    .SDcols = val_cols
  ]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col in agg_names) {
    agg_result[is.infinite(get(col)), (col) := NA_real_]
  }

  # Step E: Merge back to original data
  setkey(cell_dt, id, year)
  setkey(agg_result, id, year)

  # Remove old neighbor columns if they exist (idempotency)
  old_cols <- intersect(agg_names, names(cell_dt))
  if (length(old_cols) > 0) cell_dt[, (old_cols) := NULL]

  cell_dt <- agg_result[cell_dt, on = c("id", "year")]

  cell_dt
}

# ---------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_dt <- compute_and_add_all_neighbor_features(
  cell_dt, edges, neighbor_source_vars
)

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)
```

### If the dynamic `bquote` aggregation is fragile, here is a simpler, equally fast alternative for Step D:

```r
# Step D alternative â€” loop over variables, one aggregation each (still vectorized)
agg_list <- list()
for (v in val_cols) {
  agg_v <- merged[,
    {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(nmax = NA_real_, nmin = NA_real_, nmean = NA_real_)
      } else {
        list(nmax = max(vals), nmin = min(vals), nmean = mean(vals))
      }
    },
    by = .(id, year)
  ]
  setnames(agg_v,
    c("nmax", "nmin", "nmean"),
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  )
  agg_list[[v]] <- agg_v
}

# Merge all aggregation results together
agg_result <- Reduce(function(a, b) merge(a, b, on = c("id", "year"), all = TRUE), agg_list)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-element R list via `paste`+named vector | 1.37M-row `data.table` edge list |
| Stats computation | 6.46M Ã— 5 `lapply` iterations | 5 vectorized `data.table` grouped aggregations |
| Estimated runtime | 86+ hours | **5â€“15 minutes** |
| Peak memory | ~8â€“12 GB (lists + strings) | ~4â€“6 GB (data.table joins) |
| Numerical result | Identical | **Identical** (same max/min/mean, same NA handling) |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
