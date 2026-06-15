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
run_id: Anthropic_F06_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) list construction
`build_neighbor_lookup` iterates over **~6.46 million rows** and, for each row, performs string-pasting, named-vector lookups, and `NA` filtering. The `paste()`/`idx_lookup` pattern is essentially a hash-table join done row-by-row in an R `lapply`. With ~6.46M iterations, each doing multiple character operations and named-vector lookups, this alone can take many hours.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated per-row R-level loops
`compute_neighbor_stats` iterates over the 6.46M-element `neighbor_lookup` list, extracting values and computing `max`/`min`/`mean` in pure R. This is called **5 times** (once per source variable), yielding ~32.3M R-level loop iterations with subsetting and aggregation.

### Why raster focal/kernel operations are **not** a drop-in replacement
Focal operations assume a regular rectangular grid with a fixed kernel. Rook contiguity on an irregular or masked spatial grid (with missing cells, coastlines, borders) changes the neighbor set per cell. A focal approach would require padding, masking, and would alter results at boundaries. Since the Random Forest model is already trained on features computed with the exact rook-neighbor logic, **we must preserve the original numerical estimand**. We use the rook-neighbor logic but implement it with vectorized operations.

---

## Optimization Strategy

1. **Replace string-key lookups with integer merge/join.** Use `data.table` keyed joins to map `(neighbor_id, year)` â†’ row index in O(n log n) instead of O(n) per row with R character hashing overhead.

2. **Expand the neighbor list to an edge table once**, then join against the data. This converts the entire neighbor lookup + stat computation into a single grouped aggregation â€” no R-level row loop at all.

3. **Compute all 5 variables' stats in one pass** over the edge table, eliminating 5 separate loops.

4. **Memory budget check:** The edge table has ~1,373,394 directed rook pairs Ã— 28 years â‰ˆ **38.5M rows** with a few integer/double columns â€” roughly 1â€“2 GB, well within 16 GB RAM.

**Expected speedup:** From 86+ hours to **minutes** (typically 2â€“10 minutes depending on disk I/O).

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # -------------------------------------------------------------------
  # Step 1: Convert to data.table and create a row-index column
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # -------------------------------------------------------------------
  # Step 2: Build an edge table from the nb object

  #   rook_neighbors_unique is a list of length = number of spatial cells.
  #   rook_neighbors_unique[[i]] gives integer indices (into id_order)
  #   of the rook neighbors of cell id_order[i].
  # -------------------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as 0L; filter those out
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  # edges now has columns: focal_id, neighbor_id
  # This represents ALL directed rook-neighbor pairs (spatial, time-invariant).

  cat(sprintf("Edge table: %d directed spatial pairs\n", nrow(edges)))

  # -------------------------------------------------------------------
  # Step 3: Create a lookup from (id, year) -> row index + variable values
  # -------------------------------------------------------------------
  # We only need the id, year, row_idx, and the source variables
  cols_needed <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- dt[, ..cols_needed]
  setkey(lookup, id, year)

  # -------------------------------------------------------------------
  # Step 4: Get unique years
  # -------------------------------------------------------------------
  years <- sort(unique(dt$year))

  # -------------------------------------------------------------------
  # Step 5: For each year, join edges with data to get neighbor values,
  #         then aggregate.  We process year-by-year to control memory.
  # -------------------------------------------------------------------

  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0("max_neighbor_", var_name), value = NA_real_)
    set(dt, j = paste0("min_neighbor_", var_name), value = NA_real_)
    set(dt, j = paste0("mean_neighbor_", var_name), value = NA_real_)
  }

  cat(sprintf("Processing %d years x %d variables...\n",
              length(years), length(neighbor_source_vars)))

  for (yr in years) {
    # Subset lookup to this year
    lk_yr <- lookup[year == yr]
    setkey(lk_yr, id)

    # Join edges: for each (focal_id, neighbor_id), get neighbor's values
    # First, get the focal cell's row index
    focal_info <- lk_yr[, .(id, .row_idx)]
    setkey(focal_info, id)

    # Map focal_id -> .row_idx for this year
    edge_yr <- edges[focal_info, on = .(focal_id = id), nomatch = 0L,
                     allow.cartesian = TRUE]
    # edge_yr has columns: focal_id, neighbor_id, .row_idx (focal's row in dt)

    # Now join to get neighbor values
    neighbor_vals <- lk_yr[, c("id", neighbor_source_vars), with = FALSE]
    setkey(neighbor_vals, id)

    edge_full <- neighbor_vals[edge_yr, on = .(id = neighbor_id), nomatch = NA,
                               allow.cartesian = FALSE]
    # edge_full now has: id (=neighbor_id), <source_vars>, focal_id, .row_idx

    # Aggregate by focal cell's .row_idx
    # Build aggregation expressions dynamically
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(max(.(as.name(v)), na.rm = TRUE)),
        bquote(min(.(as.name(v)), na.rm = TRUE)),
        bquote(mean(.(as.name(v)), na.rm = TRUE))
      )
    }))
    agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
      paste0(c("max_neighbor_", "min_neighbor_", "mean_neighbor_"), v)
    }))
    names(agg_exprs) <- agg_names

    # Use a single grouped aggregation
    agg_call <- as.call(c(as.name("list"),
                          setNames(agg_exprs, agg_names)))

    stats_yr <- edge_full[, eval(agg_call), by = .row_idx]

    # Handle Inf/-Inf from max/min on all-NA (shouldn't happen if edges exist,
    # but be safe)
    for (col in agg_names) {
      vals <- stats_yr[[col]]
      vals[is.infinite(vals)] <- NA_real_
      set(stats_yr, j = col, value = vals)
    }

    # Write results back into dt using row indices
    for (col in agg_names) {
      set(dt, i = stats_yr$.row_idx, j = col, value = stats_yr[[col]])
    }

    cat(sprintf("  Year %d done (%d focal cells aggregated)\n",
                yr, nrow(stats_yr)))
  }

  # -------------------------------------------------------------------
  # Step 6: Clean up and return
  # -------------------------------------------------------------------
  dt[, .row_idx := NULL]

  # Return as data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt[])
}

# =====================================================================
# USAGE
# =====================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Now cell_data has the same 15 new columns (max/min/mean Ã— 5 vars)
# with identical numerical values to the original implementation.
# Proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level `lapply` iterations with `paste`/named-vector lookup | Single `data.table` keyed join per year (~231K rows Ã— edges) |
| **Stat computation** | 6.46M Ã— 5 = 32.3M R-level iterations | One grouped aggregation per year (vectorized C-level) |
| **Total passes over data** | 1 (lookup) + 5 (stats) = 6 | 1 pass (all 5 vars aggregated together per year) |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **Peak memory** | ~6.46M-element list of integer vectors | ~38.5M-row edge table (~1â€“2 GB) â€” fits in 16 GB |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` over same rook-neighbor sets, same `na.rm` logic |
| **RF model** | Unchanged | Unchanged â€” no retraining |
END_RESPONSE>>>
