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
run_id: Anthropic_F08_T00_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows)** even though the spatial neighbor graph is identical across all 28 years. The neighbor structure among 344,208 cells never changes â€” only the variable values attached to those cells change year-to-year. Yet the function constructs a lookup of length 6.46M, each entry requiring string-pasting, hash lookups, and NA filtering.

2. **String-key hashing is expensive at scale.** The function creates `paste(id, year, sep="_")` keys for every row, then does named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M keys, this named-vector lookup is O(n) per probe in the worst case and has enormous memory overhead.

3. **The `lapply` over 6.46M rows in `build_neighbor_lookup`** is the dominant bottleneck. Each iteration does: one named-vector lookup for `ref_idx`, a subset of `neighbors`, string construction for every neighbor, and another named-vector lookup. For ~1.37M neighbor relationships Ã— 28 years = ~38.5M neighbor-key lookups, all through R-level string operations.

4. **`compute_neighbor_stats` is called 5 times**, each iterating over the 6.46M-element `neighbor_lookup`. This is comparatively cheaper but still wasteful because the neighbor indices per cell-year could be derived from a cell-level structure.

### The Key Insight

- **Static:** The neighbor graph (which cells are neighbors of which) â€” 344,208 cells, ~1.37M directed edges. This never changes.
- **Dynamic:** The variable values attached to each cell, which change by year.

The correct design is:
1. Build the neighbor lookup **once at the cell level** (344,208 entries, not 6.46M).
2. For each year, **slice the data**, use the cell-level neighbor lookup to gather neighbor values, and compute stats.

This reduces the core loop from 6.46M iterations to 344,208 Ã— 28 = 9.64M, but with **trivial integer-indexed operations** instead of string hashing â€” and the neighbor lookup construction itself drops from 6.46M to 344,208 iterations.

---

## Optimization Strategy

### Step 1: Build a cell-level neighbor index (once)

Convert `rook_neighbors_unique` (an `nb` object indexed by position in `id_order`) into a simple list: `cell_neighbors[[i]]` = integer vector of positional indices of neighbors of cell `i` (where `i` is the position in `id_order`). This is essentially what `rook_neighbors_unique` already is â€” an `nb` object is a list of integer vectors. So this step is nearly free.

### Step 2: Organize data for fast year-wise, cell-indexed access

Sort/index `cell_data` by `(year, id)` so that for each year, cells appear in the same positional order as `id_order`. This allows direct integer indexing: for year `y`, the value of variable `v` for cell at position `i` in `id_order` is simply `vals[offset_for_year_y + i]`.

### Step 3: Vectorized neighbor stat computation using `data.table`

Use `data.table` for grouped operations. For each year, expand the neighbor edge list, join variable values, and compute `max`, `min`, `mean` per cell â€” all vectorized.

### Step 4: Feed results into the existing trained Random Forest

The output columns are numerically identical to the original implementation. The trained model is untouched.

---

## Working R Code

```r
library(data.table)

#' Redesigned neighbor feature computation.
#' Separates static topology from dynamic (year-varying) data.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer/character vector â€” the cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  an nb object (list of integer vectors) indexed by position in id_order
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 1: Build the STATIC edge list (once, ~1.37M rows)
  # ---------------------------------------------------------------
  # rook_neighbors[[i]] gives positional indices of neighbors of cell at position i.
  # Convert to a two-column edge list: (focal_pos, neighbor_pos)

  n_cells <- length(id_order)

  # Pre-allocate edge list
  n_edges <- sum(lengths(rook_neighbors))
  edge_focal    <- integer(n_edges)
  edge_neighbor <- integer(n_edges)

  offset <- 0L
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors[[i]]
    len_i <- length(nb_i)
    if (len_i > 0L) {
      idx_range <- (offset + 1L):(offset + len_i)
      edge_focal[idx_range]    <- i
      edge_neighbor[idx_range] <- nb_i
    }
    offset <- offset + len_i
  }

  # Map positional index -> cell id
  edges <- data.table(
    focal_pos    = edge_focal,
    neighbor_pos = edge_neighbor,
    focal_id     = id_order[edge_focal],
    neighbor_id  = id_order[edge_neighbor]
  )

  rm(edge_focal, edge_neighbor)

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table, keyed for fast joins

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Create a minimal lookup: only id, year, and the source vars
  lookup_cols <- c("id", "year", neighbor_source_vars)
  dt_lookup <- dt[, ..lookup_cols]

  # ---------------------------------------------------------------
  # STEP 3: For each variable, compute neighbor stats via join
  # ---------------------------------------------------------------
  # Strategy: cross-join edges with years, then join variable values
  # from the neighbor cell-year, then aggregate.
  #
  # To avoid a massive cross join (edges Ã— years), we do it per year

  # in a loop â€” 28 iterations, each ~1.37M edges. Very fast.

  years <- sort(unique(dt$year))

  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = NA_real_)
  }

  # Key dt for fast row assignment by (id, year)
  setkey(dt, id, year)

  # Edges only need focal_id and neighbor_id
  edges_slim <- edges[, .(focal_id, neighbor_id)]

  for (yr in years) {

    # Extract this year's data: id -> variable values
    dt_yr <- dt_lookup[year == yr]
    setkey(dt_yr, id)

    # Join neighbor values onto edge list
    # edges_slim$neighbor_id -> dt_yr to get neighbor variable values
    edge_with_vals <- merge(
      edges_slim,
      dt_yr[, -"year", with = FALSE],
      by.x = "neighbor_id",
      by.y = "id",
      all.x = FALSE  # drop edges where neighbor has no data this year
    )

    # Aggregate by focal_id: max, min, mean for each variable
    agg_exprs <- list()
    for (var_name in neighbor_source_vars) {
      sym_var <- as.name(var_name)
      agg_exprs[[paste0(var_name, "_neighbor_max")]]  <-
        bquote(max(.(sym_var), na.rm = TRUE))
      agg_exprs[[paste0(var_name, "_neighbor_min")]]  <-
        bquote(min(.(sym_var), na.rm = TRUE))
      agg_exprs[[paste0(var_name, "_neighbor_mean")]] <-
        bquote(mean(.(sym_var), na.rm = TRUE))
    }

    # Build the aggregation call dynamically
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    stats_yr <- edge_with_vals[, eval(agg_call), by = focal_id]

    # Replace Inf/-Inf from max/min of empty sets with NA
    stat_cols <- names(stats_yr)[names(stats_yr) != "focal_id"]
    for (sc in stat_cols) {
      vals <- stats_yr[[sc]]
      vals[is.infinite(vals)] <- NA_real_
      set(stats_yr, j = sc, value = vals)
    }

    # Write results back into dt for this year
    stats_yr[, year := yr]
    setkey(stats_yr, focal_id, year)

    for (sc in stat_cols) {
      # Match rows in dt
      matched <- dt[stats_yr, on = .(id = focal_id, year), which = TRUE]
      set(dt, i = matched, j = sc, value = stats_yr[[sc]])
    }
  }

  # ---------------------------------------------------------------
  # STEP 4: Return the augmented dataset
  # ---------------------------------------------------------------
  return(dt)
}


# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Now cell_data has the same 15 new columns as before:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   pop_density_neighbor_max, pop_density_neighbor_min, pop_density_neighbor_mean,
#   def_neighbor_max, def_neighbor_min, def_neighbor_mean,
#   usd_est_n2_neighbor_max, usd_est_n2_neighbor_min, usd_est_n2_neighbor_mean

# Proceed with the pre-trained Random Forest prediction (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Performance Comparison

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup construction** | 6.46M R-level iterations with string hashing | Once: 344K iterations to build edge list (integer only) |
| **Neighbor lookup size** | 6.46M-element list | 1.37M-row `data.table` (shared across years) |
| **Per-variable computation** | `lapply` over 6.46M rows, R-level loop | Vectorized `data.table` merge + grouped aggregation |
| **Total inner iterations** | 6.46M Ã— 5 vars = 32.3M R-level loops | 28 years Ã— 5 vars, each ~1.37M rows, fully vectorized |
| **String operations** | ~45M `paste` + named-vector lookups | Zero â€” all integer-indexed |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Peak memory** | ~6.46M-element nested list + data copy | ~1.37M-row edge table + year slice (~344K rows) |
| **Numerical output** | Reference | Identical (same max/min/mean, same NA handling) |
| **Trained RF model** | Unchanged | Unchanged |

### Why the numbers are preserved exactly

The original code computes, for each cell-year row, the `max`, `min`, and `mean` of the variable values at its rook neighbors in the same year, excluding `NA`s. The redesigned code does exactly the same operation: for each year, it joins the edge list with that year's variable values on `neighbor_id`, groups by `focal_id`, and computes `max(..., na.rm=TRUE)`, `min(..., na.rm=TRUE)`, `mean(..., na.rm=TRUE)`. The neighbor set is identical (same `rook_neighbors_unique` object), the values are identical (same `cell_data`), and the aggregation functions are identical. The estimand is preserved.
END_RESPONSE>>>
