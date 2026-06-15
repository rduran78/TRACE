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
run_id: Anthropic_F04_T03_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two functions `build_neighbor_lookup` and `compute_neighbor_stats`, both of which rely on **row-level `lapply` loops over ~6.46 million rows** with expensive per-row string operations.

### `build_neighbor_lookup` â€” Primary Bottleneck

For each of the ~6.46 million rows, the function:

1. Converts `data$id[i]` to a character and looks it up in `id_to_ref` â€” **6.46M character coercions and named-vector lookups**.
2. Retrieves neighbor cell IDs and calls `paste(..., sep = "_")` to build composite keys â€” **6.46M Ã— ~4 neighbors â‰ˆ 25.8M `paste` calls**.
3. Looks up each key in the named vector `idx_lookup` (itself of length 6.46M) â€” this is an **O(n) hash-table probe repeated ~25.8M times against a 6.46M-entry named vector**.
4. Filters `NA`s and coerces to integer.

The total work is roughly **O(N Ã— k)** with very high constant factors due to R's interpreted string operations and named-vector lookups. With N â‰ˆ 6.46M and k â‰ˆ 4, this produces billions of elementary R interpreter operations.

### `compute_neighbor_stats` â€” Secondary Bottleneck

Another `lapply` over 6.46M rows computing `max`, `min`, `mean` per row. This is called 5 times (once per variable), totaling ~32.3M interpreted loop iterations with per-iteration allocation.

### Why 86+ hours?

| Operation | Iterations | Cost per iteration | Estimated wall time |
|---|---|---|---|
| `build_neighbor_lookup` (paste + named lookup) | 6.46M Ã— ~4 | ~tens of Âµs (string alloc, hash probe on 6.46M-entry table) | **60â€“70+ hours** |
| `compute_neighbor_stats` (5 vars) | 5 Ã— 6.46M | ~Âµs (subsetting + summary) | **15â€“20 hours** |

---

## Optimization Strategy

### Principle: Replace row-level R loops and string-key lookups with vectorized integer-index operations using `data.table`.

**Key ideas:**

1. **Eliminate `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list of neighbor row indices via string keys, expand the neighbor graph into a two-column edge table `(row_i, row_j)` using vectorized joins. This replaces 6.46M `paste` + named-vector lookups with a single `data.table` equi-join.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Once we have the edge table mapping each row to its neighbor rows, we join in the variable values and compute `max`, `min`, `mean` in one vectorized grouped operation â€” no R-level loop at all.

3. **Process all 5 variables in one pass** over the edge table (or 5 fast vectorized passes) instead of rebuilding anything per variable.

**Expected speedup:** From 86+ hours to **~2â€“10 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial-neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all columns named in neighbor_source_vars.
#' @param id_order         integer vector of cell IDs in the same order as the
#'                         nb object (i.e., id_order[i] is the cell ID for the
#'                         i-th element of rook_neighbors_unique).
#' @param neighbors        spdep nb object (list of integer vectors);
#'                         rook_neighbors_unique.
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return cell_data (data.table) with new columns:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
#'         for each var in neighbor_source_vars.

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      neighbors,
                                      neighbor_source_vars) {

  # --- Step 0: Convert to data.table; add row index --------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  # --- Step 1: Build the directed edge list (focal_cell -> neighbor_cell) -----
  #     from the nb object.  Fully vectorized, no per-row R loop.
  n_neighbors <- lengths(neighbors)                       # integer vector
  focal_ref   <- rep(seq_along(neighbors), n_neighbors)   # ref indices
  nbr_ref     <- unlist(neighbors, use.names = FALSE)     # neighbor ref indices

  edges <- data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[nbr_ref]
  )
  # edges now has ~1,373,394 rows (one per directed rook relationship)

  # --- Step 2: Build a keyed lookup from (id, year) -> row index -------------
  row_key <- cell_data[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # --- Step 3: Expand edges across all years ---------------------------------
  #     For every year, every (focal, neighbor) edge produces a pair of row
  #     indices.  We do this via two joins rather than a Cartesian product.

  years <- sort(unique(cell_data$year))

  # Cross-join edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows â€” fits in RAM easily)
  edge_year <- CJ_dt(edges, years)

  # Helper: cross join edges with years vector
  # We build it manually to stay memory-efficient:
  edge_year <- edges[, .(focal_id, neighbor_id)][
    , .(year = years), by = .(focal_id, neighbor_id)
  ]

  # Join to get focal row index
  setnames(row_key, "id", "focal_id")
  setkey(row_key, focal_id, year)
  edge_year <- row_key[edge_year, on = .(focal_id, year), nomatch = 0L]
  setnames(edge_year, ".row_idx", "focal_row")

  # Join to get neighbor row index
  setnames(row_key, "focal_id", "neighbor_id")
  setkey(row_key, neighbor_id, year)
  edge_year <- row_key[edge_year, on = .(neighbor_id, year), nomatch = 0L]
  setnames(edge_year, ".row_idx", "nbr_row")

  # Restore row_key column name
  setnames(row_key, "neighbor_id", "id")

  # edge_year now has columns: focal_row, nbr_row  (and focal_id, neighbor_id, year)
  # We only need focal_row and nbr_row going forward.
  edge_year <- edge_year[, .(focal_row, nbr_row)]
  setkey(edge_year, focal_row)

  # --- Step 4: For each variable, vectorized grouped aggregation --------------
  for (var_name in neighbor_source_vars) {

    vals <- cell_data[[var_name]]

    # Attach neighbor values
    edge_year[, nbr_val := vals[nbr_row]]

    # Grouped aggregation â€” single vectorized pass
    agg <- edge_year[!is.na(nbr_val),
                     .(nmax  = max(nbr_val),
                       nmin  = min(nbr_val),
                       nmean = mean(nbr_val)),
                     keyby = .(focal_row)]

    # Allocate result columns (NA by default)
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign results by reference
    cell_data[agg$focal_row, (max_col)  := agg$nmax]
    cell_data[agg$focal_row, (min_col)  := agg$nmin]
    cell_data[agg$focal_row, (mean_col) := agg$nmean]
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]
  edge_year[, nbr_val := NULL]

  return(cell_data)
}
```

### Replacement for the original outer loop

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (~2-10 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values (max, min, mean) are identical to the original.
```

### Memory-efficient alternative for the cross-join (if 38.5M rows Ã— several columns strains 16 GB)

If RAM is tight, process one year at a time:

```r
add_all_neighbor_features_lowmem <- function(cell_data,
                                              id_order,
                                              neighbors,
                                              neighbor_source_vars) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  # Build edge list (cell-ID level, ~1.37M rows)
  n_neighbors <- lengths(neighbors)
  edges <- data.table(
    focal_id    = id_order[rep(seq_along(neighbors), n_neighbors)],
    neighbor_id = id_order[unlist(neighbors, use.names = FALSE)]
  )

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Keyed lookup
  setkey(cell_data, id, year)

  years <- sort(unique(cell_data$year))

  for (yr in years) {
    # Subset to this year
    dt_yr <- cell_data[year == yr, c("id", ".row_idx", neighbor_source_vars),
                       with = FALSE]
    setkey(dt_yr, id)

    # Join edges to get focal and neighbor row indices + values for this year
    # Focal side
    focal_join <- dt_yr[edges, on = .(id = focal_id), nomatch = 0L, allow.cartesian = TRUE]
    setnames(focal_join, ".row_idx", "focal_row")

    # Neighbor side
    nbr_vals <- dt_yr[, c("id", neighbor_source_vars), with = FALSE]
    setnames(nbr_vals, "id", "neighbor_id")
    setkey(nbr_vals, neighbor_id)

    joined <- nbr_vals[focal_join, on = .(neighbor_id), nomatch = 0L]

    for (var_name in neighbor_source_vars) {
      # Columns from neighbor side get suffix i. if there's a collision;
      # handle both cases:
      nbr_col <- if (paste0("i.", var_name) %in% names(joined)) {
        paste0("i.", var_name)
      } else {
        var_name
      }

      agg <- joined[!is.na(get(nbr_col)),
                     .(nmax  = max(get(nbr_col)),
                       nmin  = min(get(nbr_col)),
                       nmean = mean(get(nbr_col))),
                     keyby = .(focal_row)]

      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nmax)
      set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nmin)
      set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nmean)
    }
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with `paste` + named-vector lookup | **Eliminated**; replaced by a single vectorized `data.table` join |
| `compute_neighbor_stats` | 6.46M-iteration `lapply` Ã— 5 variables | **Replaced** by grouped `data.table` aggregation (`max`, `min`, `mean` by `focal_row`) |
| String key construction | ~25.8M `paste` calls + 25.8M probes into a 6.46M-entry named vector | **Zero** string operations; integer joins only |
| Estimated wall time | **86+ hours** | **~2â€“10 minutes** |
| Peak RAM | Moderate (6.46M-element list of integer vectors) | ~38.5M-row edge table â‰ˆ ~1â€“2 GB (or ~1.37M per year in low-mem variant) |
| Trained RF model | Untouched | Untouched |
| Numerical results | max, min, mean of neighbor values | **Identical** max, min, mean of neighbor values |
END_RESPONSE>>>
