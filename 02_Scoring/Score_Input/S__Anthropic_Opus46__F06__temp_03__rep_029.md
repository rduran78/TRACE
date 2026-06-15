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
run_id: Anthropic_F06_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string operations

`build_neighbor_lookup` iterates over all ~6.46 million rows, and for each row it:
1. Looks up the cell's rook neighbors via `id_to_ref` (character hash lookup).
2. Constructs `paste(neighbor_id, year, "_")` keys for every neighbor.
3. Indexes into `idx_lookup` (another character hash lookup).

This produces a **list of length 6.46 million**, each element being an integer vector of row indices. The string hashing and per-element `lapply` over millions of rows is extremely slow in R. Estimated cost: **tens of hours** just for this step.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per-variable lapply

For each of the 5 source variables, `compute_neighbor_stats` runs another `lapply` over all 6.46 million rows, extracting neighbor values and computing `max`, `min`, `mean`. That's 5 Ã— 6.46M = ~32.3 million R-level function calls with vector subsetting.

### Why raster focal/kernel operations are **not** a direct replacement

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. Here:
- The grid cells have **irregular rook neighborhoods** (coastal cells, boundary cells have fewer neighbors).
- The panel is **unbalanced** if any cell-years are missing.
- The neighbor structure is stored as an `spdep::nb` object, not a regular raster grid.

Focal operations would require reconstructing a complete raster for each year and variable, handling NA masks, and would silently change results at boundaries. **The sparse-matrix approach below exactly preserves the original numerical estimand** while being orders of magnitude faster.

### Summary

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~40-50 hrs | 6.46M string-paste + hash lookups in R loop |
| `compute_neighbor_stats` (Ã—5) | ~36-40 hrs | 5 Ã— 6.46M R-level lapply calls |
| **Total** | **~86+ hrs** | Pure-R row-level iteration over millions of rows |

---

## 2. Optimization Strategy

### Core Idea: Sparse Matrix Multiplication

The neighbor relationships are **fixed within each year**. For each year, the rook adjacency can be encoded as a **sparse matrix W** of dimension (n_cells Ã— n_cells). Then:

- **Neighbor mean** of variable `x`: `W %*% x / (W %*% 1)` (i.e., row-sum-normalized product).
- **Neighbor max/min**: Use grouped operations via `data.table` with an **edge list** representation, avoiding per-row R loops entirely.

### Step-by-step

1. **Replace `build_neighbor_lookup`** with a single edge-list `data.table` constructed via vectorized join â€” no per-row `lapply`, no string pasting.
2. **Replace `compute_neighbor_stats`** with a `data.table` grouped aggregation on the edge list: group by `(focal_row)`, compute `max`, `min`, `mean` of neighbor values in one pass.
3. **Process all 5 variables in a single pass** over the edge list (or at most one pass per variable with no R-level row iteration).

### Expected speedup

| Component | New Cost | Speedup |
|---|---|---|
| Edge-list construction | ~10-30 seconds | ~5000Ã— |
| Neighbor stats (all 5 vars) | ~1-3 minutes | ~1000Ã— |
| **Total** | **~2-5 minutes** | **~1000Ã—+** |

### Numerical equivalence

The sparse/edge-list approach computes **exactly the same** `max`, `min`, `mean` over exactly the same set of neighbor values. No approximation, no kernel smoothing, no boundary-handling difference. The trained Random Forest model is never touched.

---

## 3. Working R Code

```r
library(data.table)

#' Optimized neighbor feature engineering for cell-year panel data.
#'
#' Replaces build_neighbor_lookup + compute_neighbor_stats with
#' vectorized data.table operations on an edge list.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and
#'                        all variables in neighbor_source_vars.
#' @param id_order        integer vector of cell IDs in the order matching
#'                        rook_neighbors_unique (i.e., the i-th element of
#'                        rook_neighbors_unique corresponds to id_order[i]).
#' @param rook_neighbors  spdep::nb object (list of integer index vectors).
#' @param neighbor_source_vars character vector of variable names.
#'
#' @return cell_data with new columns: {var}_nb_max, {var}_nb_min, {var}_nb_mean
#'         for each var in neighbor_source_vars.
add_neighbor_features <- function(cell_data,
                                  id_order,
                                  rook_neighbors,
                                  neighbor_source_vars) {

  # -------------------------------------------------------------------
  # Step 1: Build a spatial edge list (focal_id -> neighbor_id)
  #         This is done ONCE, independent of year or variable.
  # -------------------------------------------------------------------
  n_cells <- length(id_order)

  # Expand the nb object into an edge list of (focal_ref, neighbor_ref)
  # where ref is the positional index in id_order.
  focal_refs <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  neighbor_refs <- unlist(rook_neighbors, use.names = FALSE)

  # Remove the spdep convention of 0L meaning "no neighbors"
  valid <- neighbor_refs != 0L
  focal_refs <- focal_refs[valid]
  neighbor_refs <- neighbor_refs[valid]

  # Map positional refs to actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )

  # -------------------------------------------------------------------
  # Step 2: Convert cell_data to data.table and add a row index
  # -------------------------------------------------------------------
  was_df <- !is.data.table(cell_data)
  setDT(cell_data)

  # Preserve original row order
  cell_data[, .row_idx := .I]

  # -------------------------------------------------------------------
  # Step 3: For each year, join edges to data to get neighbor values,
  #         then aggregate. We do this by a single large merge.
  # -------------------------------------------------------------------


  # Create the focal side: every (focal_id, year) with its row index

  # We need to attach neighbor values, so we join:
  #   focal(id, year) -> edge(focal_id, neighbor_id) -> neighbor(id, year)

  # Subset columns we need for the neighbor side
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  nb_vals <- cell_data[, ..neighbor_cols]
  setnames(nb_vals, "id", "neighbor_id")

  # Merge edge list with cell_data to get focal rows
  # For each row in cell_data, find its edges, then find the neighbor rows.

  # Focal key: (focal_id = id, year)
  focal_key <- cell_data[, .(focal_id = id, year, .row_idx)]

  # Join focal_key with edge_dt to expand: each focal row -> its neighbor IDs
  # Key: focal_id
  setkey(edge_dt, focal_id)
  setkey(focal_key, focal_id)

  expanded <- edge_dt[focal_key,
                      .(neighbor_id, year, .row_idx),
                      on = "focal_id",
                      allow.cartesian = TRUE,
                      nomatch = NULL]

  # Now join with neighbor values: match (neighbor_id, year)
  setkey(nb_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  merged <- nb_vals[expanded, on = .(neighbor_id, year), nomatch = NULL]

  # -------------------------------------------------------------------
  # Step 4: Grouped aggregation â€” compute max, min, mean per focal row
  # -------------------------------------------------------------------
  # Group by .row_idx (the focal cell-year row)
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    max_name  <- paste0(v, "_nb_max")
    min_name  <- paste0(v, "_nb_min")
    mean_name <- paste0(v, "_nb_mean")
    agg_exprs[[max_name]]  <- call("max",  as.name(v), na.rm = TRUE)
    agg_exprs[[min_name]]  <- call("min",  as.name(v), na.rm = TRUE)
    agg_exprs[[mean_name]] <- call("mean", as.name(v), na.rm = TRUE)
  }

  agg_result <- merged[, eval(as.call(c(as.name("list"), agg_exprs))),
                        by = .row_idx]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (v in neighbor_source_vars) {
    max_col <- paste0(v, "_nb_max")
    min_col <- paste0(v, "_nb_min")
    set(agg_result, which(is.infinite(agg_result[[max_col]])), max_col, NA_real_)
    set(agg_result, which(is.infinite(agg_result[[min_col]])), min_col, NA_real_)
  }

  # -------------------------------------------------------------------
  # Step 5: Join aggregated results back to cell_data
  # -------------------------------------------------------------------
  new_cols <- setdiff(names(agg_result), ".row_idx")

  # Remove any pre-existing columns with these names
  for (col in new_cols) {
    if (col %in% names(cell_data)) {
      set(cell_data, j = col, value = NULL)
    }
  }

  cell_data <- merge(cell_data, agg_result, by = ".row_idx", all.x = TRUE)

  # Restore original order and clean up
  setorder(cell_data, .row_idx)
  cell_data[, .row_idx := NULL]

  if (was_df) setDF(cell_data)

  return(cell_data)
}


# =====================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =====================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory Considerations (16 GB laptop)

The largest intermediate object is `merged`, which has one row per (focal-cell-year, neighbor) pair. With ~1.37 million directed neighbor relationships Ã— 28 years â‰ˆ **38.5 million rows** and ~8 columns â‰ˆ **2.5 GB**. Combined with `cell_data` (~6.46M Ã— 110 cols â‰ˆ 5.7 GB), peak memory is ~10-12 GB, which fits in 16 GB.

If memory is tight, process years in chunks:

```r
add_neighbor_features_chunked <- function(cell_data, id_order,
                                          rook_neighbors,
                                          neighbor_source_vars,
                                          chunk_years = 4L) {
  setDT(cell_data)
  cell_data[, .row_idx := .I]

  all_years <- sort(unique(cell_data$year))
  year_chunks <- split(all_years, ceiling(seq_along(all_years) / chunk_years))

  # Build edge list once
  focal_refs    <- rep(seq_along(id_order), times = lengths(rook_neighbors))
  neighbor_refs <- unlist(rook_neighbors, use.names = FALSE)
  valid         <- neighbor_refs != 0L
  edge_dt <- data.table(
    focal_id    = id_order[focal_refs[valid]],
    neighbor_id = id_order[neighbor_refs[valid]]
  )

  new_col_names <- unlist(lapply(neighbor_source_vars, function(v)
    paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))))

  # Pre-allocate result columns
  for (col in new_col_names) {
    set(cell_data, j = col, value = NA_real_)
  }

  neighbor_cols <- c("id", "year", neighbor_source_vars)

  for (yrs in year_chunks) {
    cat("Processing years:", paste(yrs, collapse = ", "), "\n")

    chunk_mask <- cell_data$year %in% yrs
    chunk_data <- cell_data[chunk_mask, c("id", "year", ".row_idx",
                                           neighbor_source_vars), with = FALSE]

    nb_vals <- chunk_data[, c("id", "year", neighbor_source_vars), with = FALSE]
    setnames(nb_vals, "id", "neighbor_id")

    focal_key <- chunk_data[, .(focal_id = id, year, .row_idx)]

    expanded <- edge_dt[focal_key, on = "focal_id",
                        .(neighbor_id, year, .row_idx),
                        allow.cartesian = TRUE, nomatch = NULL]

    merged <- nb_vals[expanded, on = .(neighbor_id, year), nomatch = NULL]

    agg_exprs <- list()
    for (v in neighbor_source_vars) {
      agg_exprs[[paste0(v, "_nb_max")]]  <- call("max",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0(v, "_nb_min")]]  <- call("min",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0(v, "_nb_mean")]] <- call("mean", as.name(v), na.rm = TRUE)
    }

    agg_result <- merged[, eval(as.call(c(as.name("list"), agg_exprs))),
                          by = .row_idx]

    for (v in neighbor_source_vars) {
      mc <- paste0(v, "_nb_max"); mn <- paste0(v, "_nb_min")
      set(agg_result, which(is.infinite(agg_result[[mc]])), mc, NA_real_)
      set(agg_result, which(is.infinite(agg_result[[mn]])), mn, NA_real_)
    }

    # Write results back by .row_idx
    idx <- match(agg_result$.row_idx, cell_data$.row_idx)
    for (col in new_col_names) {
      set(cell_data, i = idx, j = col, value = agg_result[[col]])
    }

    rm(chunk_data, nb_vals, focal_key, expanded, merged, agg_result)
    gc()
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

---

## 5. Verification Script

Run this to confirm numerical equivalence with the original implementation on a small subset:

```r
# Verify on a small sample
set.seed(42)
sample_rows <- sample(nrow(cell_data), 1000)

for (v in neighbor_source_vars) {
  for (stat in c("max", "min", "mean")) {
    col_name <- paste0(v, "_nb_", stat)
    orig_vals <- cell_data_original[[col_name]][sample_rows]
    new_vals  <- cell_data[[col_name]][sample_rows]
    match_pct <- mean(
      (is.na(orig_vals) & is.na(new_vals)) |
      (!is.na(orig_vals) & !is.na(new_vals) & abs(orig_vals - new_vals) < 1e-10),
      na.rm = TRUE
    )
    cat(sprintf("%s: %.6f%% match\n", col_name, match_pct * 100))
  }
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Runtime** | ~86+ hours | ~2-5 minutes |
| **Approach** | Per-row `lapply` with string keys | Vectorized `data.table` edge-list join + grouped aggregation |
| **Numerical result** | max/min/mean of rook-neighbor values | Identical |
| **RF model** | Unchanged | Unchanged |
| **Memory** | ~6-8 GB (list of 6.46M vectors) | ~10-12 GB peak (or ~4-6 GB chunked) |
| **Key insight** | â€” | The neighbor lookup is a sparse relational join, not a row-level computation; `data.table` binary-search joins + `by=` grouping eliminate all R-level loops |
END_RESPONSE>>>
