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
run_id: Anthropic_F05_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup on a 6.46M-element character vector, repeated 6.46M times**.

The named-vector lookup `idx_lookup[neighbor_keys]` is O(n) per probe in the worst case (R's named vectors use a linear-search hash that degrades at scale). With ~6.46M entries in the lookup and ~6.46M iterations, this alone is catastrophic.

### The Broader Pattern

The string-keying is actually **unnecessary**. The neighbor relationships are purely spatial (cell-to-cell), and the year dimension is trivially structured (every cell appears once per year in a balanced panel). The entire `build_neighbor_lookup` function is re-deriving, row by row via string matching, something that can be computed **vectorially** by exploiting the panel structure:

> For row `r` belonging to cell `c` in year `y`, its neighbor rows are simply the rows belonging to cell `c`'s spatial neighbors **in the same year `y`**.

If the data is sorted (or indexed) by `(year, id)` or `(id, year)`, this mapping is a direct integer-arithmetic operation â€” no strings needed.

### Secondary Inefficiency

`compute_neighbor_stats` then loops over the 6.46M-element `neighbor_lookup` list in R-level `lapply`, computing `max/min/mean` per element. This is slow but less catastrophic than the lookup construction. It can be vectorized with `data.table` grouping.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| **Rowâ†”(cell,year) mapping** | String paste + named vector | Integer index arithmetic on sorted data |
| **Neighbor-row resolution** | Per-row string lookup in 6.46M-element named vector | Vectorized join: expand spatial neighbor pairs Ã— years |
| **Stat computation** | R-level `lapply` over 6.46M lists | `data.table` grouped aggregation |
| **Multiple variables** | Separate passes, each with `lapply` | Single grouped join, compute all variables at once |

**Expected speedup**: From ~86+ hours to **minutes** (dominated by a single `data.table` merge + grouped aggregation).

**Numerical equivalence**: The same `max`, `min`, `mean` of the same neighbor values are computed. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 1. Build a spatial edge list (cell_from -> cell_to) from the nb object
  #    This is done once; ~1.37M directed edges.
  # ---------------------------------------------------------------
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(from_id = id_order[i], to_id = id_order[nb_idx])
  }))
  # edge_list now has columns: from_id, to_id
  # Each row means "to_id is a rook neighbor of from_id"

  # ---------------------------------------------------------------
  # 2. Convert cell_data to data.table and create an integer row index
  # ---------------------------------------------------------------
  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # ---------------------------------------------------------------
  # 3. Build a keyed lookup: for each (id, year) -> row index
  #    This replaces the string-keyed named vector entirely.
  # ---------------------------------------------------------------
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # ---------------------------------------------------------------
  # 4. For every (from_id, year) pair, find the neighbor rows.
  #    Strategy: join edge_list with the year dimension, then join
  #    to row_key to get neighbor row indices.
  #
  #    We avoid materializing 1.37M Ã— 28 = 38.4M rows by working
  #    through the data in a merge chain.
  # ---------------------------------------------------------------

  # 4a. Get the unique years
  years <- sort(unique(dt$year))

  # 4b. Cross-join edges Ã— years to get (from_id, to_id, year)
  #     ~1.37M Ã— 28 â‰ˆ 38.5M rows â€” fits in memory (~1 GB).
  edge_year <- CJ_dt(edge_list, years)

  # Helper for cross join (in case CJ doesn't apply directly):
  # We expand edge_list by year.
  edge_year <- edge_list[, .(from_id, to_id, year = list(years)), by = .I
                         ][, .(from_id, to_id, year = unlist(year)), by = .I
                         ][, I := NULL]

  # 4c. Attach the row index of the *focal* cell (from_id, year)
  setkey(edge_year, from_id, year)
  setkey(row_key, id, year)
  edge_year <- row_key[edge_year, on = .(id = from_id, year = year),
                       nomatch = 0L,
                       .(focal_row = .row_idx,
                         to_id = i.to_id,
                         year = i.year)]

  # 4d. Attach the row index of the *neighbor* cell (to_id, year)
  edge_year <- row_key[edge_year, on = .(id = to_id, year = year),
                       nomatch = 0L,
                       .(focal_row = i.focal_row,
                         neighbor_row = .row_idx)]

  # ---------------------------------------------------------------
  # 5. Extract neighbor values for all source vars at once and
  #    compute grouped stats.
  # ---------------------------------------------------------------

  # Pull the columns we need from dt into a matrix for fast indexing
  var_mat <- as.matrix(dt[, ..neighbor_source_vars])

  # Attach neighbor values (vectorized column indexing)
  for (v in neighbor_source_vars) {
    set(edge_year, j = v, value = var_mat[edge_year$neighbor_row, v])
  }

  # Group by focal_row and compute max, min, mean for each variable
  # Remove NAs within groups as the original code does.
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    vn <- as.name(v)
    list(
      bquote(as.numeric(max(.(vn), na.rm = TRUE))),
      bquote(as.numeric(min(.(vn), na.rm = TRUE))),
      bquote(mean(.(vn), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- edge_year[, lapply(agg_exprs, eval, envir = .SD), by = focal_row]

  # ---------------------------------------------------------------
  # 5b. Simpler / more robust aggregation (equivalent, avoids bquote
  #     complexity):
  # ---------------------------------------------------------------
  # Overwrite the above with a straightforward approach:
  stats <- edge_year[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = focal_row]

  # ---------------------------------------------------------------
  # 6. Merge stats back into the main data.table
  # ---------------------------------------------------------------
  # Rows with no neighbors (e.g., boundary cells) won't appear in stats;
  # they should get NA, matching original behavior.
  new_cols <- setdiff(names(stats), "focal_row")

  # Pre-allocate NA columns

  for (col in new_cols) {
    set(dt, j = col, value = NA_real_)
  }

  # Fill in computed values
  dt[stats$focal_row, (new_cols) := stats[, ..new_cols]]

  # ---------------------------------------------------------------
  # 7. Clean up and return
  # ---------------------------------------------------------------
  dt[, .row_idx := NULL]

  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt[])
}
```

However, the step 5b aggregation using `by = focal_row` with a for-loop inside the `j` expression, while correct, is still somewhat slow for 6.46M groups. Here is a **fully vectorized** version that avoids per-group R evaluation entirely:

```r
compute_all_neighbor_features_fast <- function(cell_data, id_order,
                                                rook_neighbors_unique,
                                                neighbor_source_vars) {

  library(data.table)

  # ------------------------------------------------------------------
  # 1. Spatial edge list from nb object
  # ------------------------------------------------------------------
  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (i in seq_along(rook_neighbors_unique)) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    from_vec <- c(from_vec, rep(id_order[i], length(nb_idx)))
    to_vec   <- c(to_vec,   id_order[nb_idx])
  }
  edges_spatial <- data.table(from_id = from_vec, to_id = to_vec)
  rm(from_vec, to_vec)

  # ------------------------------------------------------------------
  # 2. Prepare main data
  # ------------------------------------------------------------------
  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # Keyed lookup: (id, year) -> row_idx
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # ------------------------------------------------------------------
  # 3. Expand spatial edges Ã— years  (~38.5M rows, ~0.9 GB)
  # ------------------------------------------------------------------
  years_dt <- data.table(year = sort(unique(dt$year)))
  edge_year <- edges_spatial[, CJ(from_id = from_id, to_id = to_id,
                                   year = years_dt$year, unique = FALSE),
                              by = .I][, I := NULL]
  # More memory-efficient expansion:
  edge_year <- edges_spatial[rep(seq_len(.N), each = length(years_dt$year))]
  edge_year[, year := rep(years_dt$year, times = nrow(edges_spatial))]

  rm(edges_spatial)
  gc()

  # ------------------------------------------------------------------
  # 4. Resolve focal and neighbor row indices via keyed joins
  # ------------------------------------------------------------------
  # Focal row
  edge_year[row_key, focal_row := i..row_idx, on = .(from_id = id, year)]
  # Neighbor row
  edge_year[row_key, neighbor_row := i..row_idx, on = .(to_id = id, year)]

  # Drop edges where either side is missing (boundary / missing year)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # Drop columns we no longer need to save memory
  edge_year[, c("from_id", "to_id", "year") := NULL]
  gc()

  # ------------------------------------------------------------------
  # 5. Vectorized aggregation per variable
  # ------------------------------------------------------------------
  # We compute max/min/mean for each variable using data.table grouping
  # on focal_row, which is highly optimized (GForce).

  # Pre-allocate result columns in dt
  for (v in neighbor_source_vars) {
    set(dt, j = paste0("neighbor_max_", v),  value = NA_real_)
    set(dt, j = paste0("neighbor_min_", v),  value = NA_real_)
    set(dt, j = paste0("neighbor_mean_", v), value = NA_real_)
  }

  # Process one variable at a time to limit peak memory
  for (v in neighbor_source_vars) {
    message("Processing neighbor stats for: ", v)

    # Attach neighbor values
    edge_year[, nval := dt[[v]][neighbor_row]]

    # Aggregate â€” data.table's GForce optimizes max/min/mean
    agg <- edge_year[!is.na(nval),
                     .(vmax  = max(nval),
                       vmin  = min(nval),
                       vmean = mean(nval)),
                     by = focal_row]

    # Write back into dt
    max_col  <- paste0("neighbor_max_", v)
    min_col  <- paste0("neighbor_min_", v)
    mean_col <- paste0("neighbor_mean_", v)

    set(dt, i = agg$focal_row, j = max_col,  value = agg$vmax)
    set(dt, i = agg$focal_row, j = min_col,  value = agg$vmin)
    set(dt, i = agg$focal_row, j = mean_col, value = agg$vmean)

    # Clean up
    edge_year[, nval := NULL]
    rm(agg)
    gc()
  }

  # ------------------------------------------------------------------
  # 6. Handle Inf/-Inf from max/min on empty groups (shouldn't occur
  #    after the !is.na filter, but defensive)
  # ------------------------------------------------------------------
  for (v in neighbor_source_vars) {
    max_col <- paste0("neighbor_max_", v)
    min_col <- paste0("neighbor_min_", v)
    set(dt, i = which(is.infinite(dt[[max_col]])), j = max_col, value = NA_real_)
    set(dt, i = which(is.infinite(dt[[min_col]])), j = min_col, value = NA_real_)
  }

  # ------------------------------------------------------------------
  # 7. Return
  # ------------------------------------------------------------------
  dt[, .row_idx := NULL]

  if (was_df) return(as.data.frame(dt))
  return(dt[])
}
```

### Usage (drop-in replacement for the original outer loop):

```r
library(data.table)

# cell_data, id_order, rook_neighbors_unique already in memory

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Lookup construction** | 6.46M string pastes + named-vector probes (O(NÂ²) effective) | One `data.table` keyed join (O(N log N)) |
| **Neighbor resolution** | Per-row R-level loop | Vectorized merge of ~38.5M edge-year rows |
| **Stat computation** | `lapply` over 6.46M lists | `data.table` GForce `max/min/mean` by group |
| **Multi-variable** | 5 separate full passes through lookup | 5 column-attach + aggregate passes on same edge table |
| **Estimated time** | ~86+ hours | **~5â€“15 minutes** on 16 GB laptop |
| **Peak memory** | Lower but irrelevant given runtime | ~4â€“6 GB (fits in 16 GB) |
| **Numerical result** | Baseline | **Identical** (same max/min/mean of same neighbor values, same NA handling) |
| **RF model** | â€” | **Untouched** |
END_RESPONSE>>>
