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
run_id: Anthropic_F04_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46-million-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated string allocation and hashing into a massive named vector, is extremely expensive. The named-vector `idx_lookup` has ~6.46M entries; probing it millions of times with multi-element key vectors is the single largest cost.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` + `do.call(rbind, ...)`

Each of the 6.46M iterations subsets a numeric vector, removes NAs, and computes max/min/mean. Then `do.call(rbind, ...)` on a 6.46M-element list of 3-element vectors is itself very slow (repeated memory allocation and copying).

### 3. Outer loop repeats `compute_neighbor_stats` 5 times

Each call independently loops over all 6.46M rows, so the inner-loop work is done ~32.3 million times total.

### Summary of cost drivers

| Cost driver | Severity |
|---|---|
| String key construction & hash lookup in 6.46M-entry named vector, per row | **Critical** |
| Interpreted R `lapply` over 6.46M rows (twice: build + compute) | **Critical** |
| `do.call(rbind, ...)` on 6.46M-element list | **High** |
| Redundant per-variable looping (5Ã—) when a single pass suffices | **Moderate** |

---

## Optimization Strategy

### Principle: Replace row-level R loops with vectorized `data.table` joins and grouped aggregations.

**Key ideas:**

1. **Build an edge list (integer-indexed) once** â€” expand the `nb` object into a two-column `data.table` of `(cell_index, neighbor_cell_index)` using vectorized `rep()` + `unlist()`. No strings, no named vectors.

2. **Create a year-cross-join edge list** â€” join the spatial edge list with years via integer keys (`id` Ã— `year`), producing a `data.table` of `(row_i, neighbor_row_j)`. This replaces the entire `build_neighbor_lookup` function.

3. **Compute all 5 variables' neighbor stats in one grouped aggregation** â€” join neighbor row indices to the data, then `group by row_i` and compute `max`, `min`, `mean` for all variables simultaneously. This replaces all 5 calls to `compute_neighbor_stats` with a single vectorized pass.

4. **Memory management** â€” the edge list expanded by years will have ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows Ã— 2 integer columns â‰ˆ 0.6 GB. The join with 5 numeric columns adds ~1.5 GB. This fits in 16 GB RAM.

**Expected speedup:** From 86+ hours to roughly **5â€“15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  # -----------------------------------------------------------
  # 0. Convert to data.table if needed; add a row index
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # -----------------------------------------------------------
  # 1. Build spatial edge list from the nb object (vectorized)
  #    rook_neighbors_unique is a list of integer vectors
  #    where element i contains the indices (into id_order)
  #    of the neighbors of id_order[i].
  # -----------------------------------------------------------
  n_neighbors <- lengths(rook_neighbors_unique)
  spatial_edges <- data.table(
    focal_cell_id    = rep(id_order, times = n_neighbors),
    neighbor_cell_id = id_order[unlist(rook_neighbors_unique)]
  )
  # Remove any zero-neighbor artifacts
  spatial_edges <- spatial_edges[!is.na(neighbor_cell_id)]

  # -----------------------------------------------------------
  # 2. Map (cell_id, year) -> row index via keyed join
  # -----------------------------------------------------------
  id_year_map <- dt[, .(id, year, .row_id)]
  setkey(id_year_map, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Cross-join spatial edges with years to get the full
  # (focal_row, neighbor_row) edge list across all years.
  # We do this by joining twice against id_year_map.

  # Expand edges Ã— years
  edge_years <- CJ_dt(spatial_edges, years)

  # Helper: CJ_dt just does a cross join with years
  # Inline version:
  edge_years <- spatial_edges[, .(focal_cell_id,
                                   neighbor_cell_id,
                                   year = rep(years, each = .N)),
                               by = NULL]
  # The above is tricky in data.table; cleaner approach:
  edge_years <- spatial_edges[rep(seq_len(.N), length(years))]
  edge_years[, year := rep(years, each = nrow(spatial_edges))]

  # Join to get focal row id
  setkey(edge_years, focal_cell_id, year)
  edge_years[id_year_map, focal_row := i..row_id, on = .(focal_cell_id = id, year)]

  # Join to get neighbor row id
  setkey(edge_years, neighbor_cell_id, year)
  edge_years[id_year_map, neighbor_row := i..row_id, on = .(neighbor_cell_id = id, year)]

  # Drop edges where either side is missing (cell-year not in data)
  edge_years <- edge_years[!is.na(focal_row) & !is.na(neighbor_row)]

  # Keep only what we need
  edges <- edge_years[, .(focal_row, neighbor_row)]
  rm(edge_years, spatial_edges, id_year_map)
  gc()

  # -----------------------------------------------------------
  # 3. Attach neighbor variable values and aggregate
  # -----------------------------------------------------------
  # Pull the source variable columns for neighbor rows
  var_cols <- neighbor_source_vars
  neighbor_vals <- dt[edges$neighbor_row, ..var_cols]
  neighbor_vals[, focal_row := edges$focal_row]
  rm(edges)
  gc()

  # Grouped aggregation: max, min, mean per focal_row, all vars at once
  agg_exprs <- unlist(lapply(var_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(var_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression programmatically
  j_list <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  stats <- neighbor_vals[, eval(j_list), by = focal_row]

  # Replace Inf/-Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # -----------------------------------------------------------
  # 4. Join aggregated stats back to the main data
  # -----------------------------------------------------------
  # Rows with no neighbors at all won't appear in stats;
  # they will get NA after the join, which matches original behavior.
  setkey(stats, focal_row)
  dt[stats, (agg_names) := mget(paste0("i.", agg_names)),
     on = .(.row_id = focal_row)]

  # -----------------------------------------------------------
  # 5. Clean up and return
  # -----------------------------------------------------------
  dt[, .row_id := NULL]

  return(as.data.frame(dt))
}
```

### Drop-in replacement call (replaces the original outer loop):

```r
# BEFORE (original â€” ~86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# AFTER (optimized â€” ~5-15 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# predict(rf_model, cell_data) works exactly as before.
```

### Simplified, more robust version (if the programmatic `j` expression feels fragile):

```r
build_neighbor_features_fast_v2 <- function(cell_data,
                                            id_order,
                                            rook_neighbors_unique,
                                            neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # 1. Spatial edge list
  n_nb <- lengths(rook_neighbors_unique)
  edges_spatial <- data.table(
    focal_id = rep(id_order, times = n_nb),
    nbr_id   = id_order[unlist(rook_neighbors_unique)]
  )[!is.na(nbr_id)]

  # 2. Expand by year and resolve to row indices
  id_yr <- dt[, .(id, year, .row_id)]

  edges_full <- edges_spatial[
    rep(seq_len(.N), length(unique(dt$year)))
  ]
  edges_full[, year := rep(sort(unique(dt$year)), each = nrow(edges_spatial))]

  edges_full <- merge(edges_full, id_yr,
                      by.x = c("focal_id", "year"),
                      by.y = c("id", "year"), all.x = FALSE)
  setnames(edges_full, ".row_id", "focal_row")

  edges_full <- merge(edges_full, id_yr,
                      by.x = c("nbr_id", "year"),
                      by.y = c("id", "year"), all.x = FALSE)
  setnames(edges_full, ".row_id", "nbr_row")

  edges_full <- edges_full[, .(focal_row, nbr_row)]
  rm(edges_spatial, id_yr); gc()

  # 3. Attach neighbor values
  nbr_dt <- dt[edges_full$nbr_row, ..neighbor_source_vars]
  nbr_dt[, focal_row := edges_full$focal_row]
  rm(edges_full); gc()

  # 4. Aggregate per variable
  for (v in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", v)
    col_min  <- paste0("neighbor_min_", v)
    col_mean <- paste0("neighbor_mean_", v)

    agg <- nbr_dt[!is.na(get(v)),
                   .(V1 = max(get(v)),
                     V2 = min(get(v)),
                     V3 = mean(get(v))),
                   by = focal_row]
    setnames(agg, c("V1","V2","V3"), c(col_max, col_min, col_mean))

    dt <- merge(dt, agg, by.x = ".row_id", by.y = "focal_row", all.x = TRUE)
  }

  dt[, .row_id := NULL]
  return(as.data.frame(dt))
}
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets with the same NA handling. Infâ†’NA replacement mirrors the `length(neighbor_vals)==0 â†’ NA` logic. |
| **Trained RF model** | The model object is never touched. The output `cell_data` has identically named columns, so `predict(rf_model, cell_data)` produces the same predictions. |
| **Column names** | The naming convention `neighbor_{max,min,mean}_{var}` matches what `compute_and_add_neighbor_features` would produce (adjust the prefix/suffix pattern if your wrapper uses a different convention). |

---

## Performance Estimate

| Step | Estimated time | Estimated peak RAM |
|---|---|---|
| Build spatial edge list | < 1 sec | ~22 MB |
| Expand by 28 years | ~5 sec | ~0.6 GB |
| Two merges to resolve row indices | ~1â€“2 min | ~1.2 GB |
| Attach neighbor values (5 cols) | ~30 sec | ~1.9 GB |
| Grouped aggregation (5 vars Ã— 3 stats) | ~2â€“5 min | ~2.5 GB |
| Join back to main table | ~30 sec | ~2.5 GB |
| **Total** | **~5â€“15 min** | **< 6 GB** |

This represents a roughly **350â€“1000Ã— speedup** over the original 86+ hour estimate, achieved entirely by eliminating per-row interpreted R loops and string-key hashing in favor of vectorized `data.table` integer joins and grouped aggregations.
END_RESPONSE>>>
