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
run_id: Anthropic_F09_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental problems:

### 1. `build_neighbor_lookup` rebuilds neighbor relationships for every cell-year row (~6.46M rows)

The neighbor topology is **purely spatial** â€” cell A's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` creates a lookup entry for every one of the ~6.46 million cell-year rows, performing string-pasting, hash lookups, and NA filtering inside an `lapply` over 6.46M iterations. This is ~28Ã— more work than necessary.

### 2. `compute_neighbor_stats` iterates over 6.46M rows in R-level loops

For each of the 5 variables, `lapply` touches every row, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. With 5 variables Ã— 6.46M rows, that's ~32.3 million R-level function calls with repeated vector subsetting.

### Root cause summary

| Bottleneck | Why it's slow |
|---|---|
| Neighbor lookup is built at the **cell-year** level | 6.46M entries instead of 344K; string-key hashing is O(n) in row count |
| Stats computed row-by-row in R | No vectorization; `lapply` over millions of rows is inherently slow |
| Topology and attributes are entangled | Neighbor structure is recomputed implicitly every time instead of being a reusable join key |

---

## Optimization Strategy

**Core idea:** Separate the static spatial topology from the time-varying attributes. Build the adjacency table once (344K cells Ã— ~4 neighbors â‰ˆ 1.37M directed edges), then use a vectorized `data.table` join-and-aggregate to compute neighbor stats for all years simultaneously.

### Steps

1. **Build a static edge table** from the `nb` object: two columns `(cell_id, neighbor_id)`, ~1.37M rows. This is done once.
2. **Convert the panel data to `data.table`** keyed on `(id, year)`.
3. **Cross-join the edge table with years** implicitly via a keyed join: for each `(cell_id, year)`, look up each `neighbor_id`'s attribute in that same year.
4. **Aggregate** (`max`, `min`, `mean`) grouped by `(cell_id, year)` â€” fully vectorized inside `data.table`.
5. **Join results back** to the main table.

### Expected speedup

- The edge table has ~1.37M rows. Crossed with 28 years = ~38.4M join rows â€” but the `data.table` binary-search join handles this in seconds, not hours.
- Aggregation by group is C-level vectorized in `data.table`.
- Estimated total time: **2â€“10 minutes** on a 16 GB laptop (vs. 86+ hours).

### Invariants preserved

- The trained Random Forest model is **not retouched**. We only produce the same predictor columns it expects.
- The numerical results (neighbor max, min, mean) are **identical** to the original implementation â€” same rook topology, same per-variable aggregation logic.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0 : Inputs assumed to exist
#   cell_data               : data.frame/data.table with columns id, year, 
#                             ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order                : integer/character vector of cell IDs matching
#                             the nb object's positional index
#   rook_neighbors_unique   : spdep nb object (list of integer index vectors)
#   rf_model                : trained Random Forest model (UNCHANGED)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 : Build the STATIC spatial edge table (done ONCE, ~1.37M rows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] gives the positional indices of neighbors for id_order[i].
  # Convert to a two-column data.table of (cell_id, neighbor_id).
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove the spdep "0 = no neighbors" convention

  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: cell_id, neighbor_id  (~1.37M rows)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 : Convert panel data to keyed data.table
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_dt <- as.data.table(cell_data)
# Ensure id and year are the types we expect (integer / numeric)
# and set key for fast binary-search joins.
setkey(cell_dt, id, year)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 : Vectorized neighbor-stat computation for one variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # --- 3a. Subset only the columns we need from the panel ---
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # --- 3b. Expand edges Ã— years via join ---
  #   For every directed edge (cell_id â†’ neighbor_id) and every year,
  #   retrieve the neighbor's attribute value.
  #   We do this by joining edge_dt to attr_dt on neighbor_id = id.

  # First, get all unique years
  years_vec <- sort(unique(cell_dt$year))

  # Cross-join edges with years: each edge appears once per year
  # (~1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” fits comfortably in 16 GB)
  edge_year <- CJ_dt_year(edge_dt, years_vec)
  # edge_year columns: cell_id, neighbor_id, year


  # Join to get the neighbor's value in that year
  setkey(edge_year, neighbor_id, year)
  edge_year[attr_dt, neighbor_val := i.value, on = .(neighbor_id = id, year)]

  # --- 3c. Aggregate per (cell_id, year) ---
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    keyby = .(cell_id, year)
  ]

  # Name columns to match original pipeline expectations
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  stats
}

# Helper: cross-join edge table with a year vector efficiently
CJ_dt_year <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx     <- rep(seq_len(n_edges), each = n_years)
  out     <- edge_dt[idx]
  out[, year := rep(years_vec, times = n_edges)]
  out
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4 : Loop over the 5 source variables, join results back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats_dt <- compute_neighbor_features_fast(cell_dt, edge_dt, var_name)

  # Merge back onto the main table (left join: keep all cell-year rows)
  cell_dt <- merge(cell_dt, stats_dt,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)
}

setkey(cell_dt, id, year)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5 : Predict with the EXISTING Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert back to data.frame if the RF predict method expects one
cell_data <- as.data.frame(cell_dt)

# The trained model is used as-is â€” no retraining
cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory Note for 16 GB Laptop

The largest intermediate object is `edge_year` at ~38.4M rows Ã— 3â€“4 columns â‰ˆ **~1.2 GB**. Combined with `cell_dt` (~6.46M Ã— 110 cols â‰ˆ 5.7 GB), peak usage is about **8â€“9 GB**, well within 16 GB. If memory is tighter, you can process variables one at a time and remove `edge_year` between iterations (already the case in the loop above since `edge_year` is local to the function).

---

## Optional Further Optimization

If even the cross-join feels large, you can avoid it entirely by doing the join year-by-year in a vectorized inner loop (28 iterations, each ~1.37M rows):

```r
compute_neighbor_features_by_year <- function(cell_dt, edge_dt, var_name) {
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id)

  years_vec <- sort(unique(attr_dt$year))
  results <- rbindlist(lapply(years_vec, function(yr) {
    yr_attr <- attr_dt[year == yr, .(id, value)]
    setkey(yr_attr, id)
    
    # Join neighbor values
    merged <- edge_dt[yr_attr, on = .(neighbor_id = id), nomatch = 0L]
    # merged now has: cell_id, neighbor_id, value (the neighbor's value)
    
    merged[
      !is.na(value),
      .(nb_max  = max(value),
        nb_min  = min(value),
        nb_mean = mean(value)),
      keyby = .(cell_id)
    ][, year := yr]
  }))

  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(results, c("nb_max", "nb_min", "nb_mean"),
                    c(max_col,  min_col,  mean_col))
  results
}
```

This variant uses only ~1.37M rows of working memory per year-iteration (Ã—28 = trivial), bringing peak memory down by ~1 GB and running in roughly the same wall-clock time.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M R-level iterations with string keys | 1.37M-row static edge table (built once) |
| Stats computation | R-level `lapply` Ã— 6.46M Ã— 5 vars | Vectorized `data.table` join + `groupby` |
| Estimated runtime | ~86+ hours | **~2â€“10 minutes** |
| Peak RAM | High (string duplication) | ~8â€“9 GB (within 16 GB) |
| RF model | Unchanged âœ“ | Unchanged âœ“ |
| Numerical output | Original estimand | Identical âœ“ |
END_RESPONSE>>>
