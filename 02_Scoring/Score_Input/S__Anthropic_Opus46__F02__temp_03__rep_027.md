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
run_id: Anthropic_F02_T03_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction**: `paste(id, year, sep="_")` is called millions of times inside the loop body, and named-vector lookup (`idx_lookup[neighbor_keys]`) is an O(k) hash probe per neighbor key, repeated for every row.
- **Memory**: The named character vector `idx_lookup` with 6.46M entries is fine, but the output `neighbor_lookup` is a **list of 6.46 million integer vectors**. Each list element carries R object overhead (~56 bytes minimum), so the list alone consumes â‰¥ 360 MB of overhead before any actual neighbor indices are stored. With actual indices, this easily exceeds 1â€“2 GB.
- **Time**: The `lapply` is single-threaded, and the per-element work (character coercion, paste, hash lookup, NA filtering) is expensive in interpreted R. For 6.46M rows this alone can take many hours.

### 2. `compute_neighbor_stats` â€” repeated random-access gather over a 6.46M-length vector
- Called 5 times (once per variable), each time iterating over the 6.46M-element `neighbor_lookup` list, gathering neighbor values, and computing `max/min/mean`. This is again single-threaded interpreted R with heavy list overhead.
- The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (though not the dominant cost).

### Combined estimate
With ~6.46M outer iterations Ã— 5 variables, and each iteration doing string operations + subsetting, the 86+ hour estimate is consistent with pure-R interpreted overhead on a laptop.

---

## Optimization Strategy

The key insight is: **eliminate the 6.46M-element list entirely**. Replace it with a vectorized, `data.table`-based equi-join approach that:

1. **Expands the neighbor graph into an edge table** (cell_id â†’ neighbor_id), ~1.37M edges.
2. **Joins the edge table to the panel data twice** â€” once to attach the year of each focal row, and once to look up the neighbor's value in that year â€” using `data.table` keyed joins (binary search, C-level).
3. **Aggregates (max, min, mean) by focal row** using `data.table`'s grouped `j` expressions, which are executed in C.

This turns the entire pipeline into a sequence of **vectorized joins and grouped aggregations** with no per-row R interpretation, no giant list, and minimal string operations.

**Memory**: The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The largest intermediate table (edges Ã— years = 1.37M Ã— 28 â‰ˆ 38.4M rows Ã— a few columns) is ~1â€“2 GB, which fits in 16 GB RAM, especially since we process one variable at a time and discard intermediates.

**Time**: `data.table` keyed joins on integer columns over tens of millions of rows typically complete in seconds. Five variables Ã— (one join + one aggregation) should finish in **minutes, not hours**.

**Model preservation**: We only change how the 15 neighbor-derived columns (5 vars Ã— 3 stats) are computed. The numerical values are identical (same max, min, mean of the same neighbor sets), so the trained Random Forest model remains valid with no retraining.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature pipeline
#' 
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data as a data.table with new columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -----------------------------------------------------------
  # Step 1: Build a compact edge table from the nb object

  #         focal_id  ->  neighbor_id   (integer cell IDs)
  # -----------------------------------------------------------
  # rook_neighbors_unique[[i]] contains integer indices into id_order
  # for the neighbors of id_order[i].

  n_cells <- length(id_order)
  focal_idx    <- rep.int(seq_len(n_cells),
                          lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-entries that spdep uses for cells with no neighbors
  valid <- neighbor_idx != 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  rm(focal_idx, neighbor_idx, valid)

  # -----------------------------------------------------------
  # Step 2: Convert cell_data to data.table (no copy if already)
  # -----------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Ensure a row-order column so we can put results back in place
  cell_data[, .row_order := .I]

  # Key for the neighbor value lookup: (id, year)
  setkey(cell_data, id, year)

  # -----------------------------------------------------------
  # Step 3: For each variable, join + aggregate
  # -----------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    message("Processing neighbor stats for: ", var_name)

    # 3a. Build a slim lookup: neighbor_id, year -> value
    #     (only the columns we need, to save memory)
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # 3b. For every focal row, get its year and attach to edges
    #     focal_rows: focal_id, year, .row_order
    focal_rows <- cell_data[, .(focal_id = id, year, .row_order)]

    # 3c. Cross with edges to get (focal_id, year, neighbor_id, .row_order)
    #     Join focal_rows to edges on focal_id
    setkey(edges, focal_id)
    setkey(focal_rows, focal_id)
    expanded <- edges[focal_rows,
                      .(focal_id,
                        neighbor_id,
                        year = i.year,
                        .row_order = i..row_order),
                      on = "focal_id",
                      allow.cartesian = TRUE,
                      nomatch = NULL]

    # 3d. Look up the neighbor's value in the same year
    expanded[val_dt,
             neighbor_val := i.val,
             on = c(neighbor_id = "id", "year")]

    # 3e. Aggregate by focal row
    stats <- expanded[!is.na(neighbor_val),
                      .(nb_max  = max(neighbor_val),
                        nb_min  = min(neighbor_val),
                        nb_mean = mean(neighbor_val)),
                      by = .row_order]

    # 3f. Assign back to cell_data by .row_order
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    cell_data[stats, on = ".row_order",
              c(max_col, min_col, mean_col) :=
                .(i.nb_max, i.nb_min, i.nb_mean)]

    # Free intermediates
    rm(val_dt, focal_rows, expanded, stats)
    gc()
  }

  # Clean up helper column
  cell_data[, .row_order := NULL]

  # Restore original row order (setkey may have reordered)
  setkey(cell_data, NULL)

  return(cell_data)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
library(data.table)

# ---- Load your existing objects ----
# cell_data                 : your panel data.frame (~6.46M rows)
# id_order                  : integer vector of cell IDs
# rook_neighbors_unique     : spdep nb object
# rf_model                  : your already-trained Random Forest (untouched)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ...
# These are numerically identical to the original implementation.
# Proceed with prediction using the existing rf_model as before.
```

### Optional: Further reduce peak memory with chunked processing

If the `expanded` table (~38M rows) still causes memory pressure, process years in chunks:

```r
# Inside the for-loop over var_name, replace steps 3bâ€“3e with:
years <- sort(unique(cell_data$year))
stats_list <- vector("list", length(years))

for (j in seq_along(years)) {
  yr <- years[j]
  focal_yr <- cell_data[year == yr, .(focal_id = id, year, .row_order)]
  setkey(focal_yr, focal_id)

  exp_yr <- edges[focal_yr,
                  .(focal_id, neighbor_id,
                    year = i.year, .row_order = i..row_order),
                  on = "focal_id",
                  allow.cartesian = TRUE,
                  nomatch = NULL]

  exp_yr[val_dt, neighbor_val := i.val,
         on = c(neighbor_id = "id", "year")]

  stats_list[[j]] <- exp_yr[!is.na(neighbor_val),
                             .(nb_max  = max(neighbor_val),
                               nb_min  = min(neighbor_val),
                               nb_mean = mean(neighbor_val)),
                             by = .row_order]
  rm(focal_yr, exp_yr)
}
stats <- rbindlist(stats_list)
rm(stats_list)
```

This reduces peak memory from ~2 GB to ~70â€“80 MB per chunk (one year â‰ˆ 344K cells Ã— ~4 neighbors).

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Core approach** | Per-row `lapply` with string keys | Vectorized `data.table` join + grouped aggregation |
| **`build_neighbor_lookup`** | 6.46M-element R list (~1â€“2 GB) | 1.37M-row edge table (~11 MB) |
| **`compute_neighbor_stats`** | Interpreted R loop, 6.46M iterations Ã— 5 vars | C-level keyed join + `by` aggregation |
| **Estimated time** | 86+ hours | ~5â€“20 minutes (depending on disk I/O) |
| **Peak RAM** | >8 GB (list overhead + data) | ~3â€“5 GB (chunked: ~2 GB) |
| **Numerical output** | max, min, mean of rook neighbors | Identical max, min, mean of same neighbors |
| **RF model** | â€” | Untouched; no retraining needed |
END_RESPONSE>>>
