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
run_id: Anthropic_F04_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is squarely in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup` is O(NÂ²)-like in practice.** It calls `lapply` over all ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Constructs character keys** by pasting neighbor IDs with the current row's year (`paste(..., sep="_")`).
4. **Performs named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” which in R is an O(n) hash-table probe *per key*, but the named vector itself has 6.46 million entries, and this is done inside an `lapply` over 6.46 million iterations.

The total number of key lookups is ~6.46M rows Ã— ~4 neighbors/cell (rook) = ~25.8 million `paste` + named-vector lookups, all inside an interpreted R `lapply` loop. The `paste` allocations and repeated character matching against a 6.46M-entry named vector are extremely expensive. The result: an estimated 86+ hours.

**`compute_neighbor_stats`** is comparatively cheap (numeric subsetting + simple aggregation), but it is also called 5 times with an R-level `lapply` over 6.46M rows, which adds up.

---

## Optimization Strategy

### Principle: Replace row-level R loops and character-key lookups with vectorized joins via `data.table`.

1. **Vectorized neighbor-lookup construction**: Instead of looping per row, expand the `nb` object into an edge-list (cell_id â†’ neighbor_id) once, then do a single `data.table` merge-join keyed on `(neighbor_id, year)` to resolve all neighbor row indices in bulk. This eliminates all `paste` and named-vector lookups.

2. **Vectorized neighbor statistics**: Group the joined table by `(focal_row)` and compute `max`, `min`, `mean` in one `data.table` aggregation â€” no `lapply` needed.

3. **Repeat for each variable**: Each of the 5 variables is a single grouped aggregation on the same join result, so the join is done once and reused.

**Expected speedup**: The join is O(N log N) or O(N) with keys, and grouping is highly optimized in `data.table`. This should reduce 86+ hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop).

---

## Working R Code

```r
library(data.table)

#' Build a full edge list from an spdep nb object.
#' Returns a data.table with columns: cell_id, neighbor_cell_id
#' @param id_order integer vector of cell IDs in the order matching the nb object
#' @param neighbors an nb object (list of integer index vectors)
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  focal_idx <- integer(n_edges)
  neighbor_idx <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    len <- length(nb_i)
    if (len > 0L) {
      focal_idx[pos:(pos + len - 1L)] <- i
      neighbor_idx[pos:(pos + len - 1L)] <- nb_i
      pos <- pos + len
    }
  }
  
  data.table(
    cell_id          = id_order[focal_idx],
    neighbor_cell_id = id_order[neighbor_idx]
  )
}

#' Compute neighbor features for all source variables at once, fully vectorized.
#' Preserves the trained RF model â€” only adds columns to cell_data.
#'
#' @param cell_data data.frame/data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order integer vector of cell IDs matching the nb object
#' @param neighbors spdep nb object (rook_neighbors_unique)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data (data.table) with new neighbor feature columns appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {
  
  # Convert to data.table if needed (modifies in place for efficiency)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Assign a row index to every row in cell_data
  cell_data[, .row_idx := .I]
  
  # --- Step 1: Build edge list (once) ---
  edges <- build_edge_list(id_order, neighbors)
  
  # --- Step 2: Create a keyed lookup: (id, year) -> row_idx ---
  # We also need the source variable values for the neighbor rows.
  # Strategy: join edges with cell_data twice â€”
  #   (a) to get the focal row index (for grouping), and
  #   (b) to get the neighbor row's variable values.
  
  # Keyed lookup for focal rows: (cell_id, year) -> .row_idx
  # We expand edges by year via a join on cell_data's (id, year).
  
  # Focal side: for every (cell_id, year) in cell_data, get .row_idx
  focal <- cell_data[, .(focal_id = id, year, focal_row_idx = .row_idx)]
  setkey(focal, focal_id, year)
  
  # Expand edges Ã— years: join edges on focal_id = cell_id
  # This gives us (focal_row_idx, neighbor_cell_id, year) for every edge Ã— year
  setkey(edges, cell_id)
  
  # Merge: for each edge (cell_id -> neighbor_cell_id), attach all years
  # of the focal cell, yielding (focal_row_idx, neighbor_cell_id, year)
  expanded <- edges[focal, on = .(cell_id = focal_id), allow.cartesian = TRUE,
                    nomatch = NULL,
                    .(focal_row_idx = i.focal_row_idx,
                      neighbor_cell_id = x.neighbor_cell_id,
                      year = i.year)]
  
  # --- Step 3: Resolve neighbor rows ---
  # Build neighbor-side lookup: (id, year) -> row_idx + variable values
  keep_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  neighbor_lkp <- cell_data[, ..keep_cols]
  setnames(neighbor_lkp, "id", "neighbor_id")
  setnames(neighbor_lkp, ".row_idx", "neighbor_row_idx")
  setkey(neighbor_lkp, neighbor_id, year)
  
  # Join to resolve neighbor variable values
  joined <- neighbor_lkp[expanded,
                          on = .(neighbor_id = neighbor_cell_id, year),
                          nomatch = NA]
  # joined now has columns:
  #   neighbor_id, year, neighbor_row_idx, <source_vars>, focal_row_idx
  
  # --- Step 4: Grouped aggregation per (focal_row_idx) per variable ---
  for (var in neighbor_source_vars) {
    cat("Computing neighbor features for:", var, "\n")
    
    # Subset to non-NA values of this variable
    sub <- joined[!is.na(get(var)), .(focal_row_idx, val = get(var))]
    
    if (nrow(sub) == 0L) {
      cell_data[, paste0(var, "_neighbor_max")  := NA_real_]
      cell_data[, paste0(var, "_neighbor_min")  := NA_real_]
      cell_data[, paste0(var, "_neighbor_mean") := NA_real_]
      next
    }
    
    agg <- sub[, .(nb_max  = max(val),
                    nb_min  = min(val),
                    nb_mean = mean(val)),
               by = focal_row_idx]
    
    # Initialize new columns with NA
    max_col  <- paste0(var, "_neighbor_max")
    min_col  <- paste0(var, "_neighbor_min")
    mean_col <- paste0(var, "_neighbor_mean")
    
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]
    
    # Assign aggregated values back by row index
    cell_data[agg$focal_row_idx, (max_col)  := agg$nb_max]
    cell_data[agg$focal_row_idx, (min_col)  := agg$nb_min]
    cell_data[agg$focal_row_idx, (mean_col) := agg$nb_mean]
  }
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  return(cell_data)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has 15 new columns (3 per variable: _neighbor_max, _min, _mean)
# matching the original estimand exactly.
# The trained Random Forest model is untouched â€” just call predict() as before.
```

---

## Why This Preserves the Original Numerical Estimand

| Original operation | Replacement | Equivalence |
|---|---|---|
| `vals[idx]` where `idx` = neighbor row indices for a focal row | `data.table` join on `(neighbor_id, year)` resolving to the same rows | Identical row resolution; same values |
| `max`, `min`, `mean` over non-NA neighbor values | `data.table` grouped `max`, `min`, `mean` after filtering `!is.na` | Identical aggregation functions |
| Rows with no valid neighbors â†’ `NA, NA, NA` | Rows absent from `agg` remain `NA_real_` | Identical NA handling |

The column names follow the pattern `{var}_neighbor_max/min/mean`. Adjust the naming convention if the original `compute_and_add_neighbor_features` used different suffixes â€” the RF model's `predict()` call will need matching column names.

---

## Expected Performance

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| Neighbor lookup construction | ~60â€“80 hrs | ~30 sec (edge-list build + one keyed join) |
| Neighbor stats (5 vars) | ~6â€“10 hrs | ~2â€“5 min (5 grouped aggregations) |
| **Total** | **86+ hrs** | **~3â€“8 min** |

Peak RAM will spike during the `expanded` join (~25.8M edges Ã— 28 years â‰ˆ ~720M rows of 3 integer columns â‰ˆ ~8.6 GB), which fits within 16 GB. If RAM is tight, process years in batches:

```r
# Optional: batch by year to reduce peak RAM
years <- sort(unique(cell_data$year))
for (yr in years) {
  # filter expanded and joined to this year only
  # ... aggregate and assign back ...
}
```

This trades a small amount of speed for significantly lower peak memory.
END_RESPONSE>>>
