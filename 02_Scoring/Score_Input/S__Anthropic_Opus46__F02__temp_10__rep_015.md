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
run_id: Anthropic_F02_T10_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows
For every row, the function performs string pasting (`paste(id, year, sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookup in R is hash-based but with 6.46M keys the overhead per call is significant. Doing this 6.46M times inside `lapply` creates billions of character allocations and hash probes. This alone likely accounts for many hours.

### 2. `compute_neighbor_stats` â€” repeated per variable, also O(n)
Each call iterates over 6.46M entries, subsetting a numeric vector by index lists and computing `max/min/mean`. This is called 5 times (once per neighbor source variable). The per-element R-level `lapply` loop is the main cost; the subsetting and aggregation themselves are fast per call but the loop overhead for 6.46M iterations is large.

### Memory pressure
6.46M rows Ã— 110 columns is roughly 5â€“6 GB as a `data.frame` of doubles. The neighbor lookup list (6.46M elements, each a small integer vector) adds another ~1â€“2 GB. Intermediate character vectors from `paste()` add transient pressure. On a 16 GB machine this is tight but feasible if managed carefully.

### Summary
| Component | Calls | Cost driver |
|---|---|---|
| `build_neighbor_lookup` | 1 | 6.46M string-key hash lookups |
| `compute_neighbor_stats` | 5 | 6.46M R-level iterations Ã— 5 |
| **Total estimated** | â€” | **86+ hours** |

---

## Optimization Strategy

### A. Replace string-key lookup with integer join via `data.table`

Instead of building a named character vector and probing it 6.46M Ã— avg_neighbors times, we:

1. Create an integer-keyed `data.table` mapping `(id, year) â†’ row_index`.
2. Expand the neighbor list into an edge table: `(row_i, neighbor_id)` with the year carried along.
3. Perform a single **keyed equi-join** (`data.table` binary search) to resolve all neighbor row indices at once â€” vectorized, no R-level loop.

This replaces the entire `build_neighbor_lookup` function with a few vectorized operations.

### B. Replace per-row `lapply` aggregation with grouped `data.table` aggregation

Once we have an edge table `(row_i, neighbor_row_j)`, computing neighbor stats is just:

```
edge_table[, .(max_v, min_v, mean_v), by = row_i]
```

This is a single vectorized grouped aggregation â€” orders of magnitude faster than 6.46M `lapply` iterations.

### C. Process variables in the same edge table

We join the variable column onto the edge table and aggregate. We reuse the same edge structure for all 5 variables, so the expensive join-build happens only once.

### D. Memory management

- The edge table will have ~6.46M Ã— avg_neighbors_per_row rows. With an average of ~4 rook neighbors, that's ~26M rows Ã— a few integer columns â‰ˆ < 1 GB.
- We avoid duplicating the full dataset; we only join the specific variable column needed.
- We use `:=` (in-place assignment) to add new columns to the existing `data.table`.

### Expected speedup
- `build_neighbor_lookup` equivalent: from hours â†’ **seconds** (one vectorized join).
- `compute_neighbor_stats` equivalent: from hours per variable â†’ **seconds** per variable.
- **Total: minutes instead of 86+ hours.**

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature pipeline
#' 
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector â€” the cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data as data.table with new neighbor feature columns appended
optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors,
                                       neighbor_source_vars) {
  
  # --- Step 0: Convert to data.table if needed (no copy if already data.table) ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Add an explicit row index so we can map back results
  cell_data[, .row_idx := .I]
  
  # --- Step 1: Build the edge list (focal_id, neighbor_id) from the nb object ---
  # id_order[k] is the cell id for the k-th entry in rook_neighbors
  # rook_neighbors[[k]] contains integer indices into id_order for neighbors of cell k
  
  message("Building edge list from nb object...")
  
  # Pre-allocate vectors for speed
  n_edges <- sum(lengths(rook_neighbors))  # total directed edges
  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)
  
  pos <- 1L
  for (k in seq_along(rook_neighbors)) {
    nb_k <- rook_neighbors[[k]]
    if (length(nb_k) == 0L || (length(nb_k) == 1L && nb_k[1] == 0L)) next
    n_k <- length(nb_k)
    focal_ids[pos:(pos + n_k - 1L)]    <- id_order[k]
    neighbor_ids[pos:(pos + n_k - 1L)] <- id_order[nb_k]
    pos <- pos + n_k
  }
  
  # Trim if any nb entries were empty (0-sentinel in spdep)
  if (pos <= n_edges) {
    focal_ids    <- focal_ids[1:(pos - 1L)]
    neighbor_ids <- neighbor_ids[1:(pos - 1L)]
  }
  
  # This is a spatial-only edge list (no year dimension yet)
  edges_spatial <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  rm(focal_ids, neighbor_ids)
  gc()
  
  message(sprintf("  %s spatial edges.", format(nrow(edges_spatial), big.mark = ",")))
  
  # --- Step 2: Build a row-index lookup keyed on (id, year) ---
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # --- Step 3: Expand edges across years and resolve row indices ---
  # Instead of a massive cross-join (edges Ã— years), we join edges onto the data
  # grouped by focal_id â†’ that gives us (focal_id, year, neighbor_id) rows,
  # then we look up the neighbor's row index for that same year.
  
  message("Resolving neighbor row indices via keyed join...")
  
  # Get unique (id, year, row_idx) for focal cells â€” this is just row_lookup itself
  # Join: for each row in cell_data, attach its neighbors from edges_spatial
  # cell_data row â†’ focal_id=id â†’ edges_spatial gives neighbor_ids
  
  # Keyed join: cell_data rows to their spatial neighbors
  setkey(edges_spatial, focal_id)
  
  # For each row in cell_data, look up spatial neighbors
  # We do this by joining cell_data's (id) to edges_spatial's (focal_id)
  focal_dt <- cell_data[, .(focal_row = .row_idx, focal_id = id, year)]
  setkey(focal_dt, focal_id)
  
  # This is the big expansion: each cell-year row Ã— its ~4 neighbors
  expanded <- edges_spatial[focal_dt, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
  # Result columns: focal_id, neighbor_id, focal_row, year
  
  rm(focal_dt, edges_spatial)
  gc()
  
  message(sprintf("  %s expanded (cell-year, neighbor) edges.", format(nrow(expanded), big.mark = ",")))
  
  # Now resolve each neighbor's row index for the same year
  # Join expanded to row_lookup on (neighbor_id = id, year = year)
  expanded[, neighbor_row := row_lookup[.(neighbor_id, year), .row_idx, nomatch = NA]]
  
  # Drop edges where the neighbor doesn't exist in that year
  expanded <- expanded[!is.na(neighbor_row)]
  
  rm(row_lookup)
  gc()
  
  message(sprintf("  %s valid edges after filtering.", format(nrow(expanded), big.mark = ",")))
  
  # --- Step 4: Compute neighbor stats per variable ---
  # We only need: expanded[, .(focal_row, neighbor_row)] plus the variable values
  
  edge_core <- expanded[, .(focal_row, neighbor_row)]
  rm(expanded)
  gc()
  
  setkey(edge_core, focal_row)
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    
    # Attach the neighbor's value
    vals <- cell_data[[var_name]]
    edge_core[, nval := vals[neighbor_row]]
    
    # Grouped aggregation â€” exclude NAs in the variable
    stats <- edge_core[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       by = focal_row]
    
    # Prepare column names (match original naming convention)
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Initialize with NA, then fill in computed values
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
    
    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)
    
    rm(stats)
    gc()
  }
  
  # Clean up helper column
  edge_core[, nval := NULL]
  rm(edge_core)
  gc()
  
  cell_data[, .row_idx := NULL]
  
  message("Done.")
  return(cell_data)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
library(data.table)

# cell_data, id_order, and rook_neighbors_unique are already in memory
# The trained Random Forest model is untouched.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has 15 new columns (3 stats Ã— 5 variables):
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
# 
# Proceed directly to prediction with the existing trained RF model.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The `max`, `min`, and `mean` aggregations are computed on exactly the same neighbor sets with the same NA-exclusion logic. The only difference is *how* the neighbor sets are assembled (vectorized join vs. per-row string lookup). The resulting numbers are identical to floating-point precision. |
| **Trained RF model** | No model retraining occurs. We only compute input features. The column names and semantics match the original pipeline, so `predict()` works unchanged. |
| **Neighbor structure** | The `spdep::nb` object is consumed read-only. The spatial adjacency is faithfully converted into the edge table. |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~40â€“50 hours (6.46M string hash probes Ã— ~4 neighbors each) | ~30â€“60 seconds (two `data.table` keyed joins) |
| Compute stats (Ã—5 vars) | ~36+ hours (5 Ã— 6.46M `lapply` iterations) | ~30â€“90 seconds (5 Ã— one grouped aggregation) |
| **Total** | **86+ hours** | **~2â€“5 minutes** |

Peak RAM: ~10â€“12 GB (the expanded edge table of ~26M rows is the main addition), well within the 16 GB budget.
END_RESPONSE>>>
