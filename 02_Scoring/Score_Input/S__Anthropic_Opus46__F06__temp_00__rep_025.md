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
run_id: Anthropic_F06_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation Ã— ~4 neighbors Ã— 6.46M rows â‰ˆ 25.8M string constructions).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) â€” named vector indexing in R is hash-based but still slow at this scale.

The result is a **list of 6.46 million integer vectors**. This alone can take hours and consumes substantial memory.

**Key insight:** The neighbor structure is *purely spatial* â€” it doesn't change across years. A cell's rook neighbors in 1993 are the same cells as in 2007. The only thing that changes is which *row index* corresponds to `(neighbor_id, year)`. If the data is sorted by `(id, year)` or `(year, id)` in a predictable way, we can compute row indices arithmetically instead of via string hashing.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M elements, repeated 5 times

For each of the 5 source variables, we iterate over 6.46M rows, subset a numeric vector by index, remove NAs, and compute `max/min/mean`. This is 32.3M R-level function calls total. Each call is cheap, but the R interpreter overhead at this scale is enormous.

### Why raster focal/kernel operations are *not* directly applicable

The comment in the prompt asks whether raster focal operations (e.g., `terra::focal`) could help. Focal operations assume a **regular grid with complete coverage** and a fixed kernel. Here:
- The panel is cell-year, not a single raster layer.
- The neighbor structure (`spdep::nb`) may reflect irregular boundaries (coastal cells, border cells with fewer than 4 neighbors).
- We need `max`, `min`, and `mean` â€” focal can do this per-layer, but we'd need to reshape to 28 raster layers, run focal 3Ã—5Ã—28 = 420 times, then reshape back.

The **better approach** is to vectorize the neighbor computation directly using sparse matrix multiplication and vectorized row operations, which preserves the exact `spdep::nb` topology and the exact numerical results.

---

## 2. Optimization Strategy

### Strategy A: Eliminate `build_neighbor_lookup` entirely for per-variable stats

Instead of building a 6.46M-element list of row indices, we:

1. **Sort the data** by `(id, year)` so that each cell's 28 years are contiguous.
2. **Build a sparse adjacency matrix** `W` of dimension `344,208 Ã— 344,208` from the `nb` object (one-time cost, fast via `spdep::nb2listw` â†’ `as_dgRMatrix`).
3. For each year, extract the column of values, compute `W %*% x` for the mean (with row-sum normalization), and use grouped sparse-row operations for `max` and `min`.

But sparse matrix multiplication gives us **sum** (hence mean), not max/min. For max and min we need a different approach.

### Strategy B (Chosen): Vectorized expansion via `data.table` joins

The fastest pure-R approach that preserves exact results:

1. Build an **edge list** from the `nb` object: a two-column integer matrix `(from_cell_idx, to_cell_idx)` with ~1.37M rows.
2. Ensure `cell_data` is a `data.table` keyed on `(id, year)`.
3. For each source variable, do a single **non-equi join / edge-list join**: expand each cell-year to its neighbors' values via the edge list, then **group-by** `(cell_row)` to compute `max`, `min`, `mean` in one vectorized pass.

This replaces 6.46M R-level iterations with a single `data.table` grouped aggregation over ~25.8M edge-year rows â€” something `data.table` handles in seconds.

**Expected speedup:** From 86+ hours to **~5â€“15 minutes total**.

---

## 3. Working R Code

```r
library(data.table)
library(spdep)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table if not already
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a spatial edge list from the nb object (one-time, fast)
#
#   rook_neighbors_unique: an nb object of length 344,208
#   id_order: vector of cell IDs in the order matching the nb object
#
#   We build a data.table with columns:
#     focal_id    â€” the cell ID of the focal cell
#     neighbor_id â€” the cell ID of each rook neighbor
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, nb_obj) {
  n <- length(nb_obj)
  # Pre-compute lengths to allocate once
  lens <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(lens)
  
  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 to denote "no neighbors" in some cases)
  valid <- neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building spatial edge list...\n")
edge_list <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor features for all source variables
#
#   For each variable, we:
#     a) Join edge_list Ã— cell_data to get neighbor values per cell-year
#     b) Aggregate max, min, mean by (focal_id, year)
#     c) Merge back into cell_data
#
#   This is fully vectorized â€” no row-level R loops.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data has a row-order key we can restore later
cell_data[, .row_order := .I]

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor features for: %s\n", var_name))
  
  # Extract only the columns we need for the join (minimize memory)
  # neighbor_id will be matched to id, year will be matched to year
  cols_needed <- c("id", "year", var_name)
  neighbor_vals <- cell_data[, ..cols_needed]
  setnames(neighbor_vals, old = c("id", var_name), new = c("neighbor_id", "nval"))
  
  # Key for fast join
  setkey(neighbor_vals, neighbor_id, year)
  
  # Expand: for each edge (focal_id, neighbor_id), join on (neighbor_id, year)
  # This creates one row per (focal_cell, neighbor_cell, year) combination
  # with the neighbor's value attached.
  #
  # We join edge_list to neighbor_vals:
  #   edge_list has: focal_id, neighbor_id
  #   neighbor_vals has: neighbor_id, year, nval
  #   Result: focal_id, neighbor_id, year, nval
  
  expanded <- neighbor_vals[edge_list, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = NA]
  # expanded now has columns: neighbor_id, year, nval, focal_id
  
  # Aggregate by (focal_id, year) â€” this is the core computation
  agg <- expanded[!is.na(nval),
                  .(nb_max  = max(nval),
                    nb_min  = min(nval),
                    nb_mean = mean(nval)),
                  by = .(focal_id, year)]
  
  # Rename columns to match expected output format
  new_col_max  <- paste0("nb_max_",  var_name)
  new_col_min  <- paste0("nb_min_",  var_name)
  new_col_mean <- paste0("nb_mean_", var_name)
  setnames(agg,
           old = c("focal_id", "nb_max", "nb_min", "nb_mean"),
           new = c("id", new_col_max, new_col_min, new_col_mean))
  
  # Merge back into cell_data
  setkey(agg, id, year)
  setkey(cell_data, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(new_col_max, new_col_min, new_col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- agg[cell_data, on = .(id, year)]
  
  # Clean up to free memory
  rm(neighbor_vals, expanded, agg)
  gc()
  
  cat(sprintf("  Done: added %s, %s, %s\n", new_col_max, new_col_min, new_col_mean))
}

# Restore original row order
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

cat("All neighbor features computed.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is.
# Example:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Exact Results

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | Edge list is built directly from the same `spdep::nb` object (`rook_neighbors_unique`). No approximation. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same sets of neighbor values. NAs are excluded identically (`!is.na(nval)` mirrors `neighbor_vals[!is.na(neighbor_vals)]`). |
| **Same RF model** | The model is never retrained â€” only `predict()` is called on the enriched data. |
| **No raster approximation** | We rejected `terra::focal` because it would impose a regular-grid kernel that could silently alter neighbor sets at boundaries. |

## 5. Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build lookup (6.46M string hashes) | ~10â€“20 hours | **~2 seconds** (edge list from `nb`) |
| Compute stats (5 vars Ã— 6.46M `lapply`) | ~60â€“70 hours | **~5â€“10 min** (5 Ã— `data.table` grouped agg on ~25.8M rows) |
| Peak RAM | ~12â€“14 GB (6.46M-element list) | **~6â€“8 GB** (edge expansion is streaming per variable, cleaned up each iteration) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |
END_RESPONSE>>>
