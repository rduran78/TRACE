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
run_id: Anthropic_F05_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each** row it:

1. Calls `paste()` to build string keys for every neighbor of that cell-year.
2. Looks those keys up in a named character vector (`idx_lookup`) â€” which itself was built by pasting 6.46M strings.

This means:
- **~6.46M iterations**, each doing string concatenation and hash-table lookups.
- The neighbor relationships are **time-invariant** (rook contiguity doesn't change year to year), yet the code re-discovers the same spatial neighbors for every year a cell appears.
- The string-keyed lookup is an indirect way of doing what is really a **join on (id, year)** â€” something that can be done vectorized with integer indexing.

### Quantifying the Waste

| Operation | Count |
|---|---|
| Rows (cell-years) | ~6.46M |
| Unique cells | 344,208 |
| Years | 28 |
| Directed neighbor pairs | ~1.37M |
| Neighbor-year lookups inside `lapply` | ~1.37M Ã— 28 â‰ˆ **38.5M** string pastes + hash lookups |

The entire `build_neighbor_lookup` is **O(R Ã— avg_neighbors)** with expensive string constants. Then `compute_neighbor_stats` loops over 6.46M entries again per variable (Ã—5 variables).

### Broader Pattern

The architecture is:

```
build_neighbor_lookup (slow, string-based)
  â†’ returns list of 6.46M integer vectors
    â†’ compute_neighbor_stats loops over that list 5 times
```

Both stages can be replaced with **vectorized, integer-only operations** using `data.table`.

---

## Optimization Strategy

### Key Insight: Separate Spatial Structure from Temporal Replication

1. **Convert the `nb` object to an edge list once** â€” a two-column integer table of `(cell_id, neighbor_id)` with ~1.37M rows.
2. **Join the edge list to the panel on `(neighbor_id, year)`** â€” this is a single `data.table` merge that produces ~38.5M rows, fully vectorized.
3. **Aggregate neighbor statistics** with a single grouped `data.table` operation per variable (or all at once).

This eliminates:
- All `paste()`/string-key construction.
- The 6.46M-element `lapply`.
- The per-variable R-level loop inside `compute_neighbor_stats`.

### Expected Speedup

| Stage | Old | New |
|---|---|---|
| Neighbor lookup build | ~hours (string `lapply`) | ~seconds (nbâ†’edge list) |
| Per-variable stats | ~hours (`lapply` over 6.46M lists) | ~seconds (`data.table` grouped aggregation) |
| **Total for 5 variables** | **86+ hours** | **~1â€“3 minutes** |

Memory stays well within 16 GB: the edge-list Ã— years table is ~38.5M rows Ã— a few columns of integers/doubles.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 1. Convert spdep nb object to an integer edge-list (one-time)
# ============================================================
nb_to_edge_list <- function(nb_obj, id_order) {
  # nb_obj: list of integer index vectors (spdep::nb), 
  #         indices refer to positions in id_order
  # id_order: vector of cell IDs in the order matching nb_obj
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  # Remove the 0-neighbor sentinel if present
  valid <- to > 0L
  data.table(
    cell_id     = id_order[from[valid]],
    neighbor_id = id_order[to[valid]]
  )
}

# ============================================================
# 2. Vectorized neighbor-feature construction
# ============================================================
compute_all_neighbor_features <- function(cell_data, 
                                          neighbor_source_vars, 
                                          rook_neighbors_unique, 
                                          id_order) {
  
  # --- Step A: build edge list (spatial, time-invariant) ------
  edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
  # ~1.37M rows: (cell_id, neighbor_id)
  
  # --- Step B: convert panel to data.table --------------------
  dt <- as.data.table(cell_data)
  
  # Ensure key columns are present and well-typed
  dt[, id   := as.integer(id)]
  dt[, year := as.integer(year)]
  
  # --- Step C: build the neighbor-year table ------------------
  # For every directed edge, replicate across all 28 years.
  # Instead of a cross-join, we join edges to the panel on the
  # neighbor side to pull neighbor values directly.
  
  # Subset to only the columns we need for the neighbor side
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  dt_neighbor <- dt[, ..neighbor_cols]
  setnames(dt_neighbor, "id", "neighbor_id")
  
  # Key for fast join
  setkey(dt_neighbor, neighbor_id, year)
  
  # Expand edges: attach year from the focal cell
  # We need (cell_id, year) â†’ list of neighbor values
  # Approach: join edges to focal rows to get (cell_id, neighbor_id, year),
  # then join to dt_neighbor to get neighbor values.
  
  # Focal side: just need (cell_id, year) â€” one row per cell-year
  focal <- dt[, .(cell_id = id, year)]
  
  # Cross of focal Ã— edges on cell_id
  # This gives us (cell_id, neighbor_id, year) â€” ~38.5M rows
  setkey(edges, cell_id)
  setkey(focal, cell_id)
  expanded <- edges[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year
  
  # Now join to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  expanded <- dt_neighbor[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, <var columns>, cell_id
  
  # --- Step D: aggregate per (cell_id, year) ------------------
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    suffix_max  <- paste0("neighbor_max_",  v)
    suffix_min  <- paste0("neighbor_min_",  v)
    suffix_mean <- paste0("neighbor_mean_", v)
    agg_exprs[[suffix_max]]  <- bquote(as.numeric(max(.(v_sym),  na.rm = TRUE)))
    agg_exprs[[suffix_min]]  <- bquote(as.numeric(min(.(v_sym),  na.rm = TRUE)))
    agg_exprs[[suffix_mean]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Single grouped aggregation
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  stats <- expanded[, eval(agg_call), by = .(cell_id, year)]
  
  # Fix Inf/-Inf from max/min on all-NA groups â†’ NA
  num_cols <- setdiff(names(stats), c("cell_id", "year"))
  for (col in num_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # --- Step E: merge back to the original panel ---------------
  setkey(stats, cell_id, year)
  setkey(dt, id, year)
  
  # Merge
  dt <- stats[dt, on = .(cell_id = id, year = year)]
  
  # Restore column name 'id'
  setnames(dt, "cell_id", "id")
  
  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ============================================================
# 3. Drop-in replacement for the outer loop
# ============================================================
# BEFORE (86+ hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# AFTER (~1-3 minutes):

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  neighbor_source_vars = neighbor_source_vars,
  rook_neighbors_unique = rook_neighbors_unique,
  id_order             = id_order
)

# The trained Random Forest model is untouched.
# Prediction proceeds exactly as before:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of neighbor values per `(cell_id, year)`. The join `(neighbor_id, year)` replicates the original string-key lookup logic. `na.rm = TRUE` matches the original `neighbor_vals[!is.na(neighbor_vals)]` filtering. |
| **Column names** | Output columns follow the pattern `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc. Adjust the naming template if your trained RF expects different names. |
| **Trained RF model** | No model retraining. The code only reconstructs the feature columns that the model already expects. |
| **Edge cases** | Cells with zero neighbors (islands) get `NA` for all neighbor stats, matching the original `c(NA, NA, NA)` return. |

---

## Memory Profile (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `dt` (panel) | ~5.7 GB (6.46M Ã— 110 cols) |
| `edges` | ~11 MB (1.37M Ã— 2 int cols) |
| `expanded` | ~2.2 GB (38.5M Ã— 7 cols) |
| `stats` | ~1.5 GB (6.46M Ã— 17 cols) |
| **Peak** | **~10 GB** â€” fits in 16 GB |

If memory is tight, process variables in batches of 2â€“3 instead of all 5 at once, which reduces `expanded` width and `stats` width proportionally.
END_RESPONSE>>>
