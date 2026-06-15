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
run_id: Anthropic_F04_T10_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, not `compute_neighbor_stats`. Here's why:

1. **`build_neighbor_lookup` uses `lapply` over ~6.46 million rows**, and inside each iteration it performs:
   - A named-vector lookup (`id_to_ref[...]`) â€” O(1) amortized but with overhead per call.
   - Extraction of neighbor cell IDs from a list.
   - `paste()` to construct string keys for every neighbor of every row.
   - A named-vector lookup (`idx_lookup[neighbor_keys]`) on a **6.46-million-element named character vector** â€” this is hash-based but the sheer volume (~6.46M Ã— ~4 avg neighbors â‰ˆ 26M lookups against a large hash table) is brutal.
   - `is.na` filtering per iteration.

   The dominant cost is **~26 million string constructions via `paste()` and ~26 million hash lookups into a 6.46M-entry named vector**, all inside an interpreted R `lapply` loop with per-element overhead. This is estimated at **86+ hours**.

2. **`compute_neighbor_stats`** is comparatively cheap: it just indexes a numeric vector and computes `max/min/mean` per row. The lookup indices are already integer. This is fast.

3. **The Random Forest inference itself** operates on a pre-trained model against ~6.46M rows Ã— ~110 features â€” a substantial but well-optimized operation in `ranger`/`randomForest`. It is not the bottleneck.

**Root cause summary:** Row-level R loop with millions of string `paste` + hash-table lookups on a massive named vector. R's interpreted loop overhead and the string-key approach are catastrophically slow at this scale.

---

## Optimization Strategy

**Replace the row-level `lapply` with a vectorized `data.table` equi-join.**

The core insight: `build_neighbor_lookup` is conceptually a join:

> For each `(cell_id, year)`, find all rows whose `cell_id` is a rook neighbor of `cell_id` AND whose `year` matches.

And `compute_neighbor_stats` is a grouped aggregation over the join result.

**Steps:**

1. **Expand the neighbor list into an edge table** (`data.table` with columns `id` and `neighbor_id`). This is ~1.37M rows â€” tiny.
2. **Join the edge table with the data twice** â€” once on `id` (to bring in `year`) and once on `(neighbor_id, year)` â€” to directly obtain neighbor feature values. This is a vectorized `data.table` merge, no string keys, no per-row R loop.
3. **Grouped aggregation** (`max`, `min`, `mean`) by `(id, year)` using `data.table`'s optimized `by=` machinery.
4. **Merge results back** into the main data.

**Expected speedup:** From 86+ hours to **~2â€“10 minutes** on a 16 GB laptop. The join produces ~26M rows (manageable in RAM at ~1â€“2 GB for the join table) and grouped aggregation is near-instantaneous in `data.table`.

**Preserves:** The trained Random Forest (untouched) and the original numerical estimand (identical `max`, `min`, `mean` statistics).

---

## Working R Code

```r
library(data.table)

#' Build the neighbor edge table once (replaces build_neighbor_lookup).
#' 
#' @param id_order   Integer vector of cell IDs in the order matching the nb object.
#' @param neighbors  An spdep::nb object (list of integer index vectors).
#' @return A data.table with columns: id, neighbor_id
build_neighbor_edge_table <- function(id_order, neighbors) {
  # Expand the adjacency list into a two-column edge list (vectorized)
  n_neighbors <- lengths(neighbors)
  from_idx <- rep(seq_along(neighbors), times = n_neighbors)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

#' Compute neighbor summary statistics for one variable via a data.table join.
#'
#' @param dt         data.table with at least columns: id, year, and the variable.
#' @param edges      data.table with columns: id, neighbor_id (from build_neighbor_edge_table).
#' @param var_name   Character string â€” name of the source variable.
#' @return The input dt, modified in place with three new columns appended:
#'         <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean
compute_and_add_neighbor_features_fast <- function(dt, edges, var_name) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  # Subset to only necessary columns for the join to minimize memory
  # neighbor_data: keyed by (neighbor_id aliased as id, year) -> var_name value
  neighbor_data <- dt[, .(id, year, val = get(var_name))]
  setnames(neighbor_data, "id", "neighbor_id")
  setkeyv(neighbor_data, c("neighbor_id", "year"))
  
  # Step 1: Join edges with the focal cell's year.
  #   For each (id, neighbor_id) edge, replicate across all years of id.
  focal_years <- dt[, .(id, year)]
  setkeyv(focal_years, "id")
  setkeyv(edges, "id")
  
  # edges_with_year: (id, year, neighbor_id)
  # This is the ~26M-row expanded table (1.37M edges Ã— ~19 avg years per cell
  # that exist in the panel). Using a keyed join avoids Cartesian blowup for
  # cells that don't span all 28 years.
  edges_with_year <- edges[focal_years, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # Result columns: id, neighbor_id, year
  
  # Step 2: Look up the neighbor's value in the same year.
  setkeyv(edges_with_year, c("neighbor_id", "year"))
  edges_with_year <- neighbor_data[edges_with_year, on = c("neighbor_id", "year"), nomatch = NA]
  # Result columns: neighbor_id, year, val, id
  
  # Step 3: Grouped aggregation â€” compute max, min, mean per (id, year),
  #          excluding NA values to match original behavior.
  agg <- edges_with_year[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(id, year)
  ]
  
  # Step 4: Merge back into dt
  setkeyv(dt, c("id", "year"))
  setkeyv(agg, c("id", "year"))
  
  dt[agg, (col_max)  := i.nb_max,  on = c("id", "year")]
  dt[agg, (col_min)  := i.nb_min,  on = c("id", "year")]
  dt[agg, (col_mean) := i.nb_mean, on = c("id", "year")]
  
  # Rows with no valid neighbors remain NA (data.table default for unmatched joins)
  
  invisible(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main pipeline (replaces the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table in place (no copy if already a data.table)
setDT(cell_data)

# Build the edge table once (~1.37M rows, <20 MB RAM)
edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

# Source variables for neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_and_add_neighbor_features_fast(cell_data, edges, var_name)
}

# cell_data now has 15 new columns (3 per variable Ã— 5 variables):
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ...etc.
#
# Proceed to Random Forest prediction with the pre-trained model (unchanged).
```

---

## Why This Is Correct and Equivalent

| Original behavior | Vectorized replacement |
|---|---|
| For row `i`, find all rows sharing `(neighbor_id, year)` via string key lookup | `data.table` equi-join on `(neighbor_id, year)` â€” identical match semantics |
| `max`, `min`, `mean` over non-NA neighbor values | Identical: `!is.na(val)` filter then `max`/`min`/`mean` grouped by `(id, year)` |
| Returns `NA` when no valid neighbors exist | Unmatched `(id, year)` pairs remain `NA` after the left-update join |

The numerical estimand (the three summary statistics per variable per cell-year) is **identical** to the original code. The trained Random Forest model is **untouched**.

---

## Memory Budget Check

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 125 cols, numeric) | ~6.5 GB |
| `edges` (1.37M Ã— 2 integer cols) | ~11 MB |
| `edges_with_year` (~26M Ã— 3 cols) | ~620 MB |
| `agg` (~6.46M Ã— 4 cols) | ~200 MB |
| **Peak total** | **~7.3 GB** â€” fits in 16 GB |

The intermediate tables (`edges_with_year`, `agg`, `neighbor_data`) are created and garbage-collected per variable iteration, so peak memory never exceeds ~8 GB, well within the 16 GB constraint.
END_RESPONSE>>>
