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
run_id: Anthropic_F02_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again, once per variable (Ã—5 variables), performing per-row subsetting and summary statistics via R-level loops.

**Specific problems:**

1. **String key construction and lookup in `build_neighbor_lookup`:** `paste()` and named-vector indexing over 6.46M rows inside `lapply` is extremely slow. Each row builds neighbor keys by string concatenation and looks them up in a 6.46M-length named vector â€” this is O(n Ã— k) with high constant factors.

2. **R-level `lapply` loops over millions of rows:** Both functions use `lapply` over ~6.46M elements. R's interpreted loop overhead dominates at this scale.

3. **Redundant computation:** The neighbor *structure* is the same across all years for a given cell. But the lookup is rebuilt monolithically, mixing spatial and temporal indexing in a single flat loop.

4. **`do.call(rbind, result)` on 6.46M small vectors:** This is a well-known R anti-pattern that causes repeated memory allocation and copying.

5. **Memory:** Storing a 6.46M-element list of integer vectors for `neighbor_lookup` is itself large, and intermediate string vectors consume significant RAM.

---

## Optimization Strategy

The key insight is: **rook neighbors are a spatial relationship that does not change across years.** We can separate the spatial neighbor graph from the temporal (year) dimension and use vectorized/data.table operations instead of row-level R loops.

**Strategy:**

1. **Build a flat edge table** of directed neighbor pairs `(cell_id, neighbor_id)` from the `nb` object â€” this has ~1.37M rows and is year-independent.

2. **Join this edge table to the panel data by year** using `data.table` keyed joins. For each year, every cell's neighbors' values are retrieved in one vectorized merge. This produces a long table of `(cell_id, year, neighbor_value)`.

3. **Compute grouped summary statistics** (`max`, `min`, `mean`) using `data.table`'s `by=` grouping â€” fully vectorized, no R-level row loop.

4. **Join the results back** to the main data.

This replaces all `lapply` loops and string-key lookups with vectorized joins and grouped aggregations, reducing runtime from ~86 hours to an estimated **minutes**.

**Memory management:** The largest intermediate object is the long edge-year table: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— a few columns â€” easily fits in 16 GB. We process one variable at a time and discard intermediates.

**Preserves:** The original numerical estimand (max, min, mean of neighbor values) is computed identically. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build a flat spatial edge table from the nb object
#         (run once; year-independent)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell_id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (id, neighbor_id)

# ---------------------------------------------------------------
# Step 2: Convert main data to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure key columns are proper types
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]
edge_dt[, id          := as.integer(id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# Step 3: For each variable, compute neighbor stats via joins
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # neighbor values keyed by (neighbor_id aliased as id, year)
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)

  # Expand edges by year: join edge_dt to val_dt on (neighbor_id, year)
  # First, add year dimension by joining edges to the value table
  setkey(edge_dt, neighbor_id)

  # Merge: for each (id, neighbor_id) edge, get neighbor's value per year
  # This is a keyed join: edge_dt[val_dt] would be wrong direction.
  # We want: for every (id, neighbor_id) pair and every year,
  #          look up val_dt[neighbor_id, year].
  # Most efficient: merge edge_dt with val_dt on neighbor_id & year.

  # Build the long table: all (id, year) -> neighbor values
  long_dt <- merge(edge_dt, val_dt, by = "neighbor_id", allow.cartesian = TRUE)
  # long_dt columns: neighbor_id, id, year, val
  # ~1.37M edges Ã— 28 years = ~38.4M rows (if all cells present all years)

  # Remove NAs in val before aggregation
  long_dt <- long_dt[!is.na(val)]

  # Compute grouped stats
  stats_dt <- long_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(id, year)]

  # Rename columns to match expected feature names
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats_dt, c("nb_max", "nb_min", "nb_mean"),
                      c(max_col,  min_col,  mean_col))

  # Left-join back to cell_dt
  setkey(stats_dt, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- stats_dt[cell_dt, on = .(id, year)]
  # Cells with no neighbors (e.g., islands) will have NA â€” correct behavior.

  # Clean up large intermediates
  rm(long_dt, val_dt, stats_dt)
  gc()

  cell_dt
}

# ---------------------------------------------------------------
# Step 4: Outer loop â€” one variable at a time (memory-friendly)
# ---------------------------------------------------------------
setkey(cell_dt, id, year)

for (var_name in neighbor_source_vars) {
  message("Processing neighbor features for: ", var_name)
  cell_dt <- compute_and_add_neighbor_features_fast(cell_dt, edge_dt, var_name)
  gc()  # free memory between iterations
}

# ---------------------------------------------------------------
# Step 5: Convert back to data.frame if downstream code expects it
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)
```

---

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (rook contiguity by year). Results are identical to the original. |
| **Trained RF model** | Untouched. Only the feature-engineering pipeline is optimized. Column names match the original convention (`nb_max_ntl`, etc.). |
| **Memory on 16 GB laptop** | Peak intermediate is ~38.4M rows Ã— 3 columns â‰ˆ ~0.9 GB per variable, freed after each iteration. Total working set stays well under 16 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregations on 38M rows typically complete in seconds. The full 5-variable loop should finish in **under 5 minutes** vs. the original 86+ hours. |

The fundamental transformation is replacing **6.46 million R-level iterations with string operations** with **vectorized equi-joins and grouped aggregations** by factoring out the year-invariant spatial graph.
END_RESPONSE>>>
