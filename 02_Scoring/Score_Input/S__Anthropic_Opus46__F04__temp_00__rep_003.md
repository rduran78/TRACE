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
run_id: Anthropic_F04_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs character key strings by pasting cell ID and year.
4. Performs named-vector lookup on `idx_lookup` (~6.46M named entries).

Character-based named-vector lookups in R are O(n) or O(n log n) hash-table probes on millions of keys. Doing this ~6.46 million times, each time pasting and matching multiple neighbor keys against a 6.46M-entry named character vector, is catastrophically slow. The `compute_neighbor_stats` function is comparatively cheap (integer indexing into a numeric vector), but it is called 5 times, each time iterating over 6.46M list elements with `lapply` and then `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors, which is also unnecessarily slow.

**Root causes, ranked by impact:**

1. **Character key construction and lookup in a giant named vector** â€” millions of `paste()` calls and hash lookups per row.
2. **Row-level `lapply` in R** â€” 6.46M R-level function calls with no vectorization.
3. **`do.call(rbind, ...)` on millions of small vectors** â€” slow list-to-matrix coercion.

## Optimization Strategy

**Core idea:** Replace the per-row character-key lookup with a fully vectorized, integer-indexed approach using `data.table`. Pre-build a single integer matrix (or edge list) mapping every row to its neighbor rows, then compute neighbor statistics using vectorized grouped operations â€” no R-level loop over 6.46M rows.

**Steps:**

1. **Build a row-index edge list once** using `data.table` equi-joins (vectorized, hash-based). Each edge maps a `(cell, year)` row to a `(neighbor_cell, year)` row. This replaces `build_neighbor_lookup` entirely.
2. **Compute all neighbor stats via grouped `data.table` aggregation** â€” one pass per variable, fully vectorized. This replaces `compute_neighbor_stats`.
3. **Join results back** to the main table by row index.

Expected speedup: from ~86+ hours to **minutes** (the edge list is ~1.37M neighbor pairs Ã— 28 years â‰ˆ 38.5M edges; `data.table` grouped aggregation over 38.5M rows is fast).

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Convert to data.table and create integer row index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]

# â”€â”€ 1. Build the neighbor edge list (replaces build_neighbor_lookup) â”€â”€â”€â”€â”€â”€â”€
# Convert the spdep nb object into a data.table of directed edges: (cell, neighbor_cell)
# id_order is the vector mapping position in the nb list â†’ cell id.

nb_edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
    return(NULL)
  }
  data.table(id = id_order[i], neighbor_id = id_order[nb_i])
}))
# nb_edge_list has ~1,373,394 rows (directed rook-neighbor pairs)

# â”€â”€ 2. Expand edges across all years via join â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Create a lookup: (id, year) â†’ .row_idx
id_year_idx <- cell_dt[, .(id, year, .row_idx)]
setkey(id_year_idx, id, year)

# For every (cell, year) row, find its neighbor rows by joining:
#   (cell â†’ neighbor_id) Ã— year  â†’  neighbor's .row_idx
# Step 2a: attach the focal row's year and row index
edges <- merge(
  nb_edge_list,
  id_year_idx,
  by = "id",
  allow.cartesian = TRUE
)
# edges now has columns: id, neighbor_id, year, .row_idx (focal row)
setnames(edges, ".row_idx", "focal_row")

# Step 2b: attach the neighbor's row index
edges <- merge(
  edges,
  id_year_idx,
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  allow.cartesian = FALSE
)
setnames(edges, ".row_idx", "neighbor_row")

# Keep only the columns we need; key by focal_row for fast grouped ops
edges <- edges[, .(focal_row, neighbor_row)]
setkey(edges, focal_row)

# â”€â”€ 3. Compute neighbor features (replaces compute_neighbor_stats) â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {


  # Pull the variable values into the edge table by neighbor row index
  edges[, val := cell_dt[[var_name]][neighbor_row]]

  # Grouped aggregation: max, min, mean per focal row (excluding NAs)
  stats <- edges[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = focal_row
  ]

  # Initialize new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign results back by row index
  cell_dt[stats$focal_row, (max_col)  := stats$nb_max]
  cell_dt[stats$focal_row, (min_col)  := stats$nb_min]
  cell_dt[stats$focal_row, (mean_col) := stats$nb_mean]
}

# Clean up helper column
edges[, val := NULL]
cell_dt[, .row_idx := NULL]

# â”€â”€ 4. Convert back to data.frame for downstream Random Forest predict() â”€â”€
cell_data <- as.data.frame(cell_dt)

# â”€â”€ 5. Predict with the existing trained model (unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The grouped `max`, `min`, `mean` over the same set of non-NA neighbor values produces identical results to the original `compute_neighbor_stats`. |
| **Trained RF model** | The model object is never touched. Only the input feature columns are constructed, with the same names and semantics. |
| **Edge semantics** | The `merge` on `(neighbor_id, year)` replicates exactly the original logic: "for row *i*, find all rows sharing the same year whose cell ID is a rook neighbor of row *i*'s cell ID." |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build lookup / edge list | ~hours (6.46M R-level iterations with character hashing) | ~1â€“3 min (`data.table` merge, ~38.5M edge rows) |
| Compute stats (Ã—5 vars) | ~hours (6.46M `lapply` + `do.call(rbind)` per var) | ~1â€“2 min per var (vectorized grouped agg) |
| **Total** | **~86+ hours** | **~10â€“15 minutes** |

Peak memory for the `edges` table: ~38.5M rows Ã— 2 integer columns â‰ˆ 0.6 GB, well within 16 GB.
END_RESPONSE>>>
