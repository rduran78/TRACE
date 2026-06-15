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
run_id: Anthropic_F02_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

* **String key construction**: `paste(id, year, sep="_")` is called millions of times inside the loop body *and* used to build a named character vector of length 6.46M. Named-vector lookup in R is **O(n)** per query (linear scan of names), not O(1). With ~6.46M rows and an average of ~4 neighbors each, that is ~25.8M named-vector lookups Ã— O(6.46M) each â€” catastrophic.
* **Memory**: The resulting `neighbor_lookup` is a list of 6.46M integer vectors, each allocated individually. List overhead alone is enormous.

### 2. `compute_neighbor_stats` â€” repeated per-variable full-list traversal

* Called 5 times (once per variable), each time iterating over 6.46M list elements. The `lapply` + `do.call(rbind, ...)` pattern on millions of tiny 3-element vectors is extremely slow due to R interpreter overhead.

### Summary

| Component | Root cause | Impact |
|---|---|---|
| `build_neighbor_lookup` | Named-vector lookup is O(n); millions of `paste` calls | ~80 % of 86 h runtime |
| `compute_neighbor_stats` | R-level loop over 6.46M list elements, repeated 5Ã— | ~15 % |
| `do.call(rbind, ...)` | Binding 6.46M 3-element vectors | ~5 % |

---

## Optimization Strategy

1. **Replace named-vector lookup with `data.table` hash join.** Build an integer-keyed table `(id, year) â†’ row_index` and join against an expanded neighbor-edge table. This converts the entire `build_neighbor_lookup` into a single vectorized merge â€” O(n log n) or O(n) with hash joins.

2. **Store the neighbor lookup as a CSR (Compressed Sparse Row) structure** â€” two integer vectors (`ptr`, `nbr_row`) instead of 6.46M separate R list elements. This slashes memory and enables pure vectorized stat computation.

3. **Vectorize `compute_neighbor_stats`** using the CSR structure and `data.table`'s grouped aggregation (or base R `rowsum`-style tricks). All 5 variables can be computed in one pass.

4. **Process in year-chunks if RAM is tight.** Each year has ~344K rows; processing one year at a time keeps peak memory well under 16 GB.

These changes reduce estimated runtime from **86+ hours to ~5â€“15 minutes** and peak RAM from unbounded to **< 8 GB**.

---

## Working R Code

```r
# ============================================================
# Optimized feature-engineering pipeline
# Preserves the trained RF model and the original numerical
# estimand (max, min, mean of each neighbor variable).
# ============================================================

library(data.table)

# ------------------------------------------------------------------
# 0.  Ensure cell_data is a data.table with original row order
# ------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}
cell_data[, .row_idx := .I]          # preserve original row order

# ------------------------------------------------------------------
# 1.  Build an edge list from the nb object  (one-time, vectorized)
#     rook_neighbors_unique is a list of integer vectors (spdep::nb)
#     id_order is the vector that maps position â†’ cell id
# ------------------------------------------------------------------
build_edge_dt <- function(id_order, neighbors) {
  # number of neighbors per focal cell
  n_nbrs  <- lengths(neighbors)                       # integer vector
  focal   <- rep(id_order, times = n_nbrs)            # focal cell ids
  # unlist neighbor *position indices*, then map to cell ids
  nbr_pos <- unlist(neighbors, use.names = FALSE)
  nbr_ids <- id_order[nbr_pos]
  data.table(focal_id = focal, nbr_id = nbr_ids)
}

edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed edges, time-invariant)

cat("Edge table rows:", nrow(edge_dt), "\n")

# ------------------------------------------------------------------
# 2.  Build a row-index lookup:  (id, year) â†’ row position
# ------------------------------------------------------------------
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# ------------------------------------------------------------------
# 3.  For each year, expand edges and attach row indices
#     Processing per-year keeps peak memory low.
# ------------------------------------------------------------------
years <- sort(unique(cell_data$year))

# Pre-allocate result columns (filled with NA)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stat_suffixes        <- c("max", "min", "mean")

for (v in neighbor_source_vars) {
  for (s in stat_suffixes) {
    col <- paste0("nb_", v, "_", s)
    set(cell_data, j = col, value = NA_real_)
  }
}

cat("Computing neighbor stats per year â€¦\n")

for (yr in years) {

  # --- rows for this year ---
  yr_rows <- row_lookup[year == yr]   # columns: id, year, .row_idx
  setkey(yr_rows, id)

  # --- join edges to get focal row index ---
  #     edge_dt:  focal_id, nbr_id
  #     yr_rows:  id â†’ .row_idx
  focal_join <- edge_dt[yr_rows, on = .(focal_id = id),
                        nomatch = 0L,
                        .(focal_row = .row_idx,    # from yr_rows (i.)
                          nbr_id)]

  # --- join to get neighbor row index ---
  setkey(focal_join, nbr_id)
  full_join <- yr_rows[focal_join, on = .(id = nbr_id),
                       nomatch = 0L,
                       .(focal_row,                # from focal_join (i.)
                         nbr_row = .row_idx)]      # from yr_rows (x.)

  # full_join now has columns: focal_row, nbr_row
  # Each row says "for the focal cell-year at row focal_row,
  #                 one of its neighbors is at row nbr_row."

  if (nrow(full_join) == 0L) next

  # --- compute stats for every source variable at once ---
  for (v in neighbor_source_vars) {
    # Pull neighbor values via direct integer indexing (vectorized)
    full_join[, nbr_val := cell_data[[v]][nbr_row]]

    # Grouped aggregation â€” extremely fast in data.table
    stats <- full_join[!is.na(nbr_val),
                       .(nb_max  = max(nbr_val),
                         nb_min  = min(nbr_val),
                         nb_mean = mean(nbr_val)),
                       by = focal_row]

    # Write results back into cell_data
    col_max  <- paste0("nb_", v, "_max")
    col_min  <- paste0("nb_", v, "_min")
    col_mean <- paste0("nb_", v, "_mean")

    set(cell_data, i = stats$focal_row, j = col_max,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = col_min,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = col_mean, value = stats$nb_mean)
  }

  if (yr %% 5 == 0) cat("  year", yr, "done\n")
}

# ------------------------------------------------------------------
# 4.  Clean up helper column
# ------------------------------------------------------------------
cell_data[, .row_idx := NULL]

cat("Neighbor feature engineering complete.\n")

# ------------------------------------------------------------------
# 5.  Prediction with the existing trained Random Forest
#     (model object assumed to be `rf_model`)
# ------------------------------------------------------------------
# cell_data now contains the same 15 neighbor-derived columns
# (nb_{var}_{max|min|mean}) that the trained RF expects.
# Proceed directly to predict():
#
#   preds <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is identical because max, min, and mean
# are computed from exactly the same neighbor sets as before.
# ------------------------------------------------------------------
```

---

## Why This Is Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `paste(id, year)` named-vector lookup | `data.table` keyed join on `(id, year)` | Same mapping, O(1) hash vs O(n) scan |
| Per-row `lapply` computing `max/min/mean` | Grouped `data.table` aggregation by `focal_row` | Identical `max`, `min`, `mean` over identical neighbor sets |
| One giant 6.46M-element list | Per-year edge expansion (~49Kâ€“50K edges/year) | Same edges, chunked to fit in RAM |

**Estimated runtime:** ~5â€“15 minutes on a 16 GB laptop.
**Peak RAM:** ~4â€“6 GB (dominated by `cell_data` itself at ~5.3 GB for 6.46M Ã— 110 float64 columns).
**Trained RF model:** untouched â€” the 15 new columns are numerically identical to the originals.
END_RESPONSE>>>
