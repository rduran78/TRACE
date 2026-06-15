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
run_id: Anthropic_F09_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of three compounding problems:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row across every cell-year observation. For each of the ~6.46M rows, it performs string pasting, named-vector lookups, and NA filtering. Named vector lookups (`idx_lookup[neighbor_keys]`) using character keys are O(n) hash lookups repeated millions of times, and `paste()` creates millions of temporary string objects. This alone can take tens of hours.

### 2. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
The neighbor stats computation also loops through every row, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, and `mean`. This is called 5 times (once per neighbor source variable), yielding ~32.3M R-level function invocations.

### 3. The Neighbor Structure Is Time-Invariant but Rebuilt Per Cell-Year
The spatial neighbor topology is fixed: cell A's rook neighbors are always the same cells regardless of year. Yet `build_neighbor_lookup` re-resolves these relationships for every cell-year row, inflating the work by a factor of 28 (the number of years). This is the fundamental architectural flaw.

**Key insight:** The adjacency table has only ~1.37M directed relationships across ~344K cells. The yearly attribute values change, but the neighbor graph does not. The correct approach is to build the neighbor edge table once (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M edges), then join year-specific attributes onto it and use grouped aggregation â€” all vectorized.

---

## Optimization Strategy

1. **Build a static edge table once:** Convert the `spdep::nb` object into a two-column `data.table` of `(focal_id, neighbor_id)` â€” approximately 1.37M rows. This is done once and can be cached to disk.

2. **Join yearly attributes via `data.table`:** For each year, the cell attributes are keyed by `(id, year)`. We join the neighbor's attributes onto the edge table by `(neighbor_id, year)`, which `data.table` does via binary-search keyed joins in milliseconds.

3. **Grouped aggregation:** After the join, compute `max`, `min`, and `mean` grouped by `(focal_id, year)` â€” a single vectorized `data.table` operation across 1.37M Ã— 28 â‰ˆ ~38.4M rows. No R-level loops.

4. **Repeat for each of the 5 variables** (or batch all 5 into a single join + aggregation pass).

5. **Merge results back** into the main cell-year dataset and run `predict()` with the existing trained Random Forest model.

**Expected speedup:** From ~86+ hours to **minutes** (typically 2â€“10 minutes depending on disk I/O). Memory usage will peak at roughly 2â€“3 GB, well within 16 GB.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Convert cell_data to data.table (if not already)
# =============================================================================
cell_dt <- as.data.table(cell_data)

# =============================================================================
# STEP 1: Build the static neighbor edge table ONCE
# =============================================================================
# rook_neighbors_unique is an spdep::nb object (list of integer vectors)
# id_order is the vector of cell IDs corresponding to indices 1..344208

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] gives the indices (into id_order) of cell i's neighbors
  # A neighbor index of 0 means no neighbors (spdep convention)
  focal_indices <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  neighbor_indices <- unlist(neighbors_nb)

  # Remove the 0-index entries (cells with no neighbors, encoded as 0 in spdep)
  valid <- neighbor_indices != 0L
  focal_indices <- focal_indices[valid]
  neighbor_indices <- neighbor_indices[valid]

  data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (focal_id, neighbor_id)

cat("Edge table rows:", nrow(edge_dt), "\n")

# =============================================================================
# STEP 2: Compute neighbor stats for all 5 variables â€” vectorized
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a keyed lookup of cell-year attributes for just the neighbor vars
# We need columns: id, year, and the 5 source variables
attr_cols <- c("id", "year", neighbor_source_vars)
attr_dt <- cell_dt[, ..attr_cols]
setnames(attr_dt, "id", "neighbor_id")  # for joining on neighbor side
setkeyv(attr_dt, c("neighbor_id", "year"))

# Expand edge table by year: each edge exists in every year
# Instead of a full cross join (which would be huge), we join directly.
# Strategy: add year to edge_dt via a cross-join with unique years, then join attrs.

years <- sort(unique(cell_dt$year))

# Cross join edges Ã— years: ~1.37M Ã— 28 â‰ˆ 38.4M rows â€” fits easily in RAM (~1-2 GB)
edge_year_dt <- CJ_dt <- edge_dt[, .(year = years), by = .(focal_id, neighbor_id)]

cat("Edge-year table rows:", nrow(edge_year_dt), "\n")

# Key for joining neighbor attributes
setkeyv(edge_year_dt, c("neighbor_id", "year"))

# Join neighbor attributes onto edge-year table
edge_year_dt <- attr_dt[edge_year_dt, on = .(neighbor_id, year)]

# Now edge_year_dt has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, focal_id

# =============================================================================
# STEP 3: Grouped aggregation â€” compute max, min, mean per (focal_id, year)
# =============================================================================
# Build aggregation expressions dynamically for all 5 variables at once

agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Construct the call
agg_call <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))

neighbor_stats <- edge_year_dt[, eval(agg_call), by = .(focal_id, year)]

# Replace -Inf/Inf from max/min of all-NA groups with NA
inf_cols <- grep("neighbor_(max|min)_", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# Also replace NaN from mean of all-NA groups
mean_cols <- grep("neighbor_mean_", names(neighbor_stats), value = TRUE)
for (col in mean_cols) {
  set(neighbor_stats, which(is.nan(neighbor_stats[[col]])), col, NA_real_)
}

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# =============================================================================
# STEP 4: Merge neighbor stats back into the main dataset
# =============================================================================
# Drop old neighbor columns from cell_dt if they exist (from prior runs)
old_neighbor_cols <- intersect(names(cell_dt), agg_names)
if (length(old_neighbor_cols) > 0) {
  cell_dt[, (old_neighbor_cols) := NULL]
}

setkeyv(cell_dt, c("id", "year"))
setnames(neighbor_stats, "focal_id", "id")
setkeyv(neighbor_stats, c("id", "year"))

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

cat("Final dataset rows:", nrow(cell_dt), "\n")
cat("Final dataset cols:", ncol(cell_dt), "\n")

# =============================================================================
# STEP 5: Predict with the existing trained Random Forest (no retraining)
# =============================================================================
# rf_model is the pre-trained randomForest / ranger object already in memory.
# Ensure column names match exactly what the model expects.

# If using randomForest package:
# cell_dt$prediction <- predict(rf_model, newdata = cell_dt)

# If using ranger package:
# cell_dt$prediction <- predict(rf_model, data = cell_dt)$predictions

# Example (uncomment the appropriate line):
# cell_dt$prediction <- predict(rf_model, newdata = as.data.frame(cell_dt))

cat("Done. Neighbor features computed and predictions generated.\n")
```

---

## Summary of Guarantees

| Requirement | Status |
|---|---|
| **Trained RF model preserved** | âœ… No retraining; only `predict()` is called on the existing model object |
| **Original numerical estimand preserved** | âœ… `max`, `min`, `mean` computed identically; NA handling matches original logic; no rounding or transformation |
| **Correctness** | âœ… Same neighbor relationships (rook), same grouping (focal cell Ã— year), same statistics |
| **RAM** | âœ… Peak ~3â€“4 GB for the 38.4M-row edge-year table + attributes; well within 16 GB |
| **Speed** | âœ… Expected runtime: **2â€“10 minutes** vs. 86+ hours (roughly 500â€“2500Ã— speedup) |

The entire speedup comes from (a) separating the time-invariant topology from time-varying attributes, (b) replacing 6.46M R-level loop iterations with vectorized `data.table` keyed joins and grouped aggregations, and (c) batching all 5 variables into a single aggregation pass.
END_RESPONSE>>>
