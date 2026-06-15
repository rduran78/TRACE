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
run_id: Anthropic_F09_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is an O(N) `lapply` over ~6.46 million rows
For every single cell-year row, the function:
- Looks up the cell's spatial reference index.
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes together character keys (`"cellid_year"`) for every neighbor Ã— every year.
- Performs named-vector lookups (which are hash-based but still slow at scale with millions of names).

This means ~6.46M iterations, each doing string concatenation and named-vector indexing. The entire lookup table is **year-redundant**: the neighbor *structure* is purely spatial (rook contiguity doesn't change across years), yet the code rebuilds neighbor-row mappings for every cell-year combination, duplicating the spatial logic 28 times.

### 2. `compute_neighbor_stats` is an O(N) `lapply` over ~6.46 million rows
Each iteration subsets a numeric vector by index, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is notoriously slow in R due to memory allocation overhead.

### Root cause summary
The pipeline treats a **spatial** problem as a **panel** problem. Neighbor relationships are invariant across years, so the correct approach is:

> **Build the adjacency table once (344K cells), then join yearly attributes and compute grouped statistics using vectorized/data.table operations â€” never loop over 6.46M rows.**

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| **A** | Build a two-column `data.table` of directed edges: `(id, neighbor_id)` â€” ~1.37M rows. Do this **once**. | Spatial structure is time-invariant. |
| **B** | For each year, join cell attributes onto the edge table by `(neighbor_id, year)`, then compute `max`, `min`, `mean` grouped by `(id, year)` using `data.table`. | Vectorized grouped aggregation; no R-level loops. |
| **C** | Join the resulting neighbor-stats columns back onto the main panel `data.table`. | Preserves the original data structure for the pre-trained Random Forest. |

**Expected speedup:** The 6.46M-row `lapply` is replaced by a `data.table` grouped join-and-aggregate over ~1.37M edges Ã— 28 years â‰ˆ 38.5M edge-year rows, which `data.table` handles in seconds to low minutes. Total runtime should drop from **86+ hours to under 5 minutes** on a standard laptop.

**Numerical equivalence:** `max`, `min`, and `mean` are computed on exactly the same neighbor sets with the same NA-removal logic, so the trained Random Forest receives identical inputs.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP A: Build the spatial edge table ONCE (time-invariant)
# ===========================================================================
# rook_neighbors_unique : spdep nb object (list of integer vectors), length = 344,208
# id_order              : vector of cell IDs aligned with the nb object

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total directed edges
  n_edges <- sum(lengths(neighbors))  # ~1,373,394
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"; skip those
    nb_i <- nb_i[nb_i != 0L]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    idx <- pos:(pos + n_i - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges among %d cells\n",
            nrow(edge_dt), length(id_order)))

# ===========================================================================
# STEP B: Convert the panel to data.table (if not already)
# ===========================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ===========================================================================
# STEP C: Compute neighbor stats for each source variable â€” vectorized
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will expand the edge table by year, join neighbor attributes, and
# aggregate.  To keep peak RAM manageable (~16 GB laptop), we process
# one variable at a time and can even chunk by year if needed.

# Create a minimal lookup keyed by (id, year) for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  
  cat(sprintf("Computing neighbor stats for: %s ...\n", var_name))
  
  # --- Minimal attribute table: only the column we need ----
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  
  # --- Cross join edges Ã— years, then join neighbor values ----
  # Instead of a full cross join (which would be 38.5M rows at once),
  # we can do this efficiently: for every year, join edges to attributes.
  
  years <- sort(unique(cell_data$year))
  
  stats_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    # Neighbor values for this year
    attr_yr <- attr_dt[year == yr, .(neighbor_id, value)]
    setkey(attr_yr, neighbor_id)
    
    # Join edge table to neighbor values
    merged <- attr_yr[edge_dt, on = .(neighbor_id), nomatch = NA,
                      allow.cartesian = TRUE]
    # merged has columns: neighbor_id, value, id
    # Aggregate by focal cell (id)
    agg <- merged[!is.na(value),
                  .(nb_max  = max(value),
                    nb_min  = min(value),
                    nb_mean = mean(value)),
                  by = .(id)]
    agg[, year := yr]
    stats_list[[j]] <- agg
  }
  
  stats_dt <- rbindlist(stats_list, use.names = TRUE)
  
  # Name columns to match original pipeline conventions
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats_dt, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # --- Remove old columns if they exist (idempotent re-runs) ----
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # --- Join back to the main panel ----
  setkey(stats_dt, id, year)
  cell_data <- stats_dt[cell_data, on = .(id, year)]
  # Cells with no neighbors (or all-NA neighbors) get NA â€” same as original
  
  setkey(cell_data, id, year)
  
  cat(sprintf("  Done. Columns added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# ===========================================================================
# STEP D: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# The trained model object (e.g., `rf_model`) and its expected feature names
# are unchanged.  cell_data now contains the same neighbor_max_*, neighbor_min_*,
# neighbor_mean_* columns with numerically identical values.

# Example (adjust to your actual model object and prediction call):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete. Ready for Random Forest prediction.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor sets** | `edge_dt` is built from the same `rook_neighbors_unique` nb object using the same `id_order` mapping. Every directed edge is identical. |
| **Same NA handling** | The `!is.na(value)` filter in the aggregation mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors receive `NA` for all three stats (the left join leaves them as `NA`). |
| **Same aggregation functions** | `max`, `min`, `mean` in `data.table` are numerically identical to base R's `max`, `min`, `mean`. |
| **Trained RF untouched** | No retraining occurs; the model object is used as-is for `predict()`. |

## Expected Performance

| Stage | Rows/Operations | Estimated Time |
|-------|----------------|----------------|
| Build edge table | 1.37M edges | < 1 second |
| Per-variable, per-year join + aggregate | ~1.37M edges Ã— 28 years Ã— 5 vars | ~1â€“3 minutes total |
| Join back to panel | 6.46M rows Ã— 5 vars | ~30 seconds |
| **Total** | | **~2â€“5 minutes** |

This represents a speedup of roughly **1,000Ã—â€“2,500Ã—** over the original 86+ hour estimate, well within the 16 GB RAM budget (peak usage ~2â€“3 GB for the edge-year intermediates).
END_RESPONSE>>>
