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
run_id: Anthropic_F09_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(nÂ²)-like in practice
The function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** via `paste()` for every neighbor Ã— year combination.
- Performs **named vector lookups** (`idx_lookup[neighbor_keys]`) â€” named vector lookup in R is hash-based but still involves repeated character hashing across millions of calls.

With ~1.37 million directed neighbor relationships spread across 28 years, this produces roughly **38.4 million character key constructions and hash lookups** inside a serial `lapply`. The overhead of character allocation, garbage collection, and hash collisions on a 6.46M-entry named vector is enormous.

### 2. The lookup is **rebuilt identically for every run** despite the neighbor topology being static
The spatial neighbor structure (`rook_neighbors_unique`) never changes across years. Yet the current code entangles the spatial topology with the year dimension, building a single monolithic 6.46M-entry lookup. This means the function cannot exploit the fact that **neighbor relationships are year-invariant**.

### 3. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even though the neighbor index vectors are pre-resolved, the stats computation loops in R over every row, calling `max`, `min`, `mean` with subsetting and NA checks each time. This is slow for 6.46M iterations Ã— 5 variables = ~32.3 million R-level function calls.

---

## Optimization Strategy

**Core insight:** Separate the **spatial topology** (which cells are neighbors â€” static, 344K cells) from the **temporal attributes** (which values those cells have in a given year â€” varies by year). Then use vectorized `data.table` joins and grouped aggregations instead of row-level R loops.

### Step-by-step plan:

1. **Build a static edge table once** â€” a two-column `data.table` with columns `(id, neighbor_id)` representing all ~1.37M directed rook-neighbor pairs. This is built once from `rook_neighbors_unique` and `id_order`, costs seconds, and can be cached to disk.

2. **For each variable, join yearly attributes onto the edge table** â€” For a given variable (e.g., `ntl`), create a keyed `data.table` of `(id, year, value)`. Join `neighbor_id` to this table to get each neighbor's value for the same year. This is a vectorized equi-join â€” `data.table` handles millions of rows in seconds.

3. **Compute grouped aggregations** â€” Group by `(id, year)` and compute `max`, `min`, `mean` of neighbor values. This is a single vectorized `data.table` grouped operation over ~38.4M edge-year rows â€” extremely fast.

4. **Join results back** to the main `cell_data`.

**Expected speedup:** From ~86 hours to **minutes** (roughly 2â€“10 minutes total depending on I/O).

**Preservation guarantees:**
- The trained Random Forest model is untouched â€” we only change feature engineering.
- The numerical output (neighbor max, min, mean) is **identical** â€” same topology, same aggregation functions, same NA handling.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the static spatial edge table (run once, cache)
# ==============================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # id_order: vector of cell IDs in the order matching the nb object

# neighbors_nb: spdep nb object (list of integer index vectors)
  edges <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  setkey(edges, neighbor_id)
  edges
}

# Build once â€” takes seconds for 1.37M edges
edge_table <- build_edge_table(id_order, rook_neighbors_unique)

# Optional: save/load for reuse
# fst::write_fst(edge_table, "edge_table.fst")
# edge_table <- fst::read_fst("edge_table.fst", as.data.table = TRUE)


# ==============================================================
# STEP 2: Vectorized neighbor stats for one variable
# ==============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt:  data.table with columns id, year, and <var_name>
  # edge_dt:  data.table with columns id, neighbor_id (keyed on neighbor_id)

  # Extract the attribute column for joining
  attr_dt <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(attr_dt, neighbor_id, year)

  # Expand edges across all years by joining neighbor attributes
  # For each (id, neighbor_id) edge and each year, get the neighbor's value
  edge_year <- edge_dt[attr_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: id, neighbor_id, year, value

  # Compute grouped stats: for each (id, year), aggregate neighbor values
  stats <- edge_year[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Rename columns to match the variable
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  stats
}


# ==============================================================
# STEP 3: Main pipeline â€” compute all neighbor features and join
# ==============================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Define source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Loop over variables, compute stats, and join back
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")

  stats_dt <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  setkey(stats_dt, id, year)

  # Determine new column names
  new_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in new_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Left join stats back onto main table
  cell_data <- stats_dt[cell_data, on = .(id, year)]

  cat("  Done. Columns added:", paste(new_cols, collapse = ", "), "\n")
}

# Cells with no valid neighbors get NA (same as original behavior)
# Verify:
cat("Final dimensions:", nrow(cell_data), "x", ncol(cell_data), "\n")


# ==============================================================
# STEP 4: Predict with existing Random Forest (unchanged)
# ==============================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# The column names and numerical values are identical to the
# original pipeline output, so predictions are preserved exactly.
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory Consideration (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_table` (1.37M Ã— 2 cols) | ~11 MB |
| `edge_year` intermediate (max ~38.4M Ã— 4 cols) | ~1.2 GB |
| `stats` intermediate | ~200 MB |
| **Peak total** | **~8â€“9 GB** |

This fits within 16 GB. If memory is tight, process one variable at a time (as the loop already does) and call `gc()` between iterations.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor resolution | 6.46M `lapply` calls with character hashing | One-time 344K-row edge table build |
| Stats computation | 6.46M Ã— 5 = 32.3M R-level `lapply` calls | 5 vectorized `data.table` grouped aggregations |
| Estimated runtime | **86+ hours** | **2â€“10 minutes** |
| Numerical output | Neighbor max, min, mean | **Identical** |
| RF model | Unchanged | **Unchanged** |
END_RESPONSE>>>
