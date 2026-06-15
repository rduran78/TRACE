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
run_id: Anthropic_F09_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (estimated 86+ hours) because of three compounding problems:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), character-based hash lookups into `idx_lookup`, and subsetting. With ~6.46 million iterations, the overhead of R's interpreted `lapply` loop, repeated `paste` calls, and named-vector lookups (which are O(n) hash probes on a 6.46M-entry character vector) is enormous.

### 2. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
For each row, it subsets a numeric vector by variable-length index vectors, removes NAs, and computes `max/min/mean`. This is called **5 times** (once per neighbor source variable), giving ~32.3 million interpreted R iterations total.

### 3. The neighbor topology is **year-invariant** but is being rebuilt per cell-year
The rook-neighbor structure is purely spatial â€” cell A's neighbors don't change from year to year. Yet the current code builds a lookup that maps each of the 6.46M cell-year rows to its neighbor cell-year rows. This inflates the problem by a factor of 28 (the number of years). The topology only needs to be expressed once over the 344,208 cells.

---

## Optimization Strategy

**Core insight:** Separate the *time-invariant spatial topology* from the *time-varying attributes*, then use vectorized joins instead of row-level loops.

### Step-by-step plan:

1. **Build a cell-level edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_cell_id)` from `rook_neighbors_unique`. This has ~1.37M rows and never changes.

2. **For each variable, join yearly attributes onto the edge table** â€” by joining `cell_data[, .(cell_id, year, value)]` onto the edge table by `neighbor_cell_id` and `year`, every neighbor's attribute value is attached in one vectorized merge.

3. **Aggregate with `data.table` grouping** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` in one pass. This replaces millions of `lapply` iterations with a single vectorized `data.table` aggregation.

4. **Join the aggregated stats back** to `cell_data`.

**Complexity reduction:**
- The edge table has ~1.37M rows. After joining with 28 years, the working table is ~1.37M Ã— 28 â‰ˆ 38.4M rows â€” large but manageable in RAM and extremely fast to aggregate with `data.table`.
- No per-row R-level loops. Everything is vectorized C-level code inside `data.table`.

**Expected speedup:** From 86+ hours to **minutes** (typically 2â€“10 minutes total depending on I/O).

**Preservation guarantees:**
- The trained Random Forest model is not retouched.
- The numerical results (neighbor max, min, mean) are identical to the originals â€” same rook topology, same aggregation functions.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the time-invariant cell-level edge table (once)
# ==============================================================================
# rook_neighbors_unique : an spdep nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object
#
# This produces a data.table with columns: cell_id, neighbor_id
# Approximately 1,373,394 rows (directed neighbor pairs)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # spdep nb objects use 0L to denote "no neighbors"
    sum(x != 0L)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb <- neighbors_nb[[i]]
    nb <- nb[nb != 0L]
    if (length(nb) > 0L) {
      n        <- length(nb)
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb]
      pos      <- pos + n
    }
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Edge table built: %s directed neighbor pairs across %s cells.\n",
  format(nrow(edge_table), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor stats via join
# ==============================================================================
# This function:
#   - Extracts (id, year, var_value) from cell_data
#   - Joins onto edge_table so each edge row gets the neighbor's value for that year
#   - Aggregates max/min/mean by (cell_id, year)
#   - Joins the result back onto cell_data
#
# Naming convention preserved: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean

compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  # Remove old columns if they exist (to allow re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) {
      cell_dt[, (col) := NULL]
    }
  }

  # Extract the attribute slice: neighbor cell values keyed by (id, year)
  attr_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(attr_dt, neighbor_id, year)

  # Join neighbor attribute onto edge table Ã— year
  # edge_dt has (cell_id, neighbor_id); we need to cross with years
  # More efficient: join edge_dt to attr_dt directly
  # For each (cell_id --neighbor_id) edge, pull the neighbor's value for each year
  # Result: one row per (cell_id, neighbor_id, year) with the neighbor's value

  work <- merge(
    edge_dt,
    attr_dt,
    by = "neighbor_id",
    allow.cartesian = TRUE  # each neighbor_id has 28 year-rows
  )
  # work columns: neighbor_id, cell_id, year, val

  # Aggregate by (cell_id, year)
  stats <- work[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(cell_id, year)
  ]

  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           c(col_max, col_min, col_mean))

  # Join back onto cell_data
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  return(cell_dt)
}

# ==============================================================================
# STEP 4: Outer loop â€” compute for all 5 neighbor source variables
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edge_table, var_name)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat(sprintf("  %-15s done in %.1f seconds\n", var_name, elapsed))
}
cat("All neighbor features computed.\n")

# ==============================================================================
# STEP 5: Predict with the already-trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) and its expected feature names
# are exactly preserved. No retraining occurs.
#
# Example (adjust to your actual model object and prediction call):
#
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data[, predicted := predictions]
```

---

## Memory Note (16 GB laptop)

The largest intermediate object is the `work` table inside the merge: ~1.37M edges Ã— 28 years = ~38.4M rows Ã— 4 columns â‰ˆ **1.2 GB** at most. This is created and discarded once per variable, so peak memory stays well under 16 GB. If memory is tight, the loop already processes one variable at a time, so only one `work` table exists at any moment. You can add an explicit `rm(work); gc()` inside the function if needed.

---

## Summary

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup granularity | Per cell-year (6.46M R-loop iterations) | Per cell (344K rows, vectorized) |
| Stats computation | 5 Ã— 6.46M `lapply` iterations | 5 Ã— one `data.table` grouped aggregation |
| Key data structure | Named character vector (6.46M entries) | Keyed `data.table` join |
| Estimated wall time | 86+ hours | ~2â€“10 minutes |
| Trained RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical results | Original estimand | Identical (same topology, same functions) |
END_RESPONSE>>>
