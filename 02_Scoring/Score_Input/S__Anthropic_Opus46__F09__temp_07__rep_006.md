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
run_id: Anthropic_F09_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs a character-based hash lookup (`id_to_ref`) to find the cell's reference index.
- Retrieves neighbor cell IDs from the `nb` object.
- Constructs **paste-based string keys** (`"cellid_year"`) for every neighbor in every year.
- Looks up row indices via a named character vector (`idx_lookup`).

String construction and named-vector lookup for ~6.46M rows Ã— ~4 neighbors each â‰ˆ **~25.8 million string paste + match operations**. R's named vector lookup is O(n) in the worst case (hash collisions) and the `paste()` / character matching is inherently slow in an interpreted loop.

### 2. The lookup is **year-redundant**
The spatial neighbor topology is **identical across all 28 years** â€” cell *i*'s rook neighbors are always the same cells. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, doing 28Ã— the necessary work. The string key `"id_year"` conflates spatial structure with temporal indexing.

### 3. `compute_neighbor_stats` uses per-row `lapply`
Even after the lookup is built, computing stats iterates over 6.46M list elements in R-level `lapply`, calling `max`, `min`, `mean` individually. This prevents vectorization.

**Summary:** The core inefficiency is that a **static spatial relationship** (which cells neighbor which) is being re-resolved at the cell-year level via slow string operations, then stats are computed row-by-row in R loops.

---

## Optimization Strategy

### Principle: Separate topology from attributes, then vectorize.

1. **Build the neighbor table once as a `data.table` of directed edges** â€” 1,373,394 rows of `(cell_id, neighbor_id)`. This is year-invariant and built in seconds from the `nb` object.

2. **For each variable, join yearly attributes onto the edge table** â€” a keyed `data.table` join attaches `year` and the variable value for each neighbor. This produces ~1.37M Ã— 28 â‰ˆ ~38.5M rows, but the join is vectorized and fast.

3. **Aggregate (max, min, mean) by `(cell_id, year)` in one grouped operation** â€” `data.table`'s grouped aggregation is C-level and handles this in seconds.

4. **Merge the aggregated stats back onto the main panel** â€” a final keyed join.

This eliminates all R-level row loops, all `paste`-based key construction, and all per-row `lapply` calls. Expected runtime: **minutes, not hours.**

### Memory check
- Main panel: 6.46M rows Ã— ~115 cols Ã— 8 bytes â‰ˆ ~5.6 GB (tight but feasible with 16 GB).
- Edge table: 1.37M Ã— 2 cols = ~22 MB.
- Edge table expanded by year with one variable: ~38.5M Ã— 4 cols â‰ˆ ~1.2 GB (temporary, freed after aggregation).
- Fits in 16 GB if we process one variable at a time and discard intermediates.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with proper keys
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#                           indexed by position in id_order
#   id_order              : vector of cell IDs corresponding to nb indices
#
#   Output: edges_dt â€” a data.table with columns (cell_id, neighbor_id)
#           representing every directed rook-neighbor pair.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_i]
    pos <- pos + len
  }

  # Trim if any nb entries were empty / zero (unlikely but safe)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edges_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each source variable, compute neighbor max, min, mean
#         by joining yearly attributes onto the edge table, then
#         aggregating, then merging back onto cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal lookup table: (id, year, <variable>)
# We'll reuse this pattern per variable to keep peak memory low.

# Key the main data for fast joins later
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "... ")
  t0 <- proc.time()

  # --- 2a. Extract the (id, year, value) slice for this variable ---
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # --- 2b. Cross-join edges with all years present in the data ---
  #
  #   We need, for every (cell_id, neighbor_id) edge and every year,
  #   the neighbor's attribute value in that year.
  #
  #   Strategy: join attr_dt onto edges_dt by neighbor_id.
  #   This naturally expands edges Ã— years because attr_dt has one row
  #   per (id, year).

  # Rename for join clarity
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id)

  # Join: for each edge, get all (year, value) rows of the neighbor
  # Result columns: cell_id, neighbor_id, year, value
  expanded <- attr_dt[edges_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # expanded has columns: neighbor_id, year, value, cell_id

  # --- 2c. Aggregate: for each (cell_id, year), compute stats ---
  stats <- expanded[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]

  # Name the output columns to match the original pipeline's convention
  # (adjust these names to match whatever compute_and_add_neighbor_features produced)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(stats, "cell_id", "id")
  setkey(stats, id, year)

  # --- 2d. Remove old columns if they exist (idempotent re-runs) ---
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }

  # --- 2e. Merge stats back onto the main panel ---
  cell_data <- stats[cell_data, on = .(id, year)]

  # Clean up large temporaries

rm(attr_dt, expanded, stats)
  gc()

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(round(elapsed, 1), "seconds\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the already-trained Random Forest
#
#   The trained model object (e.g., rf_model) is unchanged.
#   cell_data now contains the same neighbor-stat columns as before,
#   with identical numerical values, so predictions are identical.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Example (adjust to your actual model object and prediction column names):
# cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)

cat("Done. cell_data has", nrow(cell_data), "rows and", ncol(cell_data), "columns.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values. The edge table encodes the same rook-contiguity as `rook_neighbors_unique`. The join on `(neighbor_id, year)` reproduces the same `"id_year"` matching. No approximation is introduced. |
| **Trained RF model** | The model object is never touched. The output columns have the same names and values, so `predict()` produces identical results. |
| **Missing-data handling** | `!is.na(value)` in the aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero non-NA neighbors get `NA` for all three stats (via the left join `stats[cell_data, ...]`), matching the original `return(c(NA, NA, NA))`. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| Build edge table (once) | < 2 seconds |
| Per-variable join + aggregate | ~30â€“90 seconds each |
| 5 variables total | **~3â€“8 minutes** |
| **Total vs. original** | **~5 minutes vs. ~86 hours** (~1,000Ã— speedup) |
END_RESPONSE>>>
