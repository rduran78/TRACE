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
run_id: Anthropic_F09_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all 6.46M cell-year rows
For every single cell-year row, it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste(neighbor_id, year)` keys and matches them against a named vector of 6.46M entries. Named-vector lookup in R is hash-based per call, but doing this 6.46M times with variable-length key vectors creates enormous overhead.
- The result is a **list of 6.46M integer vectors**, each containing row indices of that cell-year's neighbors in the full panel. This list itself is enormous in memory.

**The core inefficiency:** The spatial neighbor topology is **year-invariant** â€” cell 42's neighbors are the same in 1992 as in 2019. Yet the lookup rebuilds neighbor relationships at the cell-year level, exploding the problem by a factor of 28 (the number of years).

### 2. `compute_neighbor_stats` iterates over 6.46M list elements in R
`lapply` over 6.46M elements, each extracting a subset of a numeric vector and computing `max/min/mean`, is inherently slow in interpreted R. This is called 5 times (once per source variable), totaling ~32.3M R-level function invocations.

### 3. Memory pressure
The `neighbor_lookup` list has 6.46M elements. Each element is an integer vector of ~4 neighbors (rook). That's ~6.46M list entries Ã— overhead â‰ˆ several GB just for the list structure, straining a 16 GB laptop.

---

## Optimization Strategy

**Key insight:** Separate the **time-invariant spatial topology** from the **time-varying attributes**.

1. **Build the adjacency table once** as a two-column `data.table` of `(cell_id, neighbor_cell_id)` â€” only ~1.37M rows (the directed rook-neighbor pairs). This is tiny and reusable.

2. **For each variable, join yearly attributes onto this table** by `(neighbor_cell_id, year)`, then group-by `(cell_id, year)` to compute `max`, `min`, `mean`. This is a classic `data.table` equi-join + grouped aggregation â€” highly optimized in C, vectorized, and cache-friendly.

3. **No R-level loops over 6.46M rows.** Everything is vectorized via `data.table`.

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes 5 keyed joins on ~1.37M Ã— 28 â‰ˆ 38.4M rows, plus grouped aggregation on 6.46M groups â€” all in-memory columnar operations).

**Preservation guarantees:**
- The trained Random Forest model is untouched (we only compute the same input features).
- The numerical estimand is identical: for each cell-year, neighbor max/min/mean of each variable are computed over the same rook-neighbor set with the same NA handling.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the time-invariant spatial adjacency table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   We produce a data.table with columns:  id, neighbor_id
#   containing every directed rook-neighbor pair (~1.37M rows).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_adjacency_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors
  # Expand into a long edge list
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no neighbors" sentinel (integer(0) produces nothing via

  # unlist, but nb objects sometimes store 0L as a sentinel)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” trivially small

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each source variable, join + aggregate to produce
#         neighbor_max, neighbor_min, neighbor_mean
#         then merge back onto cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-set the key on cell_data for fast repeated joins
# (id, year) is the natural key for the panel
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Computing neighbor features for:", var_name, "\n")

  # --- 2a. Extract the minimal lookup table: (id, year, value) ----------
  #     We only need the column we're aggregating over.
  val_table <- cell_data[, .(id, year, value = get(var_name))]
  setkey(val_table, id, year)

  # --- 2b. Join neighbor attribute values onto the adjacency table ------
  #     For every (id, neighbor_id) pair, cross with every year,
  #     and look up the neighbor's value in that year.
  #
  #     Efficient approach: expand adj_table Ã— years via a merge with

  #     val_table keyed on (neighbor_id, year).

  # Rename for the join: we want to look up by (neighbor_id, year)
  setnames(val_table, "id", "neighbor_id")
  setkey(val_table, neighbor_id, year)

  # This join attaches (year, value) to every edge â€” result has
  # nrow(adj_table) Ã— n_years rows â‰ˆ 1.37M Ã— 28 â‰ˆ 38.4M rows

  # but data.table handles this very efficiently.
  edge_vals <- val_table[adj_table, on = "neighbor_id", allow.cartesian = TRUE,
                         nomatch = NA]
  # edge_vals columns: neighbor_id, year, value, id
  # Each row = "cell <id> has neighbor <neighbor_id> in <year> with value <value>"

  # --- 2c. Aggregate: group by (id, year) â†’ max, min, mean of value ----
  #     NA handling: na.rm = TRUE mirrors the original code which filters NAs
  #     before computing stats; groups with all-NA neighbors â†’ NA.
  agg <- edge_vals[, .(
    nbr_max  = if (all(is.na(value))) NA_real_ else max(value, na.rm = TRUE),
    nbr_min  = if (all(is.na(value))) NA_real_ else min(value, na.rm = TRUE),
    nbr_mean = if (all(is.na(value))) NA_real_ else mean(value, na.rm = TRUE)
  ), by = .(id, year)]

  # --- 2d. Name the new columns to match the original pipeline ----------
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, c("nbr_max", "nbr_min", "nbr_mean"), new_names)

  # --- 2e. Merge back onto cell_data ------------------------------------
  setkey(agg, id, year)
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)

  # Clean up to free memory before next iteration
  rm(val_table, edge_vals, agg)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the already-trained Random Forest
#         (model object and predict call are unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data now contains all 110 predictor columns including the 15
# newly computed neighbor features (5 vars Ã— 3 stats), numerically
# identical to the original pipeline's output.
#
# Example (adjust to your actual model object and feature set):
# cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Identical

| Original code | Optimized code |
|---|---|
| For each cell-year row, find neighbor row indices via `paste(id, year)` key lookup | For each `(id, neighbor_id)` edge, join on `(neighbor_id, year)` â€” same pairing |
| Extract `vals[idx]`, drop NAs, compute `max/min/mean` | Group by `(id, year)`, compute `max/min/mean` with `na.rm=TRUE`; all-NA â†’ `NA` |
| Result: one `max`, `min`, `mean` per cell-year per variable | Identical |

The set of neighbor values aggregated for each `(cell, year)` is exactly the same, so the statistics are identical. The Random Forest model is never retrained â€” only its input features are recomputed (with the same values).

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup structure | 6.46M-element R list | 1.37M-row `data.table` (two int cols) |
| Per-variable computation | 6.46M R `lapply` iterations | One keyed join + one grouped aggregation |
| Estimated wall time | ~86+ hours | **~5â€“15 minutes** |
| Peak RAM (approx.) | 10â€“14 GB (list overhead) | ~4â€“6 GB (columnar `data.table`) |
| Numerical result | Baseline | Identical |
END_RESPONSE>>>
