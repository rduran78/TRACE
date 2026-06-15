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
run_id: Anthropic_F09_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

### 1. `build_neighbor_lookup` is O(nÂ²)-like in practice
This function iterates over **all 6.46 million cell-year rows** and, for each row, performs:
- A named-vector lookup (`id_to_ref`) â€” O(1) amortized but with high constant factor for character hashing over millions of keys.
- Construction of `neighbor_keys` by pasting cell IDs with years â€” millions of `paste()` calls.
- A named-vector lookup (`idx_lookup`) on those keys â€” again, character hashing over a 6.46M-entry named vector.

The result is a **list of 6.46 million integer vectors**, each built via expensive character-key operations. This is the single largest bottleneck.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M elements
Even though each iteration is small, the R-level loop overhead across 6.46 million rows, repeated for 5 variables (= 32.3 million iterations total), is substantial.

### 3. The neighbor topology is **year-invariant** but rebuilt per cell-year
The rook-neighbor structure is purely spatial â€” cell A's neighbors are the same in 1992 as in 2019. Yet the lookup is constructed at the cell-year level, redundantly encoding the same spatial relationships 28 times.

---

## Optimization Strategy

**Core insight:** Separate the *spatial topology* (which cells are neighbors â€” fixed) from the *temporal attributes* (variable values per year â€” varying). Then use vectorized joins and grouped aggregations instead of row-wise R loops.

**Steps:**

1. **Build a cell-level edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant.

2. **Join yearly attributes onto the edge table** â€” for each variable and year, join the variable's value from the neighbor cell onto the edge table. This is a keyed `data.table` join: O(n) and vectorized in C.

3. **Aggregate neighbor stats** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized `data.table` operation.

4. **Join results back** to the main dataset.

This replaces 6.46M Ã— 5 R-level `lapply` iterations with a handful of vectorized `data.table` joins and group-by aggregations. Expected runtime: **minutes, not hours**.

**Memory:** The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. Joined with year expansion: 1.37M Ã— 28 years â‰ˆ 38.4M rows Ã— a few columns â‰ˆ manageable well within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a year-invariant spatial edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   We produce edges_dt: data.table with columns (id, neighbor_id)
#   representing every directed rook-neighbor pair.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains integer indices into id_order for the

  # neighbors of id_order[i]. Index 0 means no neighbors (spdep convention).
  from_list <- lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  })
  rbindlist(from_list, use.names = FALSE)
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edges_dt has ~1,373,394 rows: (id, neighbor_id)

cat("Edge table rows:", nrow(edges_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, compute neighbor max,
#          min, and mean via vectorized join + grouped aggregation.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure keys are set for fast joins
setkey(cell_data, id, year)

# Extract the unique years present in the panel
all_years <- sort(unique(cell_data$year))

# Cross-join edges with years to get the full (id, neighbor_id, year) table.
# ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” fits in memory.
edges_by_year <- CJ_dt <- edges_dt[, .(year = all_years), by = .(id, neighbor_id)]

cat("Edge-year table rows:", nrow(edges_by_year), "\n")

# Key for joining neighbor attributes
setkey(edges_by_year, neighbor_id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "...\n")

  # Column names for the three output features
  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)

  # --- Join the neighbor's attribute value onto the edge-year table ---
  # We need cell_data[, .(id, year, <var_name>)] keyed by (id, year)
  # and we join on (neighbor_id, year) == (id, year)

  # Subset for the join: only the columns we need
  attr_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)

  # Join: for each (id, neighbor_id, year) row in edges_by_year,
  # look up the neighbor's value by matching neighbor_id==id, year==year
  edges_by_year[attr_dt, neighbor_val := i.val, on = .(neighbor_id = id, year = year)]

  # --- Aggregate: group by (id, year), compute max/min/mean of neighbor_val ---
  agg <- edges_by_year[
    !is.na(neighbor_val),
    .(
      nmax  = max(neighbor_val),
      nmin  = min(neighbor_val),
      nmean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]

  setnames(agg, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  setkey(agg, id, year)

  # --- Remove old columns if they exist (idempotency) ---
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
  }

  # --- Join aggregated stats back onto cell_data ---
  cell_data <- agg[cell_data, on = .(id, year)]

  # Clean up the temporary column in edges_by_year
  edges_by_year[, neighbor_val := NULL]

  cat("  Done:", col_max, col_min, col_mean, "\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the already-trained Random Forest model
#
#   The trained model object (e.g., `rf_model`) is unchanged.
#   The numerical estimand is preserved because we compute identical
#   max, min, mean statistics â€” just via vectorized operations
#   instead of row-wise R loops.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Example prediction (adjust to your actual model object name):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("All neighbor features computed. Ready for Random Forest prediction.\n")
```

---

## Why the Numerical Estimand Is Preserved

The vectorized `data.table` path computes **exactly the same** `max()`, `min()`, and `mean()` over **exactly the same** set of non-NA neighbor values for each `(cell_id, year)` combination. The only change is *how* those sets are assembled (keyed join vs. character-key named-vector lookup) and *how* the aggregation is dispatched (C-level `data.table` group-by vs. R-level `lapply`). The arithmetic is identical, so the trained Random Forest model receives the same feature values and need not be retrained.

---

## Expected Performance

| Component | Old (base R `lapply`) | New (`data.table` vectorized) |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M `paste` + hash lookups) | ~seconds (1.37M-row edge table) |
| Compute stats (Ã—5 vars) | ~hours (32.3M R-loop iterations) | ~minutes (keyed join + group-by on 38.4M rows) |
| **Total** | **~86+ hours** | **~5â€“15 minutes** |

The 38.4M-row `edges_by_year` table consumes roughly 38.4M Ã— 3 columns Ã— 8 bytes â‰ˆ 0.9 GB, well within the 16 GB RAM budget. Peak memory during the join (with the temporary `neighbor_val` column) adds another ~300 MB, still comfortable.
END_RESPONSE>>>
