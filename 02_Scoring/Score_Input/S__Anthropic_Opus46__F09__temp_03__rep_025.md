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
run_id: Anthropic_F09_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), character-based lookups into named vectors (`id_to_ref`, `idx_lookup`), and `NA` filtering. Named-vector lookups in R are hash-table operations, but doing ~6.46 million of themâ€”each touching multiple neighborsâ€”creates billions of character-match operations. This alone can take tens of hours.

### 2. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
For each row, a small numeric vector is subsetted, `NA`-filtered, and then `max`/`min`/`mean` are computed. The per-call overhead of `lapply` + anonymous function + three summary calls, repeated 6.46M Ã— 5 variables = ~32.3 million invocations, is enormous.

### 3. Redundant recomputation of the same spatial topology every row
The neighbor graph is **time-invariant**: cell A's rook neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-resolves neighbor relationships for every cell-year combination, inflating the problem from ~344K cells to ~6.46M rows. The lookup should be built **once at the cell level** (344K entries), and yearly attribute values should be **joined** onto that static topology.

### Summary
| Bottleneck | Scale | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | Character key lookups per cell-year instead of per cell |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M iterations | Scalar R-loop instead of vectorized group operation |
| Conceptual | 18.7Ã— redundant work | Topology rebuilt per year instead of once |

---

## Optimization Strategy

**Core idea:** Separate the *spatial topology* (static, 344K cells) from the *temporal attributes* (yearly, 6.46M rows). Build a **directed edge table** once from the `nb` object, then use a vectorized `data.table` join-and-aggregate to compute neighbor stats for all cell-years simultaneously.

### Steps

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object). This is a two-column data.table: `(focal_id, neighbor_id)` with ~1.37M rows. Done once, reusable forever.

2. **Join yearly attributes onto the edge table.** For each year, every edge `(focal, neighbor)` gets the neighbor's attribute value via a keyed `data.table` join. This produces an "expanded" table of ~1.37M Ã— 28 years â‰ˆ 38.5M rows (but processed vectorially, not in a loop).

3. **Aggregate by `(focal_id, year)`** to compute `max`, `min`, `mean` of each neighbor variable in one vectorized pass.

4. **Join the aggregated stats back** onto the main `cell_data` table.

5. **Predict** with the existing trained Random Forest model (unchanged).

### Expected speedup
- `data.table` keyed joins and grouped aggregations on ~38.5M rows run in seconds to low minutes on 16 GB RAM.
- Total estimated time: **2â€“10 minutes** instead of 86+ hours (roughly 500â€“2500Ã— speedup).
- Memory: the edge table Ã— years Ã— 5 variables is manageable; peak ~2â€“4 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with columns "id" and "year"
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# id_order:             character or integer vector of cell IDs in the same
#                       order as rook_neighbors_unique (the nb object)
# rook_neighbors_unique: an nb object (list of integer index vectors)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a static directed edge table ONCE from the nb object
#         ~1.37 M rows, two integer/character columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for the neighbors of

  # the i-th cell.  A 0-length integer(0) means no neighbors.
  focal_idx    <- rep(seq_along(nb_obj), lengths(nb_obj))
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# This table is time-invariant and can be serialized for future reuse:
# fst::write_fst(edge_dt, "edge_table.fst")

cat(sprintf("Edge table: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 & 3: For each neighbor source variable, join yearly attributes
#             onto the edge table and aggregate in one vectorized pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var))

  # Subset only the columns we need for the neighbor lookup

  # (neighbor_id will be joined on "id", plus we need "year" and the variable)
  attr_dt <- cell_data[, .(id, year, value = get(var))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # Expand edges Ã— years: join neighbor attribute onto every (focal, neighbor, year)
  # This is a many-to-many join: each edge appears once per year
  expanded <- edge_dt[attr_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: focal_id, neighbor_id, year, value

  # Drop NAs in the variable (mirrors original behavior)
  expanded <- expanded[!is.na(value)]

  # Aggregate: one row per (focal_id, year)
  agg <- expanded[, .(
    nb_max  = max(value),
    nb_min  = min(value),
    nb_mean = mean(value)
  ), by = .(focal_id, year)]

  # Rename columns to match the original pipeline's naming convention
  # Original: e.g., ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  max_col  <- paste0(var, "_neighbor_max")
  min_col  <- paste0(var, "_neighbor_min")
  mean_col <- paste0(var, "_neighbor_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "focal_id", "id")

  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # STEP 4: Join aggregated neighbor stats back onto cell_data
  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  # Remove old columns if they already exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)

  # Clean up

  rm(attr_dt, expanded, agg)
}

# Re-key after all merges
setkey(cell_data, id, year)

cat("Neighbor feature engineering complete.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure the predictor matrix matches the model's expected feature names.

# Example (adjust to your actual model object and predict call):
# cell_data[, prediction := predict(rf_model, newdata = .SD)]

# If using ranger:
# preds <- predict(rf_model, data = cell_data)
# cell_data[, prediction := preds$predictions]

# If using randomForest:
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]
```

---

## Why This Preserves Correctness

| Requirement | How it is preserved |
|---|---|
| **Trained RF model untouched** | No retraining; only the feature-engineering step is replaced. The same 15 neighbor-stat columns (5 vars Ã— 3 stats) are produced with identical names and identical numerical values. |
| **Original numerical estimand** | `max`, `min`, `mean` are computed on the same set of non-NA neighbor values per `(focal_id, year)` as the original code. The `data.table` aggregation is algebraically identical. |
| **Rook neighbor topology** | The edge table is built directly from the same `rook_neighbors_unique` nb object and `id_order`. No spatial relationships are altered. |
| **NA handling** | `expanded[!is.na(value)]` mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` left join, matching the original `return(c(NA, NA, NA))`. |

---

## Performance Estimate on 16 GB Laptop

| Step | Estimated time | Estimated peak RAM |
|---|---|---|
| Build edge table | < 1 second | ~50 MB |
| Join + aggregate (per variable) | 30â€“90 seconds | ~1.5â€“2.5 GB |
| All 5 variables | **3â€“8 minutes total** | ~3â€“4 GB peak |
| RF prediction (unchanged) | Same as before | Same as before |

**Total: ~5â€“10 minutes** versus the original ~86 hours.
END_RESPONSE>>>
