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
run_id: Anthropic_F07_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The lookup is O(n) in list overhead** â€” `idx_lookup` is a named character vector of length ~6.46M. Named lookups on such a vector are O(n) or at best O(1)-amortized via hashing, but R's internal `match()` on named vectors rebuilds hash tables unpredictably.
3. **`compute_neighbor_stats`** then loops over the 6.46M-element list again, extracting values one-at-a-time. This is pure R-level iteration with no vectorization.

**Net effect:** ~6.46M R-level iterations Ã— (string ops + hash lookups + subsetting) â‰ˆ 86+ hours.

### Why the raster shortcut is unsafe
The document correctly notes that the cell topology may be irregular/masked. A naive `focal()` on a rectangular raster would compute neighbors for cells that don't exist in the panel or miss masked cells. The neighbor structure in `rook_neighbors_unique` (an `spdep::nb` object) is the ground truth and must be respected.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized / data.table operations:**

1. **Explode the neighbor graph into an edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_cell_id)` with ~1.37M rows. This is year-invariant.
2. **Cross-join with years vectorially** â€” instead of pasting keys 6.46M times, join `cell_data` to the edge table on `(neighbor_cell_id, year)` using `data.table` keyed joins. This is a single merge, fully vectorized in C.
3. **Compute grouped stats in one pass per variable** â€” `data.table`'s `[, .(max, min, mean), by=.(id, year)]` computes all three stats in one vectorized grouped aggregation.
4. **Memory:** The edge table Ã— 28 years â‰ˆ 1.37M Ã— 28 â‰ˆ 38.5M rows Ã— a few columns of integers/doubles â€” well within 16 GB.

**Expected speedup:** From 86+ hours to **minutes** (typically 5â€“15 min total for all 5 variables).

**Preservation guarantees:**
- The trained Random Forest model is untouched (we only rebuild feature columns with identical values).
- The numerical estimand is identical: same neighbor sets, same max/min/mean formulas, same NA handling.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Ensure cell_data is a data.table (non-destructive copy if needed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a vectorized edge table from the spdep::nb object (once)
#
#     rook_neighbors_unique : list of integer vectors (spdep nb object)
#     id_order              : vector mapping position -> cell id
#     Edge table columns    : focal_id, neighbor_id
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  n_edges <- sum(lengths(neighbors))
  focal_idx <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep zero-neighbor sentinel (integer(0) already handled

  # by lengths==0, but guard against 0L entries)
  valid <- neighbor_idx > 0L
  data.table(
    focal_id    = id_order[focal_idx[valid]],
    neighbor_id = id_order[neighbor_idx[valid]]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for one variable â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  # Columns we need from the neighbor side
  # Build a slim lookup: (id, year, value)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Join edges â†’ neighbor values.
  # For every (focal_id, year) we need the neighbor's value in the same year.
  # Strategy: cross the edge table with the value table on neighbor_id == id.
  # This is a keyed join â€” very fast.
  setkey(edge_dt, neighbor_id)
  merged <- edge_dt[val_dt,
    on = .(neighbor_id = id),
    .(focal_id, year, val),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # Drop NAs in the variable (mirrors original: neighbor_vals[!is.na()])
  merged <- merged[!is.na(val)]

  # Grouped aggregation
  stats <- merged[,
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(focal_id, year)
  ]

  # Rename to match original column naming convention
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))

  stats
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Outer loop â€” compute and attach features for every source variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "...\n")

  new_cols <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))

  # Drop old columns if they exist (safe re-run)
  for (col in new_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  setkeyv(stats_dt, c("focal_id", "year"))

  # Left-join back onto cell_data.  Rows with no neighbors get NA (correct).
  cell_data <- stats_dt[cell_data, on = .(focal_id = id, year)]

  # The join puts focal_id as the key; rename back to id
  setnames(cell_data, "focal_id", "id")
  setkey(cell_data, id, year)

  cat("  Done.\n")
}

cat("All neighbor features computed. Rows:", nrow(cell_data), "\n")
```

---

## Why This Is Correct (Estimand Preservation Proof)

| Original code | Optimized code | Equivalence |
|---|---|---|
| For row *i*, find all `neighbor_cell_ids` from `rook_neighbors_unique` | `edge_dt` encodes the identical directed pairs from the same `nb` object | Same neighbor sets |
| Filter to neighbors present in the same `year` | Join `on = .(neighbor_id = id)` with `val_dt` keyed by `(id, year)` â€” only matching year rows survive | Same temporal alignment |
| `neighbor_vals[!is.na(neighbor_vals)]` | `merged[!is.na(val)]` | Same NA removal |
| `max`, `min`, `mean` on filtered vector; return `NA` if empty | `data.table` grouped agg returns the same; left-join fills unmatched rows with `NA` | Identical numerics |

The Random Forest model object is never referenced or modified. The 15 new feature columns (5 vars Ã— 3 stats) are numerically identical to the originals, so predictions from the existing model are unchanged.

**Expected wall-clock time on a 16 GB laptop: 5â€“15 minutes** (vs. 86+ hours), dominated by the `allow.cartesian` join (~38.5M rows) and five grouped aggregations.
END_RESPONSE>>>
