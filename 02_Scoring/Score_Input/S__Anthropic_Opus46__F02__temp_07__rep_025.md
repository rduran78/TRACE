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
run_id: Anthropic_F02_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (which are hash-table probes). With ~6.46M rows this creates millions of temporary character vectors and millions of hash lookups. The named-vector `idx_lookup` itself is a character-keyed hash of 6.46M entries, and probing it inside a per-row `lapply` is extremely slow. Estimated wall time for this step alone: **20â€“40 hours**.

### 2. `compute_neighbor_stats` â€” repeated per variable, also O(n)
Each call iterates over 6.46M rows, subsetting a numeric vector by index vectors of varying length. This is called 5 times (once per neighbor source variable). The `do.call(rbind, ...)` on a 6.46M-element list of 3-element vectors is itself a major allocation bottleneck. Estimated: **~10 hours per variable Ã— 5 = 50 hours**.

### Memory
Storing `neighbor_lookup` as a list of 6.46M integer vectors is also memory-heavy (~2â€“4 GB depending on neighbor counts), and the repeated `do.call(rbind, ...)` temporarily doubles memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste`/hash lookup in `build_neighbor_lookup` | Replace with a **vectorized merge/join** using `data.table`. Build a neighbor-edge table once, then join to get row indices. Eliminates all per-row `lapply`. |
| Storing lookup as list-of-vectors | Store as a **`data.table` of edges** (`from_row`, `to_row`). This is a flat table of ~1.37M Ã— 28 â‰ˆ 38.4M edge-rows (directed, per year). Compact and joinable. |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with **grouped `data.table` aggregation**: group by `from_row`, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass. |
| `do.call(rbind, ...)` on 6.46M-element list | Eliminated entirely â€” `data.table` returns a single result table. |
| 5 separate passes over the edge table | Compute all 5 variables' neighbor stats in a **single grouped aggregation** or a tight loop of vectorized ops. |

**Expected speedup**: from ~86 hours to **~5â€“15 minutes** on the same laptop. Memory peak: ~4â€“6 GB (well within 16 GB).

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0 â€” Convert cell_data to data.table (if not already) and ensure key cols
# ===========================================================================
cell_dt <- as.data.table(cell_data)          # non-destructive copy
cell_dt[, row_idx := .I]                     # preserve original row order

# ===========================================================================
# STEP 1 â€” Build a flat edge table from the nb object (one-time, vectorized)
#
#   rook_neighbors_unique is a list of length N_cells (344,208).
#   rook_neighbors_unique[[i]] contains integer indices into id_order
#   of the neighbors of cell id_order[i].
#
#   id_order is a vector of cell IDs of length 344,208.
# ===========================================================================

# --- 1a. Expand the nb list into a two-column data.table of (from_id, to_id)
from_ref <- rep(seq_along(rook_neighbors_unique),
                lengths(rook_neighbors_unique))
to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove the spdep placeholder 0 (nb objects use 0 for "no neighbors")
valid <- to_ref != 0L
from_ref <- from_ref[valid]
to_ref   <- to_ref[valid]

edge_cells <- data.table(
  from_id = id_order[from_ref],
  to_id   = id_order[to_ref]
)
rm(from_ref, to_ref, valid)                  # free memory

# --- 1b. Create a lookup from (id, year) â†’ row_idx in cell_dt
key_dt <- cell_dt[, .(id, year, row_idx)]

# --- 1c. Cross-join edges with years to get per-year edge table,
#          then map each (id, year) to its row_idx.
#
#   Instead of a full cross-join (which would be huge), we merge twice:
#     â€¢ first  merge: edge_cells â‹ˆ key_dt  on from_id = id  â†’ gives (from_row, to_id, year)
#     â€¢ second merge: result     â‹ˆ key_dt  on to_id = id AND year â†’ gives (from_row, to_row)
#
#   This naturally restricts to (cell, year) pairs that actually exist.

setnames(key_dt, c("id", "year", "row_idx"),
                 c("from_id", "year", "from_row"))
setkey(key_dt, from_id)
setkey(edge_cells, from_id)

# First merge â€” attach from_row and year
edge_year <- edge_cells[key_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
#   columns: from_id, to_id, year, from_row

# Second merge â€” attach to_row
setnames(key_dt, c("from_id", "year", "from_row"),
                 c("to_id",   "year", "to_row"))
setkey(key_dt, to_id, year)
setkey(edge_year, to_id, year)

edge_year <- key_dt[edge_year, on = c("to_id", "year"), nomatch = 0L]
#   columns: to_id, year, to_row, from_id, from_row

# Keep only the columns we need
edge_year <- edge_year[, .(from_row, to_row)]
rm(key_dt, edge_cells)
gc()

cat("Edge-year table:", format(nrow(edge_year), big.mark = ","), "rows\n")

# ===========================================================================
# STEP 2 â€” Compute neighbor stats for all variables in one pass
#
#   For each (from_row) we need max, min, mean of the neighbor values
#   (the values at to_row) for each of the 5 source variables.
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach the neighbor (to_row) values to the edge table
# We pull only the columns we need to keep memory tight.
val_cols <- neighbor_source_vars
to_vals  <- cell_dt[edge_year$to_row, ..val_cols]
edge_aug <- cbind(edge_year, to_vals)
edge_aug[, from_row := edge_year$from_row]
rm(to_vals)
gc()

# Grouped aggregation â€” one pass over the edge table
agg_exprs <- unlist(lapply(val_cols, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(val_cols, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call dynamically
agg_list <- setNames(agg_exprs, agg_names)

# data.table aggregation
neighbor_stats <- edge_aug[,
  lapply(agg_list, eval),
  by = from_row
]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

rm(edge_aug)
gc()

# ===========================================================================
# STEP 3 â€” Join the neighbor stats back onto cell_dt
# ===========================================================================

# Rows with no neighbors will be NA after the join (correct behavior).
setkey(neighbor_stats, from_row)
setkey(cell_dt, row_idx)

# Remove any pre-existing neighbor columns to avoid conflicts
existing <- intersect(names(cell_dt), agg_names)
if (length(existing)) cell_dt[, (existing) := NULL]

cell_dt <- neighbor_stats[cell_dt, on = c(from_row = "row_idx")]

# Clean up helper columns
cell_dt[, from_row := NULL]

# ===========================================================================
# STEP 4 â€” Restore original row order and convert back to data.frame
#           (preserves compatibility with the trained Random Forest model)
# ===========================================================================

setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

cell_data <- as.data.frame(cell_dt)          # overwrite original object
rm(cell_dt, neighbor_stats, edge_year)
gc()
```

---

## Why This Preserves the Trained Random Forest and the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Column names** | The output columns are named identically to what `compute_and_add_neighbor_features` would have produced (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.). The trained model's feature name references are satisfied. |
| **Numerical values** | `max`, `min`, `mean` with `na.rm = TRUE` are the same functions applied to the same neighbor index sets. The flat edge table is an exact materialization of the original `neighbor_lookup` list. No approximation or sampling is involved, so the numerical estimand is identical to machine precision. |
| **No retraining** | The code only constructs prediction features. The model object is never touched. |

---

## Summary of Expected Gains

| Metric | Before (original) | After (data.table) |
|---|---|---|
| `build_neighbor_lookup` | ~20â€“40 h | ~1â€“3 min (vectorized joins) |
| `compute_neighbor_stats` (Ã—5) | ~50 h | ~2â€“5 min (single grouped agg) |
| Peak RAM | ~10â€“12 GB+ | ~4â€“6 GB |
| **Total wall time** | **~86 h** | **~5â€“15 min** |
END_RESPONSE>>>
