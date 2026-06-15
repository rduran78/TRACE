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
run_id: Anthropic_F04_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction functions. Here is why:

### `build_neighbor_lookup` â€” O(N) calls with expensive string operations

For each of the ~6.46 million rows, the function:
1. Converts an integer cell ID to a character, then looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs, then **pastes** each neighbor ID with the current row's year to create string keys.
3. Looks up those string keys in another named character vector (`idx_lookup`) of length 6.46 million.

Named-vector lookup in R is hash-based per call but still carries constant overhead per stringâ€hashed probe. With ~6.46M rows and an average of ~4 rook neighbors per cell (1,373,394 directed pairs / 344,208 cells â‰ˆ 4), that is roughly **26 million `paste` + hash-lookup operations**, all inside a sequential `lapply` over 6.46M iterations. The R-level loop overhead alone (function dispatch, memory allocation per iteration, intermediate character vectors) makes this extremely slow.

### `compute_neighbor_stats` â€” called 5 times, each iterating 6.46M rows

Each call subsets a numeric vector by index and computes `max`, `min`, `mean`. This is lighter per iteration than `build_neighbor_lookup`, but 5 Ã— 6.46M R-level `lapply` iterations with per-row allocation still accumulates significant overhead.

### Estimated wall-clock breakdown (86+ hours)

| Stage | Estimated share |
|---|---|
| `build_neighbor_lookup` (string pasting & hash lookup) | ~70â€“80% |
| `compute_neighbor_stats` Ã— 5 vars | ~20â€“30% |
| Random Forest `predict()` | < 1% |

---

## Optimization Strategy

### Principle: Replace row-level R loops and string operations with vectorized integer-indexed joins via `data.table`.

**Key ideas:**

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, construct a flat edge-list `data.table` mapping every `(id, year)` â†’ each neighbor's `(neighbor_id, year)`, then use `data.table` keyed joins to pull neighbor variable values. No `paste`, no named-vector hash lookup, no R-level `lapply` over 6.46M rows.

2. **Compute all five variables' neighbor statistics in a single grouped aggregation** on the joined edge table, instead of five separate `lapply` passes.

3. **Memory management:** The edge-list, after expanding by 28 years, will have ~1.37M Ã— 28 â‰ˆ 38.5M rows (directed edges Ã— years). Each row stores two integer IDs, one integer year, and (during the join) a few numeric columns â€” this fits comfortably in 16 GB RAM.

**Expected speedup:** From 86+ hours to roughly **2â€“10 minutes**, depending on disk I/O and `data.table` thread count.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Starting point: objects already in memory
#       cell_data              : data.frame or data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order               : integer vector of cell IDs in the order used by the nb object
#       rook_neighbors_unique  : spdep::nb list (length = length(id_order))
#       rf_model               : trained Random Forest model (untouched)
# ---------------------------------------------------------------

# Convert to data.table (in-place if already data.table; copy otherwise)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---------------------------------------------------------------
# 1.  Build a flat directed edge-list of (cell_id -> neighbor_id)
#     from the nb object.  This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep::nb encodes "no neighbors" as a single 0L

if (length(nb_idx) == 1L && nb_idx == 0L) {
    return(data.table(id = integer(0), neighbor_id = integer(0)))
  }
  data.table(
    id          = id_order[i],
    neighbor_id = id_order[nb_idx]
  )
}))

# This gives ~1.37 M rows (one per directed rook-neighbor pair).
# No year dimension yet â€” we will join on year below.

# ---------------------------------------------------------------
# 2.  Expand the edge list across all years by joining to cell_data.
#     Then pull each neighbor's variable values through a second join.
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Slim lookup table: only the columns we need for neighbor stats
# (id, year, and the five source variables)
val_cols <- c("id", "year", neighbor_source_vars)
vals_dt  <- cell_data[, ..val_cols]

# Key for fast join
setkey(vals_dt, id, year)

# Unique (id, year) combinations that exist in the dataset
id_year <- unique(cell_data[, .(id, year)])

# Cross-join with edge_list to create (id, year, neighbor_id) triples
# â€” an equi-join on 'id' replicates each edge across all years that cell appears.
setkey(id_year, id)
setkey(edge_list, id)

edges_by_year <- edge_list[id_year, on = "id", allow.cartesian = TRUE, nomatch = NULL]
# Result columns: id, neighbor_id, year
# Expected rows: ~1.37M edges Ã— (years per cell, â‰ˆ28) â‰ˆ 38.5 M

# ---------------------------------------------------------------
# 3.  Join neighbor variable values onto the edge table.
# ---------------------------------------------------------------

setkey(edges_by_year, neighbor_id, year)

# Bring in neighbor values (join neighbor_id == id and same year)
edges_by_year <- vals_dt[edges_by_year,
                          on = c("id" = "neighbor_id", "year" = "year"),
                          nomatch = NA]

# After this join the columns from vals_dt are the NEIGHBOR's values.
# Rename to avoid confusion: the original 'id' from vals_dt is
# actually the neighbor_id; 'i.id' (from edges_by_year) is the focal cell.
# data.table names the columns as:  id (=neighbor_id), year, ntl, â€¦, i.id
# Rename for clarity.
setnames(edges_by_year, "i.id", "focal_id")
# 'id' column is the neighbor; keep it for transparency but we group by focal_id, year.

# ---------------------------------------------------------------
# 4.  Compute neighbor max / min / mean for all five variables
#     in ONE grouped aggregation.
# ---------------------------------------------------------------

# Build the aggregation expression programmatically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Create a single combined call
agg_call <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

neighbor_stats <- edges_by_year[,
                                 eval(agg_call),
                                 by = .(focal_id, year)]

# Replace Inf / -Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
# 5.  Merge the 15 new neighbor-feature columns back to cell_data.
# ---------------------------------------------------------------

setkey(neighbor_stats, focal_id, year)
setkey(cell_data, id, year)

# Remove old neighbor columns if they already exist (idempotency)
old_cols <- intersect(agg_names, names(cell_data))
if (length(old_cols)) cell_data[, (old_cols) := NULL]

cell_data <- neighbor_stats[cell_data, on = c("focal_id" = "id", "year" = "year")]

# Restore column name so 'id' is present as expected downstream
setnames(cell_data, "focal_id", "id")

# ---------------------------------------------------------------
# 6.  Predict with the EXISTING Random Forest model (unchanged).
# ---------------------------------------------------------------

# Ensure column order / names match what rf_model expects
# (adjust 'predict' call to your specific RF package: ranger, randomForest, etc.)
cell_data[, predicted_gdp := predict(rf_model, newdata = cell_data)$predictions]
# If using randomForest::predict, use:
#   cell_data[, predicted_gdp := predict(rf_model, newdata = cell_data)]

# ---------------------------------------------------------------
# Done.  cell_data now has the same 15 neighbor features and
# the same RF predictions as the original pipeline, produced in
# minutes rather than days.
# ---------------------------------------------------------------
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same directed rook-neighbor pairs feed the same `max`, `min`, `mean` aggregations, producing identical floating-point results (modulo summation order, which is IEEE-754 negligible for `mean` of â‰¤ 8 neighbors). |
| **Trained RF model untouched** | The model object `rf_model` is only used in a `predict()` call; it is never retrained or modified. |
| **Column semantics** | The 15 new columns (`neighbor_ntl_max`, `neighbor_ntl_min`, `neighbor_ntl_mean`, â€¦) carry the same names and values as the original `compute_and_add_neighbor_features` would produce. |

---

## Performance Summary

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup (string paste + hash) | ~60â€“70 hours | **Eliminated** (flat edge-list built in ~2 s) |
| Compute neighbor stats (5 vars Ã— 6.46M rows) | ~20â€“25 hours | **~1â€“3 min** (single `data.table` grouped aggregation on ~38.5M rows) |
| Join back to cell_data | trivial | **~10 s** (keyed join) |
| RF predict | ~1 min | ~1 min (unchanged) |
| **Total** | **86+ hours** | **~2â€“10 minutes** |
END_RESPONSE>>>
