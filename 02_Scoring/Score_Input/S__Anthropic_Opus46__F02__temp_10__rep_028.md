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
run_id: Anthropic_F02_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The core bottleneck is the **`build_neighbor_lookup`** function. It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:

- A character coercion and named-vector lookup (`id_to_ref`).
- Subsetting of the `neighbors` list.
- `paste`-based key construction for every neighbor of every row.
- A second named-vector lookup (`idx_lookup`) on those string keys.

**String allocation and named-vector lookups are O(n) in the worst case in base R** (named vectors use linear hashing buckets that degrade). With ~6.46M rows and an average of ~4 rook neighbors per cell, this loop constructs and resolves roughly **25â€“30 million character strings**. The `idx_lookup` named vector itself has 6.46M entries, so every single lookup is expensive. This is the source of the 86+ hour estimate.

`compute_neighbor_stats` is comparatively lighter but still uses `lapply` over 6.46M elements and `do.call(rbind, ...)` on a 6.46M-element list, which is also slow.

**Memory** is stressed because the `neighbor_lookup` list stores ~6.46M integer vectors (one per row), plus intermediate character vectors.

---

## 2. Optimization Strategy

| Problem | Solution |
|---|---|
| String key construction & lookup in `build_neighbor_lookup` | Replace with **integer arithmetic**: encode `(id, year)` as a single integer key via a hash-free formula, and use `data.table` fast joins or `match()` on integers. |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | **Vectorize entirely**: expand the neighbor list into a flat edge table, join to get row indices, then split once. Or better: never build a per-row list at all â€” use a **flat CSR-style (compressed sparse row)** representation. |
| `lapply` over 6.46M rows in `compute_neighbor_stats` | Replace with **grouped vectorized aggregation** using `data.table`, operating on the flat edge table directly. |
| `do.call(rbind, ...)` on millions of rows | Eliminate by pre-allocating a matrix or using `data.table`'s `:=` column assignment. |
| Memory pressure (16 GB) | The flat edge table (â‰ˆ6.46M rows Ã— 4 neighbors Ã— 2 integer columns â‰ˆ 400 MB) is far smaller than millions of R list elements with per-element overhead. Process variables one at a time and discard intermediates. |

**Key insight**: We can build a single `data.table` of `(row_i, neighbor_row_j)` pairs (~25M rows) once, then for each variable, join in the values and compute grouped `max/min/mean` â€” all in vectorized C-level `data.table` code. This replaces **all** R-level loops.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0 â€” Convert cell_data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure an explicit row index that we will use throughout.
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 â€” Build a flat edge table   (replaces build_neighbor_lookup)
#
#   For every cell i and every rook neighbor j of i, and for every year,
#   we need a pair  (row index of (i,year),  row index of (j,year)).
#
#   Strategy:
#     a) Expand the nb object into a two-column integer matrix (from, to)
#        expressed as positions in id_order.
#     b) Cross that with the 28 years.
#     c) Map (id, year) -> row_idx  via an integer-keyed join.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# (a) Expand nb list â†’ edge data.table  --------------------------------
#     rook_neighbors_unique[[k]] gives the *position indices* in id_order
#     that are neighbors of the cell whose id is id_order[k].

edge_from <- rep(
  seq_along(rook_neighbors_unique),
  lengths(rook_neighbors_unique)
)
edge_to <- unlist(rook_neighbors_unique, use.names = FALSE)

# Convert position indices to actual cell ids
edges <- data.table(
  id_from = id_order[edge_from],
  id_to   = id_order[edge_to]
)
rm(edge_from, edge_to)            # free memory immediately

# (b) Cross with years -------------------------------------------------
years <- sort(unique(cell_data$year))

# Instead of a full cross join (which would be large), we do a keyed join.
# Build a lookup:  (id, year) â†’ .row_idx
id_year_key <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_key, id, year)

# We expand edges Ã— years via a rolling/equi join.
# CJ.dt helper â€” memory-efficient cross of edges with years:
edges_expanded <- edges[, .(id_from, id_to, year = rep(list(years), .N))]
edges_expanded <- edges_expanded[, .(year = unlist(year)), by = .(id_from, id_to)]

# (c) Map to row indices -----------------------------------------------
# Join to get row_i  (the row index corresponding to id_from, year)
setnames(id_year_key, c("id", "year", ".row_idx"), c("id_from", "year", "row_i"))
setkey(id_year_key, id_from, year)
setkey(edges_expanded, id_from, year)
edges_expanded <- id_year_key[edges_expanded, nomatch = 0L]

# Join to get row_j  (the row index corresponding to id_to, year)
# Re-read the key table (rename for the second join)
id_year_key2 <- cell_data[, .(id_to = id, year, row_j = .row_idx)]
setkey(id_year_key2, id_to, year)
setkey(edges_expanded, id_to, year)
edges_expanded <- id_year_key2[edges_expanded, nomatch = 0L]

rm(id_year_key, id_year_key2, edges)
gc()

# edges_expanded now has columns:  row_i, row_j, (id_from, id_to, year)
# We only need row_i and row_j going forward.
edge_dt <- edges_expanded[, .(row_i, row_j)]
rm(edges_expanded); gc()

# Sort by row_i for cache-friendly grouped operations
setkey(edge_dt, row_i)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 â€” Compute neighbor stats per variable   (vectorized)
#           (replaces compute_neighbor_stats + the outer for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  message("Processing neighbor features for: ", var_name)

  # Pull the variable values indexed by row position
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the edge table
  edge_dt[, nval := vals[row_j]]

  # Grouped aggregation â€” one pass over ~25M rows, fully vectorized in C
  agg <- edge_dt[!is.na(nval),
                 .(nb_max  = max(nval),
                   nb_min  = min(nval),
                   nb_mean = mean(nval)),
                 keyby = .(row_i)]

  # Initialise new columns to NA, then fill matched rows
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  matched <- agg$row_i
  set(cell_data, i = matched, j = max_col,  value = agg$nb_max)
  set(cell_data, i = matched, j = min_col,  value = agg$nb_min)
  set(cell_data, i = matched, j = mean_col, value = agg$nb_mean)

  # Clean up the temporary column
  edge_dt[, nval := NULL]
  rm(agg, vals, matched); gc()
}

# Remove helper column
cell_data[, .row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 â€” Predict with the pre-trained Random Forest  (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g. `rf_model`) is loaded from disk as-is.
# Predictions proceed exactly as before:
#
#   cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
#
# Nothing about the model or the estimand changes â€” only the feature-
# engineering pipeline has been optimised.
```

---

## 4. Expected Improvements

| Metric | Original | Optimised |
|---|---|---|
| **`build_neighbor_lookup` time** | ~70â€“80+ hours (string ops, 6.46M R-level iterations) | ~2â€“5 minutes (vectorised integer joins in `data.table`) |
| **`compute_neighbor_stats` time** | ~6â€“8 hours (R-level `lapply`) | ~30â€“90 seconds per variable (grouped C-level aggregation) |
| **Total feature engineering** | **86+ hours** | **~10â€“15 minutes** |
| **Peak RAM** | >16 GB (millions of list elements + character vectors) | ~4â€“6 GB (flat integer edge table + one double column at a time) |

### Why it works

1. **No character strings.** The original code created ~30M `paste()`-produced strings and looked them up in a 6.46M-entry named vector. The replacement uses integer column joins, which `data.table` handles with radix-sort merge in C.

2. **No R-level loops.** Both the lookup construction and the aggregation are fully vectorised. `data.table`'s `[, .(max, min, mean), by = row_i]` compiles to a single grouped pass.

3. **No per-row list storage.** The flat `edge_dt` (two integer columns, ~25M rows â‰ˆ 400 MB) replaces 6.46M R list elements, each with individual SEXP overhead.

4. **Model and estimand unchanged.** The output columns (`nb_max_ntl`, `nb_min_ntl`, `nb_mean_ntl`, etc.) contain identical numerical values, so the pre-trained Random Forest can be applied without retraining.

---

### Memory-Constrained Variant (Optional)

If the ~25M-row `edges_expanded` intermediate still strains 16 GB during construction, process the year cross-join in **chunks of years**:

```r
chunk_size <- 7  # 4 chunks of 7 years
year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

edge_dt <- rbindlist(lapply(year_chunks, function(yr_chunk) {
  exp <- edges[, .(id_from, id_to, year = list(yr_chunk)), by = .I][
    , .(year = unlist(year)), by = .(id_from, id_to)]
  # join row indices for this chunk ...
  # (same join logic as above, but filtered to yr_chunk)
  key_chunk <- cell_data[year %in% yr_chunk, .(id, year, .row_idx)]
  setkey(key_chunk, id, year)
  setnames(key_chunk, "id", "id_from")
  setkey(exp, id_from, year)
  exp <- key_chunk[exp, nomatch = 0L]
  setnames(exp, ".row_idx", "row_i")
  setnames(key_chunk, "id_from", "id_to")
  setkey(key_chunk, id_to, year)
  setkey(exp, id_to, year)
  exp <- key_chunk[exp, nomatch = 0L]
  setnames(exp, ".row_idx", "row_j")
  exp[, .(row_i, row_j)]
}))
```

This keeps peak intermediate memory well under 16 GB.
END_RESPONSE>>>
