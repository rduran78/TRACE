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
run_id: Anthropic_F02_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction** (`paste(id, year, sep="_")`) is done 6.46M times inside the loop *and* again for every neighbor of every row.
- **Named-vector indexing** (`idx_lookup[neighbor_keys]`) on a 6.46M-element named character vector is an O(1)-amortized hash lookup *per key*, but the sheer volume (~6.46M rows Ã— ~4 neighbors each â‰ˆ 26M lookups) plus the R-level `lapply` loop makes this extremely slow.
- The output is a **list of 6.46M integer vectors**, consuming substantial memory.

### 2. `compute_neighbor_stats` â€” another `lapply` over 6.46M rows
- For each of 5 variables, it iterates over the 6.46M-element list, subsets a numeric vector, and computes `max/min/mean`. That's 5 Ã— 6.46M = 32.3M R-level function calls.

### 3. Memory
- The neighbor lookup list alone (6.46M list elements, each a small integer vector) can easily consume 2â€“4 GB.
- Intermediate copies of `cell_data` during column binding compound the problem.

### Combined effect: ~86+ hours is dominated by the R-interpreter overhead of tens of millions of iterations in pure-R loops.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate the per-row lookup list entirely** | Build a flat edge table (`data.table`) mapping each `(row_i, row_j)` pair where `j` is a neighbor of `i` in the same year. Then use grouped aggregation. |
| **Vectorize neighbor stats** | Join the edge table to the variable column, then `data.table` grouped `max/min/mean` in one pass per variable â€” no R-level loop over 6.46M rows. |
| **Avoid string keys** | Use integer compound keys (`id`, `year`) with `data.table` keyed joins instead of `paste`-based named vectors. |
| **Reduce memory** | The flat edge table stores only two integer columns (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows Ã— 8 bytes Ã— 2 cols â‰ˆ 0.6 GB) â€” far less than the 6.46M-element list. Process variables one at a time and bind columns in place. |
| **Preserve the RF model and estimand** | We only change *how* the same 15 neighbor-feature columns are computed. The numerical values are identical, so the trained model applies without retraining. |

Expected speedup: from 86+ hours to **minutes** (typically 5â€“20 min depending on disk I/O and RAM pressure).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0 â€” Convert cell_data to data.table (in-place if possible)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure 'id' and 'year' are integer for fast keyed joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Add a row index that we will use as the canonical row reference
cell_data[, .row_idx := .I]

# Key for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 â€” Build a flat directed edge table from the nb object
#
#   rook_neighbors_unique is a list of length N_cells (344,208).
#   id_order[k] gives the cell id for position k in that list.
#   rook_neighbors_unique[[k]] gives integer indices into id_order
#   of k's neighbors.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
message("Building flat edge table from nb object â€¦")

# Materialise directed edges: (from_id, to_id)
edge_from <- rep(
  as.integer(id_order),
  times = lengths(rook_neighbors_unique)
)
edge_to <- as.integer(id_order[unlist(rook_neighbors_unique)])

edges <- data.table(from_id = edge_from, to_id = edge_to)
rm(edge_from, edge_to)

message(sprintf("  %s directed edges (unique cells).", format(nrow(edges), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 â€” Cross edges with years to get the full (row_i â†” row_j) map
#
#   Instead of a massive cross join (edges Ã— 28 years), we join twice
#   against cell_data to resolve row indices, which naturally restricts
#   to year-combinations that exist in the data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
message("Resolving row-level neighbor pairs â€¦")

# Slim lookup: (id, year) â†’ .row_idx
row_lu <- cell_data[, .(id, year, .row_idx)]
setkey(row_lu, id, year)

# Get all unique years
all_years <- sort(unique(cell_data$year))

# Process year-by-year to control peak memory
#   For each year, join edges to row_lu twice to get (row_i, row_j).
pair_list <- lapply(all_years, function(yr) {
  lu_yr <- row_lu[year == yr]                 # rows in this year
  setkey(lu_yr, id)

  # from_id â†’ row_i
  tmp <- edges[lu_yr, .(row_i = i..row_idx, to_id), on = .(from_id = id), nomatch = 0L]
  # to_id â†’ row_j
  setkey(lu_yr, id)
  tmp <- tmp[lu_yr, .(row_i, row_j = i..row_idx), on = .(to_id = id), nomatch = 0L]
  tmp
})

pairs <- rbindlist(pair_list)
rm(pair_list, row_lu, edges)
gc()

message(sprintf("  %s row-level neighbor pairs.", format(nrow(pairs), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 â€” Compute neighbor stats for each variable (vectorised)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for '%s' â€¦", var_name))

  # Attach the neighbor's value to every pair row
  pairs[, nbr_val := cell_data[[var_name]][row_j]]

  # Grouped aggregation â€” one pass, fully vectorised in C (data.table)
  stats <- pairs[!is.na(nbr_val),
                 .(nb_max  = max(nbr_val),
                   nb_min  = min(nbr_val),
                   nb_mean = mean(nbr_val)),
                 keyby = .(row_i)]

  # Initialise new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Write results back by row index â€” no copy of cell_data
  set(cell_data, i = stats$row_i, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$row_i, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$row_i, j = mean_col, value = stats$nb_mean)

  # Clean up per-iteration objects
  pairs[, nbr_val := NULL]
  rm(stats)
  gc()

  message(sprintf("  âœ“ Added %s, %s, %s", max_col, min_col, mean_col))
}

# Drop the helper column
cell_data[, .row_idx := NULL]

message("Done â€” all neighbor features computed.")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor definition** | The edge table is built from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Same year matching** | Pairs are resolved within each year via keyed join, exactly replicating the `paste(id, year)` key logic. |
| **Same statistics** | `max`, `min`, `mean` over non-NA neighbor values â€” identical to the original `compute_neighbor_stats`. |
| **Same column names** | Output columns follow the same `neighbor_{max,min,mean}_{var}` naming convention (adjust the pattern if yours differs). |
| **Trained RF model untouched** | No model object is modified; the 15 new feature columns are numerically identical, so `predict(rf_model, cell_data)` yields the same predictions. |

---

## Performance Estimates (16 GB laptop)

| Phase | Original | Optimised |
|---|---|---|
| Neighbor lookup construction | ~40â€“60 h | ~2â€“5 min (flat edge table + keyed joins) |
| Neighbor stats (5 vars) | ~26â€“30 h | ~3â€“10 min (vectorised `data.table` grouped agg) |
| Peak RAM | ~8â€“12 GB (list of 6.46M vectors) | ~4â€“6 GB (flat integer pair table) |
| **Total** | **86+ h** | **~5â€“20 min** |

The key insight is replacing millions of R-interpreter-level loop iterations and string operations with `data.table`'s C-level keyed joins and grouped aggregations, which are orders of magnitude faster for this workload.
END_RESPONSE>>>
