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
run_id: Anthropic_F02_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times with string allocation and matching is extremely expensive. The resulting list of ~6.46M integer vectors also consumes significant memory.

### 2. `compute_neighbor_stats` â€” repeated per variable, pure R loop
For each of the 5 neighbor source variables, another `lapply` over 6.46M rows subsets values, removes NAs, and computes `max/min/mean`. This is called 5 times sequentially. The `do.call(rbind, ...)` on a 6.46M-element list is itself a known performance anti-pattern.

**Combined effect:** billions of R-level function calls, millions of temporary string allocations, and repeated list-to-matrix conversions produce the estimated 86+ hour runtime.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| String-key lookups in `build_neighbor_lookup` | Replace with integer arithmetic: encode `(id, year)` as a single integer key and use `data.table` hash joins or direct index math. Since years are contiguous (1992â€“2019, 28 years), we can compute row indices arithmetically: `row = (cell_index - 1) * 28 + (year - 1991)` if data is sorted by `(id, year)`. This eliminates all `paste`/string operations. |
| `lapply` over 6.46M rows in R | Vectorize the neighbor lookup by expanding the neighbor list into a flat edge table `(row_i, row_j)` and using grouped vectorized operations via `data.table`. |
| Per-variable `lapply` over 6.46M rows | Compute all 5 variables' neighbor stats in a single grouped `data.table` aggregation pass over the edge table. |
| `do.call(rbind, list_of_6.46M)` | Eliminated entirely; `data.table` returns a single result table. |
| Memory (16 GB) | The flat edge table has ~6.46M Ã— avg_neighbors â‰ˆ ~25â€“30M rows Ã— 2 integer columns â‰ˆ ~0.5 GB. The main data (~6.46M Ã— 110 cols) is ~5â€“7 GB. Feasible within 16 GB if we avoid duplication. |

**Expected speedup:** from 86+ hours to roughly 5â€“20 minutes.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert to data.table (in-place if possible to save memory)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure data is keyed/sorted by (id, year) â€” critical for index math
setorder(cell_data, id, year)

# Add a row index explicitly
cell_data[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build integer mappings (replaces build_neighbor_lookup entirely)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# id_order is the vector of unique cell IDs matching rook_neighbors_unique
# Map each cell id -> its position in id_order (1-based "ref index")
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Map each (id, year) -> row_idx using data.table keyed join
#   Because data is sorted by (id, year) and years are contiguous 1992-2019,
#   we can use direct arithmetic.  But to be safe against missing cell-years
#   we build a small keyed lookup table.
id_year_to_row <- cell_data[, .(id, year, row_idx)]
setkey(id_year_to_row, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Expand neighbor list into a flat edge table  (row_i  â†’  row_j)
#     This is the key transformation: we do it ONCE, not per variable.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 2a. Build cell-level edge list from rook_neighbors_unique (spdep nb object)
#     Each element k of the nb list gives the ref-indices of neighbors of cell k.
message("Building cell-level edge list â€¦")
n_cells <- length(id_order)
from_ref <- rep(seq_len(n_cells),
                times = lengths(rook_neighbors_unique))
to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Convert ref indices â†’ actual cell IDs
cell_edges <- data.table(
  from_id = id_order[from_ref],
  to_id   = id_order[to_ref]
)
rm(from_ref, to_ref)

# 2b. Expand to row-level edges by joining on every year.
#     For each (from_id, year) we need (to_id, year) â€” same year.
#     Strategy: cross-join cell_edges with the 28 years, then map to row_idx.
message("Expanding to row-level edge table â€¦")

years <- sort(unique(cell_data$year))  # 1992:2019

# Expand: each cell edge Ã— each year
edge_expanded <- cell_edges[, .(year = years), by = .(from_id, to_id)]
rm(cell_edges)

# Join to get row_idx for the "from" side
setkey(edge_expanded, from_id, year)
edge_expanded[id_year_to_row, row_i := i.row_idx, on = .(from_id = id, year)]

# Join to get row_idx for the "to" (neighbor) side
setkey(edge_expanded, to_id, year)
edge_expanded[id_year_to_row, row_j := i.row_idx, on = .(to_id = id, year)]

# Drop edges where either side is missing (cell-year not in data)
edge_expanded <- edge_expanded[!is.na(row_i) & !is.na(row_j),
                               .(row_i, row_j)]
setkey(edge_expanded, row_i)

message(sprintf("Edge table: %s rows", format(nrow(edge_expanded), big.mark = ",")))

rm(id_year_to_row)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute ALL neighbor stats in one vectorized pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Attaching neighbor values to edge table â€¦")

# Pull only the columns we need from cell_data (saves memory in the join)
val_cols <- c("row_idx", neighbor_source_vars)
vals_dt  <- cell_data[, ..val_cols]
setkey(vals_dt, row_idx)

# Join neighbor values onto the edge table (row_j is the neighbor)
edge_with_vals <- vals_dt[edge_expanded, on = .(row_idx = row_j), nomatch = NA]
# edge_with_vals now has columns: row_idx (=row_j), <vars>, row_i
# Rename for clarity
setnames(edge_with_vals, "row_idx", "row_j")

rm(vals_dt, edge_expanded)
gc()

message("Computing grouped neighbor statistics â€¦")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Single grouped aggregation
neighbor_stats <- edge_with_vals[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = row_i
]

rm(edge_with_vals)
gc()

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
inf_cols <- setdiff(names(neighbor_stats), "row_i")
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Merge back into cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

message("Merging neighbor features back into cell_data â€¦")

# Remove any pre-existing neighbor columns to avoid duplication
old_cols <- intersect(names(cell_data), agg_names)
if (length(old_cols)) cell_data[, (old_cols) := NULL]

setkey(neighbor_stats, row_i)
setkey(cell_data, row_idx)

cell_data[neighbor_stats, (agg_names) := mget(paste0("i.", agg_names)),
          on = .(row_idx = row_i)]

# Rows with no neighbors will already be NA (unmatched in join)

# Clean up helper column
cell_data[, row_idx := NULL]

rm(neighbor_stats)
gc()

message("Done â€” neighbor features added.")
```

---

## Why This Preserves Correctness

| Requirement | How it is met |
|---|---|
| **Trained Random Forest model unchanged** | We only add/replace the same neighbor feature columns (`neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*`) that the original code produced. No model object is touched. |
| **Original numerical estimand preserved** | `max`, `min`, `mean` with `na.rm = TRUE` on the same neighbor sets produce identical values. The edge table is derived from the same `rook_neighbors_unique` nb object and the same year-matching logic. |
| **Same column names** | The naming convention `neighbor_{stat}_{var}` matches what `compute_and_add_neighbor_features` would have produced (adjust the prefix/suffix pattern if your wrapper used a different convention). |

---

## Performance Estimate

| Stage | Estimated time | Estimated peak memory |
|---|---|---|
| Edge table construction | ~1â€“2 min | ~2 GB (edge table ~30M rows Ã— 4 int cols) |
| Join neighbor values | ~1â€“2 min | ~4 GB (edge table + 5 double cols) |
| Grouped aggregation | ~2â€“5 min | ~3 GB (result: 6.46M Ã— 16 cols) |
| Merge back | ~30 sec | negligible additional |
| **Total** | **~5â€“10 min** | **~12â€“14 GB peak (fits in 16 GB)** |

If memory is still tight, the grouped aggregation can be chunked by variable (5 passes instead of 1), trading a small amount of speed for lower peak memory. But the single-pass approach should fit within 16 GB.
END_RESPONSE>>>
