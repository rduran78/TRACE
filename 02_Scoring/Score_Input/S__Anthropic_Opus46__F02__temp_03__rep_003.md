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
run_id: Anthropic_F02_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing ~6.46M of themâ€”each touching multiple neighborsâ€”creates billions of small operations. The `compute_neighbor_stats` function then iterates over the same 6.46M entries again per variable, and this is repeated for 5 variables.

**Specific problems:**

1. **`build_neighbor_lookup`**: Creates ~6.46M character keys (`paste(id, year)`), then for each row, pastes neighbor keys and looks them up. This is an O(N Ã— K) string operation where K is the average neighbor count (~4 for rook contiguity). With N â‰ˆ 6.46M, that's ~25.8M string constructions plus hash lookups inside `lapply`â€”all in serial, interpreted R.

2. **`compute_neighbor_stats`**: Uses `lapply` over 6.46M elements, calling `max`, `min`, `mean` on small vectors. The overhead of 6.46M R function calls dominates. `do.call(rbind, result)` on a 6.46M-element list is also expensive.

3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors has significant R list overhead (~6.46M SEXP pointers + per-element allocation).

4. **Repeated iteration**: The stats computation is called 5 times (once per variable), each time iterating over the full 6.46M rows.

---

## Optimization Strategy

**Replace the row-level R loop with a vectorized join-and-group-by approach using `data.table`.**

The key insight: the neighbor lookup is a **join** operation. Each `(cell_id, year)` pair needs to be joined to its neighbors' `(neighbor_id, same_year)` rows. This is a classic equi-join that `data.table` handles extremely efficiently in C.

**Steps:**

1. **Build an edge table** from the `nb` object: a two-column `data.table` of `(id, neighbor_id)` â€” ~1.37M rows.
2. **Join** the edge table to the panel data on `(neighbor_id, year)` to get neighbor values â€” this produces ~6.46M Ã— ~4 â‰ˆ ~25.8M rows, which at ~5 numeric columns is ~1 GB (fits in 16 GB RAM).
3. **Group-by aggregate** `(id, year)` to compute `max`, `min`, `mean` for all 5 variables **simultaneously** in one pass.
4. **Join** the aggregated results back to the original data.

This eliminates all R-level loops and leverages `data.table`'s parallelized, cache-optimized C internals. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert panel data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure 'id' and 'year' columns exist and are keyed for fast joins
if (!("id" %in% names(cell_data))) stop("cell_data must have an 'id' column")
if (!("year" %in% names(cell_data))) stop("cell_data must have a 'year' column")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build edge list from the nb object
#
# id_order is the vector of cell IDs in the order matching the nb object.
# rook_neighbors_unique is the nb object (list of integer index vectors).
# We expand it into a two-column data.table: (id, neighbor_id).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  n <- length(neighbors)
  # Pre-allocate lengths
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  from_idx <- rep.int(seq_len(n), lens)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Join edges to panel data to retrieve neighbor variable values
#
# For each (id, year), we want the variable values of all neighbors
# in the same year. This is an equi-join:
#   edges[cell_data, on = .(neighbor_id == id)]  â€” gets neighbor rows
# then joined again by year.
#
# Strategy: 
#   - Start from cell_data: take (id, year) and join to edges to get
#     neighbor_id for each row.
#   - Then join on (neighbor_id, year) to cell_data to get neighbor values.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim lookup table with only the columns we need
# This minimizes memory during the large join
lookup_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..lookup_cols]
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)

# Create a slim version of cell_data with just id and year for the first join
cell_keys <- cell_data[, .(id, year)]

# Join cell_keys to edges to get (id, year, neighbor_id) for every cell-year-neighbor combo
# This is ~6.46M rows Ã— ~4 neighbors = ~25.8M rows
setkey(edges, id)
cell_neighbors <- edges[cell_keys, on = .(id), allow.cartesian = TRUE, nomatch = NULL]
# Result columns: id, neighbor_id, year

# Now join to get the neighbor variable values
setkey(cell_neighbors, neighbor_id, year)
cell_neighbors <- neighbor_vals_dt[cell_neighbors, on = .(neighbor_id, year), nomatch = NA]
# Result columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Free the lookup table
rm(neighbor_vals_dt)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Compute grouped aggregates (max, min, mean) per (id, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build aggregation expressions dynamically for all 5 variables Ã— 3 stats
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)),  na.rm = TRUE)))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation in one pass (data.table does this in C)
agg_result <- cell_neighbors[,
  eval(as.call(c(quote(list), agg_exprs))),
  by = .(id, year)
]

# Handle Inf/-Inf from max/min on all-NA groups â†’ convert to NA
inf_cols <- grep("neighbor_(max|min)_", names(agg_result), value = TRUE)
for (col in inf_cols) {
  set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
}

# Also handle NaN from mean on all-NA groups
mean_cols <- grep("neighbor_mean_", names(agg_result), value = TRUE)
for (col in mean_cols) {
  set(agg_result, which(is.nan(agg_result[[col]])), col, NA_real_)
}

# Free the large intermediate table
rm(cell_neighbors)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Join aggregated neighbor features back to cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove any pre-existing neighbor columns to avoid conflicts
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

setkey(agg_result, id, year)
setkey(cell_data, id, year)

cell_data <- agg_result[cell_data, on = .(id, year)]

rm(agg_result)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Done. cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... (same pattern for pop_density, def, usd_est_n2)
#
# The trained Random Forest model is untouched.
# The numerical values (max, min, mean) are identical to the originals.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level `lapply` iterations with string ops | One vectorized `data.table` join (~seconds) |
| **Stats computation** | 5 Ã— 6.46M R-level `lapply` iterations | One grouped `data.table` aggregation in C |
| **Peak memory** | ~6.46M R list elements (high overhead) | ~25.8M-row `data.table` (~1â€“2 GB, fits in 16 GB) |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Numerical output** | max, min, mean per neighbor set | Identical max, min, mean per neighbor set |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |

The optimization replaces interpreted R loops with `data.table`'s C-level equi-joins and grouped aggregations, which are orders of magnitude faster and more memory-efficient for this class of problem.
END_RESPONSE>>>
