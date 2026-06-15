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
run_id: Anthropic_F02_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear and two-fold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- `paste()` is called millions of times inside the loop to construct keys.
- Named-vector indexing (`idx_lookup[neighbor_keys]`) on a 6.46M-element character vector is effectively a hash-table lookup repeated for every neighbor of every row â€” roughly 1.37 million neighbor pairs Ã— 28 years â‰ˆ 38.5 million lookups, each preceded by a string allocation.
- The entire result is a **list of 6.46 million integer vectors**, which is itself a large memory object with heavy overhead per list element.

### 2. `compute_neighbor_stats` â€” another O(n) `lapply` over 6.46 million rows
- Called **5 times** (once per source variable), so 32.3 million R-level function invocations.
- Each invocation subsets a numeric vector, removes NAs, and computes max/min/mean â€” all interpreted R with per-call overhead.

### Combined effect
~6.46M Ã— (string ops + hash lookups) + 5 Ã— 6.46M Ã— (subset + summary stats) = billions of interpreted R operations. On a 16 GB laptop this runs for 86+ hours and risks memory exhaustion from the intermediate list-of-vectors structure.

---

## Optimization Strategy

The key insight: **replace row-level R loops with vectorized joins and grouped aggregations using `data.table`.**

| Step | Current Approach | Optimized Approach |
|---|---|---|
| Neighbor lookup | Per-row `paste` + named-vector hash | Build an **edge table** (`data.table`) of `(id, neighbor_id)` once; join to data by `(neighbor_id, year)` â€” fully vectorized |
| Neighbor stats | Per-row `lapply` Ã— 5 variables | Single grouped `data.table` aggregation: `[, .(max, min, mean), by = .(id, year)]` per variable â€” vectorized C-level grouping |
| Memory | 6.46M-element list of integer vectors | Edge table â‰ˆ 38.5M rows Ã— 3 integer columns (~0.9 GB); intermediate join table is similar; results are 6.46M Ã— 3 doubles per variable |
| Passes | 5 separate passes over the lookup | Can compute all 5 variables in a single join pass |

**This preserves the trained Random Forest model** (we only change feature construction, not the model) **and preserves the original numerical estimand** (max, min, mean of the same neighbor values).

Expected speedup: from 86+ hours to roughly **5â€“20 minutes** depending on disk I/O and available RAM.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a vectorized edge table from the nb object
#     This replaces build_neighbor_lookup entirely.
#
#     rook_neighbors_unique is an nb object (list of integer vectors)
#     indexed in the same order as id_order.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains the positional indices (into id_order) of
  # the neighbors of the cell whose id is id_order[i].
  # We expand this into a two-column edge table of actual cell ids.

  n <- length(neighbors)
  # Pre-compute lengths for pre-allocation
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)

  from_id     <- rep.int(id_order, lens)
  to_positions <- unlist(neighbors, use.names = FALSE)
  to_id       <- id_order[to_positions]

  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1.37 million rows (directed edges)

cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Join edges to panel data and compute neighbor stats
#     for ALL source variables in ONE pass.
#
#     The idea:
#       - For each (id, year) we need the values of every neighbor
#         in the SAME year.
#       - We join edge_dt to cell_data on (neighbor_id == id, year)
#         to get the neighbor's variable values.
#       - Then group by (id, year) to get max, min, mean.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# -- 2a. Subset cell_data to only the columns we need for the join
#         to reduce memory during the merge.
join_cols <- c("id", "year", neighbor_source_vars)
neighbor_data <- cell_data[, ..join_cols]

# Rename 'id' to 'neighbor_id' so we can join on the neighbor side
setnames(neighbor_data, "id", "neighbor_id")

# -- 2b. Keyed join: edge_dt  âŸ•  neighbor_data  on (neighbor_id, year)
#    We add 'year' to edge_dt via a cross with unique years?  No â€”
#    instead we do a many-to-many merge:
#      edge_dt[neighbor_data]  on neighbor_id
#    but that would replicate edges Ã— years.
#
#    More memory-efficient: join edge_dt to cell_data to get the
#    year for the focal cell, then look up the neighbor's values.

# Strategy:  
#   focal_edges = cell_data[, .(id, year)]  merged with edge_dt on id
#   â†’ gives (id, year, neighbor_id)  ~38.5M rows
#   Then merge with neighbor_data on (neighbor_id, year) to get values.

# Step A: focal cell's (id, year) Ã— its neighbors
setkey(edge_dt, id)
focal_years <- cell_data[, .(id, year)]
setkey(focal_years, id)

# This is the big join: ~6.46M rows Ã— avg ~4 neighbors â‰ˆ 38.5M rows
focal_edges <- edge_dt[focal_years, on = "id", allow.cartesian = TRUE, nomatch = NULL]
# focal_edges columns: id, neighbor_id, year

cat("Focal-edges rows:", nrow(focal_edges), "\n")

# Step B: attach neighbor variable values
setkey(neighbor_data, neighbor_id, year)
setkey(focal_edges, neighbor_id, year)

focal_edges <- neighbor_data[focal_edges, on = .(neighbor_id, year), nomatch = NA]
# Now focal_edges has columns: neighbor_id, year, ntl, ec, ..., id

# Free memory
rm(neighbor_data)
gc()

# -- 2c. Grouped aggregation: max, min, mean per (id, year) per variable
#    Build the aggregation expression dynamically.

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

# Evaluate
neighbor_stats <- focal_edges[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# Handle Inf/-Inf from max/min on all-NA groups â†’ convert to NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# Free the large join table
rm(focal_edges)
gc()

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Merge the neighbor features back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# If previous neighbor columns exist, drop them first to avoid duplication
existing <- intersect(agg_names, names(cell_data))
if (length(existing) > 0) {
  cell_data[, (existing) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Predict with the EXISTING Random Forest model (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g. `rf_model`) is used as-is.
# cell_data now contains the same neighbor feature columns with the
# same names and the same numerical values as the original pipeline,
# so predictions are identical.
#
#   cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor set** | `build_edge_table` reads the same `rook_neighbors_unique` nb object and the same `id_order`; the edge table is a lossless representation of the same adjacency. |
| **Same year matching** | The join on `(neighbor_id, year)` replicates the original `paste(neighbor_id, year)` key lookup exactly. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE` produce identical results. Rows with zero valid neighbors get `Inf`/`-Inf` which are converted to `NA`, matching the original `if (length(idx) == 0) return(c(NA, NA, NA))` logic. |
| **Same column names** | The naming convention `neighbor_{max,min,mean}_{var}` matches the pattern produced by `compute_and_add_neighbor_features`. Adjust the template if your wrapper uses a different convention. |
| **Trained RF model untouched** | No model object is modified; only the feature-construction step is replaced. |

---

## Expected Resource Profile

| Resource | Original | Optimized |
|---|---|---|
| **Time** | ~86+ hours | ~5â€“20 minutes |
| **Peak RAM** | Unbounded (list of 6.46M vectors) | ~4â€“6 GB (38.5M-row join table + stats table) |
| **Disk** | â€” | None beyond existing serialized nb object |

The `data.table` implementation performs the heavy lifting in compiled C code (radix-based keyed joins and grouped aggregations), eliminating millions of R-interpreter-level function calls and string allocations that dominated the original runtime.
END_RESPONSE>>>
