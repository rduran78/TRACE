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
run_id: Anthropic_F02_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- For every one of the ~6.46M rows, the code pastes strings (`paste(id, year)`), then does named-vector indexing into `idx_lookup` (a character-keyed vector of length 6.46M). Named-vector lookup in R is hash-based but the constant factor is large when called millions of times inside `lapply`.
- The result is a **list of 6.46M integer vectors** â€” this alone consumes enormous memory (each list element has overhead â‰ˆ 56+ bytes even if empty, so â‰¥ 360 MB of pure overhead, plus the actual neighbor indices).

### 2. `compute_neighbor_stats` â€” another O(n) `lapply` over 6.46M rows
- Called **5 times** (once per source variable), each time iterating over every row, subsetting a numeric vector, and computing `max/min/mean`. The per-element R function-call overhead dominates.

### 3. Memory
- The 6.46M-element list of neighbor indices, the 6.46M Ã— 110 data frame, and intermediate copies easily exceed 16 GB.

**Summary:** The design is idiomatic R but relies on millions of R-level function calls and string operations. The fix is to **vectorize everything** using `data.table` joins and grouped operations, eliminating both the per-row `lapply` loops and the string-keyed lookups entirely.

---

## Optimization Strategy

| Step | What changes | Why it helps |
|---|---|---|
| **A. Replace `build_neighbor_lookup` with an edge-list + `data.table` equi-join** | Instead of building a 6.46M-element list, expand the `nb` object into a two-column edge table `(cell_id, neighbor_id)`, then join on `(neighbor_id, year)` to get neighbor row indices (or values directly). | One vectorized join replaces 6.46M `paste` + named-vector lookups. `data.table` join is C-level and memory-mapped. |
| **B. Compute all neighbor stats in one grouped aggregation** | After the join attaches neighbor values, a single `data.table` `[, .(max, min, mean), by = .(cell_row)]` computes all three stats at once. | Replaces 6.46M R-level `lapply` calls per variable with one C-level grouped aggregation. |
| **C. Process all 5 variables in one pass** | Melt or simply carry all 5 source columns through the single join, then aggregate all 5 simultaneously. | Reduces the number of large joins from 5 to 1. |
| **D. Avoid the giant neighbor-index list entirely** | We never materialise a 6.46M-element R list. The edge table + join produces a long `data.table` that is aggregated and then discarded. | Saves hundreds of MB of list overhead. |

**Estimated speedup:** The vectorized approach should finish in **minutes** (roughly 5â€“15 min depending on disk/RAM speed), not 86+ hours. Peak RAM â‰ˆ 4â€“6 GB.

---

## Working R Code

```r
# ------------------------------------------------------------------
# 0.  Libraries
# ------------------------------------------------------------------
library(data.table)

# ------------------------------------------------------------------
# 1.  Convert the nb object to a data.table edge list  (one-time, fast)
#     rook_neighbors_unique is a list of integer vectors (spdep::nb),
#     where element i contains the indices (into id_order) of neighbors
#     of id_order[i].
# ------------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: total number of directed edges
  n_edges <- sum(lengths(nb_obj))
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    ni <- nb_obj[[i]]
    len <- length(ni)
    if (len == 0L) next
    from_idx[pos:(pos + len - 1L)] <- i
    to_idx[pos:(pos + len - 1L)]   <- ni
    pos <- pos + len
  }
  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (one per directed rook relationship)

# ------------------------------------------------------------------
# 2.  Convert cell_data to data.table (if not already) and add a
#     row identifier that we will aggregate back to.
# ------------------------------------------------------------------
setDT(cell_data)
cell_data[, .row_id := .I]

# ------------------------------------------------------------------
# 3.  Subset the columns we actually need for the neighbor join
#     to keep peak memory low.
# ------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Small table: (id, year, .row_id) for the focal cells
focal <- cell_data[, .(id, year, .row_id)]

# Small table: (id, year, <source vars>) for the neighbor cells
neighbor_vals <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# ------------------------------------------------------------------
# 4.  Join:  focal --> edge_dt --> neighbor_vals
#     For every focal row, find its neighbors in the same year and
#     pull their variable values.
# ------------------------------------------------------------------
# Step 4a: attach neighbor cell ids to every focal row
#   focal  JOIN  edge_dt  ON  focal.id == edge_dt.cell_id
setkey(edge_dt, cell_id)
setkey(focal, id)
focal_with_nbr <- edge_dt[focal, on = .(cell_id = id),
                           allow.cartesian = TRUE,
                           nomatch = NA]
# Result columns: cell_id, neighbor_id, year, .row_id
# Rows â‰ˆ 6.46M * avg_neighbors (â‰ˆ 4) â‰ˆ 25.8M  (manageable)

# Drop rows where there was no neighbor (isolated cells)
focal_with_nbr <- focal_with_nbr[!is.na(neighbor_id)]

# Step 4b: attach neighbor variable values by (neighbor_id, year)
setkey(neighbor_vals, id, year)
setkey(focal_with_nbr, neighbor_id, year)
joined <- neighbor_vals[focal_with_nbr,
                        on = .(id = neighbor_id, year),
                        nomatch = NA]
# 'joined' now has columns: id (=neighbor_id), year, ntl, ec, ...,
#                            cell_id, .row_id

# ------------------------------------------------------------------
# 5.  Grouped aggregation â€” compute max, min, mean for every
#     (focal row, variable) combination in one pass.
# ------------------------------------------------------------------
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

# Build the j-expression programmatically
agg_call <- as.call(c(
  as.name("list"),
  setNames(agg_exprs, agg_names)
))

stats <- joined[, eval(agg_call), by = .row_id]

# Replace -Inf/Inf from max/min of all-NA groups with NA
for (col in agg_names) {
  set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
}

# ------------------------------------------------------------------
# 6.  Merge the aggregated neighbor features back to cell_data
# ------------------------------------------------------------------
setkey(stats, .row_id)
setkey(cell_data, .row_id)
cell_data <- stats[cell_data, on = ".row_id"]

# Clean up the helper column
cell_data[, .row_id := NULL]

# ------------------------------------------------------------------
# 7.  (Optional) free large intermediates
# ------------------------------------------------------------------
rm(focal, neighbor_vals, focal_with_nbr, joined, stats, edge_dt)
gc()

# ------------------------------------------------------------------
# 8.  Predict with the existing Random Forest model (unchanged)
# ------------------------------------------------------------------
# The trained RF model object (e.g. `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars Ã— 3 stats) that the model expects, with identical
# numerical values, so the estimand is preserved.
#
#   cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
# ------------------------------------------------------------------
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of rook-neighbor values per cell-year as the original code. The `data.table` join replicates the same `(cell, year) â†’ neighbor (cell, year)` mapping. |
| **Trained RF model** | No model retraining occurs. The code only reconstructs the **feature columns** that already existed, with identical names and values, then calls `predict()`. |
| **Column names** | The programmatic `agg_names` vector produces the same `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, â€¦ names (adjust the pattern if your original pipeline used a different naming convention). |

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| Wall-clock time | ~86+ hours | **~5â€“15 minutes** |
| Peak RAM | >16 GB (fails/swaps) | **~4â€“6 GB** |
| Largest intermediate | 6.46M-element R list | ~26M-row `data.table` (flat, column-oriented) |

The key insight is that the entire neighbor-lookup-then-aggregate pattern is just a **relational join followed by a grouped aggregation** â€” exactly what `data.table` is engineered to do at C speed with minimal memory copies.
END_RESPONSE>>>
