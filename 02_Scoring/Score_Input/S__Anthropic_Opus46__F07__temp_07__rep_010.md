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
run_id: Anthropic_F07_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` iteration that performs character coercion, string pasting, and named-vector lookups.** This is catastrophically slow in R because:

1. **Per-row string operations**: For each of ~6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called individually. Named vector lookup in R is O(n) hash probing *per call*, and the overhead of 6.46M R-level function calls in `lapply` is enormous.

2. **Redundant work across variables**: The neighbor lookup is built once (good), but `compute_neighbor_stats` also uses an R-level `lapply` over 6.46M elements, repeated for each of 5 variables = ~32.3M R-level function invocations.

3. **Memory pressure**: Storing 6.46M list elements, each an integer vector, creates massive list overhead on a 16 GB machine.

**In short**: The algorithm is correct but the implementation is O(rows Ã— avg_neighbors) with enormous R-interpreter constant factors. The 86+ hour estimate is dominated by millions of R-level function calls with string operations.

---

## Optimization Strategy

**Replace the row-level R loops with vectorized joins and grouped aggregations using `data.table`.**

The key insight: the neighbor lookup can be expressed as a **join**. Each cell-year row needs to find its neighbors' values *in the same year*. This is a standard equi-join:

1. **Build an edge table** (a two-column `data.table` of `id â†’ neighbor_id`) from the `nb` object â€” done once, ~1.37M rows.
2. **Join** `cell_data` to the edge table on `(id, year)` â†” `(neighbor_id, year)` to get all neighbor values in one vectorized operation.
3. **Group-aggregate** (`max`, `min`, `mean`) by `(id, year)` â€” fully vectorized in C via `data.table`.

This eliminates all per-row R loops, string pasting, and named-vector lookups. Expected runtime: **minutes, not days.**

The numerical results are identical because `max`, `min`, and `mean` over the same neighbor sets produce the same values. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a directed edge table from the nb object (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique is a list (spdep nb object) of length 344,208.
# id_order is the vector mapping list index â†’ cell id.

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))  # ~1,373,394
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L for no neighbors; filter those out
    nb_i <- nb_i[nb_i > 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  data.table(id = from_id[1:(pos - 1L)], neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order for downstream prediction
cell_data[, .row_order := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Compute neighbor stats for all 5 variables via a single join
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset to only the columns we need for the join (minimize memory)
join_cols <- c("id", "year", neighbor_source_vars)
neighbor_values <- cell_data[, ..join_cols]

# Set key on the neighbor side for fast join
setnames(neighbor_values, "id", "neighbor_id")
setkey(neighbor_values, neighbor_id, year)

# Set key on edge table
setkey(edge_dt, neighbor_id)

# Join: for each edge (id, neighbor_id), attach the neighbor's year and values
# We need to join on (neighbor_id, year), so we first cross edge_dt with years
# via cell_data. More efficient: join cell_data to edge_dt on id, then look up
# neighbor values.

# Approach: 
#   1. Take cell_data's (id, year) and join to edge_dt on id â†’ get (id, year, neighbor_id)
#   2. Join that to neighbor_values on (neighbor_id, year) â†’ get neighbor variable values
#   3. Aggregate by (id, year)

# Step 3a: Expand edges by year
# cell_data has (id, year); edge_dt has (id, neighbor_id)
# We need all (id, year, neighbor_id) combinations that exist.

id_year <- cell_data[, .(id, year)]
setkey(id_year, id)
setkey(edge_dt, id)

# Join: each (id, year) row gets its neighbor_ids
# This produces ~6.46M Ã— avg_neighbors â‰ˆ ~25.8M rows (4 neighbors avg for rook)
expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = NULL]
# Result columns: id, neighbor_id, year

# Step 3b: Look up neighbor values
setkey(expanded, neighbor_id, year)
setkey(neighbor_values, neighbor_id, year)

expanded <- neighbor_values[expanded, on = .(neighbor_id, year), nomatch = NA]
# Now expanded has: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Step 3c: Aggregate by (id, year) â€” compute max, min, mean for each variable
# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("n_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- expanded[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# Handle Inf/-Inf from max/min on all-NA groups (convert to NA)
inf_cols <- grep("_(max|min)$", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Alternative Step 3c (simpler, if the dynamic expression approach is
# finicky in your R version):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Uncomment below and comment out the dynamic approach above if preferred:

# neighbor_stats <- expanded[, {
#   out <- list()
#   for (v in neighbor_source_vars) {
#     vals <- get(v)
#     vals <- vals[!is.na(vals)]
#     if (length(vals) == 0L) {
#       out[[paste0("n_", v, "_max")]]  <- NA_real_
#       out[[paste0("n_", v, "_min")]]  <- NA_real_
#       out[[paste0("n_", v, "_mean")]] <- NA_real_
#     } else {
#       out[[paste0("n_", v, "_max")]]  <- max(vals)
#       out[[paste0("n_", v, "_min")]]  <- min(vals)
#       out[[paste0("n_", v, "_mean")]] <- mean(vals)
#     }
#   }
#   out
# }, by = .(id, year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Merge neighbor stats back into cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove any old neighbor columns if they exist (idempotency)
old_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# Restore original row order (important for predict() alignment)
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The model object and predict() call remain exactly as before.
# e.g.: predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over the identical neighbor sets defined by `rook_neighbors_unique` and the same `id_order` mapping. The join `(id â†’ neighbor_id) Ã— year` reproduces exactly the same cell-year neighbor pairings as the original `build_neighbor_lookup`. |
| **Irregular topology / masked cells** | The edge table is built directly from the `nb` object â€” no rectangular-grid assumption. Cell-years with no neighbors (no match in `edge_dt`) simply get `NA` for all neighbor stats, identical to the original code's `if (length(idx) == 0) return(c(NA, NA, NA))`. |
| **Trained RF model preserved** | No model retraining. The code only reconstructs the same feature columns with the same names and values. |
| **Memory on 16 GB laptop** | The expanded join table is ~26M rows Ã— 7 columns â‰ˆ ~1.5 GB. Together with `cell_data` (~6.46M Ã— 110 cols â‰ˆ ~5.5 GB), total peak is well within 16 GB. The intermediate tables can be removed with `rm(expanded, neighbor_values); gc()` if needed. |

## Expected Runtime

| Phase | Estimated Time |
|---|---|
| Build edge table | < 5 seconds |
| Expand edges Ã— years (join) | ~10â€“30 seconds |
| Look up neighbor values (join) | ~10â€“30 seconds |
| Grouped aggregation (15 stats) | ~30â€“90 seconds |
| Merge back to cell_data | ~10 seconds |
| **Total** | **~2â€“4 minutes** |

This replaces the estimated 86+ hour runtime with a ~3 minute vectorized pipeline while preserving the exact numerical output and the trained model.
END_RESPONSE>>>
