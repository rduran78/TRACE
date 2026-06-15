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
run_id: Anthropic_F02_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs a `lapply` over ~6.46 million rows. For each row, it:

1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) â€” these are O(n) hash lookups but repeated millions of times with per-call overhead from `paste`, `as.character`, and subsetting named vectors.
2. Builds a per-row list of neighbor indices, producing a **list of 6.46 million integer vectors**. This list alone consumes enormous memory (each list element has R object overhead of ~56+ bytes, so â‰¥ 362 MB just in overhead, plus the actual index data).

The secondary bottleneck is **`compute_neighbor_stats`**, which iterates over the same 6.46 million-element list, extracting values, removing NAs, and computing `max/min/mean` â€” all in interpreted R with per-element function-call overhead.

**Summary of problems:**
| Issue | Impact |
|---|---|
| Row-level `lapply` over 6.46M rows in `build_neighbor_lookup` | ~hours of interpreted R loop time |
| `paste` + named-vector lookup per row | Slow string operations Ã—6.46M |
| 6.46M-element list of integer vectors | High memory (GBs with overhead) |
| Row-level `lapply` in `compute_neighbor_stats` Ã—5 variables | Repeated slow iteration |
| No vectorization or use of `data.table` / matrix operations | Leaves performance on the table |

---

## Optimization Strategy

**Key insight:** The neighbor relationship is defined at the **cell level** (344,208 cells), not the cell-year level (6.46M rows). We should never loop over 6.46M rows to build lookups. Instead:

1. **Vectorize the neighbor lookup construction** using `data.table` joins. Convert the `nb` object into an edge list (cell_i â†’ cell_j), then join on `(neighbor_id, year)` to get row indices. This replaces the 6.46M-row `lapply` with a single merge.

2. **Compute neighbor stats via grouped aggregation** on the edge list. For each `(row_i, variable)`, the neighbor values are the variable values at the matched `(neighbor_id, year)` rows. We group by `row_i` and compute `max`, `min`, `mean` â€” all in `data.table`, which is C-optimized.

3. **Process all 5 variables in one pass** over the edge-joined table, avoiding redundant joins.

4. **Memory management:** The edge list for directed rook neighbors has ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (worst case). At ~24 bytes/row for two integer columns, this is < 1 GB â€” well within 16 GB RAM.

**Expected speedup:** From 86+ hours to **minutes** (typically 5â€“20 minutes depending on disk I/O and exact data).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table and add a row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Convert the nb object to an edge list (cell-level)
#
#   rook_neighbors_unique is a list of length 344,208 where element i
#   contains the integer indices (into id_order) of cell i's neighbors.
#   id_order maps those indices to actual cell IDs.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id (the focal cell), to_id (the neighbor cell)

cat("Edge list rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Join edges with cell_dt to get (focal_row, neighbor_row) pairs
#
#   For every focal cell-year row, we find the neighbor cell's row in
#   the same year via a keyed join.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Create a small lookup: (id, year) -> row_idx
row_lookup <- cell_dt[, .(id, year, row_idx)]
setkey(row_lookup, id, year)

# Expand edges by year: join focal cell's years onto the edge list
# First, get the unique years each focal cell appears in
focal_years <- cell_dt[, .(id, year, focal_row_idx = row_idx)]

# Join: for each (from_id, year), attach all to_id neighbors
# This gives us (from_id, year, to_id, focal_row_idx)
setkey(edge_dt, from_id)
setkey(focal_years, id)

# Merge edges with focal cell-year rows
expanded <- edge_dt[focal_years, on = .(from_id = id), allow.cartesian = TRUE,
                    nomatch = 0L]
# expanded now has columns: from_id, to_id, year, focal_row_idx

# Now find the neighbor's row index for the same year
setkey(row_lookup, id, year)
expanded[row_lookup, neighbor_row_idx := i.row_idx,
         on = .(to_id = id, year = year)]

# Drop rows where the neighbor doesn't exist in that year
expanded <- expanded[!is.na(neighbor_row_idx)]

cat("Expanded edge-year rows:", nrow(expanded), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Compute neighbor stats for all variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor variable values to the expanded table
# (only the columns we need, to save memory)
neighbor_vals <- cell_dt[expanded$neighbor_row_idx, ..neighbor_source_vars]
expanded <- cbind(expanded[, .(focal_row_idx)], neighbor_vals)

# Group by focal_row_idx and compute max, min, mean for each variable
# Build the aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Compute all stats in one grouped aggregation
stats_dt <- expanded[, 
  lapply(agg_exprs, eval, envir = .SD), 
  by = focal_row_idx
]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Alternative Step 3 (simpler, equally fast, avoids bquote complexity):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
stats_list <- list()
for (v in neighbor_source_vars) {
  sub <- expanded[, .(focal_row_idx, val = neighbor_vals[[v]])]
  sub <- sub[!is.na(val)]
  
  agg <- sub[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row_idx]
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
           paste0("neighbor_", c("max_", "min_", "mean_"), v))
  
  stats_list[[v]] <- agg
}

# Merge all stats together
stats_dt <- stats_list[[1]]
for (k in seq_along(stats_list)[-1]) {
  stats_dt <- merge(stats_dt, stats_list[[k]], by = "focal_row_idx", all = TRUE)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Attach results back to cell_dt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Replace -Inf/Inf from max/min of empty groups with NA
for (col in names(stats_dt)[-1]) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

cell_dt <- merge(cell_dt, stats_dt, by.x = "row_idx", by.y = "focal_row_idx",
                 all.x = TRUE, sort = FALSE)

# Restore original row order
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# Convert back to data.frame if the downstream RF predict() expects one
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | No retraining; we only prepare the same features and call `predict()`. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The `data.table` grouped aggregation is numerically equivalent to the original `lapply` logic (same NA handling, same neighbor resolution by `(id, year)` key). |
| **Column names** | The output columns follow the same `neighbor_{max,min,mean}_{var}` naming convention, matching what `compute_and_add_neighbor_features` would have produced. Adjust the naming pattern if your original code uses a different convention. |

---

## Expected Performance

| Phase | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M `lapply` iterations) | ~1â€“2 min (vectorized edge list + keyed join) |
| Compute neighbor stats (Ã—5 vars) | ~hours (6.46M Ã— 5 `lapply` iterations) | ~2â€“5 min (grouped `data.table` aggregation) |
| Peak memory | Very high (6.46M-element list + copies) | ~4â€“8 GB (edge table + cell_dt) â€” fits in 16 GB |
| **Total** | **86+ hours** | **~5â€“15 minutes** |
END_RESPONSE>>>
