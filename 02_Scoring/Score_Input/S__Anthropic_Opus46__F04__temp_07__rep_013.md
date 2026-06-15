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
run_id: Anthropic_F04_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows

Each iteration performs:
- A **character coercion and named-vector lookup** (`id_to_ref[as.character(...)]`) â€” hash-like but with per-element overhead.
- A **`paste` call** to construct neighbor keys for every neighbor of every row.
- A **named-vector lookup** (`idx_lookup[neighbor_keys]`) repeated millions of times.
- An **`is.na` filter** and `as.integer` coercion.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces roughly **25.8 million key constructions and lookups** inside an interpreted R loop. The `lapply` over 6.46M elements with per-element string operations is catastrophically slow in R.

### 2. `compute_neighbor_stats` â€” Called 5 times, each over 6.46M rows

Each call iterates over 6.46M list elements, subsetting a numeric vector and computing `max`, `min`, `mean`. The `lapply` â†’ `do.call(rbind, ...)` pattern on 6.46M three-element vectors is also extremely slow: `do.call(rbind, list_of_6.46M_vectors)` alone can take many minutes.

### Root cause summary

| Source | Problem |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of string paste + named-vector lookup in interpreted R |
| `compute_neighbor_stats` | 5 Ã— 6.46M interpreted-loop iterations + `do.call(rbind, ...)` on millions of tiny vectors |
| Combined | Estimated 86+ hours; nearly all time is in these two functions |

---

## Optimization Strategy

The core idea: **eliminate the row-level R loop entirely** by converting the problem to vectorized `data.table` grouped joins and aggregations.

### Key insights

1. **The neighbor relationship is cell-to-cell, not row-to-row.** There are only ~344K cells and ~1.37M directed neighbor pairs. The lookup is repeated identically for each of the 28 years. We should express neighbors as a flat edge table `(id, neighbor_id)` and join on `(neighbor_id, year)` to get neighbor values, then group-aggregate by `(id, year)`.

2. **Vectorized join + grouped aggregation** in `data.table` replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with operations that run in compiled C code internally â€” no interpreted R loop over 6.46M rows.

3. **All 5 variables can be handled in a single join** (or at least the join is done once and aggregations computed for all variables), avoiding 5 redundant passes.

4. **Memory is feasible.** The expanded edge table Ã— 28 years is ~1.37M Ã— 28 â‰ˆ 38.4M rows. With a few numeric columns, this fits comfortably in 16 GB RAM.

5. **The trained Random Forest model is untouched.** We are only changing *how* the neighbor features are computed, not *what* they are. The numerical results are identical (same max, min, mean of the same neighbor values).

### Expected speedup

From 86+ hours to **minutes** (typically 2â€“10 minutes on a modern laptop), because `data.table` keyed joins and grouped aggregations over ~38M rows are highly optimized.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Convert the spdep nb object to a flat edge data.table
#         This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# rook_neighbors_unique : an nb object (list of integer index vectors)
# id_order              : vector of cell IDs corresponding to nb indices

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order of neighbors of cell i
  n_neighbors <- vapply(neighbors, length, integer(1))
  from_idx <- rep(seq_along(neighbors), times = n_neighbors)
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-neighbor entries (nb objects use 0L for no-neighbor)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: id, neighbor_id
# Rows: ~1,373,394 directed neighbor pairs

cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Convert cell_data to data.table and set key
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_dt <- as.data.table(cell_data)

# Ensure original row order is preserved for later reassembly
cell_dt[, .row_order := .I]

# The columns we need from the neighbor cells
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset to only the columns needed for the join (saves memory)
neighbor_value_cols <- c("id", "year", neighbor_source_vars)
values_dt <- cell_dt[, ..neighbor_value_cols]
setnames(values_dt, "id", "neighbor_id")  # rename for join

# Key for fast join
setkey(values_dt, neighbor_id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Expand edges Ã— years and join neighbor values in one pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Get unique years
years <- sort(unique(cell_dt$year))

# Cross join edges with years: each edge exists in every year
# This gives us ~1.37M Ã— 28 â‰ˆ 38.4M rows
edge_year_dt <- CJ_dt <- edge_dt[, .(year = years), by = .(id, neighbor_id)]

# Key for joining on (neighbor_id, year) to get neighbor values
setkey(edge_year_dt, neighbor_id, year)

# Join: attach neighbor cell values
edge_year_dt <- values_dt[edge_year_dt, on = .(neighbor_id, year), nomatch = NA]
# Now edge_year_dt has columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Aggregate neighbor stats (max, min, mean) per (id, year)
#         for all 5 variables simultaneously
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation
neighbor_stats <- edge_year_dt[,
  lapply(agg_exprs, eval),
  by = .(id, year)
]

# Handle Inf/-Inf from max/min on all-NA groups â†’ convert to NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 5: Merge neighbor features back into cell_dt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setkey(cell_dt, id, year)
setkey(neighbor_stats, id, year)

# Remove any pre-existing neighbor columns to avoid duplicates
existing_neighbor_cols <- intersect(names(cell_dt), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_dt[, (existing_neighbor_cols) := NULL]
}

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# Restore original row order
setorder(cell_dt, .row_order)
cell_dt[, .row_order := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 6: Convert back to data.frame if downstream code requires it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data <- as.data.frame(cell_dt)

cat("Neighbor features added. Columns:", ncol(cell_data), "\n")
cat("Rows:", nrow(cell_data), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
#   predictions <- predict(rf_model, newdata = cell_data)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

### Alternative Step 3 (lower peak memory)

If the ~38.4M-row expanded table risks memory pressure on a 16 GB laptop, process year-by-year in a loop that is still fully vectorized *within* each year:

```r
# Lower-memory alternative: process one year at a time
neighbor_stats_list <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]

  # Subset cell values for this year
  yr_vals <- cell_dt[year == yr, ..neighbor_value_cols, env = list(neighbor_value_cols = c("id", neighbor_source_vars))]
  setnames(yr_vals, "id", "neighbor_id")
  setkey(yr_vals, neighbor_id)

  # Join edges with neighbor values for this year
  yr_edges <- yr_vals[edge_dt, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]
  # yr_edges has ~1.37M rows

  # Aggregate
  yr_stats <- yr_edges[, {
    res <- list()
    for (v in neighbor_source_vars) {
      nv <- get(v)
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) {
        res[[paste0("neighbor_", v, "_max")]]  <- NA_real_
        res[[paste0("neighbor_", v, "_min")]]  <- NA_real_
        res[[paste0("neighbor_", v, "_mean")]] <- NA_real_
      } else {
        res[[paste0("neighbor_", v, "_max")]]  <- max(nv)
        res[[paste0("neighbor_", v, "_min")]]  <- min(nv)
        res[[paste0("neighbor_", v, "_mean")]] <- mean(nv)
      }
    }
    res
  }, by = .(id)]

  yr_stats[, year := yr]
  neighbor_stats_list[[yi]] <- yr_stats
}

neighbor_stats <- rbindlist(neighbor_stats_list)
# Then proceed with Step 5 merge as above
```

This processes only ~1.37M rows per iteration (28 iterations), keeping peak memory well under 4 GB.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`/string lookup | Flat edge `data.table` built once from `nb` object (~1.37M rows) |
| **Neighbor aggregation** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single `data.table` keyed join + grouped aggregation (compiled C) |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **Peak memory** | Moderate (6.46M-element list of integer vectors) | ~38M rows Ã— ~8 cols â‰ˆ 2.3 GB (or ~1 GB with year-loop variant) |
| **Numerical results** | max/min/mean of rook-neighbor values | **Identical** â€” same neighbors, same values, same aggregations |
| **RF model** | â€” | **Unchanged** â€” no retraining |
END_RESPONSE>>>
