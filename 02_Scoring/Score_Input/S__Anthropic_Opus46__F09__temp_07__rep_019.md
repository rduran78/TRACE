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
run_id: Anthropic_F09_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two fundamental reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-ish in practice
The function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” these are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times with ~4 neighbors each, yielding **~25.8 million character-key hash lookups**. Character operations and named-vector indexing in R are slow.

The critical insight is that **the neighbor graph is purely spatial and identical every year**. The current code rebuilds the mapping from scratch across all cell-year rows, even though the spatial adjacency never changes. This is entirely redundant work.

### 2. `compute_neighbor_stats` uses per-row `lapply`
For each of the 5 variables, the function loops over 6.46 million rows in R-level `lapply`, calling `max`, `min`, and `mean` on small vectors. That's **~32.3 million R function calls** (5 vars Ã— 6.46M rows), each with overhead.

### Memory is not the bottleneck; R-level iteration is.

---

## Optimization Strategy

**Core idea:** Build a **year-free spatial neighbor edge table once** (a two-column data.table of `cell_id â†’ neighbor_id`, ~1.37M rows), then for each year, **join** the yearly cell attributes onto this table and compute grouped `max`, `min`, `mean` using `data.table` vectorized aggregation. This eliminates all per-row R-level loops and all character-key hashing.

### Steps:

1. **Convert `rook_neighbors_unique` (spdep nb object) into a spatial edge `data.table`** with columns `(id, neighbor_id)`. This is ~1.37M rows and is built **once**.

2. **For each year**, subset the panel, join cell attributes onto the edge table by `neighbor_id`, then group-by `id` to compute `max`, `min`, `mean` for each neighbor source variable â€” all vectorized in `data.table`.

3. **Join the resulting neighbor stats back** onto the main panel `data.table` by `(id, year)`.

4. **Predict** with the existing trained Random Forest model (unchanged).

**Expected speedup:** The entire neighbor-feature computation should drop from ~86 hours to **minutes** (typically 2â€“10 minutes depending on disk I/O), because:
- The edge table is 1.37M rows (not 6.46M).
- `data.table` grouped aggregation is C-level, vectorized, and cache-friendly.
- No character-key construction or named-vector lookup.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table if not already
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the spatial edge table ONCE from the nb object
#
#   rook_neighbors_unique : spdep nb object (list of integer vectors)
#   id_order              : vector mapping position index â†’ cell id
#
#   Result: edge_dt with columns (id, neighbor_id), ~1.37M rows
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices of neighbors for cell at position i
  from <- rep(
    seq_along(neighbors),
    times = lengths(neighbors)
  )
  to <- unlist(neighbors, use.names = FALSE)

  # Remove any zero-length / empty-neighbor entries (already handled by rep/unlist)
  # Map positional indices to actual cell IDs
  edge_dt <- data.table(
    id          = id_order[from],
    neighbor_id = id_order[to]
  )
  return(edge_dt)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Edge table built: %s directed neighbor pairs (expected ~1,373,394)\n",
  format(nrow(edge_dt), big.mark = ",")
))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all variables, all years at once
#
#   Strategy:
#     - Take the edge table (id, neighbor_id).
#     - Join neighbor attributes from cell_dt by (neighbor_id, year).
#     - Group by (id, year) and compute max, min, mean.
#
#   This is fully vectorized in data.table's C backend.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset only the columns we need for the join (save memory on 16 GB laptop)
join_cols <- c("id", "year", neighbor_source_vars)
attr_dt   <- cell_dt[, ..join_cols]

# Rename 'id' to 'neighbor_id' so we can join on neighbor_id + year
setnames(attr_dt, "id", "neighbor_id")

# Key the attribute table for fast join
setkey(attr_dt, neighbor_id, year)

# Cross the edge table with every year present in the data
years <- sort(unique(cell_dt$year))

# Expand edge_dt Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
# On 16 GB RAM this is feasible: 38.5M Ã— (2 ints + 1 int for year) â‰ˆ < 1 GB
edge_year_dt <- CJ_edge_year <- edge_dt[, .(year = years), by = .(id, neighbor_id)]

cat(sprintf(
  "Edge-year table: %s rows\n",
  format(nrow(edge_year_dt), big.mark = ",")
))

# Join neighbor attributes onto the edge-year table
setkey(edge_year_dt, neighbor_id, year)
edge_year_dt <- attr_dt[edge_year_dt, on = .(neighbor_id, year), nomatch = NA]

# Now edge_year_dt has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id
# where the variable values belong to the NEIGHBOR cell.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Grouped aggregation â€” compute max, min, mean per (id, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build the aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- edge_year_dt[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(id, year)
]

# Handle cells with no valid neighbors: max/min of empty â†’ -Inf/Inf â†’ set to NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, which(is.infinite(vals)), col_name, NA_real_)
}

cat(sprintf(
  "Neighbor stats computed: %s rows Ã— %s new columns\n",
  format(nrow(neighbor_stats), big.mark = ","),
  length(agg_names)
))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Join neighbor stats back onto the main panel
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove old neighbor columns if they exist (from prior runs)
old_cols <- intersect(agg_names, names(cell_dt))
if (length(old_cols) > 0) {
  cell_dt[, (old_cols) := NULL]
}

setkey(cell_dt, id, year)
setkey(neighbor_stats, id, year)

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

cat("Neighbor features joined to main panel.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rf_model is the pre-trained model object already in memory.
# The predictor column names in cell_dt must match those used at training.

cell_dt[, rf_prediction := predict(rf_model, newdata = .SD)]

cat("Predictions complete.\n")

# Convert back to data.frame if downstream code expects one
cell_data <- as.data.frame(cell_dt)
```

---

### If the `CJ`-style expansion is too large for 16 GB RAM

The edge-year expansion (~38.5M rows Ã— several numeric columns) may push memory on a 16 GB machine. Here is a **year-chunked** alternative for Step 2â€“3 that processes one year at a time and is still extremely fast:

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MEMORY-SAFE VARIANT: Process one year at a time
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
join_cols <- c("id", "year", neighbor_source_vars)

all_stats <- rbindlist(lapply(years, function(yr) {

  # Subset this year's attributes
  yr_attr <- cell_dt[year == yr, ..join_cols]
  setnames(yr_attr, "id", "neighbor_id")
  setkey(yr_attr, neighbor_id)

  # Join neighbor attributes onto the spatial edge table
  merged <- yr_attr[edge_dt, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]
  # merged now has: neighbor_id, year, <vars>, id

  # Aggregate by focal cell id
  stats <- merged[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 0L
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      n <- length(vals)
      k <- k + 1L; out[[k]] <- if (n > 0) max(vals)  else NA_real_
      k <- k + 1L; out[[k]] <- if (n > 0) min(vals)  else NA_real_
      k <- k + 1L; out[[k]] <- if (n > 0) mean(vals) else NA_real_
    }
    names(out) <- paste0("neighbor_",
      rep(c("max_", "min_", "mean_"), length(neighbor_source_vars)),
      rep(neighbor_source_vars, each = 3L)
    )
    out
  }, by = .(id)]

  stats[, year := yr]
  stats
}))

cat(sprintf(
  "Neighbor stats (chunked): %s rows Ã— %s new columns\n",
  format(nrow(all_stats), big.mark = ","),
  ncol(all_stats) - 2L
))

# Join back onto main panel (same as Step 4 above)
agg_names <- setdiff(names(all_stats), c("id", "year"))
old_cols  <- intersect(agg_names, names(cell_dt))
if (length(old_cols) > 0) cell_dt[, (old_cols) := NULL]

setkey(all_stats, id, year)
setkey(cell_dt,   id, year)
cell_dt <- all_stats[cell_dt, on = .(id, year)]
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with character-key hashing | 1.37M-row edge `data.table` built once |
| **Stat computation** | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R calls | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** |
| **RAM footprint** | Large named vectors + list of 6.46M integer vectors | ~1â€“2 GB peak (year-chunked variant: ~0.5 GB peak) |
| **Trained RF model** | âœ… Preserved, unchanged | âœ… Preserved, unchanged |
| **Numerical results** | `max`, `min`, `mean` of rook-neighbor values | Identical `max`, `min`, `mean` â€” same estimand |
END_RESPONSE>>>
