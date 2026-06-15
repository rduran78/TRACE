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
run_id: Anthropic_F09_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” these are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times Ã— ~4 neighbors each â‰ˆ **~26 million character-key hash lookups**.

Character allocation, pasting, and named-vector lookup are all slow in R. The resulting `neighbor_lookup` list itself consumes substantial memory (6.46M list elements, each an integer vector).

### 2. `compute_neighbor_stats` is called per-variable with an R-level `lapply` over 6.46M rows
Each call iterates 6.46M rows in pure R, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times (once per source variable), totaling ~32.3 million R-level function invocations.

### Root cause summary
The neighbor topology is **static across years** (rook contiguity doesn't change), yet the code rebuilds a year-specific lookup by cross-joining cells Ã— years at the row level. The key insight the prompt already identifies is correct: **build the spatial adjacency table once, then join yearly attributes onto it.**

---

## Optimization Strategy

1. **Build a static neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` derived from the `nb` object. This has ~1.37M rows and is year-independent.

2. **Join yearly attributes via `data.table`** â€” For each year, the cell-year attributes are already in the panel. We join the neighbor edge table to the panel keyed on `(neighbor_id, year)` to pull neighbor values, then aggregate `max/min/mean` grouped by `(cell_id, year)`. This replaces millions of R-level loops with vectorized `data.table` grouped operations.

3. **Process all 5 variables in a single grouped aggregation** rather than looping variable-by-variable, reducing the number of joins from 5 to 1.

4. **Memory**: The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The join expands to ~1.37M Ã— 28 years â‰ˆ 38.4M rows temporarily, which at ~5 numeric columns is ~1.5 GB â€” feasible on 16 GB RAM.

**Expected speedup**: From ~86 hours to **minutes** (typically 2â€“10 minutes depending on disk I/O and RAM pressure).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and
#         the 5 neighbor source variables.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial neighbor edge table (year-free).
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos     <- 1L

  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    n <- length(nbrs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 M rows, two integer columns

cat("Edge table rows:", nrow(edge_table), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Join yearly attributes and compute neighbor stats in one pass.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the columns we need for the neighbor join (keep it lean)
neighbor_vals_dt <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# Key the attribute table for fast join on (neighbor_id = id, year)
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)

# Cross-join edge_table with all years, then join neighbor attributes
# More memory-efficient: join edge_table to cell_data's year per cell_id

# First, get the unique years
years <- sort(unique(cell_data$year))

# Expand edge_table Ã— years  (~1.37M Ã— 28 â‰ˆ 38.4M rows)
# Use CJ-like expansion efficiently:
edge_year <- edge_table[, .(year = years), by = .(cell_id, neighbor_id)]

cat("Edge-year rows:", nrow(edge_year), "\n")

# Join neighbor attributes onto edge_year
setkey(edge_year, neighbor_id, year)
edge_year <- neighbor_vals_dt[edge_year, on = .(neighbor_id, year)]

# Now aggregate: for each (cell_id, year), compute max/min/mean of each variable
# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- edge_year[,
  lapply(agg_exprs, eval),
  by = .(cell_id, year)
]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
for (col_name in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col_name]])), col_name, NA_real_)
}

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Merge neighbor stats back onto cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove any pre-existing neighbor columns to avoid duplication
existing_nbr_cols <- intersect(names(cell_data), agg_names)
if (length(existing_nbr_cols) > 0) {
  cell_data[, (existing_nbr_cols) := NULL]
}

# Merge
setkey(cell_data, id, year)
setkey(neighbor_stats, cell_id, year)
cell_data <- neighbor_stats[cell_data, on = .(cell_id = id, year = year)]

# Restore the "id" column name (the join puts cell_id as the key)
setnames(cell_data, "cell_id", "id")

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the already-trained Random Forest (unchanged).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object (e.g., `rf_model`) is preserved as-is.
# Prediction proceeds exactly as before:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# No retraining is performed. The numerical estimand is identical because
# the neighbor max, min, and mean values are computed with the same
# formulas (max, min, mean with na.rm) on the same neighbor topology.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Clean up large temporaries
rm(edge_year, neighbor_vals_dt, neighbor_stats)
gc()
```

### If memory is tight (~38.4M row join table â‰ˆ 1.5 GB), process year-by-year:

```r
# â”€â”€ Memory-conservative variant: process one year at a time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

setkey(cell_data, id, year)
results_list <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]

  # Subset this year's attributes
  yr_data <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
  setnames(yr_data, "id", "neighbor_id")
  setkey(yr_data, neighbor_id)

  # Join neighbor attributes onto edge_table
  edges_with_vals <- yr_data[edge_table, on = .(neighbor_id), nomatch = NA]

  # Aggregate per cell_id
  stats_yr <- edges_with_vals[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 1L
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k]] <- NA_real_; out[[k+1L]] <- NA_real_; out[[k+2L]] <- NA_real_
      } else {
        out[[k]] <- max(vals); out[[k+1L]] <- min(vals); out[[k+2L]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- agg_names
    out
  }, by = .(cell_id)]

  stats_yr[, year := yr]
  results_list[[yi]] <- stats_yr

  if (yi %% 5 == 0) cat("Completed year", yr, "\n")
}

neighbor_stats <- rbindlist(results_list)

# Merge back (same as Step 3 above)
existing_nbr_cols <- intersect(names(cell_data), agg_names)
if (length(existing_nbr_cols) > 0) cell_data[, (existing_nbr_cols) := NULL]

setkey(neighbor_stats, cell_id, year)
setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = .(cell_id = id, year = year)]
setnames(cell_data, "cell_id", "id")

rm(results_list, neighbor_stats, edges_with_vals, yr_data)
gc()
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with character hashing | Static 1.37M-row integer edge table built once |
| **Stat computation** | 5 Ã— 6.46M R-level `lapply` calls | Single vectorized `data.table` grouped aggregation (or 28 yearly passes) |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** |
| **RAM peak** | Large list of 6.46M elements + overhead | ~1.5 GB temporary join table (or ~55 MB/year in conservative variant) |
| **RF model** | Unchanged â€” no retraining | Unchanged â€” no retraining |
| **Numerical result** | `max/min/mean` with `na.rm` | Identical `max/min/mean` with `na.rm` on same neighbor topology |
END_RESPONSE>>>
