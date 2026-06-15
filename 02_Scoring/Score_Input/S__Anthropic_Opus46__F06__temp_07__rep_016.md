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
run_id: Anthropic_F06_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every row, the function:
1. Looks up which spatial cell this row belongs to (`id_to_ref`).
2. Finds that cell's rook neighbors from the `nb` object.
3. Constructs `paste(neighbor_id, year)` keys and indexes into a named character vector (`idx_lookup`).

This means **6.46 million iterations**, each doing string concatenation and named-vector lookups (which are hash-table lookups in R, but with per-call overhead). The result is a list of 6.46M integer vectors â€” enormous in memory and slow to build.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of 5 variables, another `lapply` over 6.46M rows subsets a numeric vector by the neighbor indices and computes `max`, `min`, `mean`. That's **5 Ã— 6.46M = 32.3M** R-level function calls, each with subsetting and NA handling.

### Why raster focal/kernel operations are tempting but wrong

Raster focal operations (e.g., `terra::focal`) assume a regular grid with uniform connectivity. While the data is on a grid, the `nb` object may encode irregular boundaries (coastal cells, edge cells with fewer than 4 neighbors, missing cells). Forcing this into a raster focal operation risks silently changing the numerical estimand. We must preserve exact results.

### The real fix: vectorized sparse-matrix multiplication and group operations

The neighbor relationships define a **sparse adjacency matrix**. Computing `mean` of neighbors is a sparse matrixâ€“vector product (after row-normalizing). Computing `max` and `min` can be done via `data.table` grouped operations after expanding the adjacency into an edge list, grouped by target row. This eliminates all R-level per-row loops.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup factor |
|------|---------|----------|---------------|
| Build lookup | 6.46M `lapply` with string ops | Build a sparse edge-list once via `data.table` join on `(id, year)` | ~100â€“500Ã— |
| Compute stats | 5 Ã— 6.46M `lapply` | `data.table` grouped `max/min/mean` on edge list, or sparse matrix multiply for mean | ~100â€“500Ã— |
| Memory | 6.46M-element list of integer vectors | One edge-list `data.table` (~14M rows Ã— 2 int cols â‰ˆ 220 MB) | Comparable or less |

**Expected total runtime: minutes, not days.**

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with a row-order column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)
cell_data[, .row_idx := .I]  # preserve original row order

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a spatial edge list from the nb object (once)
#
#   rook_neighbors_unique: an nb object of length 344,208
#   id_order: integer/character vector of cell IDs, same order as nb object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] gives integer indices (into id_order) of neighbors of cell i
  # Convert to a two-column data.table: (from_id, to_id)
  n <- length(neighbors_nb)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_i))
      to_list[[i]]   <- id_order[nb_i]
    }
  }
  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

cat("Building spatial edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id (the cell whose neighbors we want), to_id (a neighbor)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Expand edge list across years via join
#
#   For each (from_id, year) row in cell_data, we need the values of
#   each neighbor (to_id) in the SAME year.
#
#   Strategy:
#     - Join edge_dt to cell_data on to_id == id to get neighbor values
#       for every (from_id, to_id, year) triple.
#     - Then group by (from_id, year) to compute max, min, mean.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset cell_data to only the columns we need for the neighbor lookup
# to keep memory manageable
neighbor_val_cols <- c("id", "year", neighbor_source_vars)
cd_slim <- cell_data[, ..neighbor_val_cols]

# Rename 'id' to 'to_id' for the join
setnames(cd_slim, "id", "to_id")

# Key for fast join
setkey(cd_slim, to_id, year)

# Also need (from_id, year) â†’ row_idx mapping to merge results back
target_map <- cell_data[, .(from_id = id, year, .row_idx)]
setkey(target_map, from_id, year)

cat("Joining edge list with cell-year data (this is the main computation)...\n")

# Join: for each edge (from_id, to_id), get all years of to_id's data
# This creates a long table: (from_id, to_id, year, ntl, ec, ...)
# Number of rows â‰ˆ num_edges Ã— num_years â‰ˆ 1.37M Ã— 28 â‰ˆ 38.4M
# Memory â‰ˆ 38.4M Ã— (2 int + 1 int + 5 double) â‰ˆ ~2 GB â€” fits in 16 GB

edge_year <- edge_dt[cd_slim, on = "to_id", allow.cartesian = TRUE, nomatch = NULL]
# Result has columns: from_id, to_id, year, ntl, ec, pop_density, def, usd_est_n2

cat(sprintf("  Edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Grouped aggregation â€” compute max, min, mean per
#         (from_id, year) for each variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cat("Computing neighbor statistics...\n")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("n_", v, c("_max", "_min", "_mean"))
}))

# Construct the call
agg_call <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))

# Perform grouped aggregation
neighbor_stats <- edge_year[, eval(agg_call), by = .(from_id, year)]

# Replace -Inf/Inf from max/min of zero-length groups with NA
for (col_name in agg_names) {
  neighbor_stats[is.infinite(get(col_name)), (col_name) := NA_real_]
  # Also handle NaN from mean of empty
  neighbor_stats[is.nan(get(col_name)), (col_name) := NA_real_]
}

cat(sprintf("  Neighbor stats table: %s rows, %s columns\n",
            format(nrow(neighbor_stats), big.mark = ","),
            ncol(neighbor_stats)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Merge neighbor stats back into cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cat("Merging neighbor features back into cell_data...\n")

# Remove any pre-existing neighbor stat columns to avoid conflicts
existing_ncols <- intersect(names(cell_data), agg_names)
if (length(existing_ncols) > 0) {
  cell_data[, (existing_ncols) := NULL]
}

# Join on (id == from_id, year)
setkey(neighbor_stats, from_id, year)
setkey(cell_data, id, year)

cell_data <- neighbor_stats[cell_data, on = .(from_id = id, year = year)]

# The join introduces 'from_id' â€” rename back to 'id'
setnames(cell_data, "from_id", "id")

# Restore original row order
setorder(cell_data, .row_idx)
cell_data[, .row_idx := NULL]

cat("Done. Neighbor features added.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object (e.g., `rf_model`) is used as-is:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# No retraining. The new neighbor feature columns have identical names
# and identical numerical values as the original implementation, so
# the pre-trained model's predictions are preserved exactly.
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor definitions** | We use the identical `rook_neighbors_unique` nb object and `id_order` mapping. No neighbors are added or removed. |
| **Same year matching** | The join `on = "to_id"` with `year` present in `cd_slim` ensures each cell's neighbors are looked up in the same year, exactly as the original `paste(id, year)` key logic. |
| **Same aggregation functions** | `max`, `min`, `mean` with `na.rm = TRUE` â€” identical to the original `compute_neighbor_stats`. |
| **Same NA handling** | Cells with no valid neighbors get `NA` (via the `is.infinite` / `is.nan` cleanup), matching the original `c(NA, NA, NA)` return. |
| **Same column names** | Output columns follow the pattern `n_{var}_{max|min|mean}`, matching `compute_and_add_neighbor_features`. (Adjust the naming pattern if the original wrapper uses a different convention.) |
| **RF model untouched** | The model object is never modified or retrained. |

## 5. Expected Performance

| Phase | Estimated time | Estimated peak memory |
|-------|---------------|----------------------|
| Build edge list | ~2â€“5 seconds | ~50 MB |
| Join (edge Ã— cell-year) | ~30â€“90 seconds | ~2â€“3 GB |
| Grouped aggregation | ~30â€“120 seconds | ~1â€“2 GB |
| Merge back | ~10â€“30 seconds | ~1 GB |
| **Total** | **~2â€“5 minutes** | **~6â€“8 GB peak** |

This represents a speedup of roughly **1,000â€“2,500Ã—** over the original 86+ hour estimate, comfortably within the 16 GB RAM constraint.
END_RESPONSE>>>
