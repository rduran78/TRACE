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
run_id: Anthropic_F06_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine â€” small constant per cell).
2. Constructs character keys by pasting cell ID and year (`paste(id, year, sep="_")`).
3. Performs a named-vector lookup (`idx_lookup[neighbor_keys]`) â€” this is a **hash lookup on a character vector of length 6.46M**, repeated 6.46M times.

The `idx_lookup` named vector has 6.46M entries. Each call indexes into it with ~4 keys (average rook neighbors â‰ˆ 1,373,394 / 344,208 â‰ˆ 4). That's ~25.8M character hash lookups total, but the overhead of `lapply` over 6.46M iterations, each allocating small vectors and calling `paste`, dominates. The result is a **list of length 6.46M**, each element a small integer vector â€” enormous memory overhead from list structure alone.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 variables, another `lapply` iterates over 6.46M list elements, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times = ~32.3M R-level loop iterations with per-element allocation.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular grid with a fixed rectangular kernel. Here the grid cells have an irregular neighbor structure (coastal cells, boundary cells have fewer neighbors), and the data is in long panel format (cell Ã— year), not a raster stack. Converting to raster stacks for 28 years and 5 variables is possible but would require reshaping and would lose the irregular-boundary handling. The better approach is to **vectorize the neighbor computation using sparse matrix algebra**, which is the direct analogue of a focal operation on an irregular grid.

### Summary

| Component | Current complexity | Core issue |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations + character hashing | Slow loop, character key construction |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M R-level iterations | Slow loop, per-element allocation |
| Total estimated wall time | 86+ hours | Both bottlenecks compound |

---

## 2. Optimization Strategy

### Key insight: Express neighbor aggregation as sparse matrix multiplication

A rook-neighbor adjacency can be represented as a sparse matrix **W** of dimension 344,208 Ã— 344,208. For a given year, the neighbor-max, neighbor-min, and neighbor-mean of a variable can be computed by operating on the sparse structure directly. But `max` and `min` are not linear, so we can't use a single matrix multiply for all three. However:

- **Neighbor mean**: `W_row_normalized %*% x` (one sparse matrix multiply per year-variable).
- **Neighbor max and min**: Iterate over the sparse structure, but do it in **vectorized C-level code** via `data.table` grouping or via the sparse matrix's row structure.

The overall strategy:

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, build a spatial-only adjacency edge list (source_cell, neighbor_cell) of ~1.37M rows. Then join on year to expand to ~1.37M Ã— 28 â‰ˆ 38.5M edge-year rows. This is a `data.table` cross-join â€” fast and memory-efficient (~38.5M rows Ã— a few columns â‰ˆ < 1 GB).

2. **Compute all neighbor stats via `data.table` grouped aggregation.** For each (source_cell, year) group, compute max, min, mean of the neighbor values. `data.table` does this in parallel C-level code â€” orders of magnitude faster than R-level `lapply`.

3. **Do all 5 variables in one pass** by joining once and aggregating all variables simultaneously.

### Expected speedup

- `data.table` grouped aggregation over 38.5M rows with ~6.46M groups: **seconds to low minutes**.
- Total pipeline: **under 5 minutes** on a 16 GB laptop, down from 86+ hours.
- Memory: edge table ~38.5M rows Ã— ~8 columns Ã— 8 bytes â‰ˆ 2.5 GB peak, feasible on 16 GB.

### Numerical equivalence

The `max`, `min`, and `mean` operations are computed on exactly the same neighbor sets with the same values. The results are **numerically identical** (not approximate). The trained Random Forest model is untouched.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Inputs assumed available
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data            : data.frame/data.table with columns id, year,
#                        ntl, ec, pop_density, def, usd_est_n2, ...
# rook_neighbors_unique: spdep nb object (list of length 344,208)
# id_order             : integer/character vector of cell IDs in the
#                        same order as rook_neighbors_unique
# rf_model             : pre-trained Random Forest model (untouched)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build spatial edge list from the nb object (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order of neighbors of

  # the i-th cell. We expand this into a two-column edge list of cell IDs.
  n <- length(nb_obj)
  # Pre-compute total edges for pre-allocation
  n_edges <- sum(lengths(nb_obj))

  from_idx <- rep(seq_len(n), times = lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep convention where 0L means "no neighbors"
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    source_id   = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table and set keys
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim table with only the columns we need for the neighbor join
# to minimize memory during the large join
keep_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..keep_cols]
setkey(neighbor_vals_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Expand edge list Ã— years and join neighbor values
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Get unique years
years <- sort(unique(cell_data$year))

# Cross-join edges with years: each spatial edge exists in every year
# This produces ~1.37M Ã— 28 â‰ˆ 38.5M rows
edge_year_dt <- CJ_dt <- edge_dt[, .(source_id, neighbor_id, year = rep(list(years), .N))]
# More memory-efficient approach: use CJ inside a merge
edge_year_dt <- edge_dt[, .(year = years), by = .(source_id, neighbor_id)]
cat("Edge-year rows:", nrow(edge_year_dt), "\n")

# Join to get neighbor variable values
# We join on neighbor_id + year to get the variable values of each neighbor
setkey(edge_year_dt, neighbor_id, year)
setnames(neighbor_vals_dt, "id", "neighbor_id")

edge_year_dt <- neighbor_vals_dt[edge_year_dt, on = .(neighbor_id, year)]

# Now edge_year_dt has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, source_id

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Grouped aggregation â€” compute max, min, mean per
#         (source_id, year) for all 5 variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Build aggregation expressions dynamically
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

# Construct the call
agg_call <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))

# Execute grouped aggregation
neighbor_stats <- edge_year_dt[, eval(agg_call), by = .(source_id, year)]

# Handle Inf/-Inf from max/min on all-NA groups (replace with NA)
inf_cols <- grep("neighbor_(max|min)_", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")
# Expected: ~6.46M (one row per cell-year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Merge neighbor stats back into cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Rename source_id back to id for the merge
setnames(neighbor_stats, "source_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The RF model expects a data.frame with the original column names.
# cell_data now contains all ~110 predictors + the 15 new neighbor features.
# Predict exactly as before:
#
# cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
#
# (Uncomment and adapt to your actual prediction call.)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLEANUP: Free large intermediate objects
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(edge_dt, edge_year_dt, neighbor_vals_dt, neighbor_stats)
gc()
```

---

## 4. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with character hashing | Eliminated; replaced by `data.table` keyed join |
| **Neighbor stats** | 5 Ã— 6.46M-iteration `lapply` with per-element allocation | Single `data.table` grouped aggregation over all 5 vars |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **Peak memory** | ~6.46M-element R list (high overhead) | ~38.5M-row `data.table` (~2â€“3 GB, fits in 16 GB) |
| **Numerical results** | max/min/mean of rook-neighbor values | **Identical** â€” same neighbor sets, same operations |
| **RF model** | Pre-trained | **Untouched** â€” no retraining |

The core insight is that the neighbor aggregation is a **group-by operation on an edge list**, which `data.table` executes in optimized C code with radix-sort grouping â€” replacing millions of R-level loop iterations and character-key hash lookups with a single vectorized pass.
END_RESPONSE>>>
