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
run_id: Anthropic_F06_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`
This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in the neighbor list.
2. Constructs string keys (`paste(id, year)`) for each neighbor.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String-based named-vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times with string construction and subsetting is extremely slow. The resulting `neighbor_lookup` is a list of ~6.46M integer vectors â€” itself a large object.

### Bottleneck 2: `compute_neighbor_stats` â€” Row-level `lapply` over 6.46M entries Ã— 5 variables
For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` of neighbor values. That's ~32.3 million R-level function calls with vector subsetting. The `do.call(rbind, result)` on a 6.46M-element list is also expensive.

### Why raster focal/kernel operations don't directly apply
Raster focal operations assume a regular grid with a fixed rectangular kernel. Here, the grid cells have **rook contiguity from an irregular or masked spatial grid** (an `spdep::nb` object), so the neighbor structure is heterogeneous (boundary cells, masked cells, etc.). Forcing this into a raster focal operation would require padding, masking, and could introduce numerical discrepancies at boundaries. **To preserve the original numerical estimand exactly**, we must use the actual `nb` object, not a raster approximation.

### Root cause summary
| Component | Calls | Cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + hash lookups | ~hours |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M R-level loops | ~hours |
| `do.call(rbind, ...)` on 6.46M-element list | 5 times | ~minutes each |

**Estimated total: 86+ hours** on a 16 GB laptop.

---

## Optimization Strategy

### Strategy 1: Vectorized sparse-matrix multiplication (for mean) + grouped operations (for min/max)

The key insight: instead of looping row-by-row in R, we can:

1. **Build a sparse adjacency matrix `W`** (rows = cell-year observations, cols = cell-year observations) where entry `(i, j) = 1` if row `j` is a rook neighbor of row `i` in the same year. This is done **once**.

2. **Neighbor mean** = `(W %*% x) / (W %*% 1_non_na)` â€” a sparse matrix-vector multiply, which is blazing fast via the `Matrix` package.

3. **Neighbor max and min** â€” these are not linear operations, so we can't use matrix multiplication. Instead, we use `data.table` grouped operations: expand the neighbor pairs into an edge list, join values, and compute `max`/`min`/`mean` grouped by the focal row.

### Strategy 2 (chosen): `data.table` edge-list join â€” simplest, fast, exact

- Convert the `nb` object into an **edge list of (focal_cell, neighbor_cell)** pairs (~1.37M directed pairs).
- Cross this with years to get **(focal_row, neighbor_row)** pairs (~1.37M Ã— 28 = ~38.5M edges). But we can be smarter: since the neighbor structure is the same every year, we build the cell-level edge list once and join by year.
- Use `data.table` to join variable values onto the neighbor rows, then `group by` focal row to compute `max`, `min`, `mean`.

**Expected speedup**: from 86+ hours to **~2â€“10 minutes**.

**Memory**: The edge list is ~38.5M rows Ã— a few columns of integers/doubles â€” roughly 1â€“2 GB, well within 16 GB.

**Numerical equivalence**: This computes exactly the same `max`, `min`, `mean` of the same neighbor values, preserving the estimand.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Prepare inputs
# ============================================================
# Assumptions about inputs already in the environment:
#   cell_data            â€” data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order             â€” integer/character vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique â€” an spdep::nb object (list of integer index vectors)

# Convert cell_data to data.table for speed (in-place if possible)
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ============================================================
# STEP 1: Build cell-level edge list from the nb object (once)
# ============================================================
# rook_neighbors_unique[[i]] gives the indices (into id_order) of
# the neighbors of the cell whose ID is id_order[i].

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate by computing total number of directed edges
  n_edges <- sum(lengths(nb_obj))
  focal_idx <- integer(n_edges)
  neighbor_idx <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    n <- length(nbrs)
    if (n > 0L) {
      focal_idx[pos:(pos + n - 1L)] <- i
      neighbor_idx[pos:(pos + n - 1L)] <- nbrs
      pos <- pos + n
    }
  }
  # Trim if any 0-neighbor cells caused over-allocation
  focal_idx <- focal_idx[1:(pos - 1L)]
  neighbor_idx <- neighbor_idx[1:(pos - 1L)]
  
  data.table(
    focal_cell_id    = id_order[focal_idx],
    neighbor_cell_id = id_order[neighbor_idx]
  )
}

cat("Building cell-level edge list...\n")
cell_edges <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed cell-level edges\n", format(nrow(cell_edges), big.mark = ",")))

# ============================================================
# STEP 2: Create a row-index column in cell_data
# ============================================================
# We need to map (id, year) -> row index efficiently.
cell_data[, .row_idx := .I]

# ============================================================
# STEP 3: Expand edge list to cell-year level by joining on year
# ============================================================
# For each year, the same cell-level edges apply.
# We join focal side to get focal row index, then neighbor side
# to get neighbor row index.

cat("Expanding edge list to cell-year level...\n")

# Key cell_data for fast joins
# Create lookup tables: (id, year) -> .row_idx
focal_lookup <- cell_data[, .(focal_cell_id = id, year, focal_row = .row_idx)]
setkey(focal_lookup, focal_cell_id, year)

neighbor_lookup_dt <- cell_data[, .(neighbor_cell_id = id, year, neighbor_row = .row_idx)]
setkey(neighbor_lookup_dt, neighbor_cell_id, year)

# Get unique years
years <- sort(unique(cell_data$year))

# Cross join edges Ã— years, then join to get row indices
# To manage memory, we do this in one shot since 1.37M Ã— 28 â‰ˆ 38.5M rows is manageable
edge_year <- CJ_dt_edges <- cell_edges[, .(focal_cell_id, neighbor_cell_id, year = rep(years, each = .N)), 
                                         by = .EACHI]

# Actually, the CJ approach above won't work directly. Let's use a cleaner method:
# Replicate the edge list for each year
edge_year <- cell_edges[, .(year = years), by = .(focal_cell_id, neighbor_cell_id)]

cat(sprintf("  %s cell-year edges\n", format(nrow(edge_year), big.mark = ",")))

# Join to get focal row index
setkey(edge_year, focal_cell_id, year)
edge_year <- focal_lookup[edge_year, on = .(focal_cell_id, year), nomatch = 0L]

# Join to get neighbor row index
setkey(edge_year, neighbor_cell_id, year)
edge_year <- neighbor_lookup_dt[edge_year, on = .(neighbor_cell_id, year), nomatch = 0L]

# Keep only what we need
edge_year <- edge_year[, .(focal_row, neighbor_row)]

# Clean up temporary lookup tables
rm(focal_lookup, neighbor_lookup_dt, cell_edges)
gc()

cat(sprintf("  Final edge table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

# ============================================================
# STEP 4: Compute neighbor stats for all variables at once
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Extract neighbor values via the edge list
  # edge_year$neighbor_row indexes into cell_data
  edge_year[, val := cell_data[[var_name]][neighbor_row]]
  
  # Remove NA values before aggregation
  edges_valid <- edge_year[!is.na(val)]
  
  # Aggregate: max, min, mean grouped by focal_row
  stats <- edges_valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = .(focal_row)]
  
  # Initialize new columns with NA
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  # Assign results back (only rows that had valid neighbors get non-NA)
  cell_data[stats$focal_row, (max_col)  := stats$nb_max]
  cell_data[stats$focal_row, (min_col)  := stats$nb_min]
  cell_data[stats$focal_row, (mean_col) := stats$nb_mean]
  
  cat(sprintf("    Done. Non-NA rows: %s / %s\n", 
              format(nrow(stats), big.mark = ","),
              format(nrow(cell_data), big.mark = ",")))
}

# Clean up the temporary val column from edge_year
edge_year[, val := NULL]

# Remove helper column
cell_data[, .row_idx := NULL]

cat("All neighbor features computed.\n")

# ============================================================
# STEP 5: Apply the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model is already in memory (e.g., `rf_model`).
# Prediction proceeds exactly as before:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The model is NOT retrained. The 15 new neighbor-stat columns
# (5 vars Ã— 3 stats) are numerically identical to the original
# implementation, so predictions are identical.
```

### Cleaner alternative for Step 3 (edge expansion)

The cross-join above may have syntax issues depending on `data.table` version. Here is a robust alternative:

```r
# ============================================================
# STEP 3 (ROBUST ALTERNATIVE): Expand edges to cell-year level
# ============================================================

cat("Expanding edge list to cell-year level...\n")

# Build a single lookup: (id, year) -> row_index
cell_data[, .row_idx := .I]
setkey(cell_data, id, year)

# Expand: for each edge, replicate across all years
years_dt <- data.table(year = sort(unique(cell_data$year)))

# Cross join: edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows, ~1 GB)
edge_year <- cell_edges[, CJ(edge_id = seq_len(.N), year = years_dt$year)]
# This doesn't quite work either. Simplest robust approach:

edge_year <- cell_edges[rep(seq_len(.N), each = length(years))]
edge_year[, year := rep(years, times = nrow(cell_edges))]

cat(sprintf("  %s cell-year edge rows\n", format(nrow(edge_year), big.mark = ",")))

# Join focal row index
edge_year[cell_data, focal_row := i..row_idx, 
          on = .(focal_cell_id = id, year)]

# Join neighbor row index
edge_year[cell_data, neighbor_row := i..row_idx, 
          on = .(neighbor_cell_id = id, year)]

# Drop edges where either side is missing (cell not observed that year)
edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

# Keep only index columns
edge_year <- edge_year[, .(focal_row, neighbor_row)]

cat(sprintf("  Final edge table: %s rows\n", format(nrow(edge_year), big.mark = ",")))
```

---

## Final Consolidated, Production-Ready Version

```r
library(data.table)

# ------------------------------------------------------------------
# Input objects expected in environment:
#   cell_data              : data.frame/data.table (id, year, ntl, ec, pop_density, def, usd_est_n2, ...)
#   id_order               : vector of cell IDs matching nb object indexing
#   rook_neighbors_unique  : spdep::nb object
#   rf_model               : pre-trained randomForest model (DO NOT retrain)
# ------------------------------------------------------------------

if (!is.data.table(cell_data)) setDT(cell_data)

# --- 1. Cell-level edge list from nb object ---
n_edges_total <- sum(lengths(rook_neighbors_unique))
focal_vec <- integer(n_edges_total)
nbr_vec   <- integer(n_edges_total)
pos <- 1L
for (i in seq_along(rook_neighbors_unique)) {
  nbrs <- rook_neighbors_unique[[i]]
  nbrs <- nbrs[nbrs != 0L]
  n <- length(nbrs)
  if (n > 0L) {
    idx_range <- pos:(pos + n - 1L)
    focal_vec[idx_range] <- i
    nbr_vec[idx_range]   <- nbrs
    pos <- pos + n
  }
}
cell_edges <- data.table(
  focal_cell_id    = id_order[focal_vec[1:(pos-1L)]],
  neighbor_cell_id = id_order[nbr_vec[1:(pos-1L)]]
)
rm(focal_vec, nbr_vec)

# --- 2. Row index on cell_data ---
cell_data[, .row_idx := .I]

# --- 3. Expand edges across years ---
years <- sort(unique(cell_data$year))
n_years <- length(years)
n_cell_edges <- nrow(cell_edges)

edge_year <- data.table(
  focal_cell_id    = rep(cell_edges$focal_cell_id,    each = n_years),
  neighbor_cell_id = rep(cell_edges$neighbor_cell_id, each = n_years),
  year             = rep(years, times = n_cell_edges)
)
rm(cell_edges)

# Join to get row indices
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

setkey(edge_year, focal_cell_id, year)
edge_year[row_lookup, focal_row := i..row_idx, on = .(focal_cell_id = id, year)]

setkey(edge_year, neighbor_cell_id, year)
edge_year[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_cell_id = id, year)]

edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row), .(focal_row, neighbor_row)]
rm(row_lookup)
gc()

# --- 4. Compute neighbor stats ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  edge_year[, val := cell_data[[var_name]][neighbor_row]]
  
  stats <- edge_year[!is.na(val), .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)
}

edge_year[, val := NULL]
cell_data[, .row_idx := NULL]

# --- 5. Predict with the pre-trained RF (unchanged) ---
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level string pastes + hash lookups | One vectorized edge-list expansion via `data.table` |
| **Stat computation** | 6.46M Ã— 5 `lapply` calls with per-row subsetting | 5 vectorized `data.table` group-by aggregations |
| **Result binding** | `do.call(rbind, list_of_6.46M)` Ã— 5 | Direct `set()` assignment by row index |
| **Estimated time** | **86+ hours** | **~2â€“10 minutes** |
| **Peak memory** | Moderate (but slow) | ~2â€“4 GB for edge table (fits in 16 GB) |
| **Numerical equivalence** | Baseline | **Exact** â€” same `max`, `min`, `mean` over same neighbor sets |
| **RF model** | Pre-trained | **Unchanged, not retrained** |
END_RESPONSE>>>
