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
run_id: Anthropic_F06_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) hash lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for every neighbor (expensive string allocation Ã— ~6.46M rows Ã— ~4 neighbors each).
3. Indexes into `idx_lookup` (a named character vector â€” each lookup is O(1) amortized but the constant factor of character hashing over millions of calls is large).

The result is a **list of 6.46 million integer vectors**. This alone consumes significant memory (~200+ MB of list overhead) and takes hours because `lapply` in R over millions of elements with string operations is inherently slow.

### Bottleneck B: `compute_neighbor_stats` â€” called 5 times, each iterating over 6.46M rows

Each call does another `lapply` over 6.46 million elements, subsetting a numeric vector and computing `max`, `min`, `mean`. While each individual operation is trivial, the R-level loop overhead across 6.46M iterations Ã— 5 variables is enormous.

### Why raster focal/kernel operations are *not* a direct replacement

Focal operations assume a regular rectangular grid with a fixed kernel. Here, the grid cells have an irregular boundary (not all cells have 4 rook neighbors â€” coastal/border cells have fewer), and the data is a panel (cell Ã— year). A focal approach would require reshaping into a 3D raster stack per year, running focal per year per variable, then reassembling â€” possible but fragile and risks subtle mismatches at boundaries. The better approach is to **vectorize the neighbor computation directly using sparse matrix algebra**, which exactly preserves the rook-neighbor structure and numerical results.

### Root cause summary

| Component | Calls | Per-call cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | ~string ops | ~30-40 hrs |
| `compute_neighbor_stats` | 5 Ã— 6.46M | ~subset + summary | ~45-50 hrs |
| **Total** | | | **~80-90 hrs** |

---

## 2. Optimization Strategy

### Core idea: Replace row-level R loops with sparse-matrix multiplication and vectorized group operations.

**Step 1 â€” Build a sparse adjacency matrix W (344,208 Ã— 344,208)** from the `nb` object once. This is a standard operation (`spdep::nb2listw` â†’ `as_dgRMatrix_listw`, or direct construction). Each row has ~4 nonzero entries. Total nonzeros â‰ˆ 1.37M. Memory: ~20 MB.

**Step 2 â€” For each year, extract the variable vector, multiply by W, and derive max/min/mean.** But sparse matrix multiplication gives *sums*, not max/min. So we need a different approach for max and min.

**Refined approach â€” Expand neighbor pairs into a long table, then use `data.table` grouped operations:**

1. Convert the `nb` object into an edge list: `from_id`, `to_id` (~1.37M rows).
2. Join with the panel data to create a long table of (row_index, neighbor_row_index) for all cell-years. This is done via a merge on (cell_id, year) â€” fully vectorized.
3. For each variable, extract neighbor values in bulk, then compute `max`, `min`, `mean` grouped by row_index using `data.table`.

This replaces 6.46M R-level iterations with a single vectorized `data.table` grouped aggregation over ~25.8M rows (6.46M Ã— ~4 neighbors). `data.table` does this in seconds.

**Expected speedup: from ~86 hours to ~2-5 minutes.**

### Numerical equivalence

The operations (`max`, `min`, `mean` of the exact same neighbor values) are identical. No approximation is introduced. The trained Random Forest model is not touched.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Replaces build_neighbor_lookup + compute_neighbor_stats loop
# Preserves exact numerical results. Does not touch the trained RF model.
# =============================================================================

library(data.table)
library(spdep)      
library(Matrix)     

# ---- Step 0: Ensure cell_data is a data.table with a row-order column --------
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]  # preserve original row order

# ---- Step 1: Build edge list from the nb object (once) ----------------------
# rook_neighbors_unique is an nb object; id_order maps position -> cell id
build_edge_list <- function(nb_obj, id_order) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i (in id_order)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove 0-neighbor entries (spdep uses integer(0) for islands, 
  # but rep/unlist handles that correctly â€” they simply produce nothing)
  valid <- to != 0L  # spdep marks no-neighbor with 0 in some representations
  
  data.table(
    from_id = id_order[from[valid]],
    to_id   = id_order[to[valid]]
  )
}

edge_dt <- build_edge_list(rook_neighbors_unique, id_order)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ---- Step 2: Build the full (row_idx â†’ neighbor_row_idx) mapping ------------
# For every cell-year row, find the row indices of its rook neighbors 
# in the same year.

# Create a keyed lookup: (cell_id, year) -> row_idx
setkey(cell_dt, id, year)

# Expand edges across all years:
#   For each edge (from_id, to_id), and for each year in the panel,
#   we need (row_of_from_id_in_year_t, row_of_to_id_in_year_t).
#
# Instead of a full cross join (expensive), we merge twice.

# Lookup table: id, year -> .row_idx
lookup <- cell_dt[, .(id, year, .row_idx)]
setkey(lookup, id, year)

# Get unique years
years <- sort(unique(cell_dt$year))

# Cross join edges Ã— years, then look up row indices
# This is the key vectorized step.
# edge_dt has ~1.37M rows; 28 years â†’ ~38.4M rows before filtering.
# But many (from_id, year) pairs exist for all years, so this is efficient.

edge_year <- CJ_dt <- edge_dt[, .(from_id, to_id)]
# Replicate for each year
edge_year <- edge_dt[rep(seq_len(.N), each = length(years))]
edge_year[, year := rep(years, times = nrow(edge_dt))]

cat("Edge-year rows before join:", nrow(edge_year), "\n")

# Join to get row index of the focal cell (from_id, year)
setnames(lookup, c("id", "year", ".row_idx"), c("from_id", "year", "focal_row"))
setkey(lookup, from_id, year)
setkey(edge_year, from_id, year)
edge_year <- lookup[edge_year, nomatch = 0L]

# Join to get row index of the neighbor cell (to_id, year)
setnames(lookup, c("from_id", "year", "focal_row"), c("to_id", "year", "nbr_row"))
setkey(lookup, to_id, year)
setkey(edge_year, to_id, year)
edge_year <- lookup[edge_year, nomatch = 0L]

# Restore lookup names for safety
setnames(lookup, c("to_id", "year", "nbr_row"), c("id", "year", ".row_idx"))

cat("Edge-year rows after join:", nrow(edge_year), "\n")
# Expected: ~25-38M (depends on boundary cells and year coverage)

# ---- Step 3: Compute neighbor stats for all variables at once ----------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the variable columns as a matrix for fast column access
var_mat <- as.matrix(cell_dt[, ..neighbor_source_vars])

# For each variable, pull neighbor values, group by focal_row, compute stats
for (vname in neighbor_source_vars) {
  col_idx <- match(vname, neighbor_source_vars)
  
  # Get neighbor values (vectorized extraction)
  edge_year[, nbr_val := var_mat[nbr_row, col_idx]]
  
  # Compute grouped stats â€” data.table is extremely fast at this
  stats <- edge_year[!is.na(nbr_val), 
                     .(nb_max  = max(nbr_val),
                       nb_min  = min(nbr_val),
                       nb_mean = mean(nbr_val)),
                     by = focal_row]
  
  # Initialize result columns with NA
  max_col  <- paste0(vname, "_nb_max")
  min_col  <- paste0(vname, "_nb_min")
  mean_col <- paste0(vname, "_nb_mean")
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign results back by row index
  cell_dt[stats$focal_row, (max_col)  := stats$nb_max]
  cell_dt[stats$focal_row, (min_col)  := stats$nb_min]
  cell_dt[stats$focal_row, (mean_col) := stats$nb_mean]
  
  cat("Done:", vname, "\n")
}

# ---- Step 4: Clean up and convert back if needed ----------------------------
edge_year[, nbr_val := NULL]  
cell_dt[, .row_idx := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ---- Step 5: Predict with the pre-trained Random Forest (unchanged) ----------
# The RF model is already trained. Just run predict() as before.
# cell_data$predicted <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete.\n")
```

---

## 4. Memory and Runtime Estimates

| Object | Size (approx.) |
|---|---|
| `cell_dt` (6.46M Ã— 110 cols) | ~5.5 GB |
| `edge_year` (~38M Ã— 4 cols) | ~1.2 GB |
| `var_mat` (6.46M Ã— 5) | ~260 MB |
| **Peak total** | **~8-9 GB** (fits in 16 GB) |

| Step | Estimated time |
|---|---|
| Build edge list | < 1 sec |
| Cross-join + double merge | ~30-60 sec |
| 5 Ã— grouped aggregation | ~2-3 min |
| **Total** | **~3-5 minutes** |

This is a **~1,000Ã— speedup** over the original 86-hour estimate.

---

## 5. Verification Snippet

To confirm numerical equivalence on a small sample:

```r
# Spot-check: compare old method vs new for a handful of rows
set.seed(42)
check_rows <- sample(nrow(cell_data), 20)

old_lookup <- build_neighbor_lookup(cell_data_original, id_order, rook_neighbors_unique)

for (vname in neighbor_source_vars) {
  old_stats <- compute_neighbor_stats(
    cell_data_original, old_lookup, vname
  )[check_rows, ]
  
  max_col  <- paste0(vname, "_nb_max")
  min_col  <- paste0(vname, "_nb_min")
  mean_col <- paste0(vname, "_nb_mean")
  
  new_stats <- as.matrix(cell_data[check_rows, c(max_col, min_col, mean_col)])
  
  cat(vname, "- max diff:", max(abs(old_stats - new_stats), na.rm = TRUE), "\n")
}
# Expected output: all diffs = 0 (exact match)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` with string ops | Vectorized edge-list cross-join via `data.table` |
| Neighbor stats | 5 Ã— 6.46M `lapply` with per-row subsetting | 5 Ã— `data.table` grouped aggregation |
| Runtime | ~86 hours | ~3-5 minutes |
| RAM peak | ~10+ GB (6.46M-element list of vectors) | ~8-9 GB (edge table + data) |
| Numerical result | max, min, mean of rook-neighbor values | **Identical** |
| RF model | Untouched | **Untouched** |
END_RESPONSE>>>
