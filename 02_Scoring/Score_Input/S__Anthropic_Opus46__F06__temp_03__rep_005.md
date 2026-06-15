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
run_id: Anthropic_F06_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine).
2. Constructs character key strings via `paste()` for every neighbor Ã— that row's year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching on a vector of length ~6.46M.

The result is a **list of 6.46 million integer vectors**. The `paste()`-based key construction and named-vector lookup are extremely slow at this scale. Named vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of character hashing over millions of entries is enormous). Doing this ~6.46M Ã— ~4 neighbors â‰ˆ 25+ million times is the primary time sink.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M elements

For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the index vectors from the lookup. This is done 5 times. The per-element overhead of `lapply` with anonymous functions over millions of elements is substantial, though secondary to Bottleneck A.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a **complete regular grid with uniform time dimension**. Here the panel is cell Ã— year, stored long, with an irregular neighbor structure (`spdep::nb`). Focal operations would require reshaping into a 3D raster stack per variable per year, running focal, then reshaping back â€” and would not naturally handle missing cells or the panel structure. The analogy is useful conceptually (we want a "moving window" summary), but the implementation should use **sparse matrix multiplication and vectorized operations** on the long-format panel, which preserves the exact numerical results.

---

## 2. Optimization Strategy

### Step 1: Replace the character-key lookup with integer arithmetic

Instead of `paste(id, year)` keys, exploit the panel structure. If we map each `(cell_index, year_index)` pair to a row number via a **dense integer matrix or direct arithmetic**, lookup becomes O(1) with no string operations.

Specifically, if we sort the data by `(id, year)` and the panel is balanced (344,208 cells Ã— 28 years = 9,637,824 potential rows; actual = 6.46M so it's unbalanced), we use an **integer matrix** `row_matrix[cell_index, year_index]` that maps to the row number in `cell_data`. This matrix has 344,208 Ã— 28 â‰ˆ 9.6M entries (just ~38 MB as integers) â€” trivially fits in RAM.

### Step 2: Build the neighbor lookup via vectorized matrix indexing

Instead of looping over 6.46M rows, we:
1. Build `row_matrix` (cell Ã— year â†’ row number, NA if missing).
2. Build a sparse neighbor edge list from the `nb` object (just ~1.37M directed edges).
3. For each row in `cell_data`, its neighbors are determined by its cell's spatial neighbors (from the edge list) at the same year. We can compute all ~25M neighbor-row indices in one vectorized operation using the edge list and `row_matrix`.

### Step 3: Replace `lapply`-based stats with sparse matrix multiplication

For `max`, `min`, and `mean`:
- Build a **sparse adjacency matrix** W of dimension (n_rows Ã— n_rows) where `W[i, j] = 1` if row j is a rook neighbor of row i (same year).
- **Mean**: `W_rowstandardized %*% x` gives neighbor means in one matrix-vector multiply.
- **Max/Min**: Use grouped operations via `data.table` on the edge list, which is fully vectorized.

This replaces 5 Ã— 6.46M `lapply` iterations with a handful of vectorized operations.

### Expected speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | ~40-60 hours | ~10-30 seconds |
| Stats computation (5 vars Ã— 3 stats) | ~20-30 hours | ~30-60 seconds |
| **Total** | **~86+ hours** | **~1-3 minutes** |

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)
library(spdep)    # for nb object structure
library(ranger)   # or randomForest â€” for prediction only

# ============================================================
# STEP 0: Ensure cell_data is a data.table with id and year
# ============================================================
cell_data <- as.data.table(cell_data)

# id_order: the vector of unique cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: the spdep nb object (list of integer index vectors)

# ============================================================
# STEP 1: Build integer mappings
# ============================================================

# Map cell IDs to integer indices (matching id_order)
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Map years to integer indices
all_years  <- sort(unique(cell_data$year))
year_to_idx <- setNames(seq_along(all_years), as.character(all_years))

n_cells <- length(id_order)
n_years <- length(all_years)

# Add integer indices to cell_data
cell_data[, cell_idx := id_to_idx[as.character(id)]]
cell_data[, year_idx := year_to_idx[as.character(year)]]

# Assign a row identifier (preserve original order for final output)
cell_data[, row_id := .I]

# ============================================================
# STEP 2: Build row_matrix â€” maps (cell_idx, year_idx) -> row_id
#   This is a dense integer matrix; NA means that cell-year is absent.
# ============================================================

row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_matrix[cbind(cell_data$cell_idx, cell_data$year_idx)] <- cell_data$row_id

cat("Row matrix built:", n_cells, "x", n_years, "\n")

# ============================================================
# STEP 3: Build directed edge list from nb object
#   Each entry: (from_cell_idx, to_cell_idx)
# ============================================================

edge_from <- rep(seq_along(rook_neighbors_unique),
                 lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique)

# Remove any 0-neighbor entries (spdep uses integer(0) for islands)
valid <- edge_to > 0L
edge_from <- edge_from[valid]
edge_to   <- edge_to[valid]

n_edges <- length(edge_from)
cat("Spatial directed edges:", n_edges, "\n")

# ============================================================
# STEP 4: Expand edge list across all years â€” vectorized
#   For each spatial edge (i -> j) and each year t,
#   if both row_matrix[i, t] and row_matrix[j, t] exist,
#   then row row_matrix[i, t] has neighbor row row_matrix[j, t].
# ============================================================

# We'll iterate over years (only 28 iterations â€” trivial)
edge_list_parts <- vector("list", n_years)

for (t in seq_len(n_years)) {
  from_rows <- row_matrix[edge_from, t]  # vectorized column extraction
  to_rows   <- row_matrix[edge_to,   t]
  
  both_exist <- !is.na(from_rows) & !is.na(to_rows)
  
  edge_list_parts[[t]] <- data.table(
    from_row = from_rows[both_exist],
    to_row   = to_rows[both_exist]
  )
}

edges_dt <- rbindlist(edge_list_parts)
rm(edge_list_parts)

cat("Total cell-year directed neighbor edges:", nrow(edges_dt), "\n")

# ============================================================
# STEP 5: Compute neighbor stats for each variable â€” vectorized
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
n_rows <- nrow(cell_data)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Extract the neighbor values for every edge
  vals <- cell_data[[var_name]]
  edges_dt[, val := vals[to_row]]
  
  # Remove edges where the neighbor value is NA
  valid_edges <- edges_dt[!is.na(val)]
  
  # Compute grouped stats: max, min, mean by from_row
  stats <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row]
  
  # Initialize result columns with NA
  max_col  <- rep(NA_real_, n_rows)
  min_col  <- rep(NA_real_, n_rows)
  mean_col <- rep(NA_real_, n_rows)
  
  # Fill in computed values
  max_col[stats$from_row]  <- stats$nb_max
  min_col[stats$from_row]  <- stats$nb_min
  mean_col[stats$from_row] <- stats$nb_mean
  
  # Add to cell_data with the expected column names
  # (Adjust naming convention to match what the trained RF model expects)
  set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = max_col)
  set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = min_col)
  set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  
  cat("  Done:", var_name, "\n")
}

# Clean up temporary columns
edges_dt[, val := NULL]
cell_data[, c("cell_idx", "year_idx", "row_id") := NULL]

# ============================================================
# STEP 6: Predict with the pre-trained Random Forest
#   (Model is NOT retrained â€” only used for prediction)
# ============================================================

# Example (adjust to your actual model object and column names):
# predictions <- predict(trained_rf_model, data = cell_data)
# cell_data[, prediction := predictions$predictions]

cat("Neighbor feature engineering complete.\n")
cat("Rows:", nrow(cell_data), " Columns:", ncol(cell_data), "\n")
```

---

## Summary of Key Design Decisions

| Decision | Rationale |
|---|---|
| **Integer matrix `row_matrix`** instead of character key lookup | Eliminates ~25M `paste()` calls and named-vector lookups. Matrix indexing is O(1). ~38 MB fits easily in 16 GB RAM. |
| **Loop over 28 years** (not 6.46M rows) to expand edges | Each iteration is fully vectorized column extraction from the matrix. 28 iterations is negligible. |
| **`data.table` grouped aggregation** for max/min/mean | `data.table`'s radix-sort-based grouping on integer keys is orders of magnitude faster than 6.46M `lapply` calls. Single pass per variable. |
| **No raster focal approach** | The panel is unbalanced, stored long-format, with an irregular `nb` structure. Reshaping to raster stacks per year would add complexity and potential for floating-point discrepancies. The `data.table` approach computes the **identical** max, min, and mean values. |
| **Pre-trained RF model untouched** | Only the feature engineering pipeline is optimized. The model object, its hyperparameters, and the prediction call remain unchanged. |
| **Numerical equivalence preserved** | `max()`, `min()`, and `mean()` are applied to the exact same sets of non-NA neighbor values as in the original code, producing identical results. |
END_RESPONSE>>>
