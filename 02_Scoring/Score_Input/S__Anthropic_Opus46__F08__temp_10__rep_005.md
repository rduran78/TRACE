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
run_id: Anthropic_F08_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` operates on cell-year rows, not cells.** It builds a lookup of length ~6.46 million (344,208 cells Ã— 28 years), where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. But the neighbor relationships are identical across all 28 years â€” the grid doesn't move. This means the function does 28Ã— redundant work discovering the same spatial topology.

2. **The lookup stores row indices into the full panel.** This means every year-specific query requires string-pasting cell IDs with years and hash-lookups into a 6.46M-entry named vector (`idx_lookup`). With ~1.37M directed neighbor edges Ã— 28 years â‰ˆ 38.4M string constructions and lookups, this is extremely expensive in R.

3. **`compute_neighbor_stats` iterates row-by-row over 6.46M rows using `lapply`.** Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. The per-element overhead of 6.46M R function calls dominates.

4. **There is no vectorization.** The entire pipeline â€” lookup construction and stats computation â€” is scalar R loops over millions of elements.

### Summary

| Component | Problem | Scale |
|---|---|---|
| `build_neighbor_lookup` | Rebuilds topology per cell-year; string ops | 6.46M iterations, 38.4M string ops |
| `compute_neighbor_stats` | Scalar R loop; per-row function calls | 6.46M `lapply` calls Ã— 5 variables |
| Overall | Static topology not separated from dynamic data | 28Ã— redundant topology work |

---

## Optimization Strategy

**Core Insight:** Separate what is static (neighbor topology) from what changes (variable values by year).

### Step 1: Build the neighbor topology once, at the cell level

Convert `rook_neighbors_unique` (an `nb` object indexed by position in `id_order`) into a simple two-column edge table: `(from_cell_position, to_cell_position)`. This is done once and has ~1.37M rows. No year dimension.

### Step 2: Reshape data for column-vectorized access by year

Create a mapping from `(cell_position, year)` â†’ row in the panel. This is a matrix of dimensions `(n_cells Ã— n_years)`, enabling O(1) lookup.

### Step 3: Compute neighbor stats via vectorized grouped operations

For each variable, use the edge table to gather all neighbor values in a single vectorized indexing operation, then compute grouped max/min/mean using `data.table` or `rowsum`-style operations â€” no R-level loops over 6.46M rows.

### Complexity Comparison

| | Current | Optimized |
|---|---|---|
| Topology construction | O(n_cells Ã— n_years) | O(n_cells) â€” once |
| Stats per variable | O(n_cells Ã— n_years) scalar loop | O(n_edges Ã— n_years) vectorized |
| Total R function calls | ~32.3M `lapply` calls | ~0 (vectorized) |
| Expected time | ~86+ hours | Minutes |

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Ensure cell_data is a data.table with original order
# ============================================================
# Preserve original row order so the RF prediction step sees
# the same data frame it expects.
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_order := .I]  # preserve original ordering

# ============================================================
# STEP 1: Build static cell-level edge table (ONCE)
# ============================================================
# rook_neighbors_unique is an nb object of length n_cells,
# indexed in the same order as id_order.
# Each element is an integer vector of neighbor positions.

build_edge_table <- function(neighbors) {
  # neighbors: list of integer vectors (nb object), position-indexed
  from <- rep(seq_along(neighbors), lengths(neighbors))
  to   <- unlist(neighbors, use.names = FALSE)
  data.table(from_pos = from, to_pos = to)
}

edge_table <- build_edge_table(rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (from_pos, to_pos) in id_order space

cat(sprintf("Edge table: %d directed edges among %d cells\n",
            nrow(edge_table), length(id_order)))

# ============================================================
# STEP 2: Build (cell_pos, year) -> panel row mapping
# ============================================================
# Create a position index for each cell ID
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_pos := id_to_pos[as.character(id)]]

# Unique sorted years
years_unique <- sort(unique(cell_dt$year))
n_years      <- length(years_unique)
year_to_col  <- setNames(seq_along(years_unique), as.character(years_unique))
cell_dt[, year_idx := year_to_col[as.character(year)]]

# Build a matrix: row_map[cell_pos, year_idx] = row index in cell_dt
# This allows O(1) lookup from (cell_pos, year) to data row.
n_cells <- length(id_order)
row_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_map[cbind(cell_dt$cell_pos, cell_dt$year_idx)] <- cell_dt$.row_order

# ============================================================
# STEP 3: Vectorized neighbor stats computation
# ============================================================
compute_neighbor_features_vectorized <- function(cell_dt, edge_table,
                                                  row_map, var_name,
                                                  years_unique, year_to_col) {
  # Output columns
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  vals <- cell_dt[[var_name]]  # full panel vector, indexed by .row_order
  n_rows <- nrow(cell_dt)

  # Pre-allocate output
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)

  # Process year-by-year (28 iterations â€” each fully vectorized)
  for (yr in years_unique) {
    yi <- year_to_col[as.character(yr)]

    # Row indices for "from" cells in this year
    from_rows <- row_map[edge_table$from_pos, yi]
    # Row indices for "to" (neighbor) cells in this year
    to_rows   <- row_map[edge_table$to_pos, yi]

    # Keep only edges where both endpoints exist in this year
    valid <- !is.na(from_rows) & !is.na(to_rows)
    fr    <- from_rows[valid]
    tr    <- to_rows[valid]

    # Get neighbor values
    nvals <- vals[tr]

    # Remove edges where the neighbor value is NA
    not_na <- !is.na(nvals)
    fr     <- fr[not_na]
    nvals  <- nvals[not_na]

    if (length(fr) == 0L) next

    # Use data.table for fast grouped aggregation
    agg_dt <- data.table(fr = fr, nv = nvals)
    agg    <- agg_dt[, .(nmax  = max(nv),
                         nmin  = min(nv),
                         nmean = mean(nv)), by = fr]

    out_max[agg$fr]  <- agg$nmax
    out_min[agg$fr]  <- agg$nmin
    out_mean[agg$fr] <- agg$nmean
  }

  # Attach to cell_dt (by reference)
  set(cell_dt, j = max_col,  value = out_max)
  set(cell_dt, j = min_col,  value = out_min)
  set(cell_dt, j = mean_col, value = out_mean)

  invisible(cell_dt)
}

# ============================================================
# STEP 4: Run for all 5 neighbor source variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  t0 <- Sys.time()
  compute_neighbor_features_vectorized(
    cell_dt, edge_table, row_map, var_name, years_unique, year_to_col
  )
  cat(sprintf("  Done in %.1f seconds\n", as.numeric(Sys.time() - t0, units = "secs")))
}

# ============================================================
# STEP 5: Clean up helper columns & restore as data.frame
# ============================================================
cell_dt[, c("cell_pos", "year_idx", ".row_order") := NULL]

# Convert back to data.frame (if the RF predict method expects one)
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model object and prediction code remain exactly
# as before â€” no retraining. The 15 new neighbor_* columns
# (5 vars Ã— {max, min, mean}) are numerically identical to the
# original implementation's output.
#
# Example (assuming your existing code looks like this):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `edge_table` is derived from the same `rook_neighbors_unique` nb object. Each `(from_pos, to_pos)` pair corresponds to the same cell-pair the original code used. |
| **Same per-year scoping** | The year loop ensures that for each cell-year row, only neighbors *in that same year* contribute â€” identical to the original `paste(id, year)` key logic. |
| **Same aggregation functions** | `max`, `min`, `mean` with NA removal â€” identical to the original. |
| **Same column names** | `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc. â€” matching the pattern in `compute_and_add_neighbor_features`. |
| **No RF retraining** | The pre-trained model object is never modified. Only the feature-engineering step is optimized. |

---

## Performance Estimate

| Operation | Work | Expected Time |
|---|---|---|
| Edge table construction | Unlist ~1.37M edges | < 1 second |
| Row map matrix | 6.46M assignments into matrix | < 2 seconds |
| Per variable, per year aggregation | ~49K edges Ã— 28 years, `data.table` grouped | ~5â€“15 sec/variable |
| **Total for 5 variables** | | **~1â€“2 minutes** |

This represents a speedup from **86+ hours to roughly 1â€“2 minutes** â€” approximately a **3,000Ã—** improvement â€” achieved entirely by honoring the static-vs-changing distinction in the data and replacing scalar R loops with vectorized operations.
END_RESPONSE>>>
