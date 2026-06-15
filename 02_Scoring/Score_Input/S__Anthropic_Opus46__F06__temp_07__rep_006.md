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
run_id: Anthropic_F06_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** iterates over ~6.46 million rows with an `lapply` call that, for each row, performs character coercion, string pasting, and named-vector lookups. This produces a list of 6.46 million integer vectors. Then **`compute_neighbor_stats`** iterates over that same list 5 times (once per variable), computing `max`, `min`, and `mean` on small vectors inside R's interpreter loop.

**Specific problems:**

| Component | Issue |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of `paste()`, character key lookups in a named vector (hash-like but still slow at scale), and `as.character()` coercions. This is O(n Ã— k) in pure R, where k â‰ˆ average neighbor count (~4 for rook). Estimated: tens of hours alone. |
| `compute_neighbor_stats` | Called 5 times, each time iterating over 6.46M list elements. The per-element work is tiny, but the R-level loop overhead on 6.46M elements is massive. |
| Memory | The `neighbor_lookup` list of 6.46M integer vectors has high overhead (each R vector has a ~128-byte header). With ~4 neighbors per cell-year, this list alone can consume several GB. |
| Key construction | Using `paste(id, year, sep="_")` and named-vector lookup is far slower than integer-arithmetic indexing. |

**Why raster focal/kernel operations are not directly applicable:** Focal operations assume a regular rectangular grid with a fixed kernel. Here the grid is irregular (not all cells have 4 rook neighbors â€” boundary cells, missing cell-years), and the data is in long panel format, not a raster stack. The analogy is useful conceptually but a sparse-matrix or direct-index approach is the correct implementation to preserve exact results.

---

## 2. Optimization Strategy

### Key insight: Eliminate the 6.46M-row R loop entirely using vectorized sparse-matrix multiplication and grouped operations.

**Step-by-step plan:**

1. **Replace string-key lookup with integer-arithmetic indexing.** Map each `(id, year)` pair to a row index using a `data.table` keyed join â€” O(n) with negligible constant.

2. **Build a sparse adjacency matrix** (cell-year Ã— cell-year) once, using the `Matrix` package. Each row `i` has non-zero entries in columns corresponding to i's rook neighbors in the same year. This replaces the 6.46M-element list.

3. **Compute neighbor stats via sparse matrix operations:**
   - **Mean:** `W %*% x / W %*% 1` (where `W` is the binary adjacency matrix, `x` is the variable vector, and `1` is a ones-vector for counting).
   - **Max and Min:** Use a loop over the *neighbor-pair edge list* (only ~1.37M Ã— 28 â‰ˆ 38.5M directed edges), grouped with `data.table`, which is orders of magnitude faster than 6.46M R-level iterations.

4. **Process all 5 variables** in one pass through the edge list for max/min, and via matrix multiplication for mean.

**Expected speedup:** From ~86 hours to **minutes** (sparse matrix multiply on 6.46M Ã— 6.46M with ~38.5M non-zeros is fast; `data.table` grouped aggregation on ~38.5M rows is seconds).

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# ============================================================
# STEP 0: Prepare data.table with row indices
# ============================================================
# Assume: cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assume: id_order is a vector of unique cell IDs (ordering matches rook_neighbors_unique)
# Assume: rook_neighbors_unique is an nb object (list of integer index vectors into id_order)

dt <- as.data.table(cell_data)
dt[, row_idx := .I]  # preserve original row order

# ============================================================
# STEP 1: Build an edge list of (source_row, neighbor_row) for
#          all cell-years, using integer indexing (no paste!)
# ============================================================

# Map cell id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Build the spatial edge list (cell-level, not cell-year-level)
# Each entry: (from_pos, to_pos) in id_order space
spatial_edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb <- rook_neighbors_unique[[i]]
  # spdep::nb uses 0 for no-neighbor; filter those out

  nb <- nb[nb > 0L]
  if (length(nb) == 0L) return(NULL)
  data.table(from_pos = i, to_pos = nb)
}))

cat("Spatial edges (directed):", nrow(spatial_edges), "\n")

# Map (pos_in_id_order, year) -> row_idx in dt
dt[, pos := id_to_pos[as.character(id)]]
setkey(dt, pos, year)

# For each spatial edge, expand across all 28 years
# This is the key: we join spatial edges to the panel index

# Create a lookup: (pos, year) -> row_idx
pos_year_lookup <- dt[, .(pos, year, row_idx)]
setkey(pos_year_lookup, pos, year)

# Get unique years
years <- sort(unique(dt$year))

# Expand spatial edges Ã— years using a cross join, then join to get row indices
cat("Building full edge list across years...\n")

edge_list <- CJ(edge_id = seq_len(nrow(spatial_edges)), year = years)
edge_list[, from_pos := spatial_edges$from_pos[edge_id]]
edge_list[, to_pos   := spatial_edges$to_pos[edge_id]]

# Join to get from_row and to_row
setkey(edge_list, from_pos, year)
edge_list[pos_year_lookup, from_row := i.row_idx, on = .(from_pos = pos, year)]

setkey(edge_list, to_pos, year)
edge_list[pos_year_lookup, to_row := i.row_idx, on = .(to_pos = pos, year)]

# Drop edges where either endpoint is missing (cell not observed in that year)
edge_list <- edge_list[!is.na(from_row) & !is.na(to_row)]
edge_list[, c("edge_id", "from_pos", "to_pos") := NULL]

cat("Full directed cell-year edges:", nrow(edge_list), "\n")

# ============================================================
# STEP 2: Compute neighbor max, min, mean for each variable
#          using data.table grouped aggregation on edge_list
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

n <- nrow(dt)

for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "\n")
  
  # Extract neighbor values via the edge list
  vals <- dt[[var_name]]
  edge_list[, nb_val := vals[to_row]]
  
  # Grouped aggregation: for each from_row, compute max, min, mean
  # of nb_val (excluding NAs)
  stats <- edge_list[!is.na(nb_val),
                     .(nb_max  = max(nb_val),
                       nb_min  = min(nb_val),
                       nb_mean = mean(nb_val)),
                     by = from_row]
  
  # Initialize result columns with NA
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)
  
  # Fill in computed values
  max_col[stats$from_row]  <- stats$nb_max
  min_col[stats$from_row]  <- stats$nb_min
  mean_col[stats$from_row] <- stats$nb_mean
  
  # Add to dt using the same naming convention as the original code
  # (adjust column names to match whatever compute_and_add_neighbor_features produced)
  set(dt, j = paste0(var_name, "_nb_max"),  value = max_col)
  set(dt, j = paste0(var_name, "_nb_min"),  value = min_col)
  set(dt, j = paste0(var_name, "_nb_mean"), value = mean_col)
  
  cat("  Done:", var_name, "\n")
}

# Clean up temporary columns
edge_list[, nb_val := NULL]
dt[, pos := NULL]

# ============================================================
# STEP 3: Convert back to data.frame (preserving row order)
#          and run the pre-trained Random Forest prediction
# ============================================================

# Restore original row order
setorder(dt, row_idx)
dt[, row_idx := NULL]

cell_data <- as.data.frame(dt)

# Now cell_data has the 15 new neighbor feature columns
# (5 vars Ã— 3 stats = 15 columns) with identical numerical values
# to the original implementation.
#
# Proceed with the pre-trained Random Forest model (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Exact Results

| Original | Optimized | Equivalence |
|---|---|---|
| `neighbor_lookup[[i]]` gives row indices of i's rook neighbors in the same year | `edge_list[from_row == i]$to_row` gives the same set | Same neighbor set; same year-matching logic |
| `max(vals[idx])` | `data.table` grouped `max()` | Identical IEEE 754 result |
| `min(vals[idx])` | `data.table` grouped `min()` | Identical |
| `mean(vals[idx])` | `data.table` grouped `mean()` | Identical (both use R's `mean()` on the same values) |
| NA handling: drops NAs before computing; returns NA if no valid neighbors | `edge_list[!is.na(nb_val), ...]` + default NA initialization | Identical |

The trained Random Forest model is never touched â€” only the feature-engineering step is optimized.

## 5. Performance Estimate

| Step | Estimated Time |
|---|---|
| Build edge list + joins | ~30â€“60 seconds |
| Grouped aggregation (5 vars) | ~30â€“90 seconds each, ~5 min total |
| **Total** | **~6â€“10 minutes** (vs. 86+ hours) |

Peak memory: the edge list with ~38.5M rows Ã— 3 integer columns â‰ˆ ~900 MB, well within 16 GB.
END_RESPONSE>>>
