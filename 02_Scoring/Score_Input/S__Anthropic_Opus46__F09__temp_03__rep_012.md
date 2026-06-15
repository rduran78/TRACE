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
run_id: Anthropic_F09_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups via `[` on character keys are O(n) in the worst case per call due to R's internal hashing, and `paste` is vectorized but called inside a scalar loop.

### 2. The neighbor lookup is year-redundant
The spatial neighbor structure is **identical across all 28 years**. A cell's rook neighbors in 1992 are the same cells as in 2019. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination (6.46M entries) instead of building it once for 344,208 cells and reusing it. This is a **28Ã— redundancy**.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
For each of the 5 variables, another `lapply` iterates over all 6.46M rows, extracting neighbor values one row at a time. This is 5 Ã— 6.46M = ~32.3M scalar iterations, each doing subsetting, `is.na` filtering, and summary statistics. R's interpreted loop overhead dominates.

### Summary of bottleneck
| Component | Iterations | Per-iteration cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | String ops + hash lookup | Very high |
| `compute_neighbor_stats` (Ã—5 vars) | 32.3M | Subset + summary stats | Very high |

The correct approach: build a **time-invariant adjacency table once** (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M rows), then use a **vectorized join** to attach yearly attributes and compute grouped summaries.

---

## Optimization Strategy

### Step 1: Build a static edge table once
Convert `rook_neighbors_unique` (an `nb` object) into a two-column data.table: `(cell_id, neighbor_id)`. This has ~1.37M rows and never changes.

### Step 2: Join yearly attributes via data.table
For each year, the neighbor attributes are obtained by joining `cell_data` onto the edge table by `(neighbor_id, year)`. This is a vectorized equi-join â€” no row-level loops.

### Step 3: Grouped aggregation
After the join, compute `max`, `min`, `mean` grouped by `(cell_id, year)` using `data.table`'s optimized `by=` grouping. This replaces millions of scalar `lapply` calls with a single grouped operation.

### Expected speedup
- Eliminates 6.46M-iteration `lapply` in `build_neighbor_lookup` entirely.
- Replaces 6.46M-iteration `lapply` per variable with one vectorized join + one grouped aggregation.
- Estimated runtime: **minutes, not hours** (data.table joins on keyed integer columns over ~1.37M Ã— 28 â‰ˆ 38M rows are very fast).

### Memory check
- Edge table: ~1.37M rows Ã— 2 int cols â‰ˆ 11 MB
- Expanded with year: ~1.37M Ã— 28 = ~38.5M rows Ã— 2 int cols â‰ˆ 308 MB
- After joining 5 variables: ~38.5M rows Ã— 7 cols â‰ˆ ~2.2 GB
- Original data: ~6.46M Ã— 110 cols â‰ˆ ~5.7 GB
- Total peak: ~8â€“10 GB â€” fits in 16 GB RAM, especially if we process one variable at a time.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Ensure cell_data is a data.table with proper types
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure 'id' and 'year' are integer for fast keyed joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ============================================================
# STEP 1: Build a STATIC spatial edge table (time-invariant)
#
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector mapping list index -> cell id
# ============================================================
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_cells <- length(neighbors)
  n_edges <- sum(vapply(neighbors, length, integer(1)))

  cell_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      idx_range <- pos:(pos + n_nb - 1L)
      cell_id[idx_range]     <- id_order[i]
      neighbor_id[idx_range] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  data.table(cell_id = cell_id, neighbor_id = neighbor_id)
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges among %s cells.\n",
            format(nrow(edge_table), big.mark = ","),
            format(length(id_order), big.mark = ",")))

# ============================================================
# STEP 2: For each neighbor source variable, compute neighbor
#          max, min, mean via vectorized join + grouped agg.
#
# This replaces both build_neighbor_lookup() and
# compute_neighbor_stats() entirely.
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))

  # --- 2a. Extract only the columns we need for the join ---
  # Columns: neighbor's id, year, and the variable value
  join_cols <- c("id", "year", var_name)
  neighbor_attrs <- cell_data[, ..join_cols]
  setnames(neighbor_attrs, c("neighbor_id", "year", "nb_val"))
  setkey(neighbor_attrs, neighbor_id, year)

  # --- 2b. Expand edge table by year via join ---
  # For every (cell_id, neighbor_id) pair, we need all years
  # that the CELL itself appears in. We get the year from the
  # cell's own data, then look up the neighbor's value.
  #
  # Strategy: join edge_table onto cell_data to get (cell_id, year),
  # then join onto neighbor_attrs to get neighbor's value.

  # Get unique (cell_id, year) pairs from cell_data
  cell_years <- cell_data[, .(id, year)]
  setnames(cell_years, c("cell_id", "year"))

  # Merge: for each (cell_id, year), attach all neighbors
  # This creates ~38.5M rows (1.37M edges Ã— 28 years)
  setkey(cell_years, cell_id)
  setkey(edge_table, cell_id)
  expanded <- edge_table[cell_years, on = "cell_id", allow.cartesian = TRUE, nomatch = NA]
  # Result columns: cell_id, neighbor_id, year

  # --- 2c. Join neighbor attribute values ---
  setkey(expanded, neighbor_id, year)
  expanded[neighbor_attrs, nb_val := i.nb_val, on = .(neighbor_id, year)]

  # --- 2d. Grouped aggregation ---
  stats <- expanded[!is.na(nb_val),
                    .(nb_max  = max(nb_val),
                      nb_min  = min(nb_val),
                      nb_mean = mean(nb_val)),
                    by = .(cell_id, year)]

  # Name the output columns to match original pipeline conventions
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  setnames(stats, c("cell_id", "year"), c("id", "year"))
  setkey(stats, id, year)

  # --- 2e. Merge back into cell_data ---
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]

  # Clean up to manage memory

rm(neighbor_attrs, cell_years, expanded, stats)
  gc()

  cat(sprintf("  Done: added %s, %s, %s\n", max_col, min_col, mean_col))
}

cat("All neighbor features computed.\n")

# ============================================================
# STEP 3: Predict with the existing trained Random Forest
#
# The trained model object is assumed to be in memory (e.g.,
# 'rf_model'). We do NOT retrain â€” just call predict().
# ============================================================
# Example (adjust variable name to your actual model object):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("Ready for Random Forest prediction.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as before â€” the edge table encodes the identical rook-neighbor structure from `rook_neighbors_unique`. Cells with no neighbors (or neighbors missing in a given year) receive `NA`, matching the original `c(NA, NA, NA)` fallback. |
| **Trained RF model** | The model object is never modified. Only the input feature table (`cell_data`) is augmented with the same 15 neighbor-stat columns (5 vars Ã— 3 stats) that the model expects. `predict()` is called as before. |
| **Column naming** | Output columns follow the same `neighbor_{max,min,mean}_{varname}` convention. Adjust the naming pattern if your original pipeline used a different convention. |

---

## Expected Performance

| Step | Estimated time |
|---|---|
| Build edge table (344K cells, 1.37M edges) | < 1 second |
| Per-variable join + aggregation (~38.5M rows) | ~30â€“90 seconds |
| All 5 variables | **~3â€“8 minutes total** |
| RF prediction (6.46M rows, 110 features) | Depends on forest size (unchanged) |

**Total neighbor-feature computation: ~3â€“8 minutes vs. ~86+ hours â€” roughly a 600â€“1,700Ã— speedup.**
END_RESPONSE>>>
