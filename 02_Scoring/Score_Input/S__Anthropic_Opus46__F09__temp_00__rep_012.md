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
run_id: Anthropic_F09_T00_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them against a named character vector (`idx_lookup`) of 6.46 million entries.

Named vector lookup in R via `[` on character names is **O(n)** per probe in the worst case (hash collisions aside, the overhead of repeated character hashing and matching across 6.46M keys is enormous). Doing this 6.46 million times produces billions of character operations.

### 2. The lookup is **rebuilt identically for every variable** (implicitly)
Although the outer loop reuses `neighbor_lookup`, the lookup itself is monolithic: it maps every cell-year row to the **row indices** of its neighbors in the same year. This is correct but the construction cost is paid once and is the dominant bottleneck. The `compute_neighbor_stats` function is comparatively fast (simple numeric indexing), but the initial build is the killer.

### 3. The fundamental inefficiency: conflating spatial topology with temporal panel structure
The spatial neighbor relationships are **time-invariant** â€” cell A is always a rook neighbor of cell B regardless of year. Yet the current code re-resolves this for every cell-year combination, effectively doing 28Ã— the necessary spatial work and embedding it in a massive string-keyed lookup.

---

## Optimization Strategy

**Core insight:** Separate the time-invariant spatial adjacency from the time-varying attributes.

1. **Build a spatial-only neighbor table once** â€” a simple two-column `data.table` of `(cell_id, neighbor_cell_id)` with ~1.37M rows. This is built from the `nb` object in milliseconds.

2. **Join yearly attributes onto the neighbor table** â€” For each variable, join the cell-year attribute values onto the neighbor table by `(neighbor_cell_id, year)`. This is a keyed `data.table` equi-join: extremely fast, vectorized, and memory-efficient.

3. **Aggregate neighbor stats by `(cell_id, year)`** â€” Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one pass. This is a single `data.table` grouped aggregation over ~1.37M Ã— 28 â‰ˆ 38.5M rows â€” takes seconds.

4. **Join the aggregated stats back** onto the main dataset.

**No `lapply` over 6.46M rows. No string key construction. No named vector probing.**

**Expected speedup:** From ~86 hours to **under 5 minutes** for all 5 variables.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build time-invariant spatial neighbor edge table ONCE
# ============================================================
# Input: id_order (vector of cell IDs in the order matching the nb object)
#        rook_neighbors_unique (spdep nb object, list of integer index vectors)
#
# Output: neighbor_edges â€” a data.table with columns (cell_id, neighbor_id)
#         representing all directed rook-neighbor pairs (~1.37M rows)

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Neighbor edge table: %d directed edges\n", nrow(neighbor_edges)))

# ============================================================
# STEP 2: Convert main data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year are keyed for fast joins
setkey(cell_data, id, year)

# ============================================================
# STEP 3: For each neighbor source variable, compute neighbor
#          max, min, mean via join + grouped aggregation
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Expand neighbor edges across all years (cross join with unique years)
# This creates ~1.37M * 28 â‰ˆ 38.5M rows â€” fits easily in 16 GB RAM
# (38.5M rows Ã— ~3 integer/numeric cols â‰ˆ < 1 GB)

unique_years <- sort(unique(cell_data$year))

# Build the full (cell_id, neighbor_id, year) table once
neighbor_year <- neighbor_edges[, .(year = unique_years), by = .(cell_id, neighbor_id)]
# This expands each edge to all 28 years

cat(sprintf("Neighbor-year table: %d rows (%.1f M)\n",
            nrow(neighbor_year), nrow(neighbor_year) / 1e6))

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  t0 <- proc.time()

  # Create a slim lookup table: (id, year) -> value
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Join neighbor attribute values onto the neighbor-year table
  # Match on neighbor_id == id AND year == year
  neighbor_year_vals <- merge(
    neighbor_year,
    val_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  # Aggregate: for each (cell_id, year), compute max/min/mean of neighbor values
  agg <- neighbor_year_vals[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match expected feature names
  # Typical naming convention: neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Join aggregated stats back onto cell_data
  # First remove these columns if they already exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(
    cell_data,
    agg,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Done in %.1f seconds\n", elapsed))
}

# Restore key
setkey(cell_data, id, year)

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
# ============================================================
# The trained model object is assumed to be already in memory (e.g., `rf_model`).
# The cell_data now contains all ~110 predictor columns including the
# newly computed neighbor_max_*, neighbor_min_*, neighbor_mean_* features.
#
# The numerical estimand is preserved because we compute identical
# max/min/mean statistics â€” just via vectorized joins instead of
# row-wise lapply.

# Example prediction (adjust variable name to your actual model object):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("Neighbor feature engineering complete. Ready for Random Forest prediction.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `build_neighbor_edge_table` reads the identical `rook_neighbors_unique` nb object and `id_order` mapping. Every `(cell, neighbor)` pair is preserved. |
| **Same statistics** | `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per `(cell_id, year)` group. Cells with no valid neighbors get `NA` via the `all.x = TRUE` left join. |
| **Trained RF model untouched** | No retraining. The feature columns are named identically and contain identical values, so `predict()` produces the same output. |
| **Numerical estimand preserved** | The aggregation functions are identical (`max`, `min`, `mean`). Floating-point results are bit-identical since the same R functions operate on the same values (just gathered via join rather than index subsetting). |

## Performance Summary

| Step | Rows Processed | Expected Time |
|---|---|---|
| Build edge table | ~1.37M edges | < 1 second |
| Expand to neighbor-year | ~38.5M rows | ~2 seconds |
| Per-variable join + aggregate (Ã—5) | ~38.5M rows each | ~10â€“30 seconds each |
| **Total** | | **~2â€“3 minutes** |

This replaces the ~86-hour runtime with a ~2â€“3 minute runtime â€” roughly a **1,500â€“2,500Ã— speedup** â€” by eliminating row-wise string operations and leveraging `data.table`'s vectorized keyed joins and grouped aggregation.
END_RESPONSE>>>
