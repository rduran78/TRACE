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
run_id: Anthropic_F04_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates over **~6.46 million rows** with an `lapply` call that, for every single row, performs:

1. A character-based named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
2. Subsetting the `neighbors` list.
3. `paste()`-based string key construction for every neighbor of every row.
4. A second named-vector string lookup (`idx_lookup[neighbor_keys]`) to resolve row indices.

Named-vector lookup in R is **O(n)** per query (it is a linear scan of names, not a hash table). With ~6.46M rows, each performing multiple named-vector lookups against vectors of length ~344K or ~6.46M, this is catastrophically slow â€” effectively **O(nÂ²)** overall. The `paste` key construction also generates millions of temporary string allocations.

**`compute_neighbor_stats`** is comparatively cheap (just numeric subsetting and three summary functions per row), but the `lapply` + `do.call(rbind, ...)` pattern over 6.46M elements is also unnecessarily slow. It can be vectorized.

**Summary of root causes:**

| Problem | Location | Severity |
|---|---|---|
| Named-vector lookup is O(n) per query, not O(1) | `build_neighbor_lookup` | **Critical** |
| String key construction via `paste()` for every neighbor of every row | `build_neighbor_lookup` | **High** |
| `lapply` over 6.46M rows with per-row string operations | `build_neighbor_lookup` | **High** |
| `lapply` + `do.call(rbind, ...)` over 6.46M rows | `compute_neighbor_stats` | **Moderate** |

---

## Optimization Strategy

### 1. Replace named-vector lookups with `data.table` hash joins or R environment-based hashing

R `environment` objects use hashing and give **O(1)** lookup. Even better: avoid string keys entirely by switching to integer-keyed `data.table` joins.

### 2. Eliminate per-row string construction â€” pre-build the full neighbor-row mapping as a two-column integer table

Instead of building a list of length 6.46M (one element per row), build a single `data.table` with columns `(row_i, neighbor_row_j)`. This is a "long-form" edge list of ~6.46M Ã— avg_neighbors â‰ˆ ~25â€“30M rows of integer pairs. Then compute grouped statistics with `data.table` grouped aggregation â€” **no R-level loop at all**.

### 3. Vectorize `compute_neighbor_stats` via `data.table` grouped aggregation

Once we have the edge table `(row_i, neighbor_row_j)`, computing max/min/mean of neighbor values is a single `data.table` group-by operation.

### Expected speedup

| Step | Before | After | Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~80+ hours | ~30â€“90 seconds | ~3000â€“10000Ã— |
| `compute_neighbor_stats` (Ã—5 vars) | ~6 hours | ~5â€“20 seconds per var | ~1000Ã— |
| **Total** | **86+ hours** | **~3â€“5 minutes** | **~1000Ã—+** |

Memory: the edge table is ~25â€“30M rows Ã— 2 integer columns â‰ˆ ~230 MB. Fits in 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build a vectorized neighbor-row edge table (replaces build_neighbor_lookup)
# ===========================================================================

build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the same order as the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  dt <- as.data.table(data)
  dt[, row_i := .I]  # original row index

  # --- Map each cell ID to its position in id_order (1-based ref index) ---
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )

  # --- Build the directed edge list at the cell level ---
  # For each ref index, list its neighbor ref indices
  edge_cell <- rbindlist(lapply(seq_along(neighbors), function(r) {
    nb <- neighbors[[r]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(ref_from = r, ref_to = nb)
  }))

  # Translate ref indices back to cell IDs
  edge_cell[, id_from := id_order[ref_from]]
  edge_cell[, id_to   := id_order[ref_to]]
  edge_cell[, c("ref_from", "ref_to") := NULL]

  # --- Expand to cell-year level by joining with the data ---
  # Get unique years
  years <- sort(unique(dt$year))

  # Build a lookup: (id, year) -> row_i
  setkey(dt, id, year)
  row_lookup <- dt[, .(id, year, row_i)]

  # Cross join edges with years: every edge exists in every year
  edge_cell_year <- CJ_dt(edge_cell, years)

  # Join to get row_i for the source row (from)
  setnames(row_lookup, c("id", "year", "row_i"), c("id_from", "year", "row_i_from"))
  setkey(row_lookup, id_from, year)
  setkey(edge_cell_year, id_from, year)
  edge_cell_year <- row_lookup[edge_cell_year, nomatch = 0L]

  # Join to get row_i for the neighbor row (to)
  setnames(row_lookup, c("id_from", "year", "row_i_from"), c("id_to", "year", "row_i_to"))
  setkey(row_lookup, id_to, year)
  setkey(edge_cell_year, id_to, year)
  edge_cell_year <- row_lookup[edge_cell_year, nomatch = 0L]

  # Return clean two-column integer edge table
  edge_cell_year[, .(row_i = row_i_from, neighbor_row_j = row_i_to)]
}

# Helper: cross join a data.table of edges with a vector of years
CJ_dt <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join via allow.cartesian
  k <- edge_dt[, dummy := 1L]
  years_dt[, dummy := 1L]
  result <- merge(k, years_dt, by = "dummy", allow.cartesian = TRUE)
  result[, dummy := NULL]
  result
}

# ===========================================================================
# STEP 2: Vectorized neighbor statistics (replaces compute_neighbor_stats)
# ===========================================================================

compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
  # data_dt: data.table with a row_i column and the variable of interest
  # edge_table: data.table with columns row_i, neighbor_row_j
  # var_name: character, name of the variable

  vals <- data_dt[[var_name]]

  # Attach neighbor values to edge table
  et <- copy(edge_table)
  et[, nval := vals[neighbor_row_j]]

  # Remove NAs in neighbor values
  et <- et[!is.na(nval)]

  # Grouped aggregation
  stats <- et[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = row_i]

  # Rename columns to match original convention
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))

  stats
}

# ===========================================================================
# STEP 3: Full pipeline (replaces outer loop)
# ===========================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  dt <- as.data.table(cell_data)
  dt[, row_i := .I]

  message("Building neighbor edge table...")
  t0 <- Sys.time()
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_table, row_i)
  message("  Done in ", round(difftime(Sys.time(), t0, units = "secs"), 1), "s. ",
          "Edge table: ", nrow(edge_table), " rows.")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()

    stats <- compute_neighbor_stats_fast(dt, edge_table, var_name)

    # Left-join stats back to main table
    # Rows with no valid neighbors will get NA (preserving original behavior)
    dt <- merge(dt, stats, by = "row_i", all.x = TRUE)

    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), "s.")
  }

  # Remove helper column, restore original row order
  setorder(dt, row_i)
  dt[, row_i := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }

  dt
}

# ===========================================================================
# USAGE (drop-in replacement for the original outer loop)
# ===========================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed directly to Random Forest prediction with the existing trained model:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Key Guarantees

| Requirement | How it is preserved |
|---|---|
| **Trained RF model untouched** | Code only constructs features; no model retraining. |
| **Numerical estimand identical** | `max`, `min`, `mean` computed on the same neighbor sets with the same NA handling. `data.table` uses the same IEEE 754 arithmetic. |
| **Column names match** | Output columns are named `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` â€” adjust the suffix pattern in `compute_neighbor_stats_fast` if your original wrapper used a different convention. |
| **16 GB RAM** | Edge table â‰ˆ 230 MB; `data.table` in-place operations keep peak memory well under 10 GB. |
| **No new dependencies beyond `data.table`** | Already a standard R package; no compiled C++ code or Rcpp required. |
END_RESPONSE>>>
