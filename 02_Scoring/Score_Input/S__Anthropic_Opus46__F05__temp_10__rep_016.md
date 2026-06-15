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
run_id: Anthropic_F05_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) per-row `lapply` with string operations

`build_neighbor_lookup` calls `lapply` over all ~6.46 million rows. For each row it:

1. **Pastes** neighbor cell IDs with the current year to form string keys â€” ~4 string allocations per neighbor Ã— ~4 neighbors (rook) Ã— 6.46M rows â‰ˆ **103 million `paste` calls**.
2. **Looks up** those keys in a named character vector of length 6.46M â€” each lookup is an O(1) hash probe, but the hash table is built over **character keys of length 6.46M**, and R's named-vector lookup has substantial per-call overhead.

But more importantly, the entire `neighbor_lookup` structure (a list of 6.46M integer vectors) is **invariant across the 5 source variables**. It's correctly built once and reused â€” so that part is fine. The real bottleneck is the *construction* of `neighbor_lookup` itself via the per-row `lapply`.

### Why it takes 86+ hours

| Step | Complexity | Estimated cost |
|------|-----------|----------------|
| `paste` to build `idx_lookup` (once) | O(N) | ~seconds |
| Per-row `paste` + hash lookup inside `lapply` | O(N Ã— avg_neighbors) with high constant factor | **~hours** |
| `compute_neighbor_stats` (5 vars) | O(N Ã— avg_neighbors) but pure integer indexing | ~minutes |

The dominant cost is `build_neighbor_lookup`. The per-row string construction and named-vector lookup in an `lapply` over 6.46M rows is the killer. R's `lapply` here cannot be vectorized because each row has a *variable-length* neighbor set.

### The Deeper Insight: The neighbor graph is year-invariant

Every cell has the same rook neighbors in every year. The `nb` object defines a **spatial** adjacency that doesn't change over time. The lookup is simply: "for row `i` (cell `c` in year `t`), find all rows where cell âˆˆ neighbors(c) AND year = `t`." This is a **structural join** that can be computed entirely with vectorized integer arithmetic â€” no strings needed.

---

## Optimization Strategy

**Eliminate all string operations. Replace the per-row `lapply` with a fully vectorized merge/join approach:**

1. **Build a cell-to-row-index mapping** using integer arithmetic: create a matrix or data.table keyed by `(cell_id, year)` â†’ `row_index`.
2. **Expand the neighbor list** into an edge table: `(cell_from, cell_to)` â€” ~1.37M directed edges.
3. **Cross-join** edges with years to get `(cell_from, year, cell_to)` â€” ~1.37M Ã— 28 = ~38.5M rows.
4. **Join** to map `(cell_from, year)` â†’ `source_row` and `(cell_to, year)` â†’ `neighbor_row`.
5. **Compute grouped statistics** (max, min, mean) per `source_row` using `data.table` grouped aggregation â€” fully vectorized C-level code.

This replaces billions of R-level operations with a handful of vectorized joins and group-bys.

**Estimated time: minutes instead of days.**

---

## Working R Code

```r
library(data.table)

#' Vectorized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats entirely.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all var_names
#' @param id_order          integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors    spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Assign a row index to every row ---
  dt[, .row_idx := .I]

  # --- Step 2: Build edge list from the nb object ---
  # rook_neighbors[[k]] contains integer indices into id_order
  # So cell id_order[k] has neighbors id_order[ rook_neighbors[[k]] ]
  edges <- rbindlist(lapply(seq_along(rook_neighbors), function(k) {
    nb_idx <- rook_neighbors[[k]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(cell_from = id_order[k], cell_to = id_order[nb_idx])
  }))
  # edges is ~1.37M rows: (cell_from, cell_to)

  cat(sprintf("Edge table: %s directed edges\n", format(nrow(edges), big.mark = ",")))

  # --- Step 3: Build a keyed lookup from (id, year) -> row_idx ---
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --- Step 4: Get unique years ---
  years <- sort(unique(dt$year))

  # --- Step 5: Cross-join edges Ã— years ---
  # This gives us ~38.5M rows: (cell_from, cell_to, year)
  # Memory: ~38.5M Ã— 3 int columns â‰ˆ 462 MB â€” fits in 16 GB
  edge_year <- CJ_edges_years(edges, years)

  cat(sprintf("Edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

  # --- Step 6: Map cell_from,year -> source row index ---
  setkey(edge_year, cell_from, year)
  edge_year[row_lookup, source_row := i..row_idx, on = .(cell_from = id, year = year)]

  # --- Step 7: Map cell_to,year -> neighbor row index ---
  setkey(edge_year, cell_to, year)
  edge_year[row_lookup, neighbor_row := i..row_idx, on = .(cell_to = id, year = year)]

  # --- Step 8: Drop any unmatched (boundary cells with neighbors outside data) ---
  edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

  cat(sprintf("Matched edge-year rows: %s\n", format(nrow(edge_year), big.mark = ",")))

  # --- Step 9: For each source variable, pull neighbor values and aggregate ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Pull the neighbor's value for this variable
    edge_year[, nval := dt[[var_name]][neighbor_row]]

    # Grouped aggregation: max, min, mean per source_row (excluding NAs)
    agg <- edge_year[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = source_row]

    # Create full-length result columns (NA for cells with no valid neighbors)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))

    max_col[agg$source_row]  <- agg$nb_max
    min_col[agg$source_row]  <- agg$nb_min
    mean_col[agg$source_row] <- agg$nb_mean

    # Assign to dt with the same naming convention as the original code
    # Adjust column names to match whatever compute_and_add_neighbor_features produced
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }

  # Clean up temporary column
  edge_year[, nval := NULL]
  dt[, .row_idx := NULL]

  return(dt)
}


#' Helper: Cross-join edges with years without full CJ explosion in memory at once.
#' For ~1.37M edges Ã— 28 years this is fine to do directly.
CJ_edges_years <- function(edges, years) {
  # Replicate each edge for every year
  n_edges <- nrow(edges)
  n_years <- length(years)

  edge_year <- data.table(
    cell_from = rep(edges$cell_from, times = n_years),
    cell_to   = rep(edges$cell_to,   times = n_years),
    year      = rep(years, each = n_edges)
  )
  return(edge_year)
}
```

### Usage â€” drop-in replacement for the original outer loop:

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data is now a data.table with the 15 new neighbor columns appended.
# Convert back to data.frame if needed downstream:
# cell_data <- as.data.frame(cell_data)
```

### Column Name Alignment

If the original `compute_and_add_neighbor_features` used different naming conventions (e.g., `ntl_max_neighbor` instead of `ntl_neighbor_max`), adjust these three lines in the loop:

```r
set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
```

to match whatever column names the trained Random Forest model expects. The **numerical values are identical** â€” same max, min, mean over the same neighbor sets â€” so the trained model needs no retraining.

---

## Performance Analysis

| Metric | Original | Optimized |
|--------|----------|-----------|
| String paste calls | ~103M+ | **0** |
| R-level loop iterations | ~6.46M | **0** (vectorized) |
| Hash lookups on 6.46M-length named vector | ~26M | **0** (keyed join) |
| `data.table` grouped aggregation | N/A | 5 passes over ~38.5M rows |
| **Estimated wall time** | **86+ hours** | **5â€“15 minutes** |
| Peak memory (edge_year table) | ~varies | ~1.2 GB (fits in 16 GB) |

### Why this is correct

- The `nb` object indexes into `id_order`. Edge `(id_order[k], id_order[j])` means cell `k` and cell `j` are rook neighbors.
- The join on `(id, year)` maps each spatial edge to the correct temporal row â€” exactly what the original string-key approach did.
- `max`, `min`, `mean` with `NA` removal matches the original `compute_neighbor_stats` logic.
- Cells with zero valid neighbors get `NA` for all three statistics â€” matching the original `c(NA, NA, NA)` return.
END_RESPONSE>>>
