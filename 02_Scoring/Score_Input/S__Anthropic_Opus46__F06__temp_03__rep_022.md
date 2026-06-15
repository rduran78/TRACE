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
run_id: Anthropic_F06_T03_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(neighbor_id, year, "_")`).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String pasting and named-vector lookup for 6.46M rows Ã— ~4 neighbors each â‰ˆ **25.8 million string operations and hash lookups**. The named-vector lookup in R is O(n) per probe in the worst case because `setNames` creates a character-indexed vector, not a true hash table. This alone can take tens of hours.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M rows, repeated 5Ã—

For each of 5 variables, another `lapply` iterates over 6.46M rows computing `max`, `min`, `mean` on small vectors. That's **32.3 million R-level function calls** with per-element subsetting.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular grid with a fixed rectangular kernel. Here, the grid cells have an irregular neighbor structure (coastal cells, borders, missing cells in certain years), and the panel dimension (year) means neighbors are only within the same year. A focal approach would require reshaping into a 3D raster stack and masking â€” possible but fragile and likely to change numerical results at boundaries. **The correct strategy is to vectorize the existing logic, not change the topology.**

---

## 2. Optimization Strategy

| Step | Technique | Speedup Factor |
|------|-----------|---------------|
| Replace string-key lookup with integer arithmetic | Eliminate `paste()` and named-vector lookup entirely | ~50â€“100Ã— |
| Replace `lapply` in neighbor lookup with `data.table` merge | Vectorized join on `(cell_id, year)` â†’ row index | ~100Ã— |
| Replace per-row `lapply` in stats with sparse-matrix or vectorized group operations | Use `data.table` grouped aggregation on an edge list | ~50Ã— |
| Compute all 5 variables in one pass over the edge list | Avoid rebuilding structures 5 times | 5Ã— |

**Expected total: from ~86 hours â†’ minutes (5â€“15 min on a 16 GB laptop).**

The key insight: instead of iterating row-by-row, build an **edge table** â€” one row per (focal_row, neighbor_row) pair â€” then join the variable values and do grouped `max`/`min`/`mean` via `data.table`.

---

## 3. Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # -------------------------------------------------------------------
  # STEP 1: Convert to data.table and create integer row indices

# -------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # -------------------------------------------------------------------
  # STEP 2: Build an edge list of (focal_cell_id, neighbor_cell_id)
  #         from the spdep nb object (cell-level, year-independent)
  # -------------------------------------------------------------------
  # rook_neighbors_unique is a list of length = length(id_order)
  # rook_neighbors_unique[[i]] gives integer indices into id_order
  # for the neighbors of id_order[i].

  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  cat("Edge list rows (directed cell-level):", nrow(edge_list), "\n")

  # -------------------------------------------------------------------
  # STEP 3: Expand edge list to panel level by joining on year.
  #         focal row <-> neighbor row within the same year.
  # -------------------------------------------------------------------
  # Create a lookup: (id, year) -> row_idx
  id_year_lookup <- dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Cross join edges Ã— years, then map to row indices
  # To avoid a huge cross join in memory, we do it via merge.

  # Focal side: map focal_id Ã— year -> focal_row_idx
  # Neighbor side: map neighbor_id Ã— year -> neighbor_row_idx

  # Create the full panel edge list efficiently:
  # For each year, the same cell-level edges apply.
  panel_edges <- CJ_dt_edges(edge_list, years, id_year_lookup)

  cat("Panel edge list rows:", nrow(panel_edges), "\n")

  # -------------------------------------------------------------------
  # STEP 4: For each source variable, compute grouped stats
  # -------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")

    # Attach neighbor values to the edge list
    panel_edges[, neighbor_val := dt[[var_name]][neighbor_row_idx]]

    # Remove NAs in neighbor values
    valid_edges <- panel_edges[!is.na(neighbor_val)]

    # Grouped aggregation: max, min, mean by focal_row_idx
    stats <- valid_edges[, .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ), by = focal_row_idx]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values
    dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]

    # Clean up
    panel_edges[, neighbor_val := NULL]
  }

  # -------------------------------------------------------------------
  # STEP 5: Return as data.frame (preserving compatibility)
  # -------------------------------------------------------------------
  dt[, row_idx := NULL]
  return(as.data.frame(dt))
}


# Helper: expand cell-level edges to panel-level edges via year
CJ_dt_edges <- function(edge_list, years, id_year_lookup) {
  # Replicate edge_list for each year
  year_dt <- data.table(year = years)
  # Cross join: every edge Ã— every year
  panel_edges <- edge_list[, .(focal_id, neighbor_id)][
    , CJ_year := 1L  # dummy for cross join
  ]

  # More memory-efficient: use rep
  n_edges <- nrow(edge_list)
  n_years <- length(years)

  panel_edges <- data.table(
    focal_id    = rep(edge_list$focal_id, times = n_years),
    neighbor_id = rep(edge_list$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )

  # Map focal_id + year -> focal_row_idx
  setkey(id_year_lookup, id, year)

  panel_edges[id_year_lookup,
              focal_row_idx := i.row_idx,
              on = .(focal_id = id, year = year)]

  panel_edges[id_year_lookup,
              neighbor_row_idx := i.row_idx,
              on = .(neighbor_id = id, year = year)]

  # Drop edges where either side is missing (cell not observed that year)
  panel_edges <- panel_edges[!is.na(focal_row_idx) & !is.na(neighbor_row_idx)]

  # Keep only needed columns
  panel_edges[, c("focal_id", "neighbor_id") := NULL]

  return(panel_edges)
}


# =====================================================================
# USAGE
# =====================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is then applied as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory Budget Check

| Object | Approximate Size |
|--------|-----------------|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `panel_edges` (1.37M edges Ã— 28 years Ã— 2 int cols) | ~293 MB |
| `id_year_lookup` (6.46M Ã— 3 cols) | ~155 MB |
| Temporary stats per variable | ~50 MB |
| **Total peak** | **~7â€“8 GB** â†’ fits in 16 GB |

---

## 5. Why This Preserves the Original Numerical Estimand

- **Same topology**: The edge list is constructed from the identical `rook_neighbors_unique` nb object â€” no approximation, no kernel substitution.
- **Same aggregation functions**: `max`, `min`, `mean` computed on exactly the same neighbor sets, with the same NA handling (`!is.na` filter before aggregation, `NA` output when no valid neighbors exist).
- **Same output columns**: The 15 new columns (`neighbor_{max,min,mean}_{var}`) are numerically identical to the original implementation.
- **No model retraining**: The pre-trained Random Forest is applied downstream unchanged.

The optimization is purely computational â€” replacing R-level row iteration and string operations with vectorized `data.table` joins and grouped aggregations â€” producing **bit-identical results** in approximately **5â€“15 minutes** instead of 86+ hours.
END_RESPONSE>>>
