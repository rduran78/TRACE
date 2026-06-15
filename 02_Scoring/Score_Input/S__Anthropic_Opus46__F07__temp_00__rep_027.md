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
run_id: Anthropic_F07_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-like overhead
- **`lapply` over ~6.46 million rows**, each performing string pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and `NA` filtering.
- `idx_lookup` is a **named character vector** with ~6.46M entries. Each lookup via `idx_lookup[neighbor_keys]` does **linear hashing on character keys** â€” this is extremely slow at scale.
- The function builds ~6.46M list elements, each containing integer vectors. Memory allocation and GC pressure are enormous.

### 2. `compute_neighbor_stats` â€” repeated per variable
- Another `lapply` over 6.46M rows **per variable** (Ã—5 variables = ~32.3M iterations).
- Each iteration subsets `vals[idx]`, removes NAs, and computes max/min/mean â€” all in interpreted R with no vectorization.

### 3. Combined effect
- ~6.46M list-element iterations for the lookup build.
- ~32.3M list-element iterations for stats.
- Estimated 86+ hours is consistent with character-key lookups and per-row R-level loops at this scale.

---

## Optimization Strategy

### A. Replace character-key lookup with integer indexing via `data.table`

Instead of building a named character vector of 6.46M entries and doing string-match lookups, use `data.table` keyed joins. Create an integer-indexed mapping from `(id, year)` â†’ row number. This turns the lookup from O(n) string matching to O(1) hash-table lookup.

### B. Vectorize the neighbor lookup build using edge-list expansion

Convert the `nb` object into a flat edge list `(cell_id, neighbor_cell_id)`. Cross-join with years. Join against the row-index table. This replaces the 6.46M-iteration `lapply` with a single vectorized `data.table` merge â€” typically seconds instead of hours.

### C. Vectorize neighbor stats with grouped `data.table` aggregation

Once we have an edge list `(row_i, neighbor_row_j)`, computing neighbor max/min/mean is a single grouped aggregation per variable â€” no R-level loop at all.

### D. Preserve the trained model and numerical estimand

The output columns have the same names and identical numerical values (max, min, mean of the same neighbor sets). The Random Forest model sees the same feature matrix. Nothing changes except speed.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {

  # ---------------------------------------------------------------
  # 0.  Convert to data.table (by reference if already; copy if not)
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order for downstream compatibility

  cell_data[, .row_id := .I]

  # ---------------------------------------------------------------
  # 1.  Build (id, year) â†’ row_id mapping

  # ---------------------------------------------------------------
  row_map <- cell_data[, .(id, year, .row_id)]
  setkey(row_map, id, year)

  # ---------------------------------------------------------------
  # 2.  Convert nb object to a flat edge list of cell IDs
  #     nb object: list of length N_cells, each element is an

  #     integer vector of neighbor *indices* into id_order.
  # ---------------------------------------------------------------
  # Build edge list: from_id -> to_id (cell-level, time-invariant)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as 0L (single element)
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(from_id = id_order[i], to_id = id_order[nb_idx])
  }))
  # This should have ~1,373,394 rows (directed rook-neighbor pairs)

  cat("Edge list rows:", nrow(edge_list), "\n")

  # ---------------------------------------------------------------
  # 3.  Expand edge list across all years
  #     Result: (from_id, year, to_id) â€” one row per directed
  #     neighbor-pair-year.
  # ---------------------------------------------------------------
  years <- sort(unique(cell_data$year))
  # Cross join edge_list Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows)
  # This fits comfortably in RAM (~1 GB for 3 integer columns)
  edges_by_year <- CJ_dt(edge_list, years)

  cat("Edges Ã— years rows:", nrow(edges_by_year), "\n")

  # ---------------------------------------------------------------
  # 4.  Attach row indices for both the focal cell and the neighbor
  # ---------------------------------------------------------------
  # Focal cell row index
  setnames(edges_by_year, c("from_id", "to_id", "year"))
  edges_by_year[row_map, focal_row := i..row_id,
                on = .(from_id = id, year = year)]

  # Neighbor cell row index
  edges_by_year[row_map, nb_row := i..row_id,
                on = .(to_id = id, year = year)]

  # Drop edges where either focal or neighbor is missing
  edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(nb_row)]

  cat("Valid edges Ã— years:", nrow(edges_by_year), "\n")

  # ---------------------------------------------------------------
  # 5.  Compute neighbor stats per variable â€” fully vectorized
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor stats for:", var_name, "\n")

    # Pull the variable values and attach to neighbor rows
    edges_by_year[, nb_val := cell_data[[var_name]][nb_row]]

    # Grouped aggregation: max, min, mean per focal row
    stats <- edges_by_year[!is.na(nb_val),
                           .(nb_max  = max(nb_val),
                             nb_min  = min(nb_val),
                             nb_mean = mean(nb_val)),
                           by = focal_row]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign results
    cell_data[stats$focal_row, (max_col)  := stats$nb_max]
    cell_data[stats$focal_row, (min_col)  := stats$nb_min]
    cell_data[stats$focal_row, (mean_col) := stats$nb_mean]

    # Clean up temp column
    edges_by_year[, nb_val := NULL]

    cat("  Done:", var_name, "\n")
  }

  # ---------------------------------------------------------------
  # 6.  Clean up and return
  # ---------------------------------------------------------------
  cell_data[, .row_id := NULL]

  return(cell_data)
}


# --- Helper: Cross join a data.table with a vector of years ---
CJ_dt <- function(edge_dt, years_vec) {
  years_dt <- data.table(year = years_vec)
  # Cross join via merge with allow.cartesian
  result <- edge_dt[, .(from_id, to_id)][
    , CJ_year := 1L][
    years_dt[, CJ_year := 1L],
    on = "CJ_year",
    allow.cartesian = TRUE]
  result[, CJ_year := NULL]
  return(result)
}


# ================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the same 15 neighbor feature columns
# (neighbor_max_*, neighbor_min_*, neighbor_mean_* for each of 5 vars)
# with identical numerical values as the original implementation.
# The trained Random Forest model can be applied directly.
```

---

## Performance Comparison

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~6.46M `lapply` iterations with character-key lookups | Single vectorized `data.table` join (~seconds) |
| Stats per variable | ~6.46M `lapply` iterations Ã— 5 vars | Single grouped `data.table` aggregation Ã— 5 vars |
| Total estimated time | **86+ hours** | **~2â€“10 minutes** |
| Peak RAM | ~6â€“8 GB (list of 6.46M vectors) | ~4â€“6 GB (edge table ~38.5M rows Ã— 5 int cols) |
| Numerical output | Identical | **Identical** (same max/min/mean over same neighbor sets) |
| Trained RF model | Unchanged | **Unchanged** |

### Why this is safe and exact:
1. **Same neighbor topology**: The edge list is derived from the identical `rook_neighbors_unique` nb object â€” no raster focal assumptions.
2. **Same aggregation functions**: `max`, `min`, `mean` computed over exactly the same sets of non-NA neighbor values.
3. **Same column names**: Output columns follow the same naming convention, so the trained Random Forest model's feature expectations are met exactly.
4. **Irregular/masked topology preserved**: Cells with no neighbors or missing year-observations naturally produce `NA` â€” identical to the original behavior.
END_RESPONSE>>>
