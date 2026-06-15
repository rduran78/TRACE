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
run_id: Anthropic_F07_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries. Doing this 6.46M times yields an effective O(nÂ²) operation.
2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector subsetting, and NA filtering. The per-iteration overhead of R's interpreted loop is enormous at this scale.

Secondary cost: `compute_neighbor_stats` also uses `lapply` over 6.46M elements, but each iteration is cheap (a few numeric operations). Still, 6.46M R-level function calls add up.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~80+ hours (dominant)
- `compute_neighbor_stats` Ã— 5 variables: ~5â€“6 hours
- Total: ~86+ hours

## Optimization Strategy

### 1. Replace named-vector lookup with `data.table` hash joins (O(1) amortized)

Instead of building a 6.46M-entry named character vector and indexing into it row-by-row, we:
- Create a `data.table` keyed on `(id, year)` with a row-index column.
- Expand the neighbor list into an edge table: `(source_row, neighbor_cell_id)`.
- Join the edge table against the keyed `data.table` to resolve `(neighbor_cell_id, year)` â†’ `neighbor_row` in one vectorized hash join.

This replaces 6.46M interpreted R iterations with a single vectorized join.

### 2. Replace `lapply`-based stats with `data.table` grouped aggregation

Once we have an edge table `(source_row, neighbor_row)`, we pull the variable values for all neighbor rows, then `group by source_row` to compute `max`, `min`, `mean` â€” all vectorized in C via `data.table`.

### 3. Memory management

- The edge table will have ~6.46M Ã— 4 neighbors â‰ˆ 26M rows (but actually ~1.37M directed edges Ã— 28 years â‰ˆ 38.5M rows). At ~3 integer columns, this is ~900 MB â€” fits in 16 GB.
- We process one variable at a time and discard intermediate objects.

**Expected runtime: ~2â€“5 minutes total** (down from 86+ hours).

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table if not already.
#         Assumes cell_data has columns: id, year, and the source vars.
#         Assumes id_order is a vector of cell IDs in the same order as
#         rook_neighbors_unique (the spdep nb object).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so downstream predictions are aligned
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build the spatial edge list (cell-level, time-invariant)
#
#   rook_neighbors_unique[[i]] gives the indices (into id_order) of
#   the rook neighbors of cell id_order[i].
#
#   We expand this into a two-column data.table:
#     (focal_id, neighbor_id)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (spdep nb object)
  n <- length(nb_obj)
  # Pre-compute lengths for pre-allocation
  lens <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(lens)

  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep convention where 0 means "no neighbors"
  valid <- neighbor_idx != 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

cat(sprintf("Edge list: %d directed cell-level edges\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Expand edges across years by joining to the panel
#
#   For every (focal_id, year) row in cell_data, we need the row
#   indices of (neighbor_id, year).  We do this with two keyed joins.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Create a lookup: (id, year) -> row index in cell_data
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# Cross the edge list with all years present in the data
all_years <- sort(unique(cell_data$year))

# Expand: each spatial edge Ã— each year
# Memory: ~38.5M rows Ã— 4 cols (ints) â‰ˆ manageable
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = all_years)
edge_year[, focal_id    := edge_dt$focal_id[edge_idx]]
edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year[, edge_idx := NULL]

# Join to get focal row index
setkey(edge_year, focal_id, year)
edge_year <- row_lookup[edge_year, on = .(id = focal_id, year = year), nomatch = 0L]
setnames(edge_year, ".row_idx", "focal_row")

# Join to get neighbor row index
setkey(edge_year, neighbor_id, year)
edge_year <- row_lookup[edge_year, on = .(id = neighbor_id, year = year), nomatch = 0L]
setnames(edge_year, ".row_idx", "neighbor_row")

# Keep only what we need
edge_year <- edge_year[, .(focal_row, neighbor_row)]
setkey(edge_year, focal_row)

cat(sprintf("Panel edge table: %d directed cell-year edges\n", nrow(edge_year)))

# Free memory
rm(row_lookup)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Compute neighbor stats for each variable â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Pull neighbor values into the edge table
  edge_year[, val := cell_data[[var_name]][neighbor_row]]

  # Aggregate by focal_row, dropping NAs
  stats <- edge_year[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = focal_row
  ]

  # Initialize new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign results back by row index
  cell_data[stats$focal_row, (max_col)  := stats$nb_max]
  cell_data[stats$focal_row, (min_col)  := stats$nb_min]
  cell_data[stats$focal_row, (mean_col) := stats$nb_mean]

  # Clean up the temporary column in edge_year
  edge_year[, val := NULL]

  cat(sprintf("  Done: %s â€” %d rows with neighbor stats\n", var_name, nrow(stats)))
  rm(stats)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Restore original row order and clean up
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setorder(cell_data, .row_idx)
cell_data[, .row_idx := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_data)

cat("Neighbor feature engineering complete.\n")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `build_edge_list` reads directly from the same `rook_neighbors_unique` nb object and `id_order`. No relationships are added or dropped. |
| **Same numerical estimand** | `max`, `min`, `mean` are computed on exactly the same sets of non-NA neighbor values per `(focal_id, year)` as the original code. The `data.table` aggregation is numerically identical (IEEE 754 double arithmetic, same operations). |
| **Trained Random Forest untouched** | No model code is modified. The output columns have the same names and values, so `predict(rf_model, cell_data)` produces identical results. |
| **Handles irregular topology** | Cells with zero neighbors in the nb object produce zero edges, so they get `NA` for all neighbor stats â€” identical to the original `if (length(idx) == 0) return(c(NA, NA, NA))` logic. Boundary cells with fewer than 4 neighbors are handled naturally. |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Build lookup / edge table | ~80 hours | ~10â€“30 seconds |
| Compute stats (5 vars) | ~6 hours | ~1â€“3 minutes |
| **Total** | **~86 hours** | **~2â€“5 minutes** |

The speedup factor is roughly **1,000â€“2,500Ã—**, achieved entirely by replacing interpreted row-by-row string operations with vectorized integer hash joins in `data.table`.
END_RESPONSE>>>
