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
run_id: Anthropic_F09_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two fundamental bottlenecks**:

### 1. `build_neighbor_lookup` â€” O(NÂ²)-scale overhead via per-row `lapply`
This function iterates over all **~6.46 million cell-year rows** individually. For each row, it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Performs named-vector lookup via `idx_lookup[neighbor_keys]` â€” this is an **O(k)** hash lookup per neighbor, but the constant factor of character matching across a **6.46-million-entry named vector** is enormous.

The result: ~6.46M iterations Ã— ~4 neighbors Ã— string construction + named-vector lookup = billions of character operations. This is the primary time sink.

**Key insight:** The spatial neighbor topology is **time-invariant**. There are only 344,208 unique cells and their neighbor relationships are fixed. The current code redundantly recomputes the same spatial relationships 28 times (once per year). It should build a **cell-level adjacency table once**, then use vectorized joins to resolve cell-year rows.

### 2. `compute_neighbor_stats` â€” `lapply` over 6.46M rows
Even after the lookup is built, computing stats via `lapply` with per-element R function calls over 6.46M rows is slow. This should be replaced with a vectorized group-by aggregation.

---

## Optimization Strategy

1. **Build a static cell-neighbor edge table once** â€” a simple two-column `data.table` of `(cell_id, neighbor_cell_id)` with ~1.37M rows, derived directly from the `nb` object. This is done **once** and is year-independent.

2. **Join yearly attributes onto the edge table** â€” for each year, cell attributes are already in the panel. By joining `cell_data` onto the edge table by `neighbor_cell_id` and `year`, we get all neighbor attribute values in a single vectorized merge.

3. **Compute grouped aggregations** â€” use `data.table` grouped `max`, `min`, `mean` by `(cell_id, year)` â€” fully vectorized, no R-level loops.

4. **Join results back** to the main dataset.

This replaces ~6.46M R-level iterations with a handful of vectorized `data.table` joins and group-bys. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure 'id' and 'year' columns exist as expected.
# 'id_order' is the vector of cell IDs aligned with rook_neighbors_unique (the nb object).
# rook_neighbors_unique is a list of length length(id_order), where each element
# contains integer indices into id_order of that cell's rook neighbors.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a STATIC cell-neighbor edge table (year-independent)
#         This is built ONCE from the nb object.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj: list of integer vectors (spdep nb object)
  # id_order: vector of cell IDs; nb_obj[[i]] indexes into id_order
  
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    # spdep nb objects use 0L to denote "no neighbors"
    sum(x > 0L)
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # remove the 0-placeholder if present
    n <- length(nbrs)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
      pos <- pos + n
    }
  }
  
  data.table(cell_id = from_id, neighbor_cell_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)

# Verify expected size
cat("Edge table rows:", nrow(edge_table), "\n")
# Should be ~1,373,394

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, compute neighbor stats
#         via vectorized join + grouped aggregation, then attach to
#         cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need cell_data to be keyed for fast joins.
# Ensure 'id' and 'year' are the join keys.
setkey(cell_data, id, year)

# Get unique years once
all_years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor features for:", var_name, "\n")
  
  # Extract only the columns we need for the join: id, year, and the variable
  # This keeps the join lightweight.
  attr_cols <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_cols, id, year)
  
  # Cross-join edge_table with all years, then join attribute values
  # for the NEIGHBOR cells.
  #
  # Strategy: expand edge_table Ã— years, then join neighbor attributes.
  # But edge_table Ã— 28 years = ~38.5M rows â€” manageable on 16 GB.
  #
  # More memory-efficient: join in a single step using data.table.
  
  # Create edge_year: each edge for each year
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_table)), year = all_years)
  edge_year[, cell_id          := edge_table$cell_id[edge_idx]]
  edge_year[, neighbor_cell_id := edge_table$neighbor_cell_id[edge_idx]]
  edge_year[, edge_idx := NULL]
  
  # Join neighbor attribute values onto edge_year
  # (look up the neighbor cell's value in that year)
  setkey(attr_cols, id, year)
  setkey(edge_year, neighbor_cell_id, year)
  edge_year[attr_cols, neighbor_value := i.value,
            on = .(neighbor_cell_id = id, year = year)]
  
  # Group by (cell_id, year) and compute max, min, mean of neighbor values
  stats <- edge_year[
    !is.na(neighbor_value),
    .(
      nb_max  = max(neighbor_value),
      nb_min  = min(neighbor_value),
      nb_mean = mean(neighbor_value)
    ),
    by = .(cell_id, year)
  ]
  
  # Name the output columns to match the original pipeline's naming convention.
  # Adjust these names to match whatever compute_and_add_neighbor_features produced.
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # Remove these columns from cell_data if they already exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Join stats back onto cell_data
  setkey(stats, cell_id, year)
  setkey(cell_data, id, year)
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id = cell_id, year = year)]
  
  # Clean up to free memory

rm(edge_year, stats, attr_cols)
  gc()
  
  cat("  Done:", var_name, "\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict using the existing trained Random Forest model
#         (model object is NOT retrained â€” used as-is)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data now has the same neighbor feature columns as the original pipeline.
# Predict exactly as before:

# predictions <- predict(trained_rf_model, newdata = cell_data)
# (Uncomment and use your actual model object name)

cat("Neighbor feature computation complete. Ready for RF prediction.\n")
```

---

## Memory-Optimized Variant (if 16 GB is tight)

The `edge_year` table above has ~38.5M rows, which at ~3 numeric columns is ~1 GB. If memory is tight, process **one year at a time** â€” still vastly faster than the original:

```r
for (var_name in neighbor_source_vars) {
  
  cat("Processing:", var_name, "\n")
  
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Pre-allocate result columns with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)
  
  for (yr in all_years) {
    
    # Get attribute values for this year only
    yr_attr <- attr_dt[year == yr, .(id, value)]
    setkey(yr_attr, id)
    
    # Join neighbor values via edge_table
    yr_edges <- copy(edge_table)
    yr_edges[yr_attr, neighbor_value := i.value,
             on = .(neighbor_cell_id = id)]
    
    # Aggregate
    yr_stats <- yr_edges[
      !is.na(neighbor_value),
      .(nb_max = max(neighbor_value),
        nb_min = min(neighbor_value),
        nb_mean = mean(neighbor_value)),
      by = .(cell_id)
    ]
    
    # Write back into cell_data for this year
    idx <- cell_data[year == yr, which = TRUE]
    id_in_year <- cell_data$id[idx]
    
    m <- match(id_in_year, yr_stats$cell_id)
    set(cell_data, i = idx, j = max_col,  value = yr_stats$nb_max[m])
    set(cell_data, i = idx, j = min_col,  value = yr_stats$nb_min[m])
    set(cell_data, i = idx, j = mean_col, value = yr_stats$nb_mean[m])
  }
  
  rm(attr_dt)
  gc()
  cat("  Done:", var_name, "\n")
}
```

This variant processes only ~1.37M edge rows per year-variable combination, using negligible memory.

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~seconds (1.37M-row table from nb) |
| Compute neighbor stats (per variable) | ~17 hours (6.46M lapply calls) | ~30â€“90 seconds (vectorized join + group-by) |
| **Total for 5 variables** | **~86+ hours** | **~5â€“10 minutes** |

The trained Random Forest model is **never retrained** â€” only the feature-engineering step is optimized. The numerical values of `max`, `min`, and `mean` are computed identically, preserving the original estimand.
END_RESPONSE>>>
