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
run_id: Anthropic_F09_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` rebuilds the full cell-year neighbor mapping from scratch every time.** It iterates over all ~6.46 million rows, and for each row it does string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per call in the worst case, and string operations on 6.46M rows are inherently slow. This produces a list of 6.46 million integer vectors â€” one per row â€” which is memory-heavy and slow to construct.

2. **`compute_neighbor_stats` iterates over that 6.46M-element list with `lapply`, calling `max`, `min`, and `mean` individually per row.** This is pure row-level R looping with no vectorization.

3. **The neighbor topology is static across years** (the grid doesn't change), yet the code re-resolves neighbor relationships at the cell-year level, effectively duplicating work 28 times and entangling spatial structure with temporal attributes.

**Key insight:** The neighbor adjacency is a property of the *spatial grid*, not of the panel. There are only 344,208 cells and ~1.37M directed neighbor pairs. The yearly attribute values should be *joined onto* this small, static adjacency table, and then neighbor stats should be computed via grouped vectorized aggregation â€” not per-row R loops.

---

## Optimization Strategy

1. **Build a static neighbor edge table once** â€” a `data.table` with columns `(cell_id, neighbor_id)`, derived from the `spdep::nb` object. This table has ~1.37M rows and never changes.

2. **For each year and each variable, join the neighbor's attribute value onto the edge table**, then compute `max`, `min`, and `mean` by `cell_id` using `data.table` grouped aggregation. This replaces all `lapply` loops with vectorized, in-memory columnar operations.

3. **Merge the resulting neighbor stats back** onto the main `cell_data` data.table by `(id, year)`.

This reduces the problem from 6.46M Ã— per-row R function calls to ~28 vectorized grouped joins on a 1.37M-row table â€” a speedup of roughly **100â€“500Ã—**, bringing runtime from 86+ hours to **minutes**.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static neighbor edge table ONCE
#         Input: id_order (vector of 344,208 cell IDs)
#                rook_neighbors_unique (spdep nb object, list of length 344,208)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: neighbors[[i]] gives integer indices into id_order
  n <- length(id_order)
  
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_counts)
  
  # Build vectors
  from_id <- rep(id_order, times = edge_counts)
  to_idx  <- unlist(neighbors, use.names = FALSE)
  to_id   <- id_order[to_idx]
  
  edge_dt <- data.table(cell_id = from_id, neighbor_id = to_id)
  
  # Verify
  message(sprintf(
    "Neighbor edge table: %s rows (directed edges) for %s cells.",
    format(nrow(edge_dt), big.mark = ","),
    format(n, big.mark = ",")
  ))
  
  return(edge_dt)
}

# Build it once
edge_table <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor features for all variables via vectorized joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_table, var_names) {
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  for (var_name in var_names) {
    message(sprintf("Processing neighbor stats for: %s", var_name))
    
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Subset only the columns we need for the neighbor lookup
    # (id, year, and the variable of interest)
    attr_dt <- cell_data[, .(id, year, value = get(var_name))]
    setkey(attr_dt, id, year)
    
    # For each year, join neighbor values and aggregate
    # We process all years at once by expanding the edge table across years
    
    # Create a cross of edge_table Ã— years
    year_dt <- data.table(year = years)
    edges_by_year <- CJ_dt(edge_table, year_dt)
    
    # Join the neighbor's attribute value
    # edges_by_year has (cell_id, neighbor_id, year)
    # We want the value of var_name for (neighbor_id, year)
    setkey(attr_dt, id, year)
    setnames(attr_dt, "id", "neighbor_id")
    
    edges_by_year <- merge(
      edges_by_year,
      attr_dt,
      by = c("neighbor_id", "year"),
      all.x = TRUE
    )
    
    # Aggregate: for each (cell_id, year), compute max, min, mean of neighbor values
    stats_dt <- edges_by_year[
      !is.na(value),
      .(
        n_max  = max(value),
        n_min  = min(value),
        n_mean = mean(value)
      ),
      by = .(cell_id, year)
    ]
    
    # Rename for merge
    setnames(stats_dt, c("cell_id", "year", "n_max", "n_min", "n_mean"),
             c("id", "year", col_max, col_min, col_mean))
    setkey(stats_dt, id, year)
    
    # Remove old columns if they exist (in case of re-run)
    for (cc in c(col_max, col_min, col_mean)) {
      if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
    }
    
    # Merge back onto cell_data
    cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
    setkey(cell_data, id, year)
    
    message(sprintf("  Done: %s", var_name))
  }
  
  return(cell_data)
}

# Helper: cross join two data.tables (like CJ but for tables)
CJ_dt <- function(dt1, dt2) {
  dt1[, .___cj_key___ := 1L]
  dt2[, .___cj_key___ := 1L]
  result <- merge(dt1, dt2, by = ".___cj_key___", allow.cartesian = TRUE)
  result[, .___cj_key___ := NULL]
  dt1[, .___cj_key___ := NULL]
  dt2[, .___cj_key___ := NULL]
  return(result)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# The trained RF model object is preserved as-is.
# Example (adjust to your actual model object and predict call):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

### Memory-Conscious Variant

The cross join above creates ~1.37M Ã— 28 â‰ˆ 38.5M rows, which at ~3 columns of doubles is roughly **1â€“1.5 GB** â€” fine for 16 GB RAM. However, if memory is tight or you want even more control, process year-by-year:

```r
compute_neighbor_features_by_year <- function(cell_data, edge_table, var_names) {
  
  years <- sort(unique(cell_data$year))
  setkey(cell_data, id, year)
  
  # Pre-allocate result columns with NA
  for (var_name in var_names) {
    cell_data[, paste0("neighbor_max_",  var_name) := NA_real_]
    cell_data[, paste0("neighbor_min_",  var_name) := NA_real_]
    cell_data[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  for (yr in years) {
    message(sprintf("Year: %d", yr))
    
    # Slice this year's data
    yr_data <- cell_data[year == yr, c("id", var_names), with = FALSE]
    setkey(yr_data, id)
    
    for (var_name in var_names) {
      # Build lookup: neighbor_id -> value for this year
      val_dt <- yr_data[, .(neighbor_id = id, value = get(var_name))]
      setkey(val_dt, neighbor_id)
      
      # Join neighbor values onto edge table
      edges_with_val <- merge(edge_table, val_dt, by = "neighbor_id", all.x = FALSE)
      
      # Aggregate by cell_id
      stats <- edges_with_val[
        !is.na(value),
        .(n_max = max(value), n_min = min(value), n_mean = mean(value)),
        by = .(cell_id)
      ]
      
      # Write back into cell_data for this year
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Match rows
      idx <- cell_data[year == yr, which = TRUE]
      matched <- match(cell_data$id[idx], stats$cell_id)
      
      set(cell_data, i = idx, j = col_max,  value = stats$n_max[matched])
      set(cell_data, i = idx, j = col_min,  value = stats$n_min[matched])
      set(cell_data, i = idx, j = col_mean, value = stats$n_mean[matched])
    }
  }
  
  return(cell_data)
}

# Run the memory-friendly version instead:
cell_data <- compute_neighbor_features_by_year(cell_data, edge_table, neighbor_source_vars)
```

---

## Summary

| Aspect | Before | After |
|---|---|---|
| **Neighbor resolution** | Per cell-year row (6.46M `lapply` iterations with string ops) | Static edge table (1.37M rows), built once |
| **Stats computation** | Row-level `lapply` with per-element `max`/`min`/`mean` | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | **5â€“20 minutes** |
| **Peak RAM** | Large list of 6.46M integer vectors | ~1â€“2 GB for edge joins |
| **RF model** | Unchanged | Unchanged |
| **Numerical output** | Neighbor max, min, mean | Identical neighbor max, min, mean |
END_RESPONSE>>>
