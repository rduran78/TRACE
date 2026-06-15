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
run_id: Anthropic_F12_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and "repeated list binding" in `compute_neighbor_stats()` are the main bottleneck is **incorrect**. Here's why:

**`compute_neighbor_stats()` is actually efficient.** It uses `lapply` to build a list of fixed-length vectors (length 3), then calls `do.call(rbind, ...)` once. There is no repeated list binding â€” the list is built in one pass and bound once. For ~6.46M rows, `do.call(rbind, list_of_vectors)` on a list of 3-element numeric vectors is fast (seconds, not hours).

**The true bottleneck is `build_neighbor_lookup()`.** This function runs a `lapply` over every one of the ~6.46 million rows, and inside each iteration it:

1. Calls `as.character()` on a single ID and does a named-vector lookup (`id_to_ref`).
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” constructs character keys per row.
4. Performs named-vector lookup `idx_lookup[neighbor_keys]` â€” this is a **hash-table lookup on a named vector of 6.46M entries**, repeated for every neighbor of every row.

The named-vector lookups (`id_to_ref[...]` and `idx_lookup[...]`) are O(n) scans in base R for each call when the vector is large. Across 6.46M rows Ã— ~4 neighbors each, this is catastrophic: roughly **25+ million named-vector lookups against a 6.46M-entry vector**. This is the 86-hour bottleneck.

Additionally, `build_neighbor_lookup` is called **once** but produces a list of 6.46M integer vectors, each allocated separately â€” enormous memory overhead and GC pressure on a 16 GB laptop.

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins or environment-based hashing.** R environments use true hash tables with O(1) lookup, unlike named vectors.

2. **Vectorize the neighbor lookup entirely.** Instead of iterating row-by-row, expand the neighbor relationships into a flat edge table, join on (neighbor_id, year) to get row indices, then group-by to compute stats â€” all vectorized via `data.table`.

3. **Eliminate the 6.46M-element list.** Instead of storing a per-row list of neighbor indices, compute neighbor stats directly from the flat join.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing identical numerical columns.

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  # --- Step 1: Build a flat edge table from the nb object ---
  # rook_neighbors_unique is a list of integer vectors (spdep nb object).
  # neighbors[[i]] gives the indices into id_order that are neighbors of id_order[i].
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(focal_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(
      focal_id    = rep(id_order[i], length(nb)),
      neighbor_id = id_order[nb]
    )
  }))
  # edge_list has ~1,373,394 rows (directed relationships), independent of years.
  
  # --- Step 2: Create a row-index column in dt ---
  dt[, row_idx := .I]
  
  # --- Step 3: For each variable, vectorized join + grouped aggregation ---
  for (var_name in neighbor_source_vars) {
    
    # Subset to needed columns for the join: neighbor's id, year, and value
    neighbor_vals_dt <- dt[, .(neighbor_id = id, year, val = get(var_name))]
    
    # Join: for each (focal_id, year), find all neighbor rows
    # First, cross edge_list with years implicitly by joining on neighbor_id + year
    # We need focal (id, year) -> neighbor (id, year) -> value
    
    # Focal rows: each row has (id, year). We join to edge_list on id == focal_id.
    focal_edges <- merge(
      dt[, .(row_idx, focal_id = id, year)],
      edge_list,
      by = "focal_id",
      allow.cartesian = TRUE
    )
    # focal_edges now has columns: focal_id, row_idx, year, neighbor_id
    # Each row says: "for focal row row_idx, one neighbor is neighbor_id in the same year"
    
    # Now join to get the neighbor's value in that year
    focal_edges <- merge(
      focal_edges,
      neighbor_vals_dt,
      by = c("neighbor_id", "year"),
      all.x = TRUE
    )
    # focal_edges now has: row_idx, neighbor_id, year, focal_id, val
    
    # Remove NA values (matching original logic: neighbor_vals[!is.na(neighbor_vals)])
    focal_edges_clean <- focal_edges[!is.na(val)]
    
    # Aggregate by row_idx
    stats <- focal_edges_clean[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = row_idx]
    
    # Assign back to dt; rows with no valid neighbors get NA (matching original)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    dt[stats, on = "row_idx", (max_col)  := i.nb_max]
    dt[stats, on = "row_idx", (min_col)  := i.nb_min]
    dt[stats, on = "row_idx", (mean_col) := i.nb_mean]
  }
  
  dt[, row_idx := NULL]
  
  # Return as data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# --- Usage (replaces the entire outer loop) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The output columns are numerically identical to the original pipeline.
```

### Optional further optimization â€” avoid repeated `merge` with `edge_list`:

```r
# If memory allows (~16 GB should be fine for this), precompute focal_edges once:
compute_all_neighbor_features_v2 <- function(cell_data, id_order, rook_neighbors_unique,
                                             neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  # Step 1: Flat edge table
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  
  dt[, row_idx := .I]
  
  # Step 2: Build the full focal-neighbor-year mapping ONCE
  # Key insight: this is the expensive join, do it once
  focal_map <- merge(
    dt[, .(row_idx, focal_id = id, year)],
    edge_list,
    by = "focal_id",
    allow.cartesian = TRUE
  )
  
  # Create a keyed lookup for (id, year) -> row_idx in dt for neighbor value retrieval
  setkey(dt, id, year)
  
  # Get neighbor row indices
  focal_map[, nb_row_idx := dt[.(neighbor_id, year), which = TRUE, mult = "first"]]
  focal_map <- focal_map[!is.na(nb_row_idx)]
  
  # Step 3: For each variable, just index + aggregate (very fast)
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    focal_map[, val := vals[nb_row_idx]]
    
    stats <- focal_map[!is.na(val), .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = row_idx]
    
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    dt[stats, on = "row_idx", `:=`(
      (max_col)  = i.nb_max,
      (min_col)  = i.nb_min,
      (mean_col) = i.nb_mean
    )]
  }
  
  dt[, row_idx := NULL]
  setkey(dt, NULL)
  if (!is.data.table(cell_data)) return(as.data.frame(dt))
  return(dt)
}
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` in `compute_neighbor_stats` | Not the bottleneck â€” single call on a pre-built list is fast |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M iterations with per-row `paste()` + named-vector hash lookups against a 6.46M-entry vector |
| **Root cause** | â€” | Named-vector lookup in R is O(n) per access; 25M+ lookups Ã— 6.46M entries = catastrophic |
| **Fix** | â€” | Replace with `data.table` keyed joins: build flat edge table once, join once, aggregate vectorized |
| **Expected speedup** | â€” | From ~86 hours to **minutes** (vectorized joins + grouped aggregation) |
| **RF model** | â€” | Preserved, untouched; output columns are numerically identical |
END_RESPONSE>>>
