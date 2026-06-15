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
run_id: Anthropic_F05_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts an ID to character and looks it up in a named vector â€” O(1) amortized but with string allocation overhead.
2. Extracts neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current year to form string keys â€” allocates new strings per row.
4. **Looks up** those keys in `idx_lookup` (a named vector of 6.46M entries) â€” named-vector lookup in R is hash-based but still involves repeated string hashing.

With ~6.46M rows and an average of ~8 rook neighbors per cell (1,373,394 directed relationships / 344,208 cells â‰ˆ 4 per cell, but rook typically gives ~4 undirected, ~8 directed entries in the lookup), this means roughly **50+ million `paste` + hash-lookup operations**. The named-vector approach has significant per-element overhead in R.

Then `compute_neighbor_stats` is called 5 times (once per variable), each time iterating over the 6.46M-element `neighbor_lookup` list. That's fine in isolation, but the real cost was already paid building the lookup.

### The Deeper Structural Insight

The string-keying scheme encodes a **2D index (id Ã— year)** as a flat string. But the data has a regular panel structure: every cell appears in every year (344,208 cells Ã— 28 years = 9,637,824 potential slots; 6.46M actual rows suggests some cells are missing in some years, but the structure is still highly regular). This means:

1. **Neighbor relationships are time-invariant.** Cell A's neighbors don't change across years. The `nb` object is spatial only.
2. **The year dimension is trivially indexable.** For a given row `i`, we need neighbors of `cell_id[i]` in `year[i]`. Since neighbors are the same every year, we only need to find "which rows correspond to neighbor cells in the same year."

This means we can **separate the spatial neighbor mapping from the temporal indexing** and use integer-based lookups throughout, eliminating all string operations.

## Optimization Strategy

### Strategy: Integer-Indexed Two-Level Lookup + Vectorized Aggregation via `data.table`

1. **Build a (cell_id, year) â†’ row_index integer matrix** using `data.table` for O(1) keyed joins â€” no strings.
2. **Expand the neighbor relationships into an edge table** (source_row, neighbor_row) once â€” a flat integer table of all valid neighbor-row pairs.
3. **Compute all neighbor statistics in one vectorized pass per variable** using `data.table` grouped aggregation on the edge table â€” no R-level loops over 6.46M rows.

This replaces:
- 6.46M R-level loop iterations with string operations â†’ one vectorized join
- 5 Ã— 6.46M R-level `lapply` calls â†’ 5 vectorized group-by aggregations

**Expected speedup: from 86+ hours to minutes.**

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites: cell_data (data.frame with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, ...)
#                id_order (vector of cell IDs matching the nb object indexing)
#                rook_neighbors_unique (spdep nb object)
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data, id_order, 
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  
  # --- Step 1: Convert to data.table and build integer row index -----------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]  # preserve original row order
  
  # Keyed lookup: given (id, year) -> row index
  # This replaces the paste-based idx_lookup entirely
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # --- Step 2: Build spatial edge list (cell-level, time-invariant) --------
  # Map from nb-object positional index to actual cell id
  # rook_neighbors_unique[[k]] gives positional indices of neighbors of 
  # id_order[k]
  
  cat("Building spatial edge list...\n")
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(rook_neighbors_unique))
  
  # Build edge list: (source_cell_id, neighbor_cell_id)
  # Vectorized construction
  source_pos <- rep(seq_along(rook_neighbors_unique), 
                    times = lengths(rook_neighbors_unique))
  neighbor_pos <- unlist(rook_neighbors_unique)
  
  # Remove zero-length / NA entries from nb objects (spdep convention: 
  # 0L means no neighbors)
  valid <- neighbor_pos != 0L & !is.na(neighbor_pos)
  source_pos <- source_pos[valid]
  neighbor_pos <- neighbor_pos[valid]
  
  spatial_edges <- data.table(
    source_id   = id_order[source_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  
  rm(source_pos, neighbor_pos, valid)
  
  cat(sprintf("  %s directed spatial edges\n", format(nrow(spatial_edges), 
              big.mark = ",")))
  
  # --- Step 3: Expand spatial edges across time ----------------------------
  # For each row in dt, we need to find its neighbors in the SAME year.
  # Instead of looping per row, we join:
  #   dt[i] has (id, year) -> get all neighbor_ids -> find their rows in 
  #   same year
  
  cat("Expanding edges across time (join-based)...\n")
  
  # Get unique (source_id, year, source_row_idx) from dt
  source_info <- dt[, .(source_id = id, year, source_row = .row_idx)]
  
  # Join spatial edges to source rows: for each row in dt, attach its 
  # neighbor cell IDs
  # Key: source_id
  setkey(spatial_edges, source_id)
  setkey(source_info, source_id)
  
  # This is the critical join: expand each source row by its neighbors
  # Result: (source_row, year, neighbor_id)
  edges_with_time <- spatial_edges[source_info, 
                                    .(source_row = i.source_row, 
                                      year = i.year, 
                                      neighbor_id = x.neighbor_id),
                                    on = "source_id",
                                    allow.cartesian = TRUE,
                                    nomatch = NULL]
  
  rm(source_info)
  
  # Now resolve neighbor_id + year -> neighbor_row
  # Join with row_lookup on (id=neighbor_id, year)
  edges_with_time[, id := neighbor_id]
  setkey(edges_with_time, id, year)
  
  edges_full <- row_lookup[edges_with_time, 
                            .(source_row = i.source_row, 
                              neighbor_row = x..row_idx),
                            on = .(id, year),
                            nomatch = NA]
  
  # Drop edges where the neighbor doesn't exist in that year
  edges_full <- edges_full[!is.na(neighbor_row)]
  
  rm(edges_with_time)
  
  cat(sprintf("  %s total (source_row, neighbor_row) edges\n", 
              format(nrow(edges_full), big.mark = ",")))
  
  # --- Step 4: Compute neighbor stats vectorized ---------------------------
  cat("Computing neighbor statistics...\n")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    
    # Attach the neighbor's value to each edge
    edges_full[, neighbor_val := dt[[var_name]][neighbor_row]]
    
    # Compute grouped stats: max, min, mean per source_row
    # Exclude NAs in the variable
    stats <- edges_full[!is.na(neighbor_val), 
                         .(nb_max  = max(neighbor_val),
                           nb_min  = min(neighbor_val),
                           nb_mean = mean(neighbor_val)),
                         by = source_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed values back
    dt[stats$source_row, (max_col)  := stats$nb_max]
    dt[stats$source_row, (min_col)  := stats$nb_min]
    dt[stats$source_row, (mean_col) := stats$nb_mean]
    
    rm(stats)
  }
  
  # Clean up helper column
  edges_full[, neighbor_val := NULL]
  
  # --- Step 5: Return as data.frame, preserving original row order ---------
  dt[, .row_idx := NULL]
  
  cat("Done.\n")
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names match: nb_max_{var}, nb_min_{var}, nb_mean_{var}
# Numerical results are identical: same max/min/mean over same neighbor sets.
```

## Adapting Column Names to Match the Original

If your original code produced differently named columns (e.g., via `compute_and_add_neighbor_features`), adjust the naming to match exactly what the trained Random Forest expects:

```r
# If the original column names were different, rename to match the RF model.
# Example: if original names were "ntl_nb_max" instead of "nb_max_ntl":
# Inspect with: names(trained_rf_model$forest$xlevels) or similar

# Generic renamer (adjust pattern as needed):
rename_to_original <- function(dt, var_name, 
                                orig_max_fmt  = "%s_nb_max",
                                orig_min_fmt  = "%s_nb_min", 
                                orig_mean_fmt = "%s_nb_mean") {
  setnames(dt, 
           old = c(paste0("nb_max_", var_name), 
                   paste0("nb_min_", var_name), 
                   paste0("nb_mean_", var_name)),
           new = c(sprintf(orig_max_fmt, var_name), 
                   sprintf(orig_min_fmt, var_name), 
                   sprintf(orig_mean_fmt, var_name)),
           skip_absent = TRUE)
}
```

## Memory Considerations (16 GB Laptop)

The largest object is `edges_full`. Estimated size:

- ~6.46M rows Ã— ~8 neighbors = ~51.7M edge rows (after time expansion and filtering)
- Two integer columns (source_row, neighbor_row): ~51.7M Ã— 8 bytes Ã— 2 â‰ˆ **0.83 GB**
- Plus one temporary double column (neighbor_val): +0.41 GB
- Total peak for edges: **~1.3 GB**
- `dt` itself: ~6.46M Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB**
- Total peak: **~7â€“8 GB** â€” fits in 16 GB RAM.

If memory is tight, process variables one at a time and `gc()` between them (the code above already does this).

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup strategy** | String paste + named vector hash per row | Integer keyed join (data.table) |
| **Loop structure** | 6.46M R-level `lapply` iterations | Zero R-level row loops |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` with subsetting | 5 Ã— one `data.table` group-by |
| **String allocations** | ~50M+ paste operations | Zero |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Numerical result** | max/min/mean of neighbor values | Identical |
| **RF model** | Unchanged | Unchanged |
END_RESPONSE>>>
