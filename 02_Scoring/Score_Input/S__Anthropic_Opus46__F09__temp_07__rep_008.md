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
run_id: Anthropic_F09_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### A. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every one of the ~6.46M cell-year rows, the function:
1. Looks up the cell's spatial reference index.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes together character keys (`"cellid_year"`).
4. Matches those keys against a named character vector of length 6.46M.

Character-key matching via named vectors is O(n) per lookup in the worst case. Doing this 6.46M times produces roughly **O(nÂ²)** behavior. This single step likely accounts for the majority of the 86-hour runtime.

### B. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Even after the lookup is built, `compute_neighbor_stats` iterates row-by-row in R, extracting neighbor values, removing NAs, and computing `max/min/mean`. This is called 5 times (once per source variable), so ~32.3M R-level iterations with small-vector operations.

### C. Redundant recomputation of spatial topology per cell-year
The neighbor relationships are **purely spatial** â€” they do not change across years. Yet the current code embeds the year into the lookup, effectively rebuilding the spatial graph 28 times (once per year) inside a single flat structure. This is wasteful.

---

## 2. Optimization Strategy

### Core Insight: Separate Spatial Topology from Temporal Attributes

The neighbor graph is **time-invariant**. Build it once as a **cell-to-cell adjacency table** (a two-column data.table of `id â†’ neighbor_id`), then for each year, join the yearly attribute values onto this table and compute grouped aggregates. This converts the problem from 6.46M row-level R loops into a small number of **vectorized data.table joins and grouped aggregations**.

### Specific Steps

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `nb` object â†’ two-column `data.table(id, neighbor_id)` | One-time, ~1.37M rows |
| 2 | For each year, subset `cell_data` to that year's attributes | 28 iterations |
| 3 | Join attributes onto the edge table by `neighbor_id` | Vectorized, keyed join |
| 4 | Group by `id`, compute `max`, `min`, `mean` per variable | Vectorized aggregation |
| 5 | Join results back onto `cell_data` | Vectorized |

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes the data.table joins and aggregations over ~1.37M Ã— 28 = ~38.5M edge-year rows, which data.table handles in seconds per variable).

**Memory:** The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. Yearly attribute joins peak at ~38.5M rows Ã— a few columns â‰ˆ hundreds of MB. Well within 16 GB.

**Preserves:** The trained Random Forest model is untouched. The numerical outputs (neighbor max, min, mean) are identical because the same neighbor relationships and the same aggregation functions are used.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the time-invariant cell-to-cell adjacency table
#         (run once; can be serialized to disk for reuse)
# ==============================================================

build_adjacency_table <- function(id_order, nb_object) {
  # id_order: vector of cell IDs in the same order as the nb object
  # nb_object: spdep::nb list (rook_neighbors_unique)
  #
  # Returns a data.table with columns: id, neighbor_id
  # Each row is one directed neighbor relationship.
  
  n <- length(nb_object)
  
  # Pre-count total edges to pre-allocate
  edge_counts <- vapply(nb_object, length, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id     <- integer(total_edges)
  to_id       <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- nb_object[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    k <- length(nb_idx)
    if (k > 0L) {
      from_id[pos:(pos + k - 1L)] <- id_order[i]
      to_id[pos:(pos + k - 1L)]   <- id_order[nb_idx]
      pos <- pos + k
    }
  }
  
  # Trim in case some nb entries were 0-length
  adj <- data.table(id = from_id[1:(pos - 1L)],
                    neighbor_id = to_id[1:(pos - 1L)])
  return(adj)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)

# Optional: save for future reuse
# fwrite(adj_table, "adjacency_table.csv")
# or: saveRDS(adj_table, "adjacency_table.rds")


# ==============================================================
# STEP 2: Compute neighbor stats via vectorized joins
# ==============================================================

compute_all_neighbor_features <- function(cell_data, adj_table, source_vars) {
  # cell_data:   data.frame or data.table with columns: id, year, and all source_vars
  # adj_table:   data.table with columns: id, neighbor_id
  # source_vars: character vector of variable names for which to compute neighbor stats
  #
  # Returns cell_data (data.table) with new columns:

  #   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
  #   for each var in source_vars.
  
  dt <- as.data.table(cell_data)
  adj <- copy(adj_table)  # avoid modifying the original
  
  # Ensure key columns are of consistent type
  dt[, id := as.integer(id)]
  dt[, year := as.integer(year)]
  adj[, id := as.integer(id)]
  adj[, neighbor_id := as.integer(neighbor_id)]
  
  # Key the main data for fast joins
  setkey(dt, id, year)
  
  # For each source variable, compute neighbor max, min, mean
  for (var in source_vars) {
    message("Computing neighbor stats for: ", var)
    
    # Extract only the columns we need for the join (id, year, value)
    # This keeps memory usage minimal.
    attr_cols <- dt[, .(id, year, value = get(var))]
    setnames(attr_cols, "id", "neighbor_id")
    setkey(attr_cols, neighbor_id)
    
    # Expand adjacency table by year:
    # For each year, every edge id->neighbor_id gets the neighbor's attribute value.
    # We do this by joining adj_table with the attribute table on neighbor_id,
    # but we also need to match on year. Strategy:
    #   1. Cross-join adj_table with unique years? No â€” too large and wasteful.
    #   2. Better: join dt's (id, year) with adj_table on id, then join neighbor
    #      attributes on (neighbor_id, year).
    
    # Get the (id, year) pairs that exist in the data
    id_year <- dt[, .(id, year)]
    setkey(id_year, id)
    
    # For each (id, year), attach all neighbors
    # This creates ~1.37M * 28 â‰ˆ 38.5M rows if every cell appears every year
    edges_by_year <- adj[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
    # edges_by_year has columns: id, neighbor_id, year
    
    # Now join the neighbor's attribute value for that year
    setkey(edges_by_year, neighbor_id, year)
    setkey(attr_cols, neighbor_id, year)  # re-key with year
    
    # Perform the join: attach neighbor's value
    edges_by_year[attr_cols, value := i.value, on = .(neighbor_id, year)]
    
    # Compute grouped stats: for each (id, year), aggregate over neighbors
    stats <- edges_by_year[!is.na(value),
                           .(nb_max  = max(value),
                             nb_min  = min(value),
                             nb_mean = mean(value)),
                           by = .(id, year)]
    
    # Name the output columns
    max_col  <- paste0(var, "_neighbor_max")
    min_col  <- paste0(var, "_neighbor_min")
    mean_col <- paste0(var, "_neighbor_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    
    # Remove any pre-existing columns of the same name in dt (for idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(dt)) dt[, (col) := NULL]
    }
    
    # Join stats back onto the main data
    setkey(stats, id, year)
    dt[stats, (c(max_col, min_col, mean_col)) :=
         mget(paste0("i.", c(max_col, min_col, mean_col))),
       on = .(id, year)]
    
    # Clean up to free memory
    rm(attr_cols, edges_by_year, stats)
    gc()
  }
  
  return(dt)
}


# ==============================================================
# STEP 3: Run the pipeline
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data   = cell_data,
  adj_table   = adj_table,
  source_vars = neighbor_source_vars
)

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
#          (model object is unchanged)
# ==============================================================

# cell_data now contains the same neighbor feature columns as before:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ... etc.
#
# These are numerically identical to the original implementation's output
# because the same neighbor relationships and aggregation functions are used.

# Example prediction (adjust to your actual model object and column setup):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Summary of Why This Works

| Aspect | Old Approach | New Approach |
|--------|-------------|--------------|
| **Neighbor lookup** | 6.46M character-key matches via named vector | One-time construction of ~1.37M-row integer edge table |
| **Per-variable computation** | `lapply` over 6.46M rows in R | Vectorized `data.table` join + `group by` aggregation |
| **Year handling** | Year baked into character keys, re-matched per row | Natural join on `(neighbor_id, year)` â€” no string operations |
| **Total R-level iterations** | ~32.3M (6.46M Ã— 5 vars) | 5 (one per variable; inner work is C-level data.table) |
| **Expected runtime** | ~86+ hours | **~5â€“15 minutes** on a 16 GB laptop |
| **Numerical result** | max, min, mean of rook neighbors | Identical max, min, mean of same rook neighbors |
| **RF model** | â€” | **Untouched; no retraining** |
END_RESPONSE>>>
