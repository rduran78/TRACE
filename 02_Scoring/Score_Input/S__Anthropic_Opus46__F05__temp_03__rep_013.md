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
run_id: Anthropic_F05_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Specifically:

1. **`idx_lookup`** (the named vector of all 6.46M keys) is built once â€” that's fine.
2. But **inside the `lapply`**, for each of the 6.46M rows, the code:
   - Looks up the cell's reference index in `id_to_ref`
   - Extracts neighbor cell IDs from the `nb` object
   - Calls `paste()` to build string keys for each neighbor
   - Performs named-vector lookup (`idx_lookup[neighbor_keys]`)

With ~4 rook neighbors per cell on average, this is **~25.8 million `paste` + hash-lookup operations** wrapped in an R-level loop. R's `lapply` over 6.46M iterations with string allocation is extremely slow.

### The Broader Symptom

The entire approach is **row-centric** (process each cell-year one at a time) when the problem is naturally **vectorizable** (all neighbors of all cells in a given year can be resolved in one batch operation). The neighbor topology is **year-invariant** â€” the same spatial adjacency applies to every year. This means:

- The neighbor lookup can be built once as a **sparse adjacency structure over cell indices**, then replicated across years via simple integer arithmetic â€” no strings needed at all.
- `compute_neighbor_stats` already operates vectorially over the lookup result, but it's fed a list built by the slow row-wise method.
- The 5-variable loop (`compute_and_add_neighbor_features`) is fine structurally, but it depends on the slow `neighbor_lookup`.

**Estimated cost of current approach:**
- ~6.46M R-level loop iterations Ã— (string alloc + hash lookup) â‰ˆ 86+ hours as reported.

## Optimization Strategy

### Key Insight: Separate Spatial Topology from Temporal Indexing

Since the `nb` object is purely spatial and the panel is balanced (every cell appears in every year), we can:

1. **Build a sparse adjacency matrix once** from the `nb` object (344K Ã— 344K, ~1.37M non-zero entries).
2. **Map cell-year rows to a (cell_index, year_index) grid** using integer factoring â€” no strings.
3. **Compute neighbor statistics per variable using sparse matrix multiplication** or, more precisely, using vectorized grouped operations on the adjacency list â€” all in one shot per variable, not per row.

### Approach: Vectorized Neighbor Stats via `data.table` + Integer Adjacency

We convert the `nb` object to an edge list of `(from_cell_idx, to_cell_idx)`, join it with the data by cell index and year, and compute grouped `max/min/mean` â€” all vectorized.

**Expected speedup:** From 86+ hours to **minutes** (the bottleneck becomes a few `data.table` grouped joins over ~26M edge-year pairs Ã— 5 variables).

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build integer-indexed edge list from the nb object (one-time)
# ---------------------------------------------------------------
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] gives the indices (into id_order) of neighbors of cell i
  # We build a data.table of (from_cell_id, to_cell_id)
  from_idx <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb)
  
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(to_idx) & to_idx != 0L
  
  data.table(
    from_id = id_order[from_idx[valid]],
    to_id   = id_order[to_idx[valid]]
  )
}

# ---------------------------------------------------------------
# 2. Vectorized neighbor feature computation
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          neighbor_source_vars) {
  # Convert to data.table if not already (non-destructive copy)
  dt <- as.data.table(copy(cell_data))
  
  # Build edge list once
  cat("Building edge list...\n")
  edges <- build_edge_list(id_order, neighbors_nb)
  cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edges), big.mark = ",")))
  
  # For each variable, join edges with data to get neighbor values,

  # then compute grouped stats
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Subset: only the columns we need for the join
    # "from" side: each row in dt is a focal cell-year
    # "to" side: we look up the neighbor's value in the same year
    
    # Step A: Create a slim lookup of (id, year, value)
    val_dt <- dt[, .(id, year, val = get(var_name))]
    
    # Step B: For every (from_id, year) pair, find all to_id neighbors
    #         and look up their values in the same year.
    #
    # Start from edges, cross with years present in data.
    # But since the panel is balanced, every cell appears every year,
    # so we can do a direct join.
    
    # Join focal rows with edges to get neighbor IDs
    # focal row key: (from_id = id, year)
    # Then look up neighbor value by (to_id, year)
    
    # Efficient approach: 
    #   1. Merge edges with val_dt on to_id = id to get neighbor values per year
    #   2. Then group by (from_id, year) to get max/min/mean
    
    # Rename for clarity
    neighbor_vals <- merge(
      edges,
      val_dt,
      by.x = "to_id",
      by.y = "id",
      allow.cartesian = TRUE  # each edge Ã— each year
    )
    # neighbor_vals now has columns: to_id, from_id, year, val
    # where val is the neighbor (to_id) cell's value in that year
    
    # Remove NA values before aggregation
    neighbor_vals <- neighbor_vals[!is.na(val)]
    
    # Compute stats grouped by (from_id, year)
    stats <- neighbor_vals[,
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = .(from_id, year)
    ]
    
    # Rename columns to match original naming convention
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    setnames(stats, "from_id", "id")
    
    # Left join back to dt
    dt <- merge(dt, stats, by = c("id", "year"), all.x = TRUE)
    
    cat(sprintf("  Added: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  # Restore original row order
  # (merge may reorder; use the original data's id+year as key)
  setorderv(dt, c("id", "year"))
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ---------------------------------------------------------------
# 3. Usage (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The numerical output (max, min, mean of non-NA neighbor values) is identical.
```

## Memory Optimization (if 16 GB is tight)

The `merge` of edges (~1.37M) Ã— years (28) produces ~38.4M rows per variable. At ~3 numeric columns that's roughly 900 MB per variable â€” manageable but snug. If memory is a concern, process one variable at a time (as the loop above does) and remove intermediates:

```r
# Inside the loop, after merging stats back to dt:
rm(neighbor_vals, stats)
gc()
```

Alternatively, if even that is too large, chunk by year:

```r
compute_neighbor_features_chunked <- function(dt, edges, var_name) {
  val_dt <- dt[, .(id, year, val = get(var_name))]
  years <- sort(unique(dt$year))
  
  stats_list <- lapply(years, function(yr) {
    val_yr <- val_dt[year == yr, .(id, val)]
    # Join edges with single-year values
    nb <- merge(edges, val_yr, by.x = "to_id", by.y = "id")
    nb <- nb[!is.na(val)]
    nb[, .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
       by = .(from_id)][ , year := yr]
  })
  
  rbindlist(stats_list)
}
```

This processes ~1.37M edges Ã— 1 year at a time (~11 MB per chunk), keeping peak memory very low.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Paradigm** | Row-wise `lapply` over 6.46M rows | Vectorized `data.table` join + group-by |
| **Key construction** | `paste()` string keys per row | Integer ID joins (no strings in hot path) |
| **Neighbor resolution** | Named vector hash lookup per row | Single `merge` on integer keys |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Memory** | Low (but slow) | ~1â€“2 GB peak (chunked: ~50 MB) |
| **Numerical result** | max/min/mean of non-NA neighbor vals | Identical |
| **RF model** | Unchanged | Unchanged |
END_RESPONSE>>>
