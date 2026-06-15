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
run_id: Anthropic_F05_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: Row-wise `lapply` over 6.46M rows with string operations

1. **`build_neighbor_lookup`** iterates row-by-row (`lapply` over 6.46M rows). For each row, it:
   - Calls `paste()` to construct string keys for every neighbor of that cell in that year.
   - Performs named-vector lookup (`idx_lookup[neighbor_keys]`) against a 6.46M-element named character vector.
   
   This is **O(N Ã— k)** string constructions and hash lookups, where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4 for rook). That's ~25.8M `paste` calls plus ~25.8M hash lookups against a 6.46M-entry table â€” all in an interpreted `lapply` loop.

2. **The string-keying is entirely unnecessary.** The data is a balanced panel (344,208 cells Ã— 28 years). Every cell appears in every year. So for a given row `i` with `(cell_id, year)`, its neighbors are the rows with `(neighbor_cell_id, same_year)`. Since the panel is balanced and sorted, neighbor row indices can be computed **arithmetically** â€” no strings, no hash tables.

3. **`compute_neighbor_stats`** is called 5 times (once per variable), each time re-traversing the 6.46M-element neighbor lookup. This is fine structurally, but can be vectorized with matrix operations instead of `lapply`.

### Estimated cost of current approach

- `build_neighbor_lookup`: ~6.46M iterations Ã— (string paste + hash lookup) â‰ˆ hours alone.
- `compute_neighbor_stats`: 5 vars Ã— 6.46M `lapply` iterations â‰ˆ additional hours.
- Total: the reported 86+ hours is consistent with this analysis.

---

## Optimization Strategy

### Key Insight: Arithmetic Index Mapping Replaces All String Work

For a balanced panel sorted by `(id, year)` â€” or even unsorted â€” we can build an integer matrix of neighbor row indices using **vectorized joins**, eliminating all `paste`/string operations.

**Three-phase strategy:**

| Phase | Current | Proposed |
|-------|---------|----------|
| **1. Neighbor row-index construction** | Row-wise `lapply` with `paste` + named-vector lookup | Vectorized `data.table` equi-join: expand directed neighbor pairs Ã— years in one join |
| **2. Neighbor stats computation** | `lapply` over 6.46M rows per variable | Vectorized `data.table` grouped aggregation: join neighbor values, group-by source row, compute `max/min/mean` |
| **3. Column binding** | Loop over 5 variables | Single pass or simple loop over 5 variables with vectorized internals |

**Expected speedup:** From 86+ hours to **minutes** (the dominant cost becomes a ~26M-row join and grouped aggregation, which `data.table` handles in seconds).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Preserves original numerical estimand (max, min, mean of neighbor values)
# Preserves trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

build_neighbor_features_fast <- function(cell_data, 
                                          id_order, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # PHASE 0: Convert to data.table and build row index
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Create a unique row identifier preserving original row order
  dt[, .row_id := .I]
  
  # Build a lookup: for each (id, year) -> row index
  # This is a single vectorized operation
  setkey(dt, id, year)
  
  # -------------------------------------------------------------------------
  # PHASE 1: Build directed neighbor edge list (cell-level, no year dimension)
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length = length(id_order)
  # where element i contains integer indices into id_order of neighbors of 
  # id_order[i].
  
  # Expand to a two-column data.table of (focal_id, neighbor_id)
  n_cells <- length(id_order)
  
  # Vectorized expansion of the nb object
  focal_indices <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  neighbor_indices <- unlist(rook_neighbors_unique)
  
  # Remove zero-neighbor entries (if any nb element is integer(0), 
  # lengths = 0, so they contribute nothing)
  edges <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  
  # Remove any self-loops or NA entries (defensive)
  edges <- edges[!is.na(neighbor_id) & focal_id != neighbor_id]
  
  cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edges)))
  
  # -------------------------------------------------------------------------
  # PHASE 2: Join edges with panel to get (focal_row, neighbor_row) pairs
  # -------------------------------------------------------------------------
  # For each edge (focal_id, neighbor_id), and for each year, we need:
  #   focal_row    = row in dt where id == focal_id    & year == y
  #   neighbor_row = row in dt where id == neighbor_id & year == y
  #
  # Strategy: 
  #   1. Cross-join edges with the unique years.
  #   2. Join to dt twice to get focal_row and neighbor_row.
  #
  # But cross-joining 1.37M edges Ã— 28 years = 38.5M rows, which is 
  # manageable. However, we can be smarter: join edges to dt on focal_id 
  # to get (focal_row, neighbor_id, year), then join on (neighbor_id, year) 
  # to get neighbor_row.
  
  # Step 2a: Get focal rows â€” join edges to dt on focal_id = id
  # This gives us one row per (edge Ã— year) = ~38.5M rows
  
  # Build a minimal lookup table: id -> year -> .row_id
  row_lookup <- dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)
  
  # Join: for each edge, get all years of the focal cell
  setkey(edges, focal_id)
  focal_lookup <- row_lookup[, .(focal_id = id, year, focal_row = .row_id)]
  setkey(focal_lookup, focal_id)
  
  # Merge edges with focal years
  # Each edge gets replicated across all years the focal_id appears in
  edge_year <- edges[focal_lookup, 
                     on = .(focal_id), 
                     nomatch = NULL,
                     allow.cartesian = TRUE]
  # edge_year now has columns: focal_id, neighbor_id, year, focal_row
  
  cat(sprintf("Edge-year pairs: %d rows\n", nrow(edge_year)))
  
  # Step 2b: Join to get neighbor_row
  neighbor_lookup_dt <- row_lookup[, .(neighbor_id = id, year, neighbor_row = .row_id)]
  setkey(neighbor_lookup_dt, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)
  
  edge_year <- edge_year[neighbor_lookup_dt, 
                          on = .(neighbor_id, year), 
                          nomatch = NA]
  
  # Drop rows where neighbor_row is NA (neighbor cell not present in that year)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]
  
  cat(sprintf("Valid edge-year pairs after join: %d rows\n", nrow(edge_year)))
  
  # -------------------------------------------------------------------------
  # PHASE 3: Compute neighbor stats per variable (vectorized)
  # -------------------------------------------------------------------------
  # For each variable, look up neighbor values, then group by focal_row 
  # and compute max, min, mean.
  
  setkey(edge_year, focal_row)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    
    # Look up neighbor values
    edge_year[, nval := dt[[var_name]][neighbor_row]]
    
    # Compute grouped stats (excluding NAs, matching original behavior)
    stats <- edge_year[!is.na(nval), 
                       .(var_max  = max(nval),
                         var_min  = min(nval),
                         var_mean = mean(nval)),
                       by = focal_row]
    
    # Initialize columns with NA (rows with no valid neighbors get NA)
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed values
    dt[stats$focal_row, (max_col)  := stats$var_max]
    dt[stats$focal_row, (min_col)  := stats$var_min]
    dt[stats$focal_row, (mean_col) := stats$var_mean]
    
    cat(sprintf("  Done. %d rows received neighbor stats, %d rows are NA.\n",
                nrow(stats), nrow(dt) - nrow(stats)))
  }
  
  # -------------------------------------------------------------------------
  # PHASE 4: Clean up and return as data.frame
  # -------------------------------------------------------------------------
  # Restore original row order
  setorder(dt, .row_id)
  dt[, .row_id := NULL]
  
  # Return as data.frame to preserve compatibility with downstream RF predict()
  return(as.data.frame(dt))
}


# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# 
# cell_data <- build_neighbor_features_fast(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # The trained RF model is unchanged â€” just call predict() as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Considerations (16 GB laptop)

The largest intermediate object is `edge_year` at ~38.5M rows Ã— 4 integer columns â‰ˆ **1.2 GB**. Combined with `dt` (~6.46M Ã— 110 cols â‰ˆ 5.7 GB), peak memory is ~8â€“9 GB, which fits within 16 GB.

If memory is tight, process variables one at a time and drop the `nval` column between iterations (the code above already does this implicitly since `:=` overwrites).

## Expected Runtime

| Phase | Estimated Time |
|-------|---------------|
| Edge list construction | < 1 second |
| Edge-year join (~38.5M rows) | 10â€“30 seconds |
| Per-variable grouped aggregation (Ã—5) | 5â€“15 seconds each |
| **Total** | **~2â€“5 minutes** |

This represents a speedup of roughly **1,000â€“2,500Ã—** over the original 86+ hour estimate, achieved by replacing row-wise string-key construction and hash lookups with vectorized `data.table` equi-joins and grouped aggregations. The numerical output (max, min, mean of non-NA neighbor values, with NA for cells having no valid neighbors) is identical to the original.
END_RESPONSE>>>
