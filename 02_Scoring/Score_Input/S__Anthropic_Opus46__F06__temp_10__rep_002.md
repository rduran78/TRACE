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
run_id: Anthropic_F06_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`
This function iterates over **all ~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs paste-key strings (`"id_year"`) for each neighbor-year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

The string pasting, named-vector indexing, and per-row `lapply` over 6.46M rows is extremely slow. Named vector lookup in R is hash-based but still carries overhead per call, and doing it ~6.46M Ã— ~4 neighbors â‰ˆ 25M+ times with string construction is brutal.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated R-level loops
For each of the 5 variables, `compute_neighbor_stats` iterates over all 6.46M rows again in an `lapply`, subsetting a numeric vector by index, removing NAs, and computing `max/min/mean`. That's 5 Ã— 6.46M = ~32.3M R-level function calls. The per-element overhead of `lapply` with anonymous functions dominates.

### Why raster focal/kernel operations are not appropriate
Focal operations assume a regular complete grid with uniform rectangular neighborhoods. This panel has:
- Potentially irregular boundaries (not all cells have 4 rook neighbors).
- A temporal dimension (neighbors must match within the same year).
- The neighbor structure is precomputed as an `spdep::nb` object, not a kernel.

Focal operations would require restructuring into 3D arrays per year, handling boundaries, and risk introducing subtle numerical differences. The better approach is to **vectorize the existing graph-based computation using `data.table` joins**.

---

## Optimization Strategy

### Strategy: Replace both functions with a single vectorized `data.table` join-and-aggregate

1. **Build an edge list once** from `rook_neighbors_unique` â€” a two-column data.table of `(from_id, to_id)` with ~1.37M directed edges.
2. **Cross-join with years** â€” expand the edge list to `(from_id, year, to_id)` giving ~1.37M Ã— 28 â‰ˆ ~38.5M rows (but only edges, not NÂ²).
3. **Join neighbor values** â€” for each `(to_id, year)`, look up the variable value from the main data. This is an equi-join, which `data.table` does in near-O(n) time with keys.
4. **Aggregate** â€” group by `(from_id, year)` and compute `max`, `min`, `mean` in one pass.
5. **Join back** to the main data.

This replaces ~86 hours of R-level loops with a handful of vectorized `data.table` operations that should complete in **minutes**.

### Complexity comparison
| Step | Current | Proposed |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` calls with string ops | One-time edge list build (~1.37M rows) |
| Stats per variable | 6.46M `lapply` calls Ã— 5 vars | Keyed join + `groupby` aggregate Ã— 5 vars |
| Estimated time | 86+ hours | ~5â€“15 minutes |

### Numerical equivalence
The `max`, `min`, and `mean` computations are identical â€” same neighbor sets, same values, same aggregation functions. The Random Forest model sees identical input features and is never retrained.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build directed edge list from spdep::nb object (one-time, fast)
# ==============================================================================
# rook_neighbors_unique: list of length = number of cells (344,208)
#   each element is an integer vector of neighbor indices into id_order
# id_order: vector of cell IDs of length 344,208

build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (indices into id_order)
  # An nb element of 0 (integer(0)) means no neighbors.
  n <- length(neighbors_nb)
  
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(neighbors_nb, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb uses 0 to indicate no neighbors in some representations
    nb_idx <- nb_idx[nb_idx > 0L]
    len <- length(nb_idx)
    if (len > 0L) {
      from_id[pos:(pos + len - 1L)] <- id_order[i]
      to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
      pos <- pos + len
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list built:", nrow(edges), "directed edges\n")

# ==============================================================================
# STEP 2: Convert main data to data.table and set keys
# ==============================================================================
dt <- as.data.table(cell_data)

# Ensure column names for join keys
# 'id' = cell identifier, 'year' = panel year
setkey(dt, id, year)

# ==============================================================================
# STEP 3: Expand edge list across all years (vectorized cross-join)
# ==============================================================================
years <- sort(unique(dt$year))  # 1992:2019, 28 values

# Cross join edges with years: ~1.37M * 28 â‰ˆ 38.5M rows
# This is the full set of (focal_cell, year, neighbor_cell) triples
edges_by_year <- CJ_dt <- edges[, .(from_id, to_id, year = rep(list(years), .N)), 
                                  by = .I][, .(from_id, to_id, year = unlist(year))]

# More memory-efficient approach:
edges_by_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
edges_by_year[, `:=`(from_id = edges$from_id[edge_idx],
                      to_id   = edges$to_id[edge_idx])]
edges_by_year[, edge_idx := NULL]

cat("Edge-year table:", nrow(edges_by_year), "rows\n")

# Key for joining neighbor values
setkey(edges_by_year, to_id, year)

# ==============================================================================
# STEP 4: For each variable, join neighbor values, aggregate, merge back
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Extract only the columns needed for the join (minimal memory)
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join: for each edge-year row, attach the neighbor's value
  # edges_by_year$to_id matches val_dt$id, edges_by_year$year matches val_dt$year
  merged <- val_dt[edges_by_year, on = .(id = to_id, year = year), nomatch = NA,
                   .(from_id, year, val = x.val)]
  
  # Remove NA values (matches original: neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)])
  merged <- merged[!is.na(val)]
  
  # Aggregate: max, min, mean grouped by (from_id, year)
  agg <- merged[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = .(from_id, year)]
  
  # Rename columns to match expected output pattern
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Merge back into main data.table
  # Left join: all original rows preserved; cells with no valid neighbors get NA
  dt <- agg[dt, on = .(from_id = id, year = year)]
  
  # The join puts 'from_id' as the key column; rename back to 'id'
  setnames(dt, "from_id", "id")
  setkey(dt, id, year)
  
  cat("  Done:", max_col, min_col, mean_col, "\n")
}

# ==============================================================================
# STEP 5: Convert back to data.frame if needed for the RF predict step
# ==============================================================================
cell_data <- as.data.frame(dt)

cat("All neighbor features computed. Ready for prediction.\n")

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained model object is used as-is. Example:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Optimized Alternative for Step 3

If the ~38.5M-row `edges_by_year` table strains 16 GB RAM, process year-by-year:

```r
for (var_name in neighbor_source_vars) {
  cat("Processing:", var_name, "\n")
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Initialize columns with NA
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  for (yr in years) {
    # Subset current year's data
    yr_dt <- dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_dt, id)
    
    # Join edges with neighbor values for this year
    yr_edges <- yr_dt[edges, on = .(id = to_id), nomatch = NA,
                      .(from_id, val = x.val)]
    yr_edges <- yr_edges[!is.na(val)]
    
    # Aggregate
    yr_agg <- yr_edges[, .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                        keyby = .(from_id)]
    
    # Update in place
    dt[yr_agg, on = .(id = from_id, year = yr),
       `:=`(
         (max_col)  = i.nb_max,
         (min_col)  = i.nb_min,
         (mean_col) = i.nb_mean
       ), 
       # Need exact year match
       which = TRUE]
    
    # Simpler update approach:
    idx <- dt[year == yr, which = TRUE]
    match_idx <- match(dt$id[idx], yr_agg$from_id)
    set(dt, i = idx[!is.na(match_idx)], j = max_col,  value = yr_agg$nb_max[match_idx[!is.na(match_idx)]])
    set(dt, i = idx[!is.na(match_idx)], j = min_col,  value = yr_agg$nb_min[match_idx[!is.na(match_idx)]])
    set(dt, i = idx[!is.na(match_idx)], j = mean_col, value = yr_agg$nb_mean[match_idx[!is.na(match_idx)]])
  }
  
  cat("  Done:", var_name, "\n")
}

cell_data <- as.data.frame(dt)
```

---

## Cleaner Year-by-Year Version (Recommended for 16 GB Laptop)

```r
library(data.table)

# --- Build edge list (once) ---
build_edge_list <- function(id_order, neighbors_nb) {
  from <- rep(seq_along(neighbors_nb), 
              lengths(neighbors_nb))
  to   <- unlist(neighbors_nb)
  valid <- to > 0L
  data.table(from_id = id_order[from[valid]], 
             to_id   = id_order[to[valid]])
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
setkey(edges, to_id)

# --- Convert to data.table ---
dt <- as.data.table(cell_data)
setkey(dt, id, year)
years <- sort(unique(dt$year))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  dt[, c(max_col, min_col, mean_col) := NA_real_]
  
  for (yr in years) {
    # Lookup table: cell -> value for this year
    vals <- dt[year == yr, .(id, val = get(var_name))]
    setkey(vals, id)
    
    # Attach neighbor values via join on to_id
    nb <- vals[edges, on = .(id = to_id), nomatch = NA,
               .(from_id, val = x.val)]
    nb <- nb[!is.na(val)]
    
    # Aggregate per focal cell
    agg <- nb[, .(vmax = max(val), vmin = min(val), vmean = mean(val)),
              keyby = .(from_id)]
    
    # Write results back into dt for this year
    dt[agg, on = .(id = from_id), 
       `:=`(
         (max_col)  = fifelse(year == yr, i.vmax, get(max_col)),
         (min_col)  = fifelse(year == yr, i.vmin, get(min_col)),
         (mean_col) = fifelse(year == yr, i.vmean, get(mean_col))
       )]
  }
  cat("Done:", var_name, "\n")
}

cell_data <- as.data.frame(dt)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` with string paste + named vector lookup | One vectorized edge-list build (~1.37M rows) |
| **Stats computation** | 6.46M `lapply` Ã— 5 vars = 32.3M R function calls | `data.table` keyed join + grouped aggregation |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Peak RAM** | Moderate (but slow) | ~4â€“6 GB (year-by-year variant stays under 8 GB) |
| **Numerical results** | Baseline | Identical (`max`, `min`, `mean` on same neighbor sets) |
| **RF model** | Preserved | Preserved (no retraining) |
END_RESPONSE>>>
