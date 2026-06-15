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
run_id: Anthropic_F09_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### A. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With 6.46M rows, this produces millions of small character-allocation and hash-lookup operations in pure R. Named vector lookups via `idx_lookup[neighbor_keys]` are O(n) hash probes per call, but the overhead of constructing `neighbor_keys` strings 6.46M times dominates.

### B. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing `max/min/mean` is done via an R-level `lapply` over 6.46M ragged index vectors. Each iteration allocates a small vector, subsets, removes NAs, and computes three summary statistics. The per-iteration overhead of R's interpreter and garbage collector makes this extremely slow at scale.

### C. The neighbor topology is **time-invariant**, but the lookup is rebuilt as if it were time-varying
Rook contiguity is a property of the spatial grid, not of time. The current code mixes spatial topology with panel structure by embedding `year` into the lookup key, forcing 6.46M lookups instead of 344,208. This is the fundamental architectural mistake.

### Summary of bottlenecks

| Bottleneck | Scale | Root cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops | Mixing spatial topology with time |
| `compute_neighbor_stats` | 6.46M R-level iterations with subsetting | Row-level R loop instead of vectorized join |
| Memory for `neighbor_lookup` | 6.46M list elements, each a variable-length integer vector | Redundant: same neighbor set repeated 28Ã— per cell |

---

## 2. Optimization Strategy

### Core insight: **Separate topology from attributes, then use vectorized joins**

The neighbor graph has only **344,208 nodes** and ~1.37M directed edges. This is a small, static table. The yearly attributes (ntl, ec, pop_density, def, usd_est_n2) change, but the *topology does not*. Therefore:

1. **Build the edge list once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is done once and is year-independent.

2. **Join yearly attributes onto the edge list** â€” for each year, join the cell-year attributes onto the `neighbor_id` column. This is a keyed `data.table` equi-join, which is O(n) and vectorized in C.

3. **Aggregate by `(cell_id, year)`** â€” compute `max`, `min`, `mean` of neighbor values using `data.table`'s grouped aggregation (also C-level, vectorized).

4. **Join the aggregated neighbor stats back** onto the main `cell_data` table.

This eliminates all 6.46M R-level iterations and replaces them with a handful of vectorized `data.table` operations over ~1.37M Ã— 28 = ~38.4M edge-year rows (which `data.table` handles in seconds).

### Expected speedup

| Step | Current | Proposed |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | ~seconds (expand 1.37M edges Ã— 28 years via join) |
| Compute stats per variable | ~hours (6.46M R iterations) | ~seconds (grouped `data.table` aggregation) |
| Total for 5 variables | ~86+ hours | **~1â€“3 minutes** |

### Constraints preserved
- The trained Random Forest model is **not retouched**.
- The original numerical estimand (neighbor max, min, mean for each variable, per cell-year) is **exactly preserved**.
- Memory: the edge-year table is ~38.4M rows Ã— 3 columns â‰ˆ ~900 MB, which fits in 16 GB RAM. We process one variable at a time to keep peak memory low.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table
# Inputs:
#   cell_data             â€” data.frame/data.table with columns: id, year, and
#                           the 5 neighbor source variables
#   id_order              â€” character/integer vector of cell IDs in the order
#                           matching rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
# Output:
#   cell_data with 15 new columns: {var}_neighbor_max, _min, _mean Ã— 5 vars
# =============================================================================

library(data.table)

# --------------------------------------------------------------------------
# Step 1: Build the static spatial edge list ONCE (topology only, no year)
# --------------------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  # We expand it into a two-column data.table: (cell_id, neighbor_id)
  n <- length(id_order)
  
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) next
    # Remove the spdep "no neighbors" sentinel (0L) if present
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) next
    len <- length(nb_idx)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
    pos <- pos + len
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  edge_dt <- data.table(cell_id = from_id[1:(pos - 1L)],
                        neighbor_id = to_id[1:(pos - 1L)])
  return(edge_dt)
}

# --------------------------------------------------------------------------
# Step 2: For one variable, compute neighbor max/min/mean via joins
# --------------------------------------------------------------------------
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must be a data.table with columns: id, year, and var_name
  # edge_dt is the static edge list: (cell_id, neighbor_id)
  
  # Create a slim attribute table: (id, year, value)
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  
  # Join: for every edge Ã— year, get the neighbor's attribute value.
  # We do this by joining attr_dt onto edge_dt by neighbor_id == id.
  # First, create edge-year combinations by joining cell years onto edges.
  
  # Get the unique years
  years <- sort(unique(attr_dt$year))
  
  # Cross join edges with years: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows
  # But we can be smarter: join through the cell_dt itself.
  
  # For each (cell_id, year) in cell_dt, we need the neighbor values.
  # Approach: join edge_dt onto attr_dt by (neighbor_id = id), broadcasting year.
  
  # Rename for clarity in the join
  setkey(attr_dt, id, year)
  
  # Expand edges by year: for each edge (cell_id, neighbor_id), 
  # look up neighbor's value for every year the cell_id appears.
  # 
  # Efficient approach: 
  #   1. Join cell_dt's (id, year) with edge_dt on cell_id to get 
  #      (cell_id, year, neighbor_id) â€” this is the "request" table.
  #   2. Join the request table with attr_dt on (neighbor_id = id, year) 
  #      to get neighbor values.
  #   3. Aggregate by (cell_id, year).
  
  # Step 2a: Build request table
  # cell_years: unique (id, year) pairs from cell_dt
  cell_years <- cell_dt[, .(cell_id = id, year)]
  
  # Join with edge_dt on cell_id
  setkey(cell_years, cell_id)
  setkey(edge_dt, cell_id)
  
  # This is an equi-join: for each cell-year, expand to all its neighbors
  request <- edge_dt[cell_years, on = .(cell_id), allow.cartesian = TRUE, nomatch = 0L]
  # request now has columns: cell_id, neighbor_id, year
  # Rows: ~38.4M (1.37M edges Ã— 28 years)
  
  # Step 2b: Look up neighbor values
  setkey(request, neighbor_id, year)
  setkey(attr_dt, id, year)
  
  request[attr_dt, neighbor_value := i.value, on = .(neighbor_id = id, year)]
  
  # Step 2c: Aggregate by (cell_id, year), dropping NAs
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  agg <- request[!is.na(neighbor_value),
                 .(nb_max  = max(neighbor_value),
                   nb_min  = min(neighbor_value),
                   nb_mean = mean(neighbor_value)),
                 by = .(cell_id, year)]
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(col_max,  col_min,  col_mean))
  
  return(agg)
}

# --------------------------------------------------------------------------
# Step 3: Main pipeline
# --------------------------------------------------------------------------

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build the static edge list ONCE
message("Building static edge list from rook neighbor topology...")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
setkey(edge_dt, cell_id)
message(sprintf("  Edge list: %s directed edges across %s cells.",
                format(nrow(edge_dt), big.mark = ","),
                format(length(id_order), big.mark = ",")))

# Compute neighbor features for each variable and merge back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for '%s'...", var_name))
  t0 <- proc.time()
  
  agg <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  
  # Merge back onto cell_data
  # Use id and year as the join key
  cell_data <- merge(cell_data, agg,
                     by.x = c("id", "year"),
                     by.y = c("cell_id", "year"),
                     all.x = TRUE)
  
  elapsed <- (proc.time() - t0)["elapsed"]
  message(sprintf("  Done in %.1f seconds.", elapsed))
}

# --------------------------------------------------------------------------
# Step 4: Predict with the existing trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# The trained RF model object (e.g., `rf_model`) is used as-is.
# The new neighbor feature columns have identical names and semantics
# to those produced by the original pipeline, so prediction is unchanged.
#
# Example (uncomment and adapt to your actual model object name):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)

message("Neighbor feature computation complete. Ready for RF prediction.")
```

---

## 4. Why This Is Correct (Numerical Equivalence)

| Original operation | Optimized equivalent |
|---|---|
| For row `i`, find all rows sharing `(neighbor_cell_id, same year)` | `edge_dt[cell_years, on = cell_id]` then `request[attr_dt, on = .(neighbor_id=id, year)]` â€” same logical join |
| `max(neighbor_vals)` after removing NAs | `request[!is.na(neighbor_value), max(neighbor_value), by = .(cell_id, year)]` â€” identical |
| `min(neighbor_vals)` after removing NAs | Same grouped aggregation â€” identical |
| `mean(neighbor_vals)` after removing NAs | Same grouped aggregation â€” identical (R's `mean` on the same non-NA values) |
| Cells with no valid neighbors â†’ `NA` | `all.x = TRUE` in the final merge leaves unmatched cell-years as `NA` â€” identical |

The column names, their semantics, and their numerical values are preserved exactly, so the trained Random Forest model produces identical predictions without retraining.

---

## 5. Performance Summary

| Metric | Original | Optimized |
|---|---|---|
| R-level loop iterations | ~32.3M (6.46M Ã— 5 vars) | **0** |
| String allocations in lookup | ~6.46M `paste()` calls | **0** |
| Core operation | R `lapply` + named vector lookup | C-level `data.table` keyed join + grouped aggregation |
| Estimated wall time | **86+ hours** | **1â€“3 minutes** |
| Peak RAM (approx.) | ~8â€“10 GB (6.46M-element list) | ~4â€“6 GB (38.4M-row intermediate table, one var at a time) |
| RF model | Unchanged | Unchanged |
END_RESPONSE>>>
