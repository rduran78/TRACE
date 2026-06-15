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
run_id: Anthropic_F09_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups in R via `[` on character keys are O(n) in the worst case per call due to hashing overhead at scale.

### 2. Redundant recomputation of the same spatial topology per year
The rook-neighbor structure is **purely spatial**â€”it is identical across all 28 years. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, effectively repeating the same spatial adjacency resolution 28 times (once per year embedded in each row). This inflates the work by a factor of 28.

### 3. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46M list elements, each containing a small numeric vector, is dominated by R-level loop overhead and repeated memory allocations.

**Summary:** The core inefficiency is treating a **spatial** problem as a **cell-year** problem, and using R-level loops instead of vectorized/join-based operations.

---

## Optimization Strategy

The key insight: **build the neighbor table once at the cell level (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M rows), then join yearly attributes onto it.**

### Step-by-step plan:

1. **Build a static edge table once:** Convert `rook_neighbors_unique` (the `nb` object) into a two-column `data.table` of `(cell_id, neighbor_id)` â€” ~1.37M rows. This is year-independent.

2. **For each year, join cell attributes onto the edge table:** For a given variable (e.g., `ntl`), join the variable's value for each `neighbor_id` in that year onto the edge table. This is a keyed `data.table` join â€” extremely fast.

3. **Aggregate by `(cell_id, year)`:** Compute `max`, `min`, `mean` of the neighbor values in one grouped aggregation.

4. **Join the aggregated stats back onto the main dataset.**

5. **Repeat for each of the 5 neighbor source variables.**

**Expected speedup:** From ~86 hours to **minutes**. The edge table has ~1.37M rows Ã— 28 years = ~38.4M join rows, but `data.table` keyed joins and grouped aggregations handle this trivially. No R-level row loops at all.

**Memory:** The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The yearly-expanded version is ~38.4M rows Ã— 3 columns â‰ˆ 920 MB. Well within 16 GB.

**Preserves:** The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per cell-year per variable) is identical because the same neighbor relationships and the same aggregation functions are used.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table if not already
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build the static spatial edge table ONCE
#
# rook_neighbors_unique is an nb object (list of integer vectors)
# id_order is the vector of cell IDs corresponding to each
# element of the nb object (i.e., id_order[i] is the cell ID
# for the i-th element of rook_neighbors_unique).
# ============================================================
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) > 0L) {
      n <- length(nb_idx)
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
      pos <- pos + n
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_table)))

# ============================================================
# STEP 2: Function to compute neighbor stats for one variable
#          using join + grouped aggregation
# ============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Extract only the columns we need for the join:
  # neighbor_id + year -> variable value
  lookup_cols <- c("id", "year", var_name)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, old = c("id", var_name), new = c("neighbor_id", "nb_val"))
  setkey(lookup, neighbor_id, year)
  
  # Expand edge table by year (cross join edges Ã— years)
  # More memory-efficient: join edges onto the main data's (id, year) pairs,
  # then look up neighbor values.
  
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross join: edge_table Ã— years
  # This gives us every (cell_id, neighbor_id, year) combination
  edges_by_year <- edge_dt[, .(year = years), by = .(cell_id, neighbor_id)]
  setkey(edges_by_year, neighbor_id, year)
  
  # Join neighbor values onto the expanded edge table
  edges_by_year[lookup, nb_val := i.nb_val, on = .(neighbor_id, year)]
  
  # Aggregate: for each (cell_id, year), compute max, min, mean
  # of neighbor values (excluding NAs)
  agg <- edges_by_year[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    by = .(cell_id, year)
  ]
  
  # Rename columns to match expected output naming convention
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # Join aggregated stats back onto cell_dt
  setkey(agg, cell_id, year)
  
  # Return the aggregation table (to be merged externally)
  agg
}

# ============================================================
# STEP 2b: Memory-optimized version that avoids the full
#           cross join materialization (for 16 GB RAM safety)
# ============================================================
compute_neighbor_features_lean <- function(cell_dt, edge_dt, var_name) {
  cat(sprintf("  Computing neighbor features for: %s\n", var_name))
  
  # Build a lookup: (id, year) -> value
  lookup <- cell_dt[, .(neighbor_id = id, year, nb_val = get(var_name))]
  setkey(lookup, neighbor_id, year)
  
  # For each year, join edge_dt with that year's values and aggregate
  years <- sort(unique(cell_dt$year))
  
  result_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    
    # Subset lookup to this year
    lk_yr <- lookup[year == yr, .(neighbor_id, nb_val)]
    setkey(lk_yr, neighbor_id)
    
    # Join neighbor values onto edge table
    edges_yr <- edge_dt[lk_yr, on = .(neighbor_id), nomatch = NULL]
    
    # Aggregate by cell_id
    agg_yr <- edges_yr[
      !is.na(nb_val),
      .(
        nb_max  = max(nb_val),
        nb_min  = min(nb_val),
        nb_mean = mean(nb_val)
      ),
      by = .(cell_id)
    ]
    agg_yr[, year := yr]
    
    result_list[[j]] <- agg_yr
  }
  
  agg <- rbindlist(result_list)
  
  # Rename columns
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  setkey(agg, cell_id, year)
  agg
}

# ============================================================
# STEP 3: Outer loop â€” compute for all 5 variables and merge
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

cat("Computing neighbor features for all variables...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  agg <- compute_neighbor_features_lean(cell_data, edge_table, var_name)
  
  # Determine the new column names
  new_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  
  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- intersect(new_cols, names(cell_data))
  if (length(old_cols) > 0) {
    cell_data[, (old_cols) := NULL]
  }
  
  # Merge onto cell_data
  # Use id = cell_id mapping
  setnames(agg, "cell_id", "id")
  setkey(agg, id, year)
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  cat(sprintf("  -> Merged %s features. cell_data now has %d columns.\n",
              var_name, ncol(cell_data)))
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done. Total time: %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
#
# The trained model object (e.g., `rf_model`) is unchanged.
# The feature columns are numerically identical to the original
# pipeline's output, so predictions are preserved exactly.
# ============================================================
# Example (uncomment and adapt to your actual model object):
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Produces Identical Results

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[i]]` | Same `nb` object â†’ same edge table |
| Neighbor value lookup | `vals[idx]` per row | `data.table` keyed join on `(neighbor_id, year)` â€” same values |
| Aggregation | `max()`, `min()`, `mean()` on non-NA neighbor values | Identical functions with `!is.na()` filter |
| Output columns | `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` | Same names, same values |
| RF model | Untouched | Untouched â€” only `predict()` is called |

The numerical estimand is preserved exactly because the same neighbor pairs, the same variable values, and the same summary functions (max, min, mean with NA removal) are used. No approximation or sampling is introduced.

## Expected Performance

| Stage | Rows Processed | Estimated Time |
|---|---|---|
| Build edge table | ~1.37M edges | < 1 second |
| Per variable, per year: join + aggregate | ~1.37M edges Ã— 28 years | ~5â€“15 seconds per variable |
| All 5 variables | 5 Ã— above | ~1â€“2 minutes total |
| **Total** | | **~2â€“5 minutes** vs. ~86 hours |

The speedup factor is roughly **1,000â€“2,500Ã—**, achieved entirely by eliminating R-level row iteration and exploiting the year-invariant spatial topology.
END_RESPONSE>>>
