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
run_id: Anthropic_F09_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it rebuilds neighbor relationships redundantly for every year.
The neighbor topology is **purely spatial** â€” cell A's rook neighbors are the same in 1992 as in 2019. Yet the function creates a lookup keyed on `(id, year)` pairs, performing string-pasting and hash-table lookups **6.46 million times**, each time scanning the neighbor list. This means the same spatial neighbor resolution is repeated 28 times per cell, totaling ~6.46M `lapply` iterations with string operations and named-vector lookups inside each.

### 2. `compute_neighbor_stats` iterates over 6.46M rows in R-level `lapply`.
For each of the 5 variables, it loops through all 6.46M rows in interpreted R, extracting and summarizing neighbor values. That's ~32.3M R-level loop iterations total.

### 3. The combination is catastrophic.
String concatenation (`paste`), named vector lookups, and per-row `lapply` over millions of rows in base R are orders of magnitude slower than vectorized or table-join approaches.

---

## Optimization Strategy

**Core insight:** Separate the **spatial topology** (fixed) from the **temporal attributes** (varying by year). Build the neighbor edge table once (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M edges), then use a vectorized `data.table` join-and-aggregate approach per year.

**Steps:**

1. **Build a static edge table** from the `nb` object: a two-column `data.table` with `(focal_id, neighbor_id)` â€” ~1.37M rows. This is done **once**.

2. **For each variable**, join the cell-year attribute data onto the edge table by `(neighbor_id, year)`, then group-by `(focal_id, year)` to compute `max`, `min`, `mean` â€” all in vectorized `data.table` operations.

3. **No R-level loops over 6.46M rows.** The `data.table` grouped aggregation is executed in C and handles the entire computation in seconds per variable.

**Expected speedup:** From ~86 hours to **~1â€“3 minutes total** (edge table build + 5 variable aggregations).

**Preservation guarantees:**
- The trained Random Forest model is untouched â€” we only compute the same input features.
- The numerical estimand is identical: `max`, `min`, `mean` of the same rook-neighbor values per cell-year.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the static spatial edge table (run ONCE)
# ============================================================
# Inputs:
#   id_order            â€” vector of cell IDs in the order matching the nb object
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))  # ~1.37M
  
  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    n <- length(nb_idx)
    if (n > 0L) {
      focal_ids[pos:(pos + n - 1L)]    <- id_order[i]
      neighbor_ids[pos:(pos + n - 1L)] <- id_order[nb_idx]
      pos <- pos + n
    }
  }
  
  data.table(focal_id = focal_ids[1:(pos - 1L)],
             neighbor_id = neighbor_ids[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows, two columns: focal_id, neighbor_id

cat("Edge table built:", nrow(edge_dt), "directed edges\n")

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Set key for fast joins
setkey(cell_data, id, year)

# ============================================================
# STEP 3: Compute neighbor stats for all variables (vectorized)
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We join edge_dt Ã— year with cell_data to get neighbor attribute values,
# then aggregate by (focal_id, year).

# Get the unique years
all_years <- sort(unique(cell_data$year))

# Expand edge table across all years: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
# On 16GB RAM this is feasible (38.5M rows Ã— 3 integer cols â‰ˆ ~460 MB)
# But we can be smarter: do it per-variable to limit peak memory.

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  
  t0 <- proc.time()
  
  # Extract only the columns we need for the join
  # cell_data[, .(id, year, <var>)]
  attr_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Cross-join edge table with years, then join neighbor attributes
  # More memory-efficient: join edge_dt with attr_dt by neighbor_id and year
  # We need to pair each edge with each year, then look up the neighbor's value.
  
  # Approach: 
  #   1. Create edges_with_years by CJ of edge rows and years? No â€” 
  #      each edge applies to ALL years. So:
  #   2. Join attr_dt onto edge_dt by neighbor_id = id, for all years.
  
  # Rename for clarity in join
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  
  # This join: for each (focal_id, neighbor_id) edge and each year,
  # get the neighbor's value. Result: ~38.5M rows.
  merged <- edge_dt[attr_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # merged columns: focal_id, neighbor_id, year, val
  
  # Aggregate by (focal_id, year)
  agg <- merged[!is.na(val), 
                .(nb_max  = max(val),
                  nb_min  = min(val),
                  nb_mean = mean(val)),
                by = .(focal_id, year)]
  
  # Build target column names (matching original pipeline naming convention)
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "focal_id", "id")
  setkey(agg, id, year)
  
  # Remove old columns if they exist (from a prior run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # Join aggregated stats back onto cell_data
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds\n")
  
  # Clean up to free memory
  rm(attr_dt, merged, agg)
  gc()
}

cat("All neighbor features computed.\n")
cat("cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
# ============================================================
# The trained RF model object is assumed to be already in memory (e.g., `rf_model`).
# cell_data now contains all ~110+ predictor columns including the new neighbor features.
# Predict exactly as before:

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor resolution | Per cell-year (6.46M `lapply` iters with string ops) | Static edge table built once (~344K cells) |
| Stats computation | R-level `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` join + grouped aggregation |
| String operations | ~6.46M `paste()` + named vector lookups | Zero â€” integer key joins only |
| Estimated runtime | **~86+ hours** | **~2â€“5 minutes** |
| Numerical output | max/min/mean of rook-neighbor values | **Identical** max/min/mean of rook-neighbor values |
| Trained RF model | Untouched | **Untouched** |
| Peak RAM (~16 GB) | Lower but irrelevant given runtime | ~4â€“6 GB peak (well within budget) |

The key principle: **the spatial topology is time-invariant, so encode it once as an edge table and let `data.table`'s C-level join and grouping engine do the rest.**
END_RESPONSE>>>
