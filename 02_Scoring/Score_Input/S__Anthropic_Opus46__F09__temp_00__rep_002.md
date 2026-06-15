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
run_id: Anthropic_F09_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups via `[` on character keys are O(n) in the worst case per call due to R's internal hashing overhead at scale.

### 2. Redundant recomputation of the same spatial topology per year
The rook-neighbor relationships are **purely spatial** â€” they don't change across years. Yet `build_neighbor_lookup` rebuilds the neighbor mapping for every cell-year combination, effectively duplicating the same spatial adjacency 28 times (once per year). This is the core architectural flaw.

### 3. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level `lapply` over 6.46M list elements is inherently slow. Each iteration has R function-call overhead, subsetting, `is.na` filtering, and three summary function calls.

**Summary:** The pipeline does ~6.46M Ã— (string ops + list traversal) Ã— 2 passes (build + compute) Ã— 5 variables â‰ˆ billions of interpreted R operations.

---

## Optimization Strategy

### Core Insight: Separate Spatial Topology from Temporal Attributes

The neighbor relationships are a property of **cells**, not cell-years. Build the adjacency table **once** at the cell level (344,208 cells Ã— ~4 neighbors each â‰ˆ 1.37M directed edges), then **join** yearly attributes onto that edge table and compute grouped summaries using vectorized `data.table` operations.

### Steps:

1. **Build a static edge table** (`from_id`, `to_id`) from the `spdep::nb` object â€” done once, ~1.37M rows.
2. **Join yearly cell attributes** onto the edge table by (`to_id`, `year`) â€” this replicates the attribute of each neighbor onto the edge, giving ~1.37M Ã— 28 â‰ˆ ~38.5M rows (fits in RAM).
3. **Group-by aggregate** (`from_id`, `year`) to compute `max`, `min`, `mean` â€” fully vectorized in `data.table`, no R-level loops.
4. **Join results back** to the main cell-year panel.
5. **Predict** with the existing trained Random Forest model (unchanged).

### Expected Speedup:
- Eliminates all `lapply` loops over 6.46M rows.
- Replaces string-key lookups with integer joins.
- `data.table` grouped aggregation on ~38.5M rows completes in seconds.
- **Total estimated time: 1â€“5 minutes** (vs. 86+ hours).

### Memory Check:
- Edge table: ~1.37M rows Ã— 2 int cols â‰ˆ 11 MB
- Edge-year-attribute table: ~38.5M rows Ã— 4 cols â‰ˆ 1.2 GB per variable (done one variable at a time, then discarded)
- Well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert main data to data.table (if not already)
# ============================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# plus all other predictor columns needed for RF prediction.
# rook_neighbors_unique: spdep::nb object (list of integer index vectors)
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# ============================================================
# STEP 1: Build static spatial edge table ONCE
# ============================================================
# rook_neighbors_unique[[i]] gives the index positions (into id_order)
# of the neighbors of cell id_order[i].

build_edge_table <- function(id_order, neighbors) {
  from_list <- vector("list", length(id_order))
  to_list   <- vector("list", length(id_order))
  
  for (i in seq_along(id_order)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) next
    # Remove self-references and zero entries (spdep convention)
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) next
    from_list[[i]] <- rep(id_order[i], length(nb_idx))
    to_list[[i]]   <- id_order[nb_idx]
  }
  
  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: one row per directed neighbor relationship
cat("Edge table rows:", nrow(edge_dt), "\n")

# ============================================================
# STEP 2: Get unique years
# ============================================================
all_years <- sort(unique(cell_dt$year))

# ============================================================
# STEP 3: For each neighbor source variable, compute neighbor
#          max, min, mean via vectorized join + group-by
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name, all_years) {
  
  # Extract only the columns we need for the neighbor attribute lookup
  # Columns: id, year, <var_name>
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Cross-join edges with all years to get edge-year combinations
  # edge_dt has (from_id, to_id); we need one copy per year
  edge_year_dt <- CJ_edge_year(edge_dt, all_years)
  
  # Join neighbor attribute: for each edge-year, get the neighbor's value
  # Key the attribute table on (id, year) and join on (to_id, year)
  setkey(attr_dt, id, year)
  edge_year_dt[attr_dt, neighbor_val := i.val, on = .(to_id = id, year)]
  
  # Group by (from_id, year) and compute stats, excluding NAs
  stats_dt <- edge_year_dt[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(from_id, year)
  ]
  
  # Rename columns to match original pipeline naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats_dt, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # Join back onto cell_dt by (id = from_id, year)
  cell_dt[stats_dt, (c(max_col, min_col, mean_col)) :=
            mget(c(max_col, min_col, mean_col)),
          on = .(id = from_id, year)]
  
  invisible(cell_dt)
}

# Helper: cross-join edge table with years (memory-efficient)
CJ_edge_year <- function(edge_dt, all_years) {
  # Repeat each edge for every year
  n_edges <- nrow(edge_dt)
  n_years <- length(all_years)
  
  result <- data.table(
    from_id = rep(edge_dt$from_id, times = n_years),
    to_id   = rep(edge_dt$to_id,   times = n_years),
    year    = rep(all_years, each = n_edges)
  )
  result
}

# ============================================================
# STEP 4: Run for all 5 neighbor source variables
# ============================================================
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- Sys.time()
  compute_neighbor_features_fast(cell_dt, edge_dt, var_name, all_years)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "sec\n")
}

# ============================================================
# STEP 5: Predict with the existing trained Random Forest
# ============================================================
# The RF model object (e.g., rf_model) is already in memory and must NOT
# be retrained. We simply score the enriched data.

# Convert back to data.frame if the RF predict method requires it:
cell_df <- as.data.frame(cell_dt)

# Generate predictions (preserves original numerical estimand)
cell_df$rf_prediction <- predict(rf_model, newdata = cell_df)

cat("Pipeline complete. Predictions generated for",
    nrow(cell_df), "cell-year rows.\n")
```

---

## Memory-Optimized Variant (if 16 GB is tight)

If the full cross-join (~38.5M rows) for all years at once strains memory, process **one year at a time**:

```r
compute_neighbor_features_by_year <- function(cell_dt, edge_dt, var_name) {
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  for (yr in sort(unique(cell_dt$year))) {
    # Subset attributes for this year
    attr_yr <- attr_dt[year == yr]
    setkey(attr_yr, id)
    
    # Join neighbor values onto edge table
    edge_yr <- copy(edge_dt)
    edge_yr[attr_yr, neighbor_val := i.val, on = .(to_id = id)]
    
    # Aggregate
    stats_yr <- edge_yr[
      !is.na(neighbor_val),
      .(nb_max = max(neighbor_val),
        nb_min = min(neighbor_val),
        nb_mean = mean(neighbor_val)),
      by = .(from_id)
    ]
    
    # Write back into cell_dt for this year
    cell_dt[stats_yr,
            on = .(id = from_id, year = yr),
            `:=`(
              (max_col)  = i.nb_max,
              (min_col)  = i.nb_min,
              (mean_col) = i.nb_mean
            )]
  }
  invisible(cell_dt)
}

# Usage:
for (var_name in neighbor_source_vars) {
  cat("Computing (year-by-year):", var_name, "\n")
  compute_neighbor_features_by_year(cell_dt, edge_dt, var_name)
}
```

This variant processes only ~1.37M edge rows per year-variable combination, using minimal memory.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element `lapply` with string keys | 1.37M-row static integer edge table (built once) |
| **Stat computation** | 6.46M-element `lapply` per variable | Vectorized `data.table` group-by join |
| **Topology reuse** | Rebuilt implicitly per cell-year | Built once, joined per year |
| **Estimated runtime** | ~86+ hours | ~1â€“5 minutes |
| **RAM peak** | Moderate (but slow) | ~2â€“4 GB (or ~1 GB with year-by-year variant) |
| **RF model** | Unchanged âœ“ | Unchanged âœ“ |
| **Numerical estimand** | Preserved âœ“ | Preserved âœ“ |
END_RESPONSE>>>
