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
run_id: Anthropic_F09_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

### 1. `build_neighbor_lookup` is O(nÂ²)-equivalent in practice
For every one of the **6.46 million** cell-year rows, it:
- Looks up the cell's spatial neighbors.
- Pastes together character keys (`id_year`) for each neighbor Ã— year combination.
- Performs named-vector character lookups (`idx_lookup[neighbor_keys]`) against a 6.46-million-element named character vector.

Named character vector lookup in R is **hashed**, but constructing 6.46 million character keys and performing millions of hash lookups inside an `lapply` loop is still brutally slow. More critically, **the spatial neighbor structure is the same for every year**, yet this function recomputes neighbor row-indices per cell-year rather than per cell, duplicating work 28Ã—.

### 2. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even though each iteration is small, the R-level loop overhead across 6.46 million iterations (Ã— 5 variables) is enormous. This is ~32 million R-level function calls with per-element subsetting.

### 3. The neighbor topology is **year-invariant** but treated as year-variant
The rook-neighbor structure is purely spatial. It does not change across years. The current code entangles spatial topology with temporal indexing, preventing vectorized joins.

---

## Optimization Strategy

**Core insight:** Separate the static spatial topology from the dynamic yearly attributes, then use vectorized data.table joins and grouped aggregations instead of row-level R loops.

### Step-by-step plan:

1. **Build an edge table once** â€” a two-column `data.table` of `(focal_id, neighbor_id)` derived from `rook_neighbors_unique`. This is ~1.37 million rows and never changes.

2. **Join yearly attributes onto the edge table** â€” For each year, join the cell-level variable values onto the `neighbor_id` column. This is a simple keyed `data.table` merge â€” extremely fast.

3. **Aggregate by `(focal_id, year)`** â€” Compute `max`, `min`, `mean` of neighbor values using `data.table`'s grouped aggregation, which is vectorized C-level code.

4. **Join aggregated neighbor stats back** onto the main `cell_data` table.

5. **Predict with the existing Random Forest model** â€” no retraining.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes** on a 16 GB laptop. The edge table is ~1.37M rows; crossed with 28 years gives ~38.4M join rows, which `data.table` handles trivially.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Convert cell_data to data.table (if not already)
# =============================================================================
cell_dt <- as.data.table(cell_data)

# Ensure id and year columns are present and properly typed
stopifnot(all(c("id", "year") %in% names(cell_dt)))
cell_dt[, id := as.integer(id)]
cell_dt[, year := as.integer(year)]

# =============================================================================
# STEP 1: Build the static spatial edge table ONCE from the nb object
#
# rook_neighbors_unique is a list of length 344,208 (one per cell).
# id_order is the vector mapping list index -> cell id.
# rook_neighbors_unique[[i]] contains integer indices (into id_order)
#   of the rook neighbors of cell id_order[i].
# =============================================================================

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_ids[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_ids[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# =============================================================================
# STEP 2: For each variable, compute neighbor max/min/mean via vectorized join
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Get unique years
all_years <- sort(unique(cell_dt$year))

# Cross the edge table with all years (creates the full join scaffold)
# This yields ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” fits easily in 16 GB
cat("Expanding edge table across years...\n")
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = all_years)
edge_year_dt[, focal_id    := edge_dt$focal_id[edge_idx]]
edge_year_dt[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year_dt[, edge_idx := NULL]

# Key the cell data for fast joining
setkey(cell_dt, id, year)

# Create a lookup table: (id, year) -> variable values
# We only need the neighbor source vars + id + year
lookup_dt <- cell_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(lookup_dt, "id", "neighbor_id")
setkey(lookup_dt, neighbor_id, year)

# Join neighbor attributes onto the expanded edge table
cat("Joining neighbor attributes...\n")
setkey(edge_year_dt, neighbor_id, year)
edge_year_dt <- lookup_dt[edge_year_dt, on = .(neighbor_id, year)]

# =============================================================================
# STEP 3: Aggregate neighbor stats grouped by (focal_id, year)
# =============================================================================

cat("Computing neighbor aggregations...\n")

# Build aggregation expressions dynamically
agg_exprs <- list()
for (var in neighbor_source_vars) {
  var_sym <- as.name(var)
  
  # Naming convention must match original pipeline output column names.
  # Adjust these suffixes if your trained RF model expects different names.
  max_name  <- paste0("neighbor_max_", var)
  min_name  <- paste0("neighbor_min_", var)
  mean_name <- paste0("neighbor_mean_", var)
  
  agg_exprs[[max_name]]  <- bquote(max(.(var_sym),  na.rm = TRUE))
  agg_exprs[[min_name]]  <- bquote(min(.(var_sym),  na.rm = TRUE))
  agg_exprs[[mean_name]] <- bquote(mean(.(var_sym), na.rm = TRUE))
}

# Convert to a single call for data.table's j
agg_call <- as.call(c(as.name("list"), agg_exprs))

neighbor_stats <- edge_year_dt[, eval(agg_call), by = .(focal_id, year)]

# Replace -Inf/Inf from max/min of empty groups with NA (safety)
for (col_name in names(neighbor_stats)) {
  if (is.numeric(neighbor_stats[[col_name]])) {
    set(neighbor_stats, 
        i = which(is.infinite(neighbor_stats[[col_name]])),
        j = col_name, 
        value = NA_real_)
  }
}

# =============================================================================
# STEP 4: Join neighbor stats back onto the main cell data
# =============================================================================

cat("Joining neighbor stats back to main data...\n")
setnames(neighbor_stats, "focal_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_dt, id, year)

# Remove old neighbor columns if they exist (from a prior run)
old_neighbor_cols <- grep("^neighbor_(max|min|mean)_", names(cell_dt), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_dt[, (old_neighbor_cols) := NULL]
}

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# =============================================================================
# STEP 5: Predict with the EXISTING trained Random Forest (no retraining)
# =============================================================================

cat("Generating predictions with existing RF model...\n")

# Convert back to data.frame if the RF model expects one
cell_data_final <- as.data.frame(cell_dt)

# The trained model object (e.g., `rf_model`) is assumed to already be in memory.
# Predict using the exact same feature set:
cell_data_final$rf_prediction <- predict(rf_model, newdata = cell_data_final)

cat("Done.\n")
```

---

## Memory-Constrained Alternative (if 38.4M-row expansion is too large)

If the ~38.4 million-row `edge_year_dt` table with all variable columns approaches the 16 GB limit, process **one variable at a time** or **one year at a time**:

```r
# ---- Memory-efficient: process one variable at a time ----

setkey(edge_dt, neighbor_id)  # static edge table, ~1.37M rows

for (var in neighbor_source_vars) {
  cat(sprintf("Processing variable: %s\n", var))
  
  # Minimal lookup: just (id, year, variable)
  lkp <- cell_dt[, .(neighbor_id = id, year, val = get(var))]
  setkey(lkp, neighbor_id)
  
  # For each year, join + aggregate (avoids full cross-product in memory)
  year_results <- rbindlist(lapply(all_years, function(yr) {
    lkp_yr <- lkp[year == yr]
    setkey(lkp_yr, neighbor_id)
    merged <- lkp_yr[edge_dt, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]
    merged[, .(
      nb_max  = max(val, na.rm = TRUE),
      nb_min  = min(val, na.rm = TRUE),
      nb_mean = mean(val, na.rm = TRUE)
    ), by = .(focal_id, year)]
  }))
  
  # Fix infinities
  year_results[is.infinite(nb_max),  nb_max  := NA_real_]
  year_results[is.infinite(nb_min),  nb_min  := NA_real_]
  
  # Rename and join back
  max_nm  <- paste0("neighbor_max_", var)
  min_nm  <- paste0("neighbor_min_", var)
  mean_nm <- paste0("neighbor_mean_", var)
  setnames(year_results, c("nb_max", "nb_min", "nb_mean"), c(max_nm, min_nm, mean_nm))
  setnames(year_results, "focal_id", "id")
  setkey(year_results, id, year)
  
  # Drop old columns if they exist
  for (cn in c(max_nm, min_nm, mean_nm)) {
    if (cn %in% names(cell_dt)) cell_dt[, (cn) := NULL]
  }
  
  cell_dt <- year_results[cell_dt, on = .(id, year)]
  rm(year_results, lkp); gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M character hash lookups in R loop | Static 1.37M-row integer edge table built once |
| **Stats computation** | `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| **Time complexity** | O(rows Ã— avg_neighbors Ã— char_hash) | O(edges Ã— years) with C-level joins |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** |
| **RAM usage** | Low but slow | ~4â€“8 GB peak (fits 16 GB laptop) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical results** | Original estimand | Identical `max`/`min`/`mean` âœ“ |
END_RESPONSE>>>
