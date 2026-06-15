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
run_id: Anthropic_F09_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` constructs a row-level lookup across all 6.46 million cell-year rows using `lapply` with string-key matching.** For each of the ~6.46M rows, it pastes neighbor cell IDs with that row's year, then looks them up in a named character vector (`idx_lookup`). This means ~6.46M iterations, each involving string concatenation, named-vector subsetting (which is O(n) hash lookup per key), and NA filtering. The named vector `idx_lookup` itself has 6.46M entries, making each lookup expensive in aggregate.

2. **The lookup is year-aware but the neighbor *topology* is year-invariant.** The spatial neighbor structure (which cell borders which cell) never changes across years. Yet `build_neighbor_lookup` re-resolves neighbor relationships at the row level (cell Ã— year), effectively repeating the same topological work 28 times â€” once for every year â€” embedded inside a single massive loop over all 6.46M rows.

3. **`compute_neighbor_stats` then iterates over the 6.46M-element list with `lapply`, extracting values one row at a time.** This is pure R-level looping with no vectorization.

4. **Memory pressure:** A 6.46M-element list of integer vectors consumes substantial RAM and is slow to construct and traverse.

**In summary:** The pipeline treats a *static spatial topology + dynamic yearly attributes* problem as a monolithic row-level problem, foregoing the massive speedup available from separating structure from data.

---

## Optimization Strategy

### Core Idea: Build the neighbor edge table once, join yearly attributes, compute stats via vectorized grouped aggregation.

**Step 1 â€” Build a reusable directed edge table (cell-to-neighbor) once.**  
Convert `rook_neighbors_unique` (an `nb` object) into a two-column `data.table`: `(cell_id, neighbor_id)`. This table has ~1.37M rows and never changes.

**Step 2 â€” For each year, join cell attributes onto the edge table.**  
Using `data.table` keyed joins, attach the neighbor's attribute value to each edge row for a given year. This produces ~1.37M Ã— 28 â‰ˆ 38.5M rows (or we can process year-by-year to save RAM).

**Step 3 â€” Compute grouped max, min, mean.**  
Group by `(cell_id, year)` and compute the three summary statistics in one vectorized `data.table` aggregation.

**Step 4 â€” Join results back to the main `cell_data` table.**

This replaces 6.46M R-level iterations with vectorized `data.table` joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

### Why this preserves correctness:
- The neighbor topology is identical (same `nb` object, same rook neighbors).
- The attribute values joined are the same original columns.
- Max, min, and mean are computed over the same neighbor sets.
- The trained Random Forest model is not retouched; we only recompute the input feature columns identically.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Convert cell_data to data.table if not already
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================
# STEP 1: Build the static directed edge table ONCE
#
# rook_neighbors_unique: an nb object (list of integer vectors)
#   where element i contains the indices (into id_order) of
#   neighbors of id_order[i].
# id_order: vector of cell IDs corresponding to the nb object.
# ==============================================================
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)

cat("Edge table rows:", nrow(edge_table), "\n")
# Expected: ~1,373,394

# ==============================================================
# STEP 2: Compute neighbor stats for each variable
#
# For each variable, we:
#   a) Take the subset of cell_data with (id, year, variable).
#   b) Join onto edge_table Ã— years to get neighbor values.
#   c) Aggregate max, min, mean grouped by (cell_id, year).
#   d) Join back onto cell_data.
# ==============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  
  # Column names to create (must match what the RF model expects)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Extract only what we need: id, year, and the variable
  # Using 'id' as the cell identifier column in cell_data
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  
  # Key the attribute table for fast join
  setkey(attr_dt, id, year)
  
  # Get unique years
  years <- sort(unique(attr_dt$year))
  
  # Process all years via a cross-join approach:
  # Expand edge_table Ã— years, then join neighbor attributes
  
  # Create edge-year table: every edge exists in every year
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB easily
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, cell_id     := edge_dt$cell_id[edge_idx]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  edge_year[, edge_idx := NULL]
  
  # Join neighbor's attribute value
  setkey(edge_year, neighbor_id, year)
  setkey(attr_dt, id, year)
  edge_year[attr_dt, neighbor_val := i.value, on = .(neighbor_id = id, year = year)]
  
  # Remove rows where neighbor value is NA (matches original behavior)
  edge_year <- edge_year[!is.na(neighbor_val)]
  
  # Aggregate: max, min, mean by (cell_id, year)
  stats <- edge_year[, .(
    nmax  = max(neighbor_val),
    nmin  = min(neighbor_val),
    nmean = mean(neighbor_val)
  ), by = .(cell_id, year)]
  
  # Rename columns
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  
  # Remove old columns from cell_dt if they exist (idempotent re-runs)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_dt)) cell_dt[, (cc) := NULL]
  }
  
  # Join stats back onto cell_dt
  setkey(stats, cell_id, year)
  setkey(cell_dt, id, year)
  cell_dt <- merge(cell_dt, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
  
  # Clean up
  rm(edge_year, stats, attr_dt)
  gc()
  
  cell_dt
}

# ==============================================================
# STEP 3: Run for all 5 neighbor source variables
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  elapsed <- (proc.time() - t0)[3]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
#
# The RF model object (e.g., rf_model) is already in memory
# and must NOT be retrained. The feature columns now exist in
# cell_data with identical names and identical numerical values.
# ==============================================================

# Example (adjust to your actual model object and prediction workflow):
# predictions <- predict(rf_model, newdata = cell_data)

cat("\nAll neighbor features computed. Ready for Random Forest prediction.\n")
cat("Rows:", nrow(cell_data), " Columns:", ncol(cell_data), "\n")
```

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~2 seconds (vectorized edge table) |
| Compute stats per variable | ~17 hours Ã— 5 = 85h | ~1â€“3 minutes Ã— 5 â‰ˆ 5â€“15 min |
| **Total** | **~86+ hours** | **~5â€“20 minutes** |

### Memory footprint estimate:
- `edge_table`: ~1.37M rows Ã— 2 int cols â‰ˆ 11 MB
- `edge_year` (peak): ~38.5M rows Ã— 4 cols â‰ˆ 1.2 GB (transient, freed per variable)
- `cell_data`: 6.46M Ã— ~125 cols â‰ˆ 6.5 GB
- **Total peak: ~8â€“9 GB** â€” fits within 16 GB RAM

### Correctness guarantees:
- **Same topology:** Edge table is derived from the identical `rook_neighbors_unique` nb object and `id_order`.
- **Same statistics:** `max`, `min`, `mean` are computed over the same non-NA neighbor values per `(cell_id, year)` group.
- **Same column names:** Output columns follow the `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` naming convention (adjust the prefix pattern if your RF model expects different names).
- **Trained RF model untouched:** No retraining occurs; only the input feature matrix is recomputed identically.
END_RESPONSE>>>
