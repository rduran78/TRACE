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
run_id: Anthropic_F09_T03_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
The function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine),
- Pastes together character keys (`paste(id, year)`),
- Performs **named-vector character matching** (`idx_lookup[neighbor_keys]`).

Named character vector lookup in R is hash-based but still involves repeated string construction and hashing for every single row. With ~6.46M rows Ã— ~4 neighbors each â‰ˆ **25.8 million string-match operations**, this is extremely expensive. Worse, the entire lookup table `idx_lookup` is a named character vector of length 6.46M, so each lookup traverses a large hash table.

### 2. The lookup is **year-redundant**
The spatial neighbor topology is **identical across all 28 years**. The cell-neighbor adjacency is purely spatialâ€”it never changes. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, doing 28Ã— more work than necessary. A cell's neighbors in 1992 are the same cells as in 2019; only the attribute values change.

### 3. `compute_neighbor_stats` uses row-wise `lapply`
Even after the lookup is built, computing stats iterates row-by-row over 6.46M rows in R-level `lapply`, calling `max`, `min`, `mean` individually. This prevents vectorization.

---

## Optimization Strategy

**Core insight:** Build the adjacency table **once** at the cell level (344,208 cells, not 6.46M cell-years), store it as a two-column data.table of `(cell_id, neighbor_id)`, then use **vectorized joins** to compute neighbor statistics in bulk.

### Steps:
1. **Build a static edge table** from `rook_neighbors_unique` â€” a simple two-column table: `(cell_id, neighbor_id)`. This has ~1.37M rows and is built once.
2. **Join yearly attributes onto the edge table** by `(neighbor_id, year)` to pull in each neighbor's variable value.
3. **Group-by aggregate** `(cell_id, year)` to compute `max`, `min`, `mean` in one vectorized pass per variable.
4. **Join results back** onto the main `cell_data`.

This replaces 6.46M R-level iterations with a handful of `data.table` joins and group-by operations, reducing runtime from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_data <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build the static spatial edge table ONCE
#         from the spdep nb object (rook_neighbors_unique)
#         and the id_order vector.
#
#         rook_neighbors_unique[[i]] contains integer indices
#         into id_order for the neighbors of id_order[i].
#         id_order is a vector of 344,208 cell IDs.
# ============================================================

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  edge_lengths <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_lengths)
  
  # Build vectors directly
  from_id <- rep(id_order, times = edge_lengths)
  to_idx  <- unlist(neighbors, use.names = FALSE)
  to_id   <- id_order[to_idx]
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)
# This is built ONCE and reused for every variable and every year.

cat("Edge table rows:", nrow(edge_table), "\n")

# ============================================================
# STEP 2: Function to compute neighbor max, min, mean for one
#         variable using vectorized data.table joins + group-by
# ============================================================

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Create a slim lookup: just (id, year, value)
  val_col <- var_name
  lookup <- cell_dt[, .(id, year, value = get(val_col))]
  setkey(lookup, id, year)
  
  # Join: for each edge (cell_id, neighbor_id), cross with all years,
  # then pull the neighbor's value for that year.
  #
  # But we don't need a full cross: we only need years that exist in cell_dt.
  # Strategy: join edge_table onto cell_dt's (id, year) to get the
  # set of (cell_id, year, neighbor_id), then look up neighbor values.
  
  # Get the (cell_id, year) combinations that exist
  cell_years <- cell_dt[, .(cell_id = id, year)]
  
  # Expand: for each (cell_id, year), attach all neighbors
  # This is a join of cell_years with edge_table on cell_id
  setkey(cell_years, cell_id)
  setkey(edge_dt, cell_id)
  
  expanded <- edge_dt[cell_years, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year
  # Rows: ~6.46M * ~4 neighbors â‰ˆ ~25.8M rows (fits in RAM)
  
  # Look up the neighbor's value for that year
  setkey(expanded, neighbor_id, year)
  expanded[lookup, on = c(neighbor_id = "id", "year"), neighbor_val := i.value]
  
  # Aggregate: group by (cell_id, year), compute stats
  # Drop NAs in neighbor_val before aggregation
  stats <- expanded[!is.na(neighbor_val),
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(cell_id, year)]
  
  # Rename columns to match expected output naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # Join back onto cell_dt
  # First remove old columns if they exist (for idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE, sort = FALSE)
  
  cell_dt
}

# ============================================================
# STEP 3: Outer loop â€” compute neighbor features for all 5 vars
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- Sys.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "seconds\n")
}

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
#         (model object is unchanged; no retraining)
# ============================================================

# Convert back to data.frame if the RF model expects one
cell_data_df <- as.data.frame(cell_data)

# Predict (the trained model object name may vary; shown as `rf_model`)
cell_data_df$prediction <- predict(rf_model, newdata = cell_data_df)

cat("Prediction complete. Rows:", nrow(cell_data_df), "\n")
```

---

## Summary of Gains

| Aspect | Before (Current) | After (Optimized) |
|---|---|---|
| **Neighbor lookup** | 6.46M row-wise string hashes | 1 vectorized edge table build (~1.37M rows) |
| **Neighbor stats** | 6.46M Ã— 5 vars R-level `lapply` | 5 Ã— one `data.table` join + group-by |
| **String operations** | ~25.8M `paste` + hash lookups | Zero |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** (5 vars Ã— ~1â€“3 min each) |
| **Peak RAM** | Moderate (but slow) | ~3â€“5 GB for the expanded table (fits in 16 GB) |
| **RF model** | Unchanged âœ“ | Unchanged âœ“ |
| **Numerical results** | Original estimand | Preserved identically (same max/min/mean logic) âœ“ |

The key transformation is: **separate topology (static) from attributes (yearly), join them with vectorized operations, and aggregate in bulk** â€” eliminating millions of R-level loop iterations and string operations.
END_RESPONSE>>>
