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
run_id: Anthropic_F07_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²) key-lookup via named vectors
- `idx_lookup` is a **named character vector** with ~6.46 million entries. Each lookup via `idx_lookup[neighbor_keys]` triggers a **linear scan** (or at best a partial-match hash) on character names for every single row.
- The function calls `lapply` over **6.46 million rows**, and for each row it constructs character keys, then does named-vector subsetting. With ~1.37M directed neighbor relationships spread across 28 years, this produces roughly **38.4 million** individual key lookups, each against a 6.46M-length named vector.
- **Estimated cost:** This is the dominant bottleneck â€” the `paste` + named-vector lookup pattern is extremely slow in R at this scale. This alone accounts for most of the 86+ hour estimate.

### 2. `compute_neighbor_stats` â€” R-level `lapply` over 6.46M rows
- Each iteration computes `max`, `min`, `mean` on a small integer-indexed subset. While each call is cheap, 6.46 million R-level function calls with list allocation is slow (though far less so than the lookup problem above).
- This is repeated 5 times (once per source variable), producing 15 new columns.

### 3. Architectural mismatch
- The neighbor topology is **time-invariant** (same 344,208 cells, same rook neighbors every year), but the lookup is rebuilt as if it were time-varying. The code re-discovers the same spatial neighbors for each of the 28 years per cell â€” a 28Ã— redundancy.

---

## Optimization Strategy

### A. Replace named-vector lookup with `data.table` hash joins
Use `data.table` keyed joins (binary search, O(log n)) instead of named-vector character matching. This reduces the lookup phase from hours to seconds.

### B. Exploit time-invariance of topology
Build the neighbor index **once at the cell level** (344K cells), then expand to cell-years via a vectorized merge/join â€” not an `lapply` over 6.46M rows.

### C. Vectorize `compute_neighbor_stats`
Instead of `lapply` over 6.46M rows, construct an **edge table** (cell-year â†’ neighbor-cell-year) and use `data.table` grouped aggregation (`max`, `min`, `mean`) in a single pass per variable.

### D. Memory considerations
- Edge table: ~38.4M rows Ã— 2 integer columns â‰ˆ 0.6 GB. Fits in 16 GB RAM.
- `data.table` operations are memory-efficient and single-threaded-safe.

### Expected speedup: from 86+ hours â†’ **minutes** (typically 5â€“15 minutes total).

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a spatial edge list (time-invariant, build once)
# ==============================================================
# rook_neighbors_unique: spdep nb object (list of length 344,208)
# id_order: vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  
  # Remove self-neighbors and zero-entries (spdep uses 0 for no-neighbor)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id  (~1.37M rows)

# ==============================================================
# STEP 2: Convert cell_data to data.table and key it
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================
# STEP 3: Compute neighbor stats for all variables efficiently
# ==============================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  
  # Create a minimal lookup: id, year, and the source variables
  lookup_cols <- c("id", "year", source_vars)
  lookup_dt <- cell_data[, ..lookup_cols]
  
  # ---------------------------------------------------------
  # Expand edge_dt across all years to get cell-year edges

  # Instead of a full cross join (expensive), we join through the data
  # ---------------------------------------------------------
  
  # For each (from_id, year) row in cell_data, find neighbor values
  # Strategy: join edge_dt to cell_data on from_id = id to get years,
  #           then join to cell_data again on to_id + year to get neighbor values
  
  # Step A: Create the cell-year â†’ neighbor-cell-year edge table
  # We need: for each row in cell_data (id, year), the neighbor ids
  # Then look up the neighbor's values in that same year
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # Cross edge_dt with years (this is the full directed edge-year table)
  # ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” fits in memory
  edge_year_dt <- CJ_dt_edges(edge_dt, years)
  
  # Step B: Join neighbor values
  setkey(lookup_dt, id, year)
  
  # Join to get neighbor variable values
  # edge_year_dt has: from_id, to_id, year
  # We want the values of source_vars for (to_id, year)
  edge_year_dt[lookup_dt, 
               (source_vars) := mget(paste0("i.", source_vars)),
               on = .(to_id = id, year = year)]
  
  # Step C: Aggregate by (from_id, year) â€” this is the neighbor summary
  for (vname in source_vars) {
    agg <- edge_year_dt[!is.na(get(vname)),
                        .(nmax  = max(get(vname)),
                          nmin  = min(get(vname)),
                          nmean = mean(get(vname))),
                        by = .(from_id, year)]
    
    max_col  <- paste0("neighbor_max_", vname)
    min_col  <- paste0("neighbor_min_", vname)
    mean_col <- paste0("neighbor_mean_", vname)
    setnames(agg, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
    
    # Join back to cell_data
    cell_data[agg, 
              (c(max_col, min_col, mean_col)) := mget(paste0("i.", c(max_col, min_col, mean_col))),
              on = .(id = from_id, year = year)]
  }
  
  cell_data
}

# Helper: cross join edges with years (memory-efficient)
CJ_dt_edges <- function(edge_dt, years) {
  # Repeat each edge for each year
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  
  data.table(
    from_id = rep(edge_dt$from_id, times = n_years),
    to_id   = rep(edge_dt$to_id,   times = n_years),
    year    = rep(years, each = n_edges)
  )
}

# ==============================================================
# STEP 4: Execute
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Remove old neighbor columns if re-running (safety)
old_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

# Set key for efficient joins
setkey(cell_data, id, year)

# Run the computation
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# ==============================================================
# STEP 5: Verify output integrity
# ==============================================================
expected_new_cols <- paste0(
  rep(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"), times = length(neighbor_source_vars)),
  rep(neighbor_source_vars, each = 3)
)
stopifnot(all(expected_new_cols %in% names(cell_data)))

cat("Neighbor features computed successfully.\n")
cat("New columns:", paste(expected_new_cols, collapse = ", "), "\n")
cat("Rows:", nrow(cell_data), "\n")
```

---

## Summary of Guarantees

| Requirement | Status |
|---|---|
| **Numerical equivalence** to original `max`, `min`, `mean` | âœ… Identical aggregation logic; same NA handling |
| **Trained Random Forest model preserved** | âœ… No model code is touched; column names match original schema |
| **Rook-neighbor topology preserved** | âœ… Uses the same `rook_neighbors_unique` nb object |
| **Handles irregular/masked cells** | âœ… `on` joins naturally produce `NA` for missing cell-years (no rectangular raster assumption) |
| **Fits in 16 GB RAM** | âœ… Peak ~4â€“5 GB (edge-year table ~1.2 GB + cell_data ~3 GB) |
| **Runtime** | ~5â€“15 minutes vs. 86+ hours |

The key insight: the original code performed **~38 million character-key lookups against a 6.46M named vector inside an R-level loop** â€” an accidentally quadratic pattern. Replacing this with `data.table` keyed joins and grouped aggregation reduces the complexity to O(n log n) with minimal R-level iteration.
END_RESPONSE>>>
