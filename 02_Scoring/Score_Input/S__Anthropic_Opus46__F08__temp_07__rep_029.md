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
run_id: Anthropic_F08_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static spatial topology with dynamic temporal data.**

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows)** rather than per cell (~344K cells). Since rook-neighbor adjacency is purely spatial and never changes across years, this is doing **28Ã— redundant work** on the topology side.

2. **String-key hashing is expensive.** The function creates `paste(id, year)` keys for all 6.46M rows, builds a named lookup vector, and then for each of the 6.46M rows, constructs neighbor keys via `paste()` again and does named-vector lookups. This involves massive string allocation and hash-table probing.

3. **`lapply` over 6.46M rows** in `compute_neighbor_stats` is inherently slow in R â€” each iteration has overhead from function dispatch, subsetting, and `is.na` checks.

4. **The neighbor lookup list itself is ~6.46M entries**, each containing integer vectors. This consumes substantial memory and creates GC pressure.

### What Is Static vs. What Changes

| Aspect | Static (invariant across years) | Dynamic (changes by year) |
|---|---|---|
| Cell IDs | âœ“ | |
| Neighbor adjacency (rook) | âœ“ | |
| Variable values (ntl, ec, â€¦) | | âœ“ |
| Neighbor stats (max, min, mean) | | âœ“ |

**Key insight:** We only need **one** neighbor lookup of ~344K cells (not 6.46M cell-years). Then for each year, we index into that year's data slice using the static topology.

---

## Optimization Strategy

### 1. Build the neighbor lookup once, at the cell level only (~344K entries)

Create a mapping from each cell's positional index (1â€¦344,208) to its neighbors' positional indices. This is done **once** and is year-independent.

### 2. Organize data so that each year's values can be accessed by cell index

Sort/ensure `cell_data` is ordered by `(id, year)` or `(year, id)`. With a consistent cell ordering, for any given year we can extract a contiguous block of rows and index into it by cell position.

### 3. Vectorize neighbor stat computation using sparse matrix multiplication

Represent the neighbor adjacency as a **sparse matrix** `W` of dimension 344,208 Ã— 344,208. Then for each year and each variable:

- `neighbor_mean = (W %*% x) / (W %*% ones)` (where non-neighbor entries are 0)
- `neighbor_max` and `neighbor_min` via grouped operations on the sparse structure

This replaces 6.46M R-level `lapply` iterations with ~28 matrix operations per variable.

### 4. Alternative: use `data.table` grouped operations with the static adjacency edge list

Convert the `nb` object to an edge list (from_cell, to_cell). Join against each year's variable values. Compute grouped max/min/mean via `data.table` â€” extremely fast.

### Expected Speedup

| Step | Old | New |
|---|---|---|
| Build lookup | ~6.46M string ops | ~344K integer ops (once) |
| Compute stats | ~6.46M Ã— 5 lapply calls | 28 Ã— 5 vectorized group-bys |
| Estimated time | 86+ hours | **~2â€“10 minutes** |

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Convert the static nb object to a static edge list (ONCE)
# ==============================================================================
# rook_neighbors_unique: an nb object (list of length 344,208)
# id_order: vector of cell IDs in the order matching the nb object

build_static_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (positional indices)
  # id_order[i] is the cell ID for position i
  
  from_pos <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  to_pos   <- unlist(neighbors_nb)
  
  # Return as data.table with positional indices and cell IDs
  data.table(
    from_pos = from_pos,
    to_pos   = to_pos,
    from_id  = id_order[from_pos],
    to_id    = id_order[to_pos]
  )
}

edge_dt <- build_static_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows and is year-independent

cat(sprintf("Static edge list: %d directed neighbor pairs\n", nrow(edge_dt)))

# ==============================================================================
# STEP 2: Ensure cell_data is a data.table with consistent ordering
# ==============================================================================
cell_data <- as.data.table(cell_data)

# Create a positional index for each cell ID (matching the nb object order)
id_pos_map <- data.table(id = id_order, cell_pos = seq_along(id_order))
cell_data  <- merge(cell_data, id_pos_map, by = "id", all.x = TRUE, sort = FALSE)

# ==============================================================================
# STEP 3: Compute neighbor stats efficiently â€” static topology, dynamic values
# ==============================================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # cell_dt must have columns: id, year, cell_pos, and all source_vars
  # edge_dt must have columns: from_pos, to_pos (static, year-independent)
  
  years <- sort(unique(cell_dt$year))
  
  # Pre-allocate result columns in cell_dt
  for (var in source_vars) {
    cell_dt[, paste0("neighbor_max_",  var) := NA_real_]
    cell_dt[, paste0("neighbor_min_",  var) := NA_real_]
    cell_dt[, paste0("neighbor_mean_", var) := NA_real_]
  }
  
  # Key cell_dt for fast lookups by (year, cell_pos)
  # We'll process year by year to keep memory bounded
  
  for (yr in years) {
    cat(sprintf("  Processing year %d ...\n", yr))
    
    # Extract this year's data: a vector of values indexed by cell_pos
    # Get row indices in cell_dt for this year
    yr_rows <- cell_dt[year == yr]
    
    # Build a lookup: cell_pos -> row index in cell_dt
    # (we need to write results back)
    yr_row_indices <- which(cell_dt$year == yr)
    
    # Build a value lookup by cell_pos for this year
    # Create a vector of length max(cell_pos), indexed by cell_pos
    n_cells <- length(unique(cell_dt$cell_pos))
    max_pos <- max(cell_dt$cell_pos, na.rm = TRUE)
    
    # Map cell_pos -> row index within yr_rows
    pos_to_yr_row <- integer(max_pos)
    pos_to_yr_row[yr_rows$cell_pos] <- seq_len(nrow(yr_rows))
    
    for (var in source_vars) {
      # Build value vector indexed by cell_pos
      val_vec <- rep(NA_real_, max_pos)
      val_vec[yr_rows$cell_pos] <- yr_rows[[var]]
      
      # Look up neighbor values using the STATIC edge list
      # For each edge (from_pos, to_pos), get the value at to_pos
      neighbor_vals_dt <- data.table(
        from_pos = edge_dt$from_pos,
        val      = val_vec[edge_dt$to_pos]
      )
      
      # Remove NA neighbor values before aggregation
      neighbor_vals_dt <- neighbor_vals_dt[!is.na(val)]
      
      # Compute grouped stats: max, min, mean per from_pos
      if (nrow(neighbor_vals_dt) > 0) {
        stats <- neighbor_vals_dt[, .(
          nmax  = max(val),
          nmin  = min(val),
          nmean = mean(val)
        ), by = from_pos]
        
        # Map from_pos back to yr_row_indices in cell_dt
        # Find which yr_rows correspond to these from_pos values
        matched_yr_local <- pos_to_yr_row[stats$from_pos]
        matched_global   <- yr_row_indices[matched_yr_local]
        
        # Write results directly into cell_dt
        set(cell_dt, i = matched_global,
            j = paste0("neighbor_max_", var),  value = stats$nmax)
        set(cell_dt, i = matched_global,
            j = paste0("neighbor_min_", var),  value = stats$nmin)
        set(cell_dt, i = matched_global,
            j = paste0("neighbor_mean_", var), value = stats$nmean)
      }
    }
  }
  
  cell_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (static topology, dynamic values)...\n")
t0 <- proc.time()

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ==============================================================================
# STEP 4: Clean up helper column, proceed to prediction
# ==============================================================================
cell_data[, cell_pos := NULL]

# The trained Random Forest model is unchanged â€” use it directly:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | How it is met |
|---|---|
| **Same numerical estimand** | For each cell-year row, the neighbor max/min/mean is computed over the same set of rook-adjacent cells' values for that same year â€” identical to the original logic. The `data.table` grouped aggregation (`max`, `min`, `mean`) produces the same IEEE 754 results. |
| **Trained RF model unchanged** | No model retraining occurs. The output columns have the same names and values, so `predict()` on the pre-trained model works identically. |
| **NA handling preserved** | `neighbor_vals_dt[!is.na(val)]` mirrors the original `neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` (the pre-allocated default). |

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string-key entries | 1 sparse edge list (~1.37M rows, built once) |
| Stat computation iterations | 6.46M Ã— 5 = 32.3M `lapply` calls | 28 years Ã— 5 vars = 140 vectorized `data.table` group-bys |
| Memory for lookup | ~6.46M-element list of int vectors | ~1.37M-row `data.table` (reused) |
| Estimated wall time | **86+ hours** | **~2â€“10 minutes** |
END_RESPONSE>>>
