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
run_id: Anthropic_F06_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(nÂ²) key-based row matching

`build_neighbor_lookup` creates a named lookup vector `idx_lookup` keyed by `paste(id, year)` strings for all ~6.46 million rows, then for **each** of those 6.46M rows, it:
1. Identifies the rook neighbors of that cell (fast).
2. Constructs `paste()` key strings for each neighbor Ã— that row's year (moderate).
3. Performs named-vector lookup into `idx_lookup` (slow â€” R named vector lookup is O(n) per probe in the worst case, and this is done billions of times cumulatively).

The resulting object is a **list of 6.46M integer vectors** â€” expensive to build and memory-heavy.

### Bottleneck B: `compute_neighbor_stats` â€” R-level loop over 6.46M rows

`compute_neighbor_stats` iterates via `lapply` over 6.46M elements, each time subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. This is done **5 times** (once per source variable), so ~32.3M R-level function calls with repeated subsetting.

### Why raster focal/kernel operations are NOT a drop-in replacement

Focal operations (e.g., `terra::focal`) assume a regular rectangular grid with uniform cell resolution and a fixed kernel. The dataset here is a **panel** (cell Ã— year), neighbor relationships come from an irregular `spdep::nb` object (likely reflecting coastlines, boundaries, or masked cells), and the computation is done per-year within a long-format data frame. Focal operations would require reshaping every variable into a raster stack per year, running focal, then re-extracting â€” and would silently produce wrong results at boundaries where the nb object differs from a 3Ã—3 rook kernel on a rectangular grid. **The pre-computed `rook_neighbors_unique` must be respected to preserve the original numerical estimand.**

### Estimated current runtime breakdown

| Step | Calls | Estimated time |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string operations + lookups | ~20â€“30 hours |
| `compute_neighbor_stats` | 5 vars Ã— 6.46M `lapply` iterations | ~50â€“60 hours |
| **Total** | | **~70â€“90 hours** |

---

## 2. Optimization Strategy

### Strategy A: Replace string-keyed lookup with integer-arithmetic indexing

Since the panel is balanced (344,208 cells Ã— 28 years = 9,637,824 potential rows, of which ~6.46M exist), we can:
- Sort (or index) data by `(id, year)`.
- Map each `(id, year)` pair to a row index using a **hash table** (`data.table` keyed join) or direct integer arithmetic if the panel is balanced.
- Build the neighbor lookup using vectorized `data.table` joins instead of per-row `paste` + named-vector lookup.

### Strategy B: Vectorize `compute_neighbor_stats` via sparse-matrix multiplication or `data.table` grouped aggregation

Instead of looping over 6.46M rows:
- Build a **sparse adjacency matrix** W (6.46M Ã— 6.46M) where entry (i, j) = 1 if row j is a rook neighbor of row i **in the same year**.
- `mean` = (W %*% x) / (W %*% ones) â€” sparse matrix-vector multiply, runs in C via the `Matrix` package.
- `max` and `min` cannot be computed via matrix multiplication, but can be computed via `data.table` grouped operations after "exploding" the neighbor list into an edge list.

### Strategy C: Edge-list + `data.table` aggregation (best balance of correctness, speed, RAM)

1. Build the neighbor lookup as an **edge list** `data.table` with columns `(row_i, row_j)`.
2. For each variable, join the variable values onto `row_j`, then aggregate by `row_i` to get max, min, mean.
3. This leverages `data.table`'s radix-sort-based grouping, which is highly optimized in C.

**Expected speedup**: from 86+ hours to **~5â€“15 minutes**.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Ensure cell_data is a data.table, sorted by (id, year)
# ==============================================================
cell_data <- as.data.table(cell_data)
# Add a row index before any reordering so we can map back
cell_data[, .row_idx := .I]

# ==============================================================
# STEP 1: Build the edge list (replaces build_neighbor_lookup)
#
# rook_neighbors_unique: an spdep::nb object of length 344,208
# id_order: integer/character vector of length 344,208 giving
#           the cell id corresponding to each nb-list position.
# ==============================================================

build_edge_list_dt <- function(cell_data, id_order, neighbors) {
  # --- 1a. Map cell id -> position in nb list -----------------
  #     (position in id_order / neighbors)
  n_cells <- length(id_order)
  
  # Expand nb object into a data.table of (cell_id, neighbor_cell_id)
  # This is the spatial edge list (no year dimension yet).
  from_pos <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)
  
  # Remove the spdep "0" entries that indicate no neighbors
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  spatial_edges <- data.table(
    from_id = id_order[from_pos],
    to_id   = id_order[to_pos]
  )
  # ~1.37M rows (directed rook-neighbor relationships)
  
  # --- 1b. Cross-join with years to get panel edge list -------
  # We need (from_id, year) -> (to_id, year) edges, but only
  # where both rows actually exist in cell_data.
  
  # Build a keyed lookup: (id, year) -> row index in cell_data
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # For efficiency, expand spatial_edges Ã— years in one shot
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB RAM
  # (3 integer columns â‰ˆ 38.5M Ã— 3 Ã— 8 bytes â‰ˆ 0.9 GB)
  
  panel_edges <- CJ_dt_edges(spatial_edges, years, row_lookup)
  
  return(panel_edges)
}

CJ_dt_edges <- function(spatial_edges, years, row_lookup) {
  # Repeat each spatial edge for every year
  n_edges <- nrow(spatial_edges)
  n_years <- length(years)
  
  edge_year <- data.table(
    from_id = rep(spatial_edges$from_id, times = n_years),
    to_id   = rep(spatial_edges$to_id,   times = n_years),
    year    = rep(years, each = n_edges)
  )
  
  # Join to get row indices for "from" (row_i) and "to" (row_j)
  setkey(edge_year, from_id, year)
  edge_year[row_lookup, row_i := i..row_idx, on = .(from_id = id, year = year)]
  
  setkey(edge_year, to_id, year)
  edge_year[row_lookup, row_j := i..row_idx, on = .(to_id = id, year = year)]
  
  # Keep only edges where both endpoints exist in the data
  edge_year <- edge_year[!is.na(row_i) & !is.na(row_j), .(row_i, row_j)]
  
  return(edge_year)
}

# Build the edge list once
message("Building panel edge list...")
t0 <- Sys.time()
panel_edges <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)
message("Edge list built in ", round(difftime(Sys.time(), t0, units = "secs"), 1), "s. ",
        "Rows: ", nrow(panel_edges))

# ==============================================================
# STEP 2: Compute neighbor stats for all variables at once
#          (replaces compute_neighbor_stats + outer loop)
# ==============================================================

compute_and_add_all_neighbor_features <- function(cell_data, panel_edges, var_names) {
  # Pre-allocate output columns
  for (var_name in var_names) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  for (var_name in var_names) {
    message("  Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Attach the neighbor's value to each edge
    vals <- cell_data[[var_name]]
    edges_with_val <- panel_edges[, .(row_i, val = vals[row_j])]
    
    # Drop edges where the neighbor value is NA
    edges_with_val <- edges_with_val[!is.na(val)]
    
    # Aggregate by row_i
    agg <- edges_with_val[, .(
      v_max  = max(val),
      v_min  = min(val),
      v_mean = mean(val)
    ), by = row_i]
    
    # Write results back into cell_data
    set(cell_data, i = agg$row_i, j = col_max,  value = agg$v_max)
    set(cell_data, i = agg$row_i, j = col_min,  value = agg$v_min)
    set(cell_data, i = agg$row_i, j = col_mean, value = agg$v_mean)
    
    message("    Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), "s")
  }
  
  return(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing all neighbor features...")
t0 <- Sys.time()
cell_data <- compute_and_add_all_neighbor_features(cell_data, panel_edges, neighbor_source_vars)
message("All neighbor features computed in ",
        round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

# ==============================================================
# STEP 3: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================
# The trained model object (e.g., `rf_model`) is loaded from disk
# and used as-is. The feature columns now exist in cell_data with
# identical names and identical numerical values, so predictions
# are identical to the original pipeline.
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | We expand the exact same `rook_neighbors_unique` nb object â€” no kernel approximation, no raster focal substitution. |
| **Same per-year scoping** | The join is on `(id, year)`, so a cell in 2005 only sees its neighbors' 2005 values. |
| **Same NA handling** | Edges where the neighbor value is `NA` are dropped before aggregation, matching `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. |
| **Same aggregation functions** | `max`, `min`, `mean` computed by `data.table` use identical base-R semantics. |
| **No-neighbor cells get NA** | Rows with no valid edges remain `NA` (pre-allocated), matching the `if (length(idx) == 0) return(c(NA, NA, NA))` path. |
| **Random Forest untouched** | The model object is loaded and called via `predict()` â€” no retraining. |

## 5. Performance Estimate

| Step | Estimated time (16 GB laptop) |
|---|---|
| Build edge list (~38.5M rows) | 30â€“90 seconds |
| Compute 5 Ã— 3 = 15 neighbor features | 3â€“8 minutes |
| **Total** | **~5â€“10 minutes** |

This represents a speedup of roughly **500â€“1000Ã—** over the original 86+ hour estimate, achieved entirely through replacing R-level per-row string operations and loops with vectorized `data.table` joins and grouped aggregations executed in C.
END_RESPONSE>>>
