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
run_id: Anthropic_F12_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions against the data facts:

### `do.call(rbind, result)` in `compute_neighbor_stats()`
- This binds ~6.46 million 3-element vectors into a matrix. While `do.call(rbind, ...)` on millions of small vectors is not ideal, it executes in seconds to low minutes on modern hardware. This is a **minor** bottleneck.

### The **true dominant bottleneck** is `build_neighbor_lookup()`

Here's why â€” examine what happens inside the `lapply` over `row_ids`:

1. **`as.character(data$id[i])` and `id_to_ref[as.character(...)]`**: Called 6.46 million times, each doing a named-vector lookup (hash lookup).
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`**: For each of the 6.46M rows, this pastes together a vector of neighbor IDs with the year. On average, each cell has ~4 rook neighbors (1,373,394 directed relationships / 344,208 cells â‰ˆ 4), so this creates ~25.8 million paste operations total.
3. **`idx_lookup[neighbor_keys]`**: Named vector lookup for each neighbor key â€” ~25.8 million hash lookups against a 6.46-million-entry named vector.

But the **critical insight** is that `build_neighbor_lookup` produces a **row-to-row** mapping that is **year-invariant in structure**. Every cell's neighbors are the same across all 28 years. The function redundantly recomputes the same spatial neighbor relationships 28 times (once per year per cell), doing `344,208 Ã— 28 = 6.46M` iterations when only `344,208` unique spatial lookups are needed. The year dimension just shifts which rows to look up, but the pattern repeats identically.

Furthermore, the entire `lapply` over 6.46 million rows with per-element `paste()` and named-vector lookups is inherently slow in R due to interpreter overhead. At even 1ms per iteration, that's 6,460 seconds (~1.8 hours) just for this function â€” and with the string operations it's likely much worse. Multiply by 1 call (it's called once), this alone could take hours.

Then `compute_neighbor_stats()` is called 5 times, each iterating over 6.46M entries. The `lapply` with per-element subsetting and `max/min/mean` is slow but not catastrophic â€” the real cost is the element-wise R-level loop over millions of entries.

**Summary of bottleneck ranking:**
1. **`build_neighbor_lookup()`**: O(6.46M) string-paste + named-vector lookups â€” the dominant bottleneck. Redundantly recomputes year-invariant spatial structure 28Ã—.
2. **`compute_neighbor_stats()` inner `lapply`**: O(6.46M Ã— 5 vars) R-level loop â€” significant but secondary.
3. **`do.call(rbind, result)`**: Minor cost relative to the above.

**I reject the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()` with its redundant per-year recomputation and millions of string-based hash lookups, not the `rbind` or list binding in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Exploit year-invariance**: Build the spatial neighbor lookup only once over the 344,208 unique cells, then expand to all years using vectorized integer arithmetic instead of string matching.
2. **Vectorize `compute_neighbor_stats()`**: Replace the R-level `lapply` with a single vectorized operation using sparse-matrix multiplication or `data.table` grouped operations.
3. **Pre-allocate output matrices** instead of `do.call(rbind, ...)`.
4. **Preserve** the trained Random Forest model and original numerical results exactly.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# Preserves the trained RF model and original numerical estimand.
# =============================================================================

library(data.table)

# ---- Step 1: Optimized neighbor lookup builder ----
# Key insight: neighbor relationships are purely spatial (year-invariant).
# Build a cell-level lookup once, then map to rows via integer arithmetic.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Unique cell IDs and their integer reference
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Unique years, sorted
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_offset <- setNames(seq_along(years) - 1L, as.character(years))
  
  # Build a mapping: for each unique cell, which row index corresponds to each year?
  # We assume data is structured so that each (id, year) pair appears at most once.
  # Create a fast lookup matrix: cell_ref x year_offset -> row_index
  
  dt[, cell_ref := id_to_ref[as.character(id)]]
  dt[, year_off := year_to_offset[as.character(year)]]
  
  n_cells <- length(id_order)
  
  # cell_year_to_row: matrix where [cell_ref, year_offset+1] = row index in data
  cell_year_to_row <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  cell_year_to_row[cbind(dt$cell_ref, dt$year_off + 1L)] <- dt$row_idx
  
  # Build spatial neighbor pairs at the cell level (done once for 344K cells)
  # neighbors is an nb object: neighbors[[i]] gives integer indices into id_order
  # that are neighbors of id_order[i].
  
  # Create edge list: (focal_cell_ref, neighbor_cell_ref)
  focal_refs <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)
  
  # Now expand to all years: for each year, map cell_ref -> row_idx
  # Build the full neighbor_lookup as a list of length nrow(data)
  # neighbor_lookup[[row_i]] = vector of row indices of neighbors in same year
  
  # But even better: build two parallel vectors (from_row, to_row) for all
  # neighbor relationships, then use data.table for grouped stats.
  
  # For each year, the focal rows and neighbor rows:
  from_rows <- vector("list", n_years)
  to_rows   <- vector("list", n_years)
  
  for (y in seq_len(n_years)) {
    focal_row_idx    <- cell_year_to_row[focal_refs, y]
    neighbor_row_idx <- cell_year_to_row[neighbor_refs, y]
    
    # Remove pairs where either focal or neighbor doesn't exist in this year
    valid <- !is.na(focal_row_idx) & !is.na(neighbor_row_idx)
    from_rows[[y]] <- focal_row_idx[valid]
    to_rows[[y]]   <- neighbor_row_idx[valid]
  }
  
  from_all <- unlist(from_rows, use.names = FALSE)
  to_all   <- unlist(to_rows, use.names = FALSE)
  
  list(
    from_row = from_all,
    to_row   = to_all,
    n_rows   = nrow(data)
  )
}

# ---- Step 2: Optimized neighbor stats computation ----
# Uses data.table grouped aggregation â€” fully vectorized, no R-level loop.

compute_neighbor_stats_fast <- function(data, edge_info, var_name) {
  
  vals <- data[[var_name]]
  n <- edge_info$n_rows
  
  # Build a data.table of edges with the neighbor's value
  edge_dt <- data.table(
    focal   = edge_info$from_row,
    nb_val  = vals[edge_info$to_row]
  )
  
  # Remove edges where neighbor value is NA
  edge_dt <- edge_dt[!is.na(nb_val)]
  
  # Grouped aggregation
  stats <- edge_dt[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = focal]
  
  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  
  out_max[stats$focal]  <- stats$nb_max
  out_min[stats$focal]  <- stats$nb_min
  out_mean[stats$focal] <- stats$nb_mean
  
  cbind(out_max, out_min, out_mean)
}

# ---- Step 3: Wrapper that adds columns (mirrors original behavior) ----

compute_and_add_neighbor_features_fast <- function(data, var_name, edge_info) {
  mat <- compute_neighbor_stats_fast(data, edge_info, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- mat[, 3]
  
  data
}

# ---- Step 4: Run the optimized pipeline ----

# Build edge list once (replaces build_neighbor_lookup)
# Expected time: seconds to ~1 minute instead of hours
edge_info <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_info)
}

# The trained Random Forest model is untouched.
# The numerical output (max, min, mean of neighbor values) is identical.
```

---

## Performance Comparison

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops + hash lookups) | ~seconds (vectorized integer matrix + 28 loop iterations over pre-built edge list) | **~1000Ã—+** |
| `compute_neighbor_stats` (Ã—5 vars) | ~hours (6.46M R-level `lapply` per var) | ~seconds (`data.table` grouped aggregation, vectorized) | **~100Ã—+** |
| `do.call(rbind, ...)` | ~minutes | Eliminated (direct vector assignment) | N/A |
| **Total estimated** | **86+ hours** | **< 5 minutes** | **~1000Ã—** |

The key optimizations are:
1. **Year-invariance exploitation**: 28Ã— reduction in spatial lookups.
2. **Elimination of string operations**: No `paste()`, no named-vector hash lookups â€” pure integer indexing.
3. **Vectorized aggregation**: `data.table` grouped `max/min/mean` replaces 6.46M R-level function calls.
4. **No list-to-matrix conversion**: Output vectors are pre-allocated and filled by reference.
END_RESPONSE>>>
