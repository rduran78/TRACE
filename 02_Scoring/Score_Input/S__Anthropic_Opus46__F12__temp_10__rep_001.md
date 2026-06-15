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
run_id: Anthropic_F12_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and "repeated list binding" inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. Here's why:

**`compute_neighbor_stats()` is actually reasonably efficient.** The `lapply` produces a list of fixed-length numeric vectors (length 3), and `do.call(rbind, ...)` on a list of equal-length vectors is a single matrix constructionâ€”an O(n) operation. There is no repeated `rbind` growth (no Schlemiel-the-Painter pattern). For ~6.46M rows, this takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.** Specifically:

1. **`paste()` and named-vector lookups at scale:** `idx_lookup` is a named integer vector with ~6.46 million entries. Named vector lookups in R use linear hashing that degrades significantly at this scale. Every call to `idx_lookup[neighbor_keys]` performs repeated name-based searches against a 6.46M-element named vector.

2. **The `lapply` over all ~6.46M rows:** For each of the 6.46M rows, the function (a) converts the id to character, (b) looks up `ref_idx` in a named vector, (c) retrieves neighbor cell IDs, (d) pastes them with the year to form keys, and (e) looks those keys up in the 6.46M-element named lookup. That's ~6.46M iterations, each doing string concatenation and hash-table lookups against a massive named vector. With ~1.37M neighbor relationships spread over 28 years, the total number of key lookups is enormous.

3. **Redundant recomputation across years:** The neighbor *structure* is identical across all 28 years for each cell. But the lookup is rebuilt per cell-year row, repeating the same neighbor-ID retrieval 28 times per cell. The only thing that changes is the year, yet the entire pipeline processes each of the 6.46M rows independently.

**Quantitative estimate:** ~6.46M iterations Ã— (string operations + named vector lookups into a 6.46M-entry table) â‰ˆ the dominant cost. The `compute_neighbor_stats` function, by contrast, just indexes a numeric vector by integer positionâ€”which is nearly instantaneous per row.

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins or environment-based hashing.** `data.table` keyed joins are orders of magnitude faster than named-vector lookup at millions of keys.

2. **Separate the spatial and temporal dimensions.** Build the neighbor lookup at the *cell level* (344K cells), not the *cell-year level* (6.46M rows). Then expand to cell-year via a vectorized integer-arithmetic mapping, eliminating all `paste()` key construction.

3. **Vectorize `compute_neighbor_stats`.** Replace the per-row `lapply` with a single grouped aggregation using `data.table`, computing max/min/mean over neighbor values in one pass.

4. **Preserve the trained Random Forest model and original numerical estimand.** The output columns are identical in name, type, and valueâ€”only the computation path changes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup + compute_neighbor_stats
# ==============================================================================

compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {
  
  # Convert to data.table for speed (non-destructive; we return a data.frame at the end if needed)
  dt <- as.data.table(cell_data)
  
  # ---- Step 1: Build cell-level neighbor edge list (spatial only, ~344K cells) ----
  # id_order is the vector of cell IDs in the order matching the nb object.
  # neighbors is the spdep::nb list: neighbors[[i]] gives integer indices into id_order.
  
  message("Building cell-level edge list...")
  
  # Pre-allocate edge list vectors
  n_edges <- sum(lengths(neighbors))  # total directed neighbor pairs
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    n_nb <- length(nb_i)
    from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
    to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_i]
    pos <- pos + n_nb
  }
  
  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  edges <- data.table(from_id = from_id, to_id = to_id)
  
  message(sprintf("Edge list: %s directed neighbor pairs.", format(nrow(edges), big.mark = ",")))
  
  # ---- Step 2: Build row-index lookup via data.table keyed join ----
  # Map (id, year) -> row index in dt
  dt[, .row_idx := .I]
  
  # We need unique years
  years <- sort(unique(dt$year))
  
  # ---- Step 3: For each variable, compute neighbor stats via vectorized join ----
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing neighbor variable: %s ...", var_name))
    
    # Expand edges Ã— years: for each (from_id, to_id) pair and each year,
    # we need the neighbor's value.
    # Instead of expanding the full cross product (which could be huge),
    # we join edges against the data twice: once to get the focal row, once to get neighbor value.
    
    # Subset to just the columns we need for this variable
    val_dt <- dt[, .(id, year, val = get(var_name), .row_idx)]
    setkey(val_dt, id, year)
    
    # For each edge (from_id -> to_id), for each year present in the data for from_id,
    # look up the neighbor (to_id) value in that same year.
    
    # Join edges with focal cell's years
    # focal_rows: all (from_id, year) combinations that exist in the data
    focal_rows <- val_dt[, .(from_id = id, year, focal_row_idx = .row_idx)]
    setkey(focal_rows, from_id)
    
    # Merge edges with focal rows to get (from_id, to_id, year, focal_row_idx)
    # This is the key expansion: each edge is repeated for each year the focal cell appears
    setkey(edges, from_id)
    expanded <- edges[focal_rows, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded has columns: from_id, to_id, year, focal_row_idx
    
    message(sprintf("  Expanded edge-year pairs: %s", format(nrow(expanded), big.mark = ",")))
    
    # Now look up the neighbor's value: join on (to_id, year) -> val
    neighbor_vals <- val_dt[, .(to_id = id, year, neighbor_val = val)]
    setkey(neighbor_vals, to_id, year)
    setkey(expanded, to_id, year)
    
    expanded <- neighbor_vals[expanded, on = c("to_id", "year"), nomatch = NA]
    # expanded now has: to_id, year, neighbor_val, from_id, focal_row_idx
    
    # Remove NA neighbor values before aggregation
    expanded_clean <- expanded[!is.na(neighbor_val)]
    
    # ---- Step 4: Grouped aggregation (the actual max/min/mean) ----
    stats <- expanded_clean[, .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ), by = focal_row_idx]
    
    # ---- Step 5: Write results back into dt ----
    # Initialize with NA
    max_col <- paste0("nb_max_", var_name)
    min_col <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]
    
    message(sprintf("  Done: %s", var_name))
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  # Return as data.frame if the input was a data.frame (preserves downstream compatibility)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names (nb_max_ntl, nb_min_ntl, nb_mean_ntl, ...) must match
# whatever names the original compute_and_add_neighbor_features() produced.
# Adjust the naming pattern above if the original used a different convention.
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | Named vector (6.46M entries), string key lookup | `data.table` keyed join (hash + radix) |
| **Key construction** | `paste(id, year)` per row in `lapply` (6.46M calls) | Eliminated; join on integer `(id, year)` columns |
| **Neighbor resolution** | Per-row `lapply` with R-level loop (6.46M iterations) | Vectorized edge-list expansion + single grouped aggregation |
| **Aggregation** | Per-row `max/min/mean` inside `lapply` | `data.table` grouped `max/min/mean` (C-level, single pass) |
| **Estimated time** | ~86+ hours | **~2â€“10 minutes** on the same laptop |

The memory footprint of the expanded edge table is approximately `1.37M edges Ã— 28 years Ã— ~40 bytes/row â‰ˆ 1.5 GB`, which fits within the 16 GB constraint. If memory is tight, the loop over `neighbor_source_vars` already processes one variable at a time, so the peak is only one expanded table at a time.

**Column naming note:** The code above produces columns named `nb_max_ntl`, `nb_min_ntl`, `nb_mean_ntl`, etc. If the original `compute_and_add_neighbor_features()` used a different naming convention (e.g., `ntl_nb_max`), adjust the `paste0()` pattern accordingly so the trained Random Forest model's expected feature names are matched exactly. The numerical values are identical to the original implementation.
END_RESPONSE>>>
