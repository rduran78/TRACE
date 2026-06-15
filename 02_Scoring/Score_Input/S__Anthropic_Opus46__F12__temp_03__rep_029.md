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
run_id: Anthropic_F12_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions against the data facts:

### `compute_neighbor_stats()` â€” Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not free â€” it's O(n) allocation and copy. And the per-element `lapply` does redundant subsetting. But this function operates on a **pre-built lookup** and does only simple numeric operations (max, min, mean) on small neighbor sets. For 5 variables, that's 5 Ã— 6.46M iterations of trivial arithmetic. This is on the order of minutes, not hours. The `do.call(rbind, ...)` on 6.46M rows is slow (~seconds to low minutes) but not 86 hours slow.

### `build_neighbor_lookup()` â€” The True Bottleneck

This is where the 86+ hours lives. Here's why:

1. **`lapply` over 6.46 million rows**, each iteration doing:
   - `as.character(data$id[i])` â€” character conversion per row
   - `id_to_ref[as.character(...)]` â€” named vector lookup (hash lookup per row)
   - `id_order[neighbors[[ref_idx]]]` â€” subsetting neighbor IDs
   - **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” string concatenation for every neighbor of every row. With ~1.37M directed neighbor relationships spread across 344K cells, the average cell has ~4 rook neighbors. Across 28 years, that's 6.46M Ã— ~4 = **~25.8 million `paste` operations**, each producing a string, each then looked up in a **named vector of 6.46 million entries** (`idx_lookup`).
   - Named vector lookup in R with 6.46M names is **not O(1)** in practice â€” R's internal hashing for named vectors degrades at this scale. Each lookup into `idx_lookup` with multiple keys triggers repeated hash probes across a 6.46M-entry hash table, **6.46 million times**.

2. **The critical insight**: The lookup is being built **row-by-row** for 6.46M rows, but the neighbor structure is **cell-level** (344K cells) and is simply **repeated identically across all 28 years**. The function redundantly recomputes the same neighbor mapping 28 times per cell.

**Verdict: REJECT the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()`, specifically the O(6.46M) loop with per-iteration string pasting and named-vector hash lookups into a 6.46M-entry table. `compute_neighbor_stats()` is a secondary, much smaller cost.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` entirely**: Exploit the panel structure. Compute neighbor indices at the **cell level** (344K cells, not 6.46M rows), then broadcast across years using integer arithmetic instead of string pasting/hashing.

2. **Replace `do.call(rbind, ...)` in `compute_neighbor_stats`** with a pre-allocated matrix and direct vectorized computation.

3. **Use `data.table` for fast keyed joins** instead of named-vector lookups.

4. **Preserve the trained Random Forest model** â€” we only change feature-engineering speed, not the features themselves. The numerical output is identical.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Key insight: neighbor relationships are defined at the CELL level (344K cells)
# and are identical across all 28 years. We compute cell-level neighbor indices
# once, then map to row indices using integer arithmetic, not string hashing.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # --- Step 1: Build a cell-level mapping ---
  # Map each unique cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Get the unique years in sorted order and map them to integer indices
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_yidx <- setNames(seq_along(years_sorted), as.character(years_sorted))
  
  # Get unique cell IDs in the order they appear, and map to integer cell index
  # We need a mapping: cell_id -> set of row indices for that cell across years
  # If data is sorted by (id, year), this is trivial. Let's not assume that.
  
  # Create a keyed lookup: for each (cell_id, year) -> row_idx
  setkey(dt, id, year)
  # Fast lookup table
  cell_year_to_row <- dt[, .(row_idx = row_idx[1]), by = .(id, year)]
  setkey(cell_year_to_row, id, year)
  
  # --- Step 2: Build cell-level neighbor list (only 344K entries) ---
  # For each cell c, get the IDs of its rook neighbors
  n_cells <- length(id_order)
  
  # Precompute: for each cell index in id_order, which other cell indices are neighbors?
  # neighbors is an nb object: neighbors[[i]] gives integer indices into id_order
  # We need to map those to actual cell IDs, then to row indices per year.
  
  # Build an edge list at the cell level
  # Each cell i has neighbors[[i]] as indices into id_order
  # Expand to (focal_cell_id, neighbor_cell_id) pairs
  
  cat("Building cell-level neighbor edge list...\n")
  
  focal_indices <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_indices <- unlist(neighbors)
  
  # Edge list with actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  
  cat("  Edge list:", nrow(edge_dt), "directed edges\n")
  
  # --- Step 3: For each (focal_id, year), find row indices of all neighbors ---
  # Cross join edges with years
  cat("Crossing edges with years...\n")
  
  years_dt <- data.table(year = years_sorted)
  edge_year <- edge_dt[, CJ_val := 1][
    years_dt[, CJ_val := 1], 
    on = "CJ_val", 
    allow.cartesian = TRUE
  ]
  edge_year[, CJ_val := NULL]
  
  # Now edge_year has columns: focal_id, neighbor_id, year
  # Look up the row index for each (neighbor_id, year)
  cat("Joining to get neighbor row indices...\n")
  
  setnames(cell_year_to_row, c("id", "year", "row_idx"), c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(cell_year_to_row, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)
  
  edge_year <- cell_year_to_row[edge_year, on = .(neighbor_id, year), nomatch = NA]
  
  # Drop NAs (neighbor cell-year combinations not present in data)
  edge_year <- edge_year[!is.na(neighbor_row_idx)]
  
  # --- Step 4: Look up the focal row index ---
  # Reset cell_year_to_row names for focal lookup
  setnames(cell_year_to_row, 
           c("neighbor_id", "year", "neighbor_row_idx"), 
           c("focal_id", "year", "focal_row_idx"))
  setkey(cell_year_to_row, focal_id, year)
  setkey(edge_year, focal_id, year)
  
  edge_year <- cell_year_to_row[edge_year, on = .(focal_id, year), nomatch = NA]
  edge_year <- edge_year[!is.na(focal_row_idx)]
  
  # --- Step 5: Build the lookup as a list indexed by row ---
  cat("Assembling lookup list...\n")
  
  n_rows <- nrow(dt)
  setkey(edge_year, focal_row_idx)
  
  # Split neighbor_row_idx by focal_row_idx
  lookup_list <- vector("list", n_rows)
  
  # Use split for efficiency
  split_result <- split(edge_year$neighbor_row_idx, edge_year$focal_row_idx)
  
  # Fill in the lookup list
  filled_indices <- as.integer(names(split_result))
  for (j in seq_along(filled_indices)) {
    lookup_list[[filled_indices[j]]] <- as.integer(split_result[[j]])
  }
  
  # Fill remaining with integer(0)
  empty_indices <- setdiff(seq_len(n_rows), filled_indices)
  for (j in empty_indices) {
    lookup_list[[j]] <- integer(0)
  }
  
  cat("Neighbor lookup built.\n")
  return(lookup_list)
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats
# =============================================================================
# Avoids do.call(rbind, ...) and uses pre-allocated matrix.
# For even more speed, uses the edge list directly for vectorized aggregation.

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    result_mat[i, 1] <- max(nv)
    result_mat[i, 2] <- min(nv)
    result_mat[i, 3] <- mean(nv)
  }
  
  result_mat
}

# =============================================================================
# EVEN FASTER: Fully vectorized stats using data.table aggregation
# =============================================================================
# This avoids the R-level loop over 6.46M rows entirely.

compute_neighbor_stats_vectorized <- function(data, edge_year_dt, var_name) {
  # edge_year_dt has columns: focal_row_idx, neighbor_row_idx
  # Compute stats by focal_row_idx using vectorized grouping
  
  dt <- data.table(
    focal_row_idx    = edge_year_dt$focal_row_idx,
    neighbor_val     = data[[var_name]][edge_year_dt$neighbor_row_idx]
  )
  
  # Remove NA neighbor values
  dt <- dt[!is.na(neighbor_val)]
  
  # Aggregate
  stats <- dt[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = focal_row_idx]
  
  # Map back to full row set
  n_rows <- nrow(data)
  result_mat <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  result_mat[stats$focal_row_idx, 1] <- stats$nb_max
  result_mat[stats$focal_row_idx, 2] <- stats$nb_min
  result_mat[stats$focal_row_idx, 3] <- stats$nb_mean
  
  result_mat
}


# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features (preserves original column names)
# =============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
  data
}


# =============================================================================
# FULL PIPELINE (drop-in replacement)
# =============================================================================

# Build the optimized neighbor lookup (runs once, ~minutes instead of ~days)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched â€” use it for prediction as before.
# All numerical outputs are identical to the original implementation.
```

---

## Summary

| Aspect | Colleague's Diagnosis | Actual Diagnosis |
|---|---|---|
| **Claimed bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | **`build_neighbor_lookup()`**: 6.46M-iteration loop with per-row string pasting and hash lookups into a 6.46M-entry named vector |
| **Why it's wrong** | `compute_neighbor_stats` does trivial arithmetic; `do.call(rbind, ...)` on 6.46M Ã— 3 is slow but ~seconds/minutes, not hours | The lookup rebuilds identical neighbor mappings 28Ã— (once per year per cell) via expensive string operations and O(n) named-vector hashing |
| **Estimated speedup** | Fixing `rbind` alone: ~minutes saved | Fixing `build_neighbor_lookup`: **~85+ hours saved** (from ~86h to ~5â€“15 minutes) |
| **Optimization** | Pre-allocate matrix (minor) | Vectorize at cell level, broadcast across years via `data.table` keyed joins, eliminate per-row string operations |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical equivalence** | Preserved âœ“ | Preserved âœ“ |
END_RESPONSE>>>
