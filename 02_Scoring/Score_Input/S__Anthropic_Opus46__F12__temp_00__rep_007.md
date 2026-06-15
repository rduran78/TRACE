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
run_id: Anthropic_F12_T00_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. Here's why:

**`compute_neighbor_stats()` is actually efficient.** It uses `lapply` to build a list of fixed-length numeric vectors (length 3), then calls `do.call(rbind, ...)` once. For ~6.46M rows, this is a single matrix-binding operation on pre-allocated 3-element vectors â€” it completes in seconds. There is no "repeated list binding" inside the function; each iteration returns a fixed-size vector, and binding happens once at the end. Even across 5 variables, this function accounts for trivial runtime.

**The true bottleneck is `build_neighbor_lookup()`.** This function is called once but executes an `lapply` over **every row** (~6.46 million rows), and inside each iteration it:

1. Performs `as.character()` conversion and a named-vector lookup (`id_to_ref[...]`).
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” but critically, **the same cell's neighbors are looked up identically for all 28 years of that cell**. This means the neighbor-cell-ID resolution is redundantly computed ~28 times per cell (28 Ã— 344,208 = 9,637,824 redundant lookups).
3. Constructs `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string concatenation for every neighbor of every row.
4. Performs named-vector lookup via `idx_lookup[neighbor_keys]` â€” named character vector lookup on a 6.46M-element vector is **O(n) hash probing per call**, repeated 6.46M times.

The string operations (`paste`, `as.character`) and named-vector lookups on millions of keys, repeated for every single row, dominate the runtime. The redundancy across years (same cell, same neighbors, just different year) is the deepest structural bottleneck.

## Optimization Strategy

1. **Eliminate per-row string operations entirely.** Replace the character-key lookup with integer-indexed lookup using a matrix or `data.table` join.
2. **Exploit the panel structure.** Compute neighbor cell IDs once per spatial cell (344K), not once per cell-year (6.46M). Then expand across years using integer arithmetic.
3. **Vectorize `compute_neighbor_stats`.** Replace the per-row `lapply` with grouped vectorized operations using `data.table` or pre-indexed matrix operations.
4. **Preserve the trained Random Forest model** â€” we only change feature-engineering speed, not the features themselves or the model.

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Key insight: neighbor relationships are spatial (cell-to-cell), not temporal.
# We resolve neighbor cell IDs once per cell, then map to row indices via
# integer-indexed lookup instead of string-keyed lookup.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  n_cells <- length(id_order)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_along(id_order)
  # If IDs are not contiguous positive integers, use a hash:
  # But typically grid cell IDs are. If not, fall back:
  # id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Unique years in sorted order
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_int <- integer(max(years))
  year_to_int[years] <- seq_along(years)
  
  # Build a fast row-index matrix: row_matrix[cell_ref, year_int] = row index in data
  # cell_ref = position in id_order; year_int = position in years vector
  # This replaces ALL string-based lookups.
  
  cell_refs <- id_to_ref[dt$id]
  year_ints <- year_to_int[dt$year]
  
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(cell_refs, year_ints)] <- dt$row_idx
  
  # Now build neighbor_lookup: for each row i, find row indices of neighbors
  # in the same year.
  # 
  # Instead of looping over 6.46M rows, loop over 344K cells and vectorize
  # across years.
  
  # Pre-compute neighbor refs for each cell (once per cell, not per cell-year)
  neighbor_refs_per_cell <- vector("list", n_cells)
  for (c_ref in seq_len(n_cells)) {
    nb <- neighbors[[c_ref]]
    if (length(nb) == 0 || (length(nb) == 1 && nb[1] == 0)) {
      neighbor_refs_per_cell[[c_ref]] <- integer(0)
    } else {
      neighbor_refs_per_cell[[c_ref]] <- as.integer(nb)
    }
  }
  
  # Allocate the full lookup list (6.46M entries)
  n_rows <- nrow(dt)
  neighbor_lookup <- vector("list", n_rows)
  
  # Fill by iterating over cells (344K) Ã— years (28) instead of rows with

  # string ops. For each cell, for each year it appears in, gather neighbor
  # row indices from row_matrix.
  
  for (c_ref in seq_len(n_cells)) {
    nb_refs <- neighbor_refs_per_cell[[c_ref]]
    
    for (y_int in seq_len(n_years)) {
      my_row <- row_matrix[c_ref, y_int]
      if (is.na(my_row)) next
      
      if (length(nb_refs) == 0) {
        neighbor_lookup[[my_row]] <- integer(0)
      } else {
        nb_rows <- row_matrix[nb_refs, y_int]
        nb_rows <- nb_rows[!is.na(nb_rows)]
        neighbor_lookup[[my_row]] <- nb_rows
      }
    }
  }
  
  neighbor_lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# =============================================================================
# The original is actually not terrible, but we can speed it up further
# by avoiding per-element lapply and using vectorized subsetting.

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Pre-allocate output matrix
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  
  # Flatten for vectorized computation
  # Build group indices and neighbor value vector
  lens <- lengths(neighbor_lookup)
  has_neighbors <- lens > 0
  
  if (any(has_neighbors)) {
    # Unlist all neighbor indices at once
    all_idx <- unlist(neighbor_lookup[has_neighbors], use.names = FALSE)
    all_vals <- vals[all_idx]
    
    # Create group vector
    grp <- rep(which(has_neighbors), lens[has_neighbors])
    
    # Use data.table for fast grouped aggregation
    agg_dt <- data.table(grp = grp, val = all_vals)
    agg_dt <- agg_dt[!is.na(val)]
    
    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(
        mx = max(val),
        mn = min(val),
        mn2 = mean(val)
      ), by = grp]
      
      out[stats$grp, 1] <- stats$mx
      out[stats$grp, 2] <- stats$mn
      out[stats$grp, 3] <- stats$mn2
    }
  }
  
  out
}

# =============================================================================
# Wrapper that matches original interface
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  
  # Preserve original column naming convention
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]
  
  data
}

# =============================================================================
# OPTIMIZED Outer Loop (drop-in replacement)
# =============================================================================
# Build lookup once (now ~minutes instead of ~days)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched.
# The numerical output (max, min, mean of neighbor values) is identical.
```

## Summary of Speedup Sources

| Component | Original Cost | Optimized Cost | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + 6.46M named-vector lookups on 6.46M-key vector | 344K cell iterations Ã— 28 year integer matrix lookups | **~500â€“1000Ã—** |
| `compute_neighbor_stats` | 6.46M `lapply` calls with per-element `max/min/mean` | Vectorized `unlist` + `data.table` grouped aggregation | **~10â€“50Ã—** |
| **Total estimated runtime** | **86+ hours** | **~5â€“15 minutes** | |

**The colleague was wrong.** The `do.call(rbind, ...)` in `compute_neighbor_stats` is a one-shot bind of a pre-built list â€” it's fast. The true bottleneck is `build_neighbor_lookup`, which performs millions of redundant string constructions and O(1)-amortized-but-high-constant-factor named-vector lookups, all while ignoring the panel structure that makes 96.5% of the work (28âˆ’1 out of 28 year-copies per cell) purely redundant.
END_RESPONSE>>>
