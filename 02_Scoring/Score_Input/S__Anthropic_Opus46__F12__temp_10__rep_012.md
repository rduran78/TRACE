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
run_id: Anthropic_F12_T10_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (5 times total) and runs in seconds-to-minutes at most. There is no "repeated list binding" inside `compute_neighbor_stats()`â€”it uses a single `lapply` that returns a pre-allocated list, then one `rbind`. This is not the bottleneck.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** The `lapply` inside `build_neighbor_lookup` iterates over every row (~6.46M). For each row, it:
   - Calls `as.character(data$id[i])` â€” character conversion.
   - Indexes into `id_to_ref` â€” named character vector lookup (linear hash probe on ~344K names).
   - Extracts `neighbors[[ref_idx]]` â€” a list index.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” creates a character vector of neighbor keys (avg ~4 neighbors per cell for rook contiguity â†’ ~4 strings).
   - Indexes into `idx_lookup` â€” a named vector with **6.46M names**, meaning each lookup is a repeated hash-table probe against a massive named vector.

2. **Scale analysis:** ~6.46M iterations Ã— ~4 neighbor lookups each = ~25.8M named-vector lookups into a 6.46M-element named vector. Named vector lookup in R is O(n) in the worst case (no true hash table) or at best a slow hash. This single function likely consumes **>95% of the 86+ hour runtime**.

3. `compute_neighbor_stats()` by contrast is pure numeric indexing (`vals[idx]`) plus simple `max/min/mean` â€” extremely fast even over 6.46M rows.

**Conclusion:** The deep bottleneck is the repeated string-key construction and named-vector lookup in `build_neighbor_lookup()`, executed 6.46 million times against a 6.46M-key lookup table.

---

## Optimization Strategy

1. **Replace the per-row `lapply` in `build_neighbor_lookup` with a vectorized, integer-keyed approach.** Instead of building string keys and probing a named vector, build an integer-indexed lookup matrix/hash using `data.table` or `match()` on integer-encoded keys. Since `(id, year)` pairs are unique row identifiers, encode them as integers and use direct indexing.

2. **Exploit the panel structure:** Every cell appears once per year. So for a given year, the neighbor rows are simply the same neighbors' rows in that year. We can build the lookup as a block operation over years rather than row-by-row.

3. **Replace `do.call(rbind, ...)` with a pre-allocated matrix** for marginal improvement.

4. **Preserve the trained RF model and original numerical estimand** â€” we only change the speed of feature construction, not the values produced.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Strategy: Instead of per-row string pasting + named-vector lookup,
# use data.table keyed joins to vectorize everything.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # --- Step 1: Create a data.table mapping (id, year) -> row_index
  dt <- data.table(
    id    = data$id,
    year  = data$year,
    row_i = seq_len(nrow(data))
  )
  
  # --- Step 2: Build an edge list of (focal_id, neighbor_id) from the nb object
  #     id_order[k] is the cell id for position k in the nb object
  #     neighbors[[k]] gives the positions of k's neighbors
  
  n_cells <- length(id_order)
  # Pre-compute lengths to allocate at once
  edge_lengths <- vapply(neighbors, length, integer(1))
  total_edges  <- sum(edge_lengths)
  
  focal_ids    <- rep(id_order, times = edge_lengths)
  neighbor_ids <- id_order[unlist(neighbors, use.names = FALSE)]
  
  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  
  # --- Step 3: For each row in data, find its neighbors' row indices
  #     A row is identified by (id, year). Its neighbors share the same year
  #     but have neighbor_id as their id.
  
  # Join edges onto data rows to get (row_i_focal, neighbor_id, year)
  # Then join again to find the neighbor's row index in that year.
  
  # First: attach focal row info
  setkey(dt, id)
  focal_dt <- dt[, .(focal_row = row_i, focal_id = id, year)]
  
  # Cross with edges: for each focal row, get its neighbor_ids
  setkey(edges, focal_id)
  # Use a merge: focal_dt joins edges on focal_id
  # Result: (focal_row, year, neighbor_id)
  expanded <- edges[focal_dt, on = .(focal_id), allow.cartesian = TRUE,
                    nomatch = NA,
                    .(focal_row, year, neighbor_id)]
  
  # Remove rows where there were no neighbors (NA neighbor_id)
  expanded <- expanded[!is.na(neighbor_id)]
  
  # Now find the row index of each (neighbor_id, year) pair
  setkey(dt, id, year)
  setnames(dt, "id", "nid_join")
  
  expanded[, neighbor_row := dt[.(expanded$neighbor_id, expanded$year),
                                  row_i, nomatch = NA]]
  
  # Restore dt column name
  setnames(dt, "nid_join", "id")
  
  # Remove unmatched

  expanded <- expanded[!is.na(neighbor_row)]
  
  # --- Step 4: Convert to the list-of-integer-vectors format expected downstream
  #     Index: focal_row (1..nrow(data))
  
  n_rows <- nrow(data)
  
  # Order by focal_row for split
  setkey(expanded, focal_row)
  
  # Split neighbor_row by focal_row
  lookup_list <- vector("list", n_rows)
  
  # Use split â€” very fast on integer keys
  splitted <- split(expanded$neighbor_row, expanded$focal_row)
  
  # splitted names are character versions of focal_row indices that had neighbors
  idx_with_neighbors <- as.integer(names(splitted))
  lookup_list[idx_with_neighbors] <- splitted
  
  # Fill remaining with integer(0)
  empty_idx <- setdiff(seq_len(n_rows), idx_with_neighbors)
  lookup_list[empty_idx] <- list(integer(0))
  
  lookup_list
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================================
# Replace do.call(rbind, ...) with pre-allocated matrix and vectorized ops

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n    <- length(neighbor_lookup)
  
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1] <- max(nv)
    out[i, 2] <- min(nv)
    out[i, 3] <- mean(nv)
  }
  
  out
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
# Wrapper that mirrors the original function's behavior

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  
  data
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================

# Build the optimized lookup (runs once â€” minutes instead of tens of hours)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched.
# The numerical estimand is preserved â€” identical values are produced.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations Ã— string key construction + named-vector probe against 6.46M keys | Eliminated |
| **Lookup construction** | `lapply` + `paste` + named vector (~86+ hrs) | `data.table` keyed join + `split` (~2â€“10 min) |
| **Stat computation** | `lapply` + `do.call(rbind,...)` (fast, not bottleneck) | Pre-allocated matrix loop (marginally faster) |
| **Numerical output** | Identical | Identical |
| **RF model** | Preserved | Preserved |
| **Expected total time** | 86+ hours | ~10â€“30 minutes |

The colleague misidentified the bottleneck. The `do.call(rbind, ...)` in `compute_neighbor_stats` operates on a pre-built list and is a minor cost. The catastrophic cost is the **O(N Ã— k) string-based named-vector lookup in `build_neighbor_lookup`**, where N = 6.46M rows and each probe hits a 6.46M-entry named vector. Replacing this with `data.table` keyed equi-joins eliminates the bottleneck entirely.
END_RESPONSE>>>
