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
run_id: Anthropic_F12_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (only 5 calls total) and each call binds fixed-length 3-element vectors â€” this is relatively cheap.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookups over 6.46M rows:** `build_neighbor_lookup` creates `idx_lookup` â€” a named vector of ~6.46 million entries keyed by `paste(id, year)`. It then calls `lapply` over all ~6.46M rows, and *inside each iteration* it:
   - Calls `as.character()` on a single id â†’ character conversion 6.46M times.
   - Looks up `id_to_ref[as.character(...)]` â€” a named-vector lookup (linear hash probe) 6.46M times.
   - Retrieves `neighbors[[ref_idx]]` to get neighbor cell IDs (typically ~4 for rook contiguity).
   - Calls `paste(neighbor_cell_ids, data$year[i], sep="_")` â€” string concatenation inside the loop, ~4Ã— per row = ~25.8M paste operations.
   - Performs `idx_lookup[neighbor_keys]` â€” named-vector lookup of ~4 keys against a 6.46M-length named vector, 6.46M times.

2. **Complexity:** Named vector lookup in R via `[` on a character-named vector is **O(n)** in the worst case per probe (R uses hashing, but the hash table is rebuilt/probed repeatedly). With ~6.46M rows Ã— ~4 neighbor lookups each = ~25.8M hash probes against a 6.46M-entry hash. This dwarfs everything in `compute_neighbor_stats()`.

3. **`compute_neighbor_stats` is comparatively cheap:** Once `neighbor_lookup` exists, each call simply does `vals[idx]` (integer subsetting â€” very fast), computes `max/min/mean` on ~4 values, and returns a length-3 vector. The `do.call(rbind, ...)` on 6.46M length-3 vectors takes seconds, not hours.

**Conclusion:** `build_neighbor_lookup()` is the dominant bottleneck, likely consuming 80%+ of the 86-hour runtime. The repeated string construction and named-vector lookups inside a 6.46M-iteration `lapply` are the root cause.

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins or environment-based hashing.** Use `match()` or `data.table` keyed joins instead of named-vector character lookups.

2. **Vectorize `build_neighbor_lookup` entirely.** Instead of looping row-by-row, expand the neighbor relationships into a full edge list (cell_i, cell_j), join on year to create (row_i, row_j) pairs, then use `data.table` grouped operations to compute neighbor stats directly â€” **eliminating `build_neighbor_lookup` and `compute_neighbor_stats` as separate steps.**

3. **Compute all 5 variables' neighbor stats in one pass** over the edge list, avoiding redundant work.

This reduces the problem from 6.46M R-level loop iterations with string operations to a single vectorized join + grouped aggregation.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Optimized pipeline: replaces build_neighbor_lookup(),
# compute_neighbor_stats(), and the outer for-loop entirely.
# ---------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Build a full directed edge list from the nb object ---
  # rook_neighbors_unique is a list of integer vectors (spdep nb object).
  # neighbors[[i]] gives the indices (into id_order) of neighbors of id_order[i].

  # Pre-allocate edge list vectors
  n_edges <- sum(lengths(rook_neighbors_unique))  # ~1,373,394
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(rook_neighbors_unique)) {
    nb_i <- rook_neighbors_unique[[i]]
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

  # --- Step 2: Create a row-index mapping: (id, year) -> row_index ---
  dt[, row_idx := .I]

  # --- Step 3: For each year, expand edges into (row_i, row_j) pairs ---
  # We need: for each row (from_id, year), find all neighbor rows (to_id, same year).
  # Strategy: join edges with dt on id to get (from_row, to_id, year),
  # then join again to get to_row.

  # Keyed lookup tables
  id_year_to_row <- dt[, .(id, year, row_idx)]
  setkey(id_year_to_row, id, year)

  # Expand: each edge Ã— each year that from_id appears in
  # Join edges with id_year_to_row on from_id = id
  setnames(id_year_to_row, "id", "from_id")
  setkey(edges, from_id)
  setkey(id_year_to_row, from_id)

  # This gives us: for each (from_id, to_id) edge, all years where from_id has data
  edge_year <- edges[id_year_to_row, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: from_id, to_id, year, row_idx (= from_row_idx)
  setnames(edge_year, "row_idx", "from_row_idx")

  # Now join to get to_row_idx: match (to_id, year)
  setnames(id_year_to_row, c("to_id", "year", "to_row_idx"))
  setkey(id_year_to_row, to_id, year)
  setkey(edge_year, to_id, year)

  edge_rows <- id_year_to_row[edge_year, on = c("to_id", "year"), nomatch = 0L]
  # Columns: to_id, year, to_row_idx, from_id, from_row_idx

  # --- Step 4: Compute neighbor stats for all variables at once ---
  # For each from_row_idx, gather neighbor values (at to_row_idx), compute max/min/mean.

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Extract neighbor values via integer indexing (very fast)
    edge_rows[, nval := dt[[var_name]][to_row_idx]]

    # Grouped aggregation â€” the core computation
    stats <- edge_rows[!is.na(nval),
                       .(nmax  = max(nval),
                         nmin  = min(nval),
                         nmean = mean(nval)),
                       by = from_row_idx]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results by row index
    dt[stats$from_row_idx, (max_col)  := stats$nmax]
    dt[stats$from_row_idx, (min_col)  := stats$nmin]
    dt[stats$from_row_idx, (mean_col) := stats$nmean]

    # Clean up temp column
    edge_rows[, nval := NULL]
  }

  # Remove helper column
  dt[, row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility (e.g., with trained RF model)
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement for the original pipeline section):
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original pipeline.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations with `paste()` + named-vector probes | Eliminated entirely |
| **Neighbor resolution** | Row-by-row string key construction & lookup | Single vectorized `data.table` keyed join |
| **Stats computation** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | `data.table` grouped aggregation (`by=`) |
| **Number of passes over edges** | 1 (build) + 5 (compute per var) = 6 | Edge expansion once; 5 lightweight grouped aggregations |
| **Expected runtime** | ~86+ hours | **Minutes** (estimated 5â€“15 min on the same laptop) |
| **RAM safety** | `edge_rows` at ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 5 int/numeric cols â‰ˆ ~1.5 GB â€” fits in 16 GB | âœ“ |
| **Trained RF model** | Preserved (no retraining) | âœ“ |
| **Numerical equivalence** | Original `max/min/mean` over non-NA neighbor values | Identical â€” same `max()`, `min()`, `mean()` over same neighbor sets | âœ“ |
END_RESPONSE>>>
