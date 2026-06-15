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
run_id: Anthropic_F02_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is **`build_neighbor_lookup`**, which calls `lapply` over ~6.46 million rows, each iteration performing character coercion, string pasting, and named-vector lookups. Named vector lookups in R (`idx_lookup[neighbor_keys]`) are hash-table searches on character keysâ€”fast individually, but catastrophically slow when repeated billions of times (6.46M rows Ã— average ~4 neighbors Ã— multiple string operations). The second bottleneck is **`compute_neighbor_stats`**, which loops over 6.46M entries five times (once per variable), each time extracting and summarizing a small numeric vector.

**Specific problems:**

1. **`build_neighbor_lookup`**: `paste()` and character-key lookups inside a 6.46M-iteration `lapply` produce ~26M+ temporary string allocations and hash lookups. This is the 86-hour wall.
2. **`compute_neighbor_stats`**: `lapply` over 6.46M elements returning 3-element vectors, then `do.call(rbind, ...)` on a 6.46M-element list, is memory-intensive and slow. This is called 5 times.
3. **Data representation**: The neighbor lookup is stored as a list of 6.46M integer vectors (row-level), which is a large, fragile R object.

**Root cause**: The algorithm is correct but uses idiomatic-R patterns (named vectors, `paste`, per-row `lapply`) that do not scale to millions of rows.

---

## Optimization Strategy

**Replace all row-level R loops and string-key lookups with vectorized join and grouped aggregation using `data.table`.**

The key insight: the neighbor lookup is a **join** problem. Each cell-year row needs to be joined to its neighbors' rows for the same year, then statistics are computed per group. This is exactly what `data.table` excels at.

**Steps:**

1. **Build an edge table** (once): Expand `rook_neighbors_unique` (the `nb` object) into a two-column integer table of `(cell_id, neighbor_cell_id)` â€” ~1.37M rows.
2. **Join edges to data by year** (vectorized): For each cell-year, find the neighbor-cell-year rows via a keyed `data.table` merge. This replaces `build_neighbor_lookup` entirely.
3. **Grouped aggregation** (vectorized): Compute `max`, `min`, `mean` per `(cell_id, year)` group in one `data.table` operation per variable (or all at once).
4. **Join results back**: Merge the aggregated neighbor stats back onto the main table.

**Expected gains:**
- `build_neighbor_lookup`: eliminated (the edge table + join replaces it).
- `compute_neighbor_stats`: replaced by `data.table` grouped aggregation, which runs in seconds on ~26M joined rows.
- Estimated total time: **minutes**, not hours.
- Peak memory: the edge table exploded by year is ~26M rows Ã— a few columns â‰ˆ manageable within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a directed edge table from the nb object
#
#   rook_neighbors_unique is a list of length N_cells (344,208)
#   where element i contains integer indices into id_order of
#   the neighbors of cell i.
#   id_order[i] gives the actual cell id for position i.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0L && !(length(nb_i) == 1L && nb_i[1] == 0L)) {
      n <- length(nb_i)
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_i]
      pos <- pos + n
    }
  }

  # Trim if any nb entries were empty/zero (pos may be < n_edges+1)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 M rows, two integer columns â€” trivial memory

cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor statistics for all variables at once
#
# Strategy:
#   - Join edge_dt to cell_data to get neighbor rows for every
#     (id, year) combination.
#   - Group by (id, year) and compute max/min/mean per variable.
#   - Join the results back to cell_data.
#
# To control peak memory we do this variable-by-variable, but
# each variable takes only seconds.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need a slim lookup: for each (neighbor_id, year) â†’ variable values
# Key cell_data for fast joins
setkey(cell_data, id, year)

compute_and_add_all_neighbor_features <- function(cell_data, edge_dt,
                                                   neighbor_source_vars) {

  # Build a lookup keyed by (id, year) containing only the columns we need
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup <- cell_data[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # For each cell-year, we need to know its year so we can match neighbors
  # in the same year.  Get the unique (id, year) pairs and their row indices.
  # We join: edge_dt (id, neighbor_id) Ã— years
  # But we should only look at (id, year) pairs that actually exist.

  # Approach: join cell_data's (id, year) to edges, then to neighbor values.

  # cell_id_year: the (id, year) combinations that exist
  # We do NOT explode edges Ã— 28 years globally; instead we join via cell_data.


  # Small intermediate: attach year to each edge via the focal cell
  # cell_data has columns id and year; edge_dt has columns id and neighbor_id
  # Result: (id, year, neighbor_id) â€” one row per directed-edge-year

  # Use a left join of cell_data[, .(id, year)] to edge_dt on id
  # That gives us ~6.46M Ã— avg_degree â‰ˆ 26M rows

  id_year <- cell_data[, .(id, year)]
  setkey(id_year, id)
  setkey(edge_dt, id)

  # This is the main expansion: ~26M rows
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year

  cat("Expanded edge-year rows:", nrow(expanded), "\n")

  # Now join to lookup to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, ntl, ec, ..., id  (from the join)

  # Aggregate per (id, year)
  # Build aggregation expressions dynamically
  agg_exprs <- lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  })

  # Flatten and name
  agg_list <- list()
  agg_names <- character()
  for (v in neighbor_source_vars) {
    vsym <- as.name(v)
    agg_list <- c(agg_list, list(
      bquote(fifelse(all(is.na(.(vsym))), NA_real_, max(.(vsym), na.rm = TRUE))),
      bquote(fifelse(all(is.na(.(vsym))), NA_real_, min(.(vsym), na.rm = TRUE))),
      bquote(fifelse(all(is.na(.(vsym))), NA_real_, mean(.(vsym), na.rm = TRUE)))
    ))
    agg_names <- c(agg_names,
                    paste0("max_neighbor_", v),
                    paste0("min_neighbor_", v),
                    paste0("mean_neighbor_", v))
  }

  # Build the single aggregation call
  agg_call <- as.call(c(as.name("list"),
                         setNames(agg_list, agg_names)))

  cat("Computing grouped aggregation...\n")
  stats <- expanded[, eval(agg_call), by = .(id, year)]
  cat("Aggregation complete. Rows:", nrow(stats), "\n")

  # Replace Inf / -Inf from max/min of empty groups with NA
  for (col_name in agg_names) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # Merge back onto cell_data
  setkey(stats, id, year)
  setkey(cell_data, id, year)

  # Remove any pre-existing neighbor columns to avoid duplication
  existing <- intersect(names(cell_data), agg_names)
  if (length(existing) > 0L) {
    cell_data[, (existing) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]

  return(cell_data)
}

cell_data <- compute_and_add_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Predict with the already-trained Random Forest
#
# The model object (e.g. `rf_model`) is preserved unchanged.
# The columns produced above (max_neighbor_*, min_neighbor_*,
# mean_neighbor_*) are numerically identical to the originals,
# so predictions will be identical.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Example (adapt to your actual model variable and feature set):
# cell_data[, predicted_gdp := predict(rf_model, newdata = .SD)]
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with `paste` + named-vector hash lookup | Single `data.table` keyed join (~26M rows built in seconds) |
| **Neighbor stats** | `lapply` over 6.46M list elements Ã— 5 vars, `do.call(rbind, ...)` | One grouped `data.table` aggregation for all 5 vars simultaneously |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** (join + aggregation on 26M rows) |
| **Peak memory** | Huge list of 6.46M integer vectors + intermediate string vectors | ~26M-row `data.table` â‰ˆ 1â€“2 GB (fits in 16 GB) |
| **Numerical results** | max / min / mean of non-NA neighbor values | Identical (`fifelse(all(is.na(...)), NA, ...)` guards match original NA logic) |
| **Trained RF model** | â€” | **Untouched**; column names and values are preserved |

The optimization replaces algorithmic inefficiency (per-row string operations in R) with `data.table`'s vectorized C-level joins and grouped aggregations, yielding roughly a **1,000Ã—** speedup while preserving exact numerical equivalence.
END_RESPONSE>>>
