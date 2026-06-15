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
run_id: Anthropic_F12_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed inefficient (it creates a matrix row-by-row). However, this is called only **5 times** (once per variable). Replacing it with a pre-allocated matrix or `vapply` is a quick win, but it shaves minutes, not hours.

### `build_neighbor_lookup()` â€” The True Bottleneck

This is where the **86+ hours** lives. Here's why:

1. **`lapply` over 6.46 million rows**, each iteration doing:
   - `as.character()` conversion
   - Named vector lookup via `id_to_ref[as.character(data$id[i])]` â€” this is **O(n) hash lookup repeated 6.46M times**, but worse:
   - `paste()` to build `neighbor_keys` â€” **string allocation per row**
   - `idx_lookup[neighbor_keys]` â€” named vector lookup with **string matching** on a vector of 6.46M names, repeated for every neighbor set, every row.

2. **String-keyed lookups on a 6.46M-element named vector (`idx_lookup`)** are the critical cost. Each `idx_lookup[neighbor_keys]` call does repeated string hashing/matching against a massive character vector. With ~4 neighbors per cell on average, that's ~25.8 million string lookups **per call**, and this function is called once, but the inner loop is 6.46M iterations each doing multiple string operations.

3. The `paste()`-based key construction creates **millions of temporary string objects**, causing heavy GC pressure on a 16 GB machine.

**Verdict: REJECT the colleague's diagnosis as the primary bottleneck.** The dominant cost is `build_neighbor_lookup()` â€” specifically the row-level `paste()` + string-keyed lookups over a 6.46M-entry named character vector inside a per-row `lapply`. `compute_neighbor_stats` is a secondary, much smaller inefficiency.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a vectorized approach using integer arithmetic instead of string keys. Instead of `paste(id, year)` â†’ string lookup, use a **two-column integer index**: create a fast `(id, year) â†’ row` mapping via an integer-keyed environment or `data.table`, then vectorize the neighbor expansion across all rows at once.

2. **Replace `compute_neighbor_stats`** with vectorized grouped operations using `data.table` or pre-allocated matrix output via `vapply`.

3. **Key insight**: The neighbor lookup is **year-invariant** â€” cell A's rook neighbors are the same cells every year. So we can build the spatial neighbor structure once (344K cells), then expand to all years via a merge/join, avoiding per-row string operations entirely.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup
# ============================================================
# Strategy: exploit the fact that spatial neighbors are fixed
# across years. Build an integer-indexed lookup using data.table
# for O(1) keyed joins instead of string matching.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Step 1: Create integer mapping from id -> reference index in id_order
  id_to_ref <- data.table(
    id = id_order,
    ref_idx = seq_along(id_order)
  )

  # Step 2: Build a flat edge table of (cell_ref_idx -> neighbor_cell_id)
  # This is done ONCE for the 344,208 cells, not per row.
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(ref_idx) {
    nb <- neighbors[[ref_idx]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(ref_idx = integer(0), neighbor_id = integer(0)))
    }
    data.table(ref_idx = ref_idx, neighbor_id = id_order[nb])
  }))

  # Step 3: Map each cell_id to its ref_idx
  # Then for each (cell_id, year) row, find neighbor rows by joining
  # on (neighbor_id, year).

  # Add ref_idx to dt
  dt_with_ref <- merge(dt, id_to_ref, by = "id", sort = FALSE)

  # Step 4: Expand edges to (source_row, year, neighbor_id)
  # For each row in dt, get its ref_idx, then get neighbor_ids from edge_list
  # Then find the row_idx of (neighbor_id, year) in dt.

  # Create a keyed lookup: (id, year) -> row_idx
  row_lookup <- dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # Expand: for each row, get its neighbors via ref_idx
  # Instead of per-row lapply, do a massive join:

  # source_info: row_idx, ref_idx, year for every row
  source_info <- dt_with_ref[, .(row_idx, ref_idx, year)]

  # Join source_info with edge_list on ref_idx to get all
  # (source_row_idx, year, neighbor_id) triples
  setkey(source_info, ref_idx)
  setkey(edge_list, ref_idx)
  expanded <- edge_list[source_info, on = "ref_idx", allow.cartesian = TRUE,
                        nomatch = NA]
  # expanded has columns: ref_idx, neighbor_id, row_idx (source), year

  # Now find the row_idx of each neighbor in the same year
  expanded <- expanded[!is.na(neighbor_id)]
  setnames(expanded, "row_idx", "source_row_idx")

  # Join to find neighbor's row_idx
  expanded[, c("id_lookup", "year_lookup") := .(neighbor_id, year)]
  neighbor_rows <- row_lookup[expanded, on = c(id = "id_lookup", year = "year_lookup"),
                              nomatch = NA]
  # neighbor_rows now has: source_row_idx, row_idx (= neighbor's row_idx)

  neighbor_rows <- neighbor_rows[!is.na(row_idx)]

  # Step 5: Build the lookup list indexed by source_row_idx
  setkey(neighbor_rows, source_row_idx)
  n_rows <- nrow(dt)

  # Group neighbor row indices by source row
  lookup_dt <- neighbor_rows[, .(neighbor_indices = list(row_idx)),
                              by = source_row_idx]

  # Initialize full list with empty integer vectors
  lookup <- vector("list", n_rows)
  for (i in seq_len(n_rows)) lookup[[i]] <- integer(0)

  # Fill in the ones that have neighbors
  lookup[lookup_dt$source_row_idx] <- lookup_dt$neighbor_indices

  return(lookup)
}


# ============================================================
# OPTIMIZED compute_neighbor_stats
# ============================================================
# Strategy: Instead of lapply + do.call(rbind, ...), use the
# flat edge table directly with data.table grouped aggregation.
# This avoids all per-row R function calls.

compute_neighbor_stats_fast <- function(data_dt, neighbor_edges, var_name) {
  # neighbor_edges: data.table with (source_row_idx, neighbor_row_idx)
  # Pull values
  vals <- data_dt[[var_name]]
  n <- nrow(data_dt)

  # Attach neighbor values
  edges <- copy(neighbor_edges)
  edges[, neighbor_val := vals[neighbor_row_idx]]
  edges <- edges[!is.na(neighbor_val)]

  # Grouped aggregation â€” fully vectorized
  agg <- edges[, .(
    nmax  = max(neighbor_val),
    nmin  = min(neighbor_val),
    nmean = mean(neighbor_val)
  ), by = source_row_idx]

  # Build output matrix, NA for rows with no valid neighbors
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[agg$source_row_idx, 1] <- agg$nmax
  out[agg$source_row_idx, 2] <- agg$nmin
  out[agg$source_row_idx, 3] <- agg$nmean

  return(out)
}


# ============================================================
# OPTIMIZED compute_and_add_neighbor_features
# ============================================================
compute_and_add_neighbor_features_fast <- function(data_dt, var_name,
                                                    neighbor_edges) {
  stats <- compute_neighbor_stats_fast(data_dt, neighbor_edges, var_name)
  data_dt[, paste0("n_max_", var_name)  := stats[, 1]]
  data_dt[, paste0("n_min_", var_name)  := stats[, 2]]
  data_dt[, paste0("n_mean_", var_name) := stats[, 3]]
  return(data_dt)
}


# ============================================================
# OPTIMIZED OUTER PIPELINE
# ============================================================
# Convert to data.table once
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# --- Build flat edge table (replaces build_neighbor_lookup) ---
# Step A: id -> ref_idx mapping
id_to_ref <- data.table(id = id_order, ref_idx = seq_along(id_order))

# Step B: Flat spatial edge list from the nb object (344K cells)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(data.table(ref_idx = integer(0), neighbor_id = integer(0)))
  }
  data.table(ref_idx = i, neighbor_id = id_order[nb])
}))

# Step C: Attach ref_idx to each row
cell_dt_ref <- merge(cell_dt, id_to_ref, by = "id", sort = FALSE)

# Step D: Expand spatial edges across all years via join
#   For each (ref_idx, year) row â†’ get all neighbor_ids
#   Then find the row_idx of (neighbor_id, same year)
source_info <- cell_dt_ref[, .(source_row_idx = row_idx, ref_idx, year)]
setkey(edge_list, ref_idx)
setkey(source_info, ref_idx)

expanded <- edge_list[source_info, on = "ref_idx",
                      allow.cartesian = TRUE, nomatch = 0L]
# Columns: ref_idx, neighbor_id, source_row_idx, year

# Step E: Keyed lookup for (id, year) -> row_idx
row_lookup <- cell_dt[, .(id, year, row_idx)]
setkey(row_lookup, id, year)

# Join to resolve neighbor row indices
neighbor_edges <- row_lookup[expanded,
                              on = c(id = "neighbor_id", year = "year"),
                              nomatch = 0L][, .(source_row_idx,
                                                 neighbor_row_idx = row_idx)]
setkey(neighbor_edges, source_row_idx)

# --- Compute features for all 5 variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_dt <- compute_and_add_neighbor_features_fast(cell_dt, var_name,
                                                     neighbor_edges)
  cat("Done:", var_name, "\n")
}

# Convert back to data.frame if needed downstream
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model (rf_model) is untouched.
# All numerical outputs (n_max_*, n_min_*, n_mean_*) are identical
# to the original implementation â€” same max/min/mean aggregations
# over the same neighbor sets.
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, ...)` | "Main bottleneck" | Minor (~5 calls, fixable but not dominant) |
| `build_neighbor_lookup` | Not mentioned | **TRUE BOTTLENECK**: 6.46M iterations of `paste()` + string-keyed lookups on a 6.46M-entry named vector |
| Root cause | List binding | **String construction & matching**: ~25.8M `paste()` calls + character hash lookups per row in a single-threaded `lapply` |
| Estimated speedup | Marginal | **~100â€“500Ã—** by replacing string ops with `data.table` integer-keyed joins |

**The colleague's diagnosis is rejected.** The correct optimization is to eliminate `build_neighbor_lookup()`'s per-row string operations entirely by (1) building a flat integer edge table from the spatial neighbor structure once, (2) expanding it across years via `data.table` keyed joins, and (3) computing grouped aggregations vectorially â€” all while preserving the trained Random Forest model and the original numerical estimand.
END_RESPONSE>>>
