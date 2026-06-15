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
run_id: Anthropic_F07_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) hash probing per call, but the real cost is doing ~6.46M `paste()` calls and ~6.46M named-vector subscript operations, each touching a names vector of length 6.46M. This is catastrophically slow.

2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector lookup, and NA filtering. The per-iteration overhead of R's interpreted loop, combined with the O(1)-amortized-but-high-constant-factor named lookups on a 6.46M-length vector, yields the estimated 86+ hour runtime.

3. **`compute_neighbor_stats`** then does a second `lapply` over the 6.46M-element list, which is comparatively fast but still unnecessarily slow because it's row-by-row interpreted R.

**Root cause summary:** The algorithm is correct but implemented with interpreted-R loops and string-keyed lookups over millions of rows. The neighbor topology is time-invariant (same grid, same neighbors every year), so the spatial join should be done once at the cell level (344K cells), then broadcast across years via vectorized merge â€” not repeated per cell-year row.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Separate space from time** | Build the neighbor edge list once over 344K cells, not 6.46M cell-years. |
| **Vectorize with `data.table`** | Replace `lapply`/`paste`/named-vector lookups with `data.table` keyed joins and grouped aggregations. |
| **Columnar neighbor stats** | For each variable, do a single vectorized join of cell-year values onto the edge list, then `group by (id, year)` to compute `max`, `min`, `mean`. |
| **Memory-safe** | The edge list is ~1.37M rows Ã— 3 columns (source_id, neighbor_id, implicit). Joined with year, it becomes ~1.37M Ã— 28 â‰ˆ 38.4M rows â€” large but fits in 16 GB as integer/double columns. We can process one variable at a time to limit peak memory. |
| **Preserve numerics exactly** | `max`, `min`, `mean` on the same neighbor sets with the same NA handling â†’ identical numerical results. |
| **No model retraining** | We only rebuild the feature columns; the trained RF object is untouched. |

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed already in memory:
#       cell_data            â€” data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#       id_order             â€” integer/character vector of cell IDs (same order as rook_neighbors_unique)
#       rook_neighbors_unique â€” spdep nb object (list of integer index vectors into id_order)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a SPATIAL edge list (once, ~1.37M rows)
#     Each row: (id, neighbor_id) meaning "neighbor_id is a rook neighbor of id"
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] gives integer indices into id_order for neighbors of id_order[i]
  n <- length(nb_obj)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))  # should be ~1,373,394
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- nb_obj[[i]]
    # spdep nb objects use 0L to denote "no neighbors" for an isolate
    if (length(nb_i) == 1L && nb_i[1L] == 0L) next
    len <- length(nb_i)
    idx <- pos:(pos + len - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + len
  }
  
  # Trim if any isolates caused fewer edges
  if (pos - 1L < n_edges) {
    from_id <- from_id[seq_len(pos - 1L)]
    to_id   <- to_id[seq_len(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building spatial edge list...\n")
edges <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edges)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for each variable via vectorized join
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need a lookup keyed on (id, year) for the variable values.
# We'll key cell_data on (id, year) for fast joins.
setkeyv(cell_data, c("id", "year"))

# Unique years for the cross-join with edges
all_years <- sort(unique(cell_data$year))

# Expand edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.4M rows)
# To save memory, we do this once and reuse.
cat("Expanding edge list across years...\n")
edge_years <- edges[, .(year = all_years), by = .(id, neighbor_id)]
# edge_years columns: id, neighbor_id, year
# Set key on (neighbor_id, year) for joining neighbor values
setkeyv(edge_years, c("neighbor_id", "year"))

cat(sprintf("  Expanded edge-year rows: %s\n", format(nrow(edge_years), big.mark = ",")))

compute_and_add_neighbor_features_dt <- function(cell_data, edge_years, var_name) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Extract just the columns we need for the join: (id, year, value)
  val_dt <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkeyv(val_dt, c("neighbor_id", "year"))
  
  # Join neighbor values onto edge_years
  # edge_years keyed on (neighbor_id, year); val_dt keyed on (neighbor_id, year)
  ey <- edge_years[val_dt, on = .(neighbor_id, year), nomatch = 0L]
  # ey now has columns: id, neighbor_id, year, val
  # where val is the neighbor's value
  
  # Aggregate by (id, year) â€” these are the stats for each cell-year
  stats <- ey[!is.na(val),
              .(nb_max  = max(val),
                nb_min  = min(val),
                nb_mean = mean(val)),
              by = .(id, year)]
  
  # Name the new columns to match original convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Merge back into cell_data
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_data, c("id", "year"))
  cell_data[stats, on = .(id, year),
            c(max_col, min_col, mean_col) := mget(paste0("i.", c(max_col, min_col, mean_col)))]
  
  # Cell-years with no valid neighbors remain NA (already the default for unmatched rows)
  invisible(NULL)
}

cat("Computing neighbor statistics...\n")
for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_dt(cell_data, edge_years, var_name)
}
cat("Done.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Clean up large temporaries
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(edges, edge_years)
gc()

# cell_data now contains the 15 new neighbor-stat columns
# (3 stats Ã— 5 variables) with numerically identical values
# to the original implementation.
#
# The trained Random Forest model is untouched and can be used
# directly for prediction on the updated cell_data.
```

---

## Why This Is Numerically Identical

| Original | Optimized | Match? |
|---|---|---|
| For each cell-year row, find neighbor cell-year rows by string key, extract values, compute `max`/`min`/`mean` excluding `NA`. | For each `(id, year)` group in the joined edge table, compute `max`/`min`/`mean` on the same neighbor values excluding `NA`. | âœ… Identical sets, identical functions. |
| Cell-years with zero valid neighbors â†’ `NA, NA, NA`. | Unmatched `(id, year)` keys after the join remain `NA` in `cell_data`. | âœ… Identical. |

The `max`, `min`, and `mean` functions are deterministic and associative â€” grouping the same values produces the same results regardless of evaluation order.

---

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~80+ hours (6.46M string ops) | ~1 sec (344K integer loop) |
| Expand edges Ã— years | â€” | ~2 sec (38.4M row CJ) |
| Per-variable join + aggregate | ~1 hour each (6.46M lapply) | ~15â€“30 sec each (data.table keyed join + groupby) |
| **Total** | **86+ hours** | **~2â€“5 minutes** |
END_RESPONSE>>>
