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
run_id: Anthropic_F09_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-ish in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste(..., year, sep="_")` key strings for every neighbor Ã— year combination (expensive string allocation).
- Indexes into a named character vector (`idx_lookup`) via string matching (slow hash lookup repeated billions of times).

This means ~6.46M iterations Ã— ~4 neighbors on average = ~25.8M string constructions and hash lookups, all inside an R-level `lapply` with no vectorization.

### 2. The lookup is **year-redundant**
The spatial neighbor structure is **identical across all 28 years**. A cell's rook neighbors in 1992 are the same cells in 2019. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year row, doing 28Ã— the necessary work. The neighbor *topology* only needs to be built once for the 344,208 cells; yearly attribute values should be joined afterward.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing max/min/mean iterates row-by-row in R over 6.46M rows. This is inherently slow without vectorization or `data.table` grouped operations.

---

## Optimization Strategy

**Core idea:** Separate topology from attributes. Build the neighbor edge list **once** (344K cells, ~1.37M edges), then use a `data.table` join-and-group strategy to compute neighbor statistics for all years simultaneously in vectorized C-level operations.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Convert `nb` object â†’ edge `data.table` (cell_id, neighbor_id): ~1.37M rows. **Once.** | O(E) |
| 2 | Join yearly attributes onto the edge table by `(neighbor_id, year)`. | O(E Ã— T) â‰ˆ 38.4M rows |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean`. | Vectorized, single pass |
| 4 | Join results back to the main panel. | O(N) |

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes the `data.table` join on ~38M rows Ã— 5 variables, which is trivial for `data.table`).

**Constraints honored:**
- The trained Random Forest model is untouched.
- The numerical output (neighbor max, min, mean per variable per cell-year) is identical.
- Peak RAM stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build the spatial edge list ONCE from the nb object
#         This replaces build_neighbor_lookup entirely.
# ===========================================================================

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj: list of length N, each element is integer vector of neighbor indices
  # id_order: vector of cell IDs corresponding to nb_obj positions
  #
  # Returns a data.table with columns: cell_id, neighbor_id
  # Represents directed rook-neighbor edges (one row per directed pair).

  n <- length(nb_obj)
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    # spdep nb objects use 0L for "no neighbors"
    len <- length(x)
    if (len == 1L && x[1L] == 0L) 0L else len
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos     <- 1L

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    k <- length(nbrs)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nbrs]
    pos <- pos + k
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build once â€” takes seconds for ~1.37M edges
edges <- build_edge_table(id_order, rook_neighbors_unique)

# ===========================================================================
# STEP 2 & 3: Compute neighbor stats for all variables, all years at once
# ===========================================================================

compute_all_neighbor_features <- function(cell_data_dt, edges, source_vars) {
  # cell_data_dt: data.table with columns 'id', 'year', and all source_vars
  # edges:        data.table with columns 'cell_id', 'neighbor_id'
  # source_vars:  character vector of variable names
  #
  # Returns cell_data_dt with new columns:
  #   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
  #   for each var in source_vars.

  # Ensure data.table and set keys for fast joins
  if (!is.data.table(cell_data_dt)) cell_data_dt <- as.data.table(cell_data_dt)

  # Subset only the columns we need for the neighbor attribute lookup
  attr_cols <- c("id", "year", source_vars)
  attrs <- cell_data_dt[, ..attr_cols]

  # Key the attribute table on (id, year) for the join

setkey(attrs, id, year)

  # Expand edges Ã— years:
  #   Join neighbor attributes onto the edge table.
  #   For each (cell_id, neighbor_id) edge, we replicate across all years
  #   by joining on neighbor_id == id.
  #
  # Result: one row per (cell_id, neighbor_id, year) with neighbor's values.

  # Rename for join clarity
  setnames(attrs, "id", "neighbor_id")
  setkey(attrs, neighbor_id, year)
  setkey(edges, neighbor_id)

  # This is the big join: edges[, .(cell_id, neighbor_id)] Ã— attrs by neighbor_id

  # data.table does this efficiently as an equi-join.
  expanded <- attrs[edges, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # expanded now has columns: neighbor_id, year, <source_vars>, cell_id

  # Group by (cell_id, year) and compute stats for each source var
  agg_exprs <- list()
  for (v in source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0(v, "_neighbor_max")]]  <-
      bquote(as.numeric(max(.(sym_v), na.rm = TRUE)), list(sym_v = sym_v))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <-
      bquote(as.numeric(min(.(sym_v), na.rm = TRUE)), list(sym_v = sym_v))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <-
      bquote(mean(.(sym_v), na.rm = TRUE), list(sym_v = sym_v))
  }

  # Build the aggregation call dynamically
  agg_list <- as.call(c(as.name("list"), agg_exprs))
  neighbor_stats <- expanded[, eval(agg_list), by = .(cell_id, year)]

  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  stat_cols <- names(neighbor_stats)[-(1:2)]
  for (col in stat_cols) {
    neighbor_stats[is.infinite(get(col)), (col) := NA_real_]
  }

  # Join back onto the main data
  setkey(neighbor_stats, cell_id, year)
  setkey(cell_data_dt, id, year)
  # Align key names for merge
  setnames(neighbor_stats, "cell_id", "id")
  cell_data_dt <- neighbor_stats[cell_data_dt, on = .(id, year)]

  return(cell_data_dt)
}

# ===========================================================================
# STEP 4: Run it
# ===========================================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# ===========================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ===========================================================================

# The trained model object (e.g., rf_model) is used as-is.
# The new neighbor feature columns have the same names and semantics
# as the original pipeline produced, so prediction is identical:
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# No retraining required. The numerical estimand is preserved because
# max, min, and mean are computed over exactly the same neighbor sets
# and the same attribute values.
```

---

## Why This Is Correct and Equivalent

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor set for cell *i* in year *t* | Rook neighbors from `nb` object, filtered to those present in year *t* | Identical â€” the `nomatch = NA` join drops missing cell-years, and `na.rm = TRUE` in aggregation mirrors the `!is.na()` filter |
| Statistics computed | `max`, `min`, `mean` of non-NA neighbor values | Identical functions, identical `na.rm` behavior |
| All-missing neighbor groups | Return `(NA, NA, NA)` | `Inf`/`-Inf` from `max`/`min` are replaced with `NA`; `mean` of empty = `NaN` â†’ `NA` |
| Column names | `<var>_neighbor_max`, `<var>_neighbor_min`, `<var>_neighbor_mean` | Identical |
| RF model | Not modified | Not modified |

## Expected Performance

| Stage | Rows/Operations | Estimated Time |
|-------|----------------|---------------|
| `build_edge_table` | 1.37M edges | ~1â€“2 seconds |
| Join (edges Ã— attrs) | ~38.4M rows (1.37M edges Ã— 28 years) | ~10â€“30 seconds |
| Group-by aggregation | 6.46M groups Ã— 5 vars Ã— 3 stats | ~1â€“3 minutes |
| Column join back | 6.46M rows | ~5 seconds |
| **Total** | | **~2â€“5 minutes** |

Peak memory: ~38.4M rows Ã— ~8 columns Ã— 8 bytes â‰ˆ **2.5 GB** (well within 16 GB).

This reduces the runtime from **~86 hours to ~2â€“5 minutes** â€” a roughly **1,000Ã— speedup** â€” by eliminating year-redundant topology construction and replacing row-level R loops with vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
