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
run_id: Anthropic_F05_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Specifically:

1. **`idx_lookup`** (the named vector of all 6.46M keys) is built once â€” that's fine.
2. But **inside the `lapply`**, for every single row `i`, the code:
   - Looks up the cell's reference index in `id_to_ref`
   - Extracts neighbor cell IDs from the `nb` object
   - Calls `paste()` to build string keys for each neighbor
   - Performs named-vector lookup into `idx_lookup`

With ~6.46M rows and an average of ~4 rook neighbors each, that's **~25.8 million `paste()` + named-vector-match operations**, all in an interpreted R loop. Named-vector lookup on a 6.46M-element character vector is O(n) per probe in the worst case (hash collisions aside, R's internal hashing on 6.46M keys is slow to build and probe).

### The Broader Pattern

This is not just about string keys. The entire algorithmic structure is:

```
For each row i (6.46M iterations):
    Find neighbors of cell i in the same year
    â†’ This is a spatial join constrained by year
```

But **the neighbor topology is time-invariant** â€” the same cell has the same rook neighbors in every year. The code re-discovers this for every cell-year pair, when it only needs to be resolved once per cell and then broadcast across years.

Furthermore, `compute_neighbor_stats` is already vectorized given the lookup â€” so the bottleneck is entirely in `build_neighbor_lookup`.

---

## Optimization Strategy

### Key Insight: Separate the spatial topology (time-invariant) from the temporal indexing

1. **Build a cell-level neighbor map once** (344K cells, not 6.46M cell-years).
2. **Use integer indexing throughout** â€” eliminate all string keys.
3. **Vectorize the neighbor-stats computation** using `data.table` grouped operations or matrix arithmetic, avoiding per-row `lapply`.

### Approach

- Create a `data.table` with a row-index column, keyed on `(id, year)` for O(1) integer-based joins.
- Expand the cell-level neighbor list into an edge table: `(focal_id, neighbor_id)`.
- Join this edge table with the data to get `(focal_row, neighbor_row)` pairs per year.
- Compute `max`, `min`, `mean` per focal row using `data.table` grouped aggregation â€” fully vectorized in C.

**Expected speedup**: from ~86 hours to **minutes** (the bottleneck becomes a few large equi-joins and grouped aggregations, all handled by `data.table`'s radix-based C internals).

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # Step 1: Convert to data.table, add integer row index

# ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # ---------------------------------------------------------------
# Step 2: Build a cell-level integer ID mapping
# ---------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching the nb object
  # Map each cell ID to its position in id_order (its "ref index")
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # ---------------------------------------------------------------
# Step 3: Build the time-invariant edge list from the nb object
  #         This is done ONCE for 344K cells, not 6.46M cell-years
# ---------------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(ref_idx) {
    nb_ref_indices <- rook_neighbors_unique[[ref_idx]]
    # nb objects use 0-length integer for no-neighbor; filter those
    if (length(nb_ref_indices) == 0L || (length(nb_ref_indices) == 1L && nb_ref_indices[1] == 0L)) {
      return(NULL)
    }
    data.table(
      focal_id    = id_order[ref_idx],
      neighbor_id = id_order[nb_ref_indices]
    )
  }))

  # ---------------------------------------------------------------
# Step 4: For each year, join edges with data to get row indices
  #         Then compute stats in one vectorized pass per variable
# ---------------------------------------------------------------

  # Key the data for fast (id, year) lookups
  setkey(dt, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  # Process year by year to control memory (each year: ~344K rows, ~1.37M edges)
  for (yr in years) {
    # Subset rows for this year
    dt_yr <- dt[year == yr, c("id", ".row_idx", neighbor_source_vars), with = FALSE]
    setkey(dt_yr, id)

    # Join edges with this year's data for focal side
    # focal_id -> get focal .row_idx
    focal_join <- dt_yr[, .(focal_id = id, focal_row_idx = .row_idx)]
    setkey(focal_join, focal_id)

    # neighbor_id -> get neighbor values
    neighbor_join <- copy(dt_yr)
    setnames(neighbor_join, "id", "neighbor_id")
    setnames(neighbor_join, ".row_idx", "neighbor_row_idx")
    setkey(neighbor_join, neighbor_id)

    # Build the full edge table for this year with values attached
    # edges: (focal_id, neighbor_id)
    yr_edges <- copy(edges)
    setkey(yr_edges, focal_id)
    yr_edges <- focal_join[yr_edges, on = .(focal_id), nomatch = 0L]

    setkey(yr_edges, neighbor_id)
    yr_edges <- neighbor_join[yr_edges, on = .(neighbor_id), nomatch = 0L]

    # Now yr_edges has: focal_id, focal_row_idx, neighbor_id, neighbor_row_idx,
    #                   and all neighbor_source_vars (from the neighbor side)

    # Compute grouped stats per focal_row_idx for each variable
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      # Aggregate: group by focal_row_idx, compute max/min/mean of neighbor's var
      stats <- yr_edges[!is.na(get(var_name)),
                        .(nmax  = max(get(var_name)),
                          nmin  = min(get(var_name)),
                          nmean = mean(get(var_name))),
                        by = focal_row_idx]

      # Write results back into dt using row indices
      if (nrow(stats) > 0L) {
        set(dt, i = stats$focal_row_idx, j = max_col,  value = stats$nmax)
        set(dt, i = stats$focal_row_idx, j = min_col,  value = stats$nmin)
        set(dt, i = stats$focal_row_idx, j = mean_col, value = stats$nmean)
      }
    }

    if (yr %% 5 == 0) message("Completed year: ", yr)
  }

  # ---------------------------------------------------------------
# Step 5: Clean up and return as data.frame (preserving compatibility)
# ---------------------------------------------------------------
  dt[, .row_idx := NULL]
  return(as.data.frame(dt))
}
```

### Usage (drop-in replacement for the original outer loop)

```r
# Original code replaced by:
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is untouched.
# Column names produced: neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.
# Numerical results are identical (same max/min/mean over same neighbor sets).
```

### If your original code used different column naming conventions, add this adapter:

```r
# Rename to match whatever the RF model expects, e.g.:
# If the original code produced columns like "ntl_neighbor_max":
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  for (stat in c("max", "min", "mean")) {
    old_name <- paste0("neighbor_", stat, "_", var_name)
    new_name <- paste0(var_name, "_neighbor_", stat)  # adjust to your convention
    if (old_name %in% names(cell_data) && old_name != new_name) {
      names(cell_data)[names(cell_data) == old_name] <- new_name
    }
  }
}
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **Outer loop** | 6.46M R-level iterations | 28 year iterations |
| **Key construction** | ~25.8M `paste()` calls inside loop | 0 string keys; integer joins |
| **Lookup method** | Named character vector (hash on 6.46M strings) | `data.table` binary search join |
| **Stats computation** | `lapply` over 6.46M lists | `data.table` grouped aggregation (C) |
| **Memory pattern** | 6.46M-element list of integer vectors | ~1.37M-row edge table per year |
| **Estimated time** | ~86+ hours | **~5â€“15 minutes** |

### Why the numerical results are preserved

The optimized code computes `max`, `min`, and `mean` over exactly the same set of neighbor values (same cell, same year, same rook adjacency, same NA-removal logic) â€” just via a vectorized join-and-aggregate instead of a per-row loop. The estimand is identical.
END_RESPONSE>>>
