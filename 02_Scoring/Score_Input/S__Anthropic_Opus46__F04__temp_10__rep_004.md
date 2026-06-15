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
run_id: Anthropic_F04_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial-neighbor functions, and specifically the way they interact with R's performance characteristics:

### `build_neighbor_lookup` â€” The Primary Bottleneck

1. **Row-level `lapply` over ~6.46 million rows.** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookup (`idx_lookup[neighbor_keys]`). These are all slow in a scalar R loop.
2. **Repeated string allocation.** For every single row, `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates a new character vector. Over 6.46M rows, this generates billions of short-lived string objects, causing enormous GC pressure.
3. **Named vector lookup is O(n) hashing per call.** `idx_lookup[neighbor_keys]` on a named vector of length 6.46M is re-hashing on every access in base R. This is done ~6.46M times.

**Estimated cost:** Each row touches ~4 neighbors on average (rook contiguity). That's ~6.46M Ã— 4 = ~25.8M string constructions and named-vector lookups, on top of the 6.46M `paste` calls for keys. This alone can take tens of hours.

### `compute_neighbor_stats` â€” A Secondary Bottleneck

1. **`lapply` + `do.call(rbind, ...)`** over 6.46M rows: `do.call(rbind, list_of_6.46M_vectors)` is notoriously slow â€” it copies and rebinds incrementally.
2. Called **5 times** (once per source variable), so the cost multiplies.

### Summary

| Component | Calls | Estimated share of 86h |
|---|---|---|
| `build_neighbor_lookup` (string ops, named-vector lookup) | 6.46M | ~60â€“70% |
| `compute_neighbor_stats` (lapply + rbind, Ã—5 vars) | 5 Ã— 6.46M | ~25â€“35% |
| Random Forest prediction | 1 | < 5% |

---

## Optimization Strategy

The core idea: **eliminate all string operations and named-vector lookups; work entirely with integer indices and vectorized/`data.table` operations.**

### Step-by-step plan

1. **Replace the string-keyed lookup with an integer-keyed join.** Build a `data.table` keyed on `(id, year)` â†’ `row_index`. Use a fast equi-join to resolve all neighbor references at once, vectorized.

2. **Expand the neighbor list into a flat edge table once.** Instead of iterating row-by-row, explode `rook_neighbors_unique` into a two-column integer table `(cell_ref, neighbor_ref)`, then join against the panel to produce `(row_i, neighbor_row_i)` in one vectorized pass.

3. **Compute all neighbor stats with grouped `data.table` aggregation.** Group the flat edge table by `row_i`, pull the variable values by `neighbor_row_i`, and compute `max`, `min`, `mean` in one grouped operation â€” no R-level loop.

4. **Process all 5 variables in one pass** over the edge table (or with minimal additional passes), avoiding redundant work.

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes depending on RAM bandwidth), because every operation becomes vectorized C-level code inside `data.table`.

**Numerical equivalence:** The aggregation functions (`max`, `min`, `mean`) are identical; only the iteration strategy changes. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 0. Convert to data.table (if not already); record original order

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx__ := .I]

  # ---------------------------------------------------------------
  # 1. Build integer cell-id â†’ ref-index mapping
  #    id_order[ref_idx] == cell_id
  # ---------------------------------------------------------------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # ---------------------------------------------------------------
  # 2. Expand the nb object into a flat edge list (ref-space)
  #    Each element of rook_neighbors_unique is an integer vector of

  #    neighbor ref-indices for that ref-index.
  # ---------------------------------------------------------------
  n_refs <- length(rook_neighbors_unique)
  # Lengths of each neighbor set
  lens <- lengths(rook_neighbors_unique)
  # "from" ref-index repeated
  from_ref <- rep(seq_len(n_refs), lens)
  # "to" ref-index concatenated
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Convert ref-indices to actual cell ids
  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref, lens)

  # ---------------------------------------------------------------
  # 3. Build a row-index lookup keyed on (id, year)
  # ---------------------------------------------------------------
  row_key <- dt[, .(id, year, row_idx__)]
  setkey(row_key, id, year)

  # ---------------------------------------------------------------
  # 4. Get unique years once
  # ---------------------------------------------------------------
  years <- sort(unique(dt$year))

  # ---------------------------------------------------------------
  # 5. Cross edges Ã— years to get the full (row_i, neighbor_row_j)
  #    mapping.  To keep memory manageable we do this by year.
  # ---------------------------------------------------------------

  # Pre-allocate result columns (NA_real_) in dt for every feature
  stat_names <- c("max", "min", "mean")
  new_cols <- character(0)
  for (v in neighbor_source_vars) {
    for (s in stat_names) {
      col <- paste0(v, "_neighbor_", s)
      new_cols <- c(new_cols, col)
      set(dt, j = col, value = NA_real_)
    }
  }

  # Pre-extract variable vectors for fast indexing
  var_vectors <- setNames(
    lapply(neighbor_source_vars, function(v) dt[[v]]),
    neighbor_source_vars
  )

  # Process year-by-year to bound memory (~344K cells Ã— ~4 neighbors)
  for (yr in years) {
    # Rows in this year
    yr_rows <- row_key[.(unique(edges$from_id), yr), nomatch = 0L,
                        on = .(id, year)]
    setnames(yr_rows, c("id", "year", "row_idx__"),
             c("from_id", "year_from", "row_i"))

    # Join edges to get (from_id -> to_id) for this year
    yr_edges <- edges[yr_rows, on = .(from_id), nomatch = 0L, allow.cartesian = TRUE]
    # yr_edges now has: from_id, to_id, year_from, row_i

    # Resolve to_id + year -> neighbor row index
    yr_edges[, year := year_from]
    neighbor_rows <- row_key[yr_edges, on = .(id = to_id, year), nomatch = 0L]
    # neighbor_rows has: id (=to_id), year, row_idx__ (=neighbor row), row_i

    # For each source variable, compute grouped stats
    for (v in neighbor_source_vars) {
      neighbor_rows[, val__ := var_vectors[[v]][row_idx__]]

      stats <- neighbor_rows[!is.na(val__),
                             .(nmax  = max(val__),
                               nmin  = min(val__),
                               nmean = mean(val__)),
                             keyby = .(row_i)]

      col_max  <- paste0(v, "_neighbor_max")
      col_min  <- paste0(v, "_neighbor_min")
      col_mean <- paste0(v, "_neighbor_mean")

      set(dt, i = stats$row_i, j = col_max,  value = stats$nmax)
      set(dt, i = stats$row_i, j = col_min,  value = stats$nmin)
      set(dt, i = stats$row_i, j = col_mean, value = stats$nmean)
    }
  }

  # ---------------------------------------------------------------
  # 6. Clean up helper column and return
  # ---------------------------------------------------------------
  dt[, row_idx__ := NULL]

  # Return as data.frame if the caller expects one
  if (inherits(cell_data, "data.frame") && !inherits(cell_data, "data.table")) {
    return(as.data.frame(dt))
  }
  return(dt[])
}

# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged; proceed to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor sets (same rook contiguity, same year matching, same `NA` removal). |
| **Trained RF model untouched** | No model retraining. The output columns have the same names and values, so `predict(rf_model, newdata = cell_data)` is unchanged. |
| **Column naming** | Assumes the original `compute_and_add_neighbor_features` created columns named `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean`. Adjust the `paste0` pattern if the original naming convention differs. |

## Expected Performance

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| Build neighbor lookup | ~50â€“60 h | <10 s (vectorized expand) |
| Compute stats (5 vars Ã— 6.46M rows) | ~25â€“30 h | ~2â€“8 min (data.table grouped agg, by year) |
| Peak RAM | Moderate (GC thrash from strings) | ~3â€“5 GB (flat edge table + panel) |
| **Total** | **~86 h** | **~3â€“10 min** |

The key insight: replacing 6.46 million R-level string-paste-and-lookup iterations with a single vectorized `data.table` equi-join eliminates the bottleneck entirely.
END_RESPONSE>>>
