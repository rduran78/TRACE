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
run_id: Anthropic_F09_T00_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it doesn't need to be.

The neighbor topology is **purely spatial** â€” it never changes across years. Yet the function builds a lookup keyed on `(id, year)` pairs, iterating over every one of the ~6.46 million rows. For each row it does string-pasting, hash-table lookups, and NA filtering. This means the same neighbor set for a given cell is redundantly resolved 28 times (once per year). The `paste()`/`setNames()` pattern over 6.46M keys is itself expensive (large character vector hashing).

### 2. `compute_neighbor_stats` uses an R-level `lapply` over 6.46M rows.

Each iteration indexes into a numeric vector, removes NAs, and computes `max/min/mean`. While each call is small, 6.46M R-level function calls with allocation of small vectors is brutally slow. This is repeated for each of the 5 neighbor source variables â†’ ~32.3M R-level iterations total.

### Summary of bottlenecks

| Component | Calls | Cost driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations | String pasting, hash lookup per cell-year |
| `compute_neighbor_stats` | 6.46M Ã— 5 vars | R-level loop, small vector alloc per row |

---

## Optimization Strategy

### Core insight: Separate topology from attributes, then vectorize.

1. **Build the neighbor table once at the cell level (not cell-year level).** The ~344K cells have ~1.37M directed rook-neighbor pairs. Store this as a two-column `data.table`: `(cell_id, neighbor_id)`. This is built once and is tiny (~11 MB).

2. **Join yearly attributes onto the neighbor table.** For each year and each variable, join the cell-year attribute onto the neighbor side of the edge table. Then use `data.table` grouped aggregation (`[, .(max, min, mean), by = cell_id]`) to compute neighbor stats in one vectorized pass. This replaces millions of R-level `lapply` iterations with `data.table`'s optimized C-level grouped operations.

3. **Avoid string keys entirely.** Use integer cell IDs and integer years as join keys â€” no `paste()`.

### Expected speedup

- The neighbor edge table has ~1.37M rows. For each of 28 years Ã— 5 variables = 140 passes, we do a `data.table` join + grouped aggregation over ~1.37M rows. Total work: ~192M row-operations, all in C-level vectorized code.
- Estimated time: **2â€“10 minutes** on a standard laptop (vs. 86+ hours).

### What is preserved

- The trained Random Forest model is untouched (no retraining).
- The numerical output (neighbor max, min, mean per cell-year per variable) is identical â€” same estimand, same values.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the spatial neighbor edge table ONCE (cell-level only)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inputs:
#   id_order             â€” integer vector of cell IDs (length 344,208),
#                          ordered to match the spdep::nb object indices.
#   rook_neighbors_unique â€” spdep::nb object (list of length 344,208),
#                          each element is an integer vector of neighbor
#                          indices into id_order.
#
# Output:
#   neighbor_edges â€” data.table with columns (cell_id, neighbor_id),
#                    ~1,373,394 rows (directed rook-neighbor pairs).

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate vectors for speed
  n_cells <- length(id_order)
  # Count total edges
  n_edges <- sum(vapply(neighbors_nb, length, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors_nb[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

# Verify
cat("Neighbor edge table rows:", nrow(neighbor_edges), "\n")
# Expected: ~1,373,394


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data must have columns: id, year, and the neighbor source variables.

cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all variables, all years at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# For each variable, we:
#   (a) Take the neighbor edge table (~1.37M rows).
#   (b) Cross-join it with all 28 years â†’ ~38.5M rows (but we do this
#       implicitly via a join, not a literal cross-join, to save memory).
#   (c) Join the variable value from cell_dt onto the neighbor side.
#   (d) Group by (cell_id, year) and compute max, min, mean.
#   (e) Join the results back onto cell_dt.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Get the unique years present in the data
all_years <- sort(unique(cell_dt$year))

compute_and_merge_neighbor_features <- function(cell_dt, neighbor_edges,
                                                 var_name, all_years) {
  # Column names for the output
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Extract only the columns we need for the join: (id, year, var_name)
  # Rename var_name to "val" for convenience
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Strategy: loop over years (28 iterations â€” trivial) to keep memory
  # bounded. For each year, join neighbor_edges with val_dt for that year,
  # then aggregate.

  results_list <- vector("list", length(all_years))

  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]

    # Subset values for this year
    yr_vals <- val_dt[year == yr, .(id, val)]
    setkey(yr_vals, id)

    # Join neighbor values: for each edge (cell_id, neighbor_id),
    # look up the neighbor's value
    # edges_with_vals has columns: cell_id, neighbor_id, val
    edges_with_vals <- yr_vals[neighbor_edges, on = .(id = neighbor_id),
                               nomatch = NA,
                               .(cell_id, val = x.val)]

    # Aggregate by cell_id
    agg <- edges_with_vals[!is.na(val),
                           .(nmax  = max(val),
                             nmin  = min(val),
                             nmean = mean(val)),
                           by = cell_id]
    agg[, year := yr]

    results_list[[yi]] <- agg
  }

  # Combine all years
  all_results <- rbindlist(results_list, use.names = TRUE)
  setnames(all_results, c("nmax", "nmin", "nmean"),
           c(col_max, col_min, col_mean))

  # Merge back onto cell_dt
  setkeyv(all_results, c("cell_id", "year"))
  setkeyv(cell_dt, c("id", "year"))

  # Remove old columns if they exist (in case of re-run)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_dt)) cell_dt[, (cc) := NULL]
  }

  cell_dt <- all_results[cell_dt, on = .(cell_id = id, year = year)]

  # The join puts cell_id as the key column; rename back to id
  setnames(cell_dt, "cell_id", "id")

  cell_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Run for all 5 variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_dt <- compute_and_merge_neighbor_features(
    cell_dt, neighbor_edges, var_name, all_years
  )
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds.\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Convert back to data.frame if needed for the RF predict call
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: Predict with the existing (already-trained) Random Forest
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., rf_model) is unchanged.
# Predictions use the same columns as before, now computed much faster.

# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `build_neighbor_edge_table` reads the same `rook_neighbors_unique` nb object and the same `id_order`, producing identical directed pairs. |
| **Same aggregation** | `max`, `min`, `mean` are computed over the same non-NA neighbor values per `(cell_id, year)` group. Rows with zero non-NA neighbors get `NA` via the left join (`nomatch = NA` + the `!is.na(val)` filter leaves them absent from `agg`, and the final right join fills them with `NA`). |
| **Same estimand** | The RF model is never retrained. The 15 new columns (3 stats Ã— 5 vars) are numerically identical to the originals. All ~110 predictor columns are preserved. |

## Expected Performance

| Step | Rows processed | Estimated time |
|---|---|---|
| Build edge table | 1.37M edges | < 2 seconds |
| Per variable, per year: join + aggregate | 1.37M Ã— 28 = 38.5M per var | ~5â€“15 sec per var |
| All 5 variables | ~192M total | **~1â€“3 minutes total** |
| RF prediction (unchanged) | 6.46M rows | Same as before |

**Total: ~2â€“10 minutes** vs. the original ~86+ hours â€” roughly a **500â€“2,500Ã— speedup**, achieved entirely by (a) separating spatial topology from temporal attributes, (b) replacing R-level `lapply` over 6.46M rows with `data.table` vectorized joins and grouped aggregations in C, and (c) eliminating all string-key operations.
END_RESPONSE>>>
