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
run_id: Anthropic_F06_T00_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration does string pasting, hash lookups in `idx_lookup`, and subsetting. The string-key approach (`paste(id, year, sep="_")`) is extremely expensive at this scale â€” it allocates millions of temporary character vectors and performs millions of named-vector lookups (which are O(n) in base R named vectors, not O(1) hash lookups).

### Bottleneck B: `compute_neighbor_stats` â€” per-row `lapply` over 6.46M entries

- For each of the 5 variables, another `lapply` iterates over 6.46M elements, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also very slow (repeated row-binding).

### Why 86+ hours?

- ~6.46M iterations Ã— 2 passes (build + compute) Ã— 5 variables for compute = ~38.8M R-level loop iterations, each with allocation-heavy string operations or subsetting. The named-vector lookup `idx_lookup[neighbor_keys]` is the single worst offender â€” it's a linear scan per lookup in base R.

### Why raster focal/kernel operations don't directly apply

The comment in the prompt asks whether raster focal operations are a useful analogy. They are conceptually analogous (computing neighborhood statistics), but the data is in **long panel format** (cell Ã— year), not a raster stack. Converting to raster, applying `focal()` per year per variable, and converting back would require reshaping 28 years Ã— 5 variables = 140 focal operations on 344K-cell rasters. This is feasible but introduces complexity around NA handling and irregular grid boundaries. The more direct and faithful optimization is to **vectorize the neighbor computation using sparse matrix multiplication and grouped operations**, which preserves the exact numerical results.

---

## 2. Optimization Strategy

### Step 1: Replace string-keyed lookup with integer indexing via `data.table`

Use `data.table` keyed joins to map `(id, year)` â†’ row index in O(1) amortized time.

### Step 2: Build a sparse adjacency matrix (cell-level), then expand to cell-year level

- The rook neighbor structure is **time-invariant**: cell *i*'s neighbors are the same in every year. So we build a 344,208 Ã— 344,208 sparse adjacency matrix **once**, then use it to compute all neighbor statistics via sparse matrixâ€“vector products.
- For `mean`: neighbor mean of variable `x` for cell `i` = `(A %*% x) / (A %*% 1)` where `A` is the adjacency matrix (applied within each year).
- For `max` and `min`: sparse matrix multiplication gives the sum, not max/min. We handle these with a **grouped operation** using `data.table` â€” explode the neighbor pairs, join the variable values, and compute `max`/`min`/`mean` grouped by `(id, year)`.

### Step 3: Vectorized grouped computation with `data.table`

- Build an edge table: `(id, neighbor_id)` from the `nb` object (~1.37M directed edges).
- Cross-join with years to get `(id, year, neighbor_id)` â€” ~1.37M Ã— 28 = ~38.5M rows.
- Join neighbor variable values from the main data.
- Group by `(id, year)`, compute `max`, `min`, `mean`.
- Join results back to the main data.

This replaces 6.46M R-level iterations with a single vectorized `data.table` grouped aggregation. Expected runtime: **minutes, not hours**.

### Memory check

- Edge table expanded: ~38.5M rows Ã— 3 integer columns â‰ˆ 460 MB
- With one double-precision variable column joined: +308 MB
- Total working memory per variable: ~800 MB â€” fits in 16 GB RAM.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert main data to data.table and key it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]
setkey(cell_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the directed edge list from the nb object (time-invariant)
#
#   rook_neighbors_unique is an nb object of length 344,208.
#   id_order is the vector mapping position in the nb list â†’ cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, nb_obj) {
  # nb objects store neighbor indices as integer vectors; 0L means no neighbors
  from_ids <- rep(id_order, times = lengths(nb_obj))
  to_positions <- unlist(nb_obj, use.names = FALSE)

  # spdep uses 0L (integer 0) for cells with no neighbors â€” remove those
  valid <- to_positions > 0L
  from_ids <- from_ids[valid]
  to_ids <- id_order[to_positions[valid]]

  data.table(id = from_ids, neighbor_id = to_ids)
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edges), "\n")
# Expected: ~1,373,394

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Expand edges across all years and compute neighbor stats
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(cell_dt$year))  # 1992:2019

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-build the (id, year, neighbor_id) table once â€” ~38.5M rows
# To save memory, we process one year at a time inside the variable loop.

# Prepare a keyed lookup table for joining variable values
# We'll subset columns as needed.

compute_and_add_all_neighbor_features <- function(cell_dt, edges, years,
                                                   neighbor_source_vars) {
  # For each variable, process all years in a single vectorized operation
  setkey(cell_dt, id, year)

  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor features for:", var_name, "\n")
    t0 <- proc.time()

    # Extract only the columns we need for the join
    val_dt <- cell_dt[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # Process in yearly chunks to limit peak memory (~1.37M edges per year)
    result_list <- vector("list", length(years))

    for (yi in seq_along(years)) {
      yr <- years[yi]

      # All edges for this year: join neighbor values
      # edges has (id, neighbor_id); we need val for each neighbor_id in this year
      yr_edges <- copy(edges)
      yr_edges[, year := yr]

      # Join to get neighbor's variable value
      setkey(yr_edges, neighbor_id, year)
      yr_edges[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]

      # Compute grouped stats: max, min, mean (excluding NAs)
      stats <- yr_edges[!is.na(neighbor_val),
                         .(nb_max = max(neighbor_val),
                           nb_min = min(neighbor_val),
                           nb_mean = mean(neighbor_val)),
                         by = .(id)]
      stats[, year := yr]
      result_list[[yi]] <- stats
    }

    # Combine all years
    all_stats <- rbindlist(result_list, use.names = TRUE)

    # Rename columns to match expected output names
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(all_stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    # Join back to main data
    setkey(all_stats, id, year)
    setkey(cell_dt, id, year)

    # Remove old columns if they exist (for idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }

    cell_dt <- all_stats[cell_dt, on = .(id, year)]

    elapsed <- (proc.time() - t0)[3]
    cat("  Done in", round(elapsed, 1), "seconds\n")
  }

  return(cell_dt)
}

cell_dt <- compute_and_add_all_neighbor_features(
  cell_dt, edges, years, neighbor_source_vars
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Convert back to data.frame if needed for the RF predict step
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, row_idx := NULL]  # clean up helper column
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Further Optimization: Fully Vectorized (No Year Loop)

If memory permits (~2â€“3 GB working set), you can eliminate the year loop entirely:

```r
compute_neighbor_features_fully_vectorized <- function(cell_dt, edges,
                                                        neighbor_source_vars) {
  setkey(cell_dt, id, year)

  # Expand edges Ã— years in one shot
  yr_dt <- CJ(year = sort(unique(cell_dt$year)),
               edge_idx = seq_len(nrow(edges)))
  yr_dt[, `:=`(id = edges$id[edge_idx],
               neighbor_id = edges$neighbor_id[edge_idx])]
  yr_dt[, edge_idx := NULL]
  # ~38.5M rows

  for (var_name in neighbor_source_vars) {
    cat("Processing:", var_name, "\n")
    t0 <- proc.time()

    val_dt <- cell_dt[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # Join neighbor values
    setkey(yr_dt, neighbor_id, year)
    yr_dt[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]

    # Grouped aggregation
    stats <- yr_dt[!is.na(neighbor_val),
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(id, year)]

    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    setkey(stats, id, year)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }
    cell_dt <- stats[cell_dt, on = .(id, year)]

    yr_dt[, neighbor_val := NULL]  # free for next variable

    cat("  Done in", round((proc.time() - t0)[3], 1), "s\n")
  }

  return(cell_dt)
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + named-vector lookups | Integer edge list from `nb` object, O(n) |
| **Neighbor stats** | 6.46M `lapply` iterations Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| **`do.call(rbind, ...)`** | 6.46M-element list row-bind | `rbindlist` (C-level) |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Peak memory** | Moderate (but slow) | ~2â€“4 GB (fits in 16 GB) |
| **Numerical results** | `max`, `min`, `mean` of rook neighbors | **Identical** â€” same operations, same NA handling |
| **RF model** | N/A | **Unchanged â€” no retraining** |

The key insight: the neighbor structure is **time-invariant**, so the expensive per-row string-key lookup is entirely unnecessary. By expressing the problem as a join between an edge table and the panel data, followed by a grouped aggregation, we replace millions of R-level loop iterations with a handful of vectorized C-level `data.table` operations.
END_RESPONSE>>>
