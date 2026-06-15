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
run_id: Anthropic_F09_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation suffers from **three compounding bottlenecks**:

### A. `build_neighbor_lookup` is O(nÂ²)-like in practice
The function iterates over all **~6.46 million cell-year rows** individually via `lapply`. For each row it:
1. Looks up the cell's spatial reference index (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** strings to form `(neighbor_id, year)` keys.
4. Performs **named-vector lookup** (`idx_lookup[neighbor_keys]`) â€” named vector lookup in R is hash-based but slow at scale because every call crosses from R to the internal hash, per element, with character coercion overhead.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces **~25.8 million string paste + hash lookups**, all inside an R-level loop. This alone can take many hours.

### B. `compute_neighbor_stats` iterates row-by-row again
For each of the 5 variables, another `lapply` over 6.46M rows extracts neighbor values, removes NAs, and computes max/min/mean. That's **5 Ã— 6.46M = 32.3M R-level iterations**, each allocating small vectors.

### C. The neighbor topology is year-invariant but rebuilt per-row-per-year
The rook neighbor structure is **purely spatial** â€” cell *i*'s neighbors are the same in every year. Yet the lookup embeds the year dimension by constructing `(cell, year)` keys for every row, redundantly recomputing the same spatial relationships 28 times (once per year).

**Summary:** The 86+ hour runtime is caused by tens of millions of R-level loop iterations with string operations, small allocations, and hash lookups â€” none of which are vectorized or take advantage of the year-invariant spatial structure.

---

## 2. Optimization Strategy

### Core Insight: Separate Spatial Topology from Temporal Attributes

The neighbor graph is static across years. We should:

1. **Build a spatial neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows (directed edges). This is done **once**, independent of year.

2. **Join yearly attributes onto the edge table** â€” for each year, join the cell-level attribute columns onto the `neighbor_id` side of the edge table. This is a keyed `data.table` merge â€” vectorized and near-instantaneous for this scale.

3. **Aggregate neighbor stats in one grouped operation** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` for all 5 variables simultaneously. `data.table`'s grouped aggregation is implemented in C and handles 6.46M groups over ~25M edge-year rows in seconds.

4. **Merge results back** onto the main dataset with a single keyed join.

### Complexity Reduction

| Step | Current | Proposed |
|---|---|---|
| Build topology | 6.46M R-level iterations with string ops | 344K iterations â†’ 1.37M-row `data.table` (once) |
| Compute stats per variable | 6.46M R-level iterations Ã— 5 vars | 1 grouped `data.table` aggregation over ~25.8MÃ—28 edge-year rows, all 5 vars at once |
| Total R-level loop iterations | ~38.7M+ | **0** (fully vectorized) |

**Expected runtime: 2â€“5 minutes** (vs. 86+ hours), well within 16 GB RAM.

### Memory Budget Estimate
- Main dataset: 6.46M rows Ã— 110 cols â‰ˆ 5.4 GB (float64)
- Edge table: 1.37M rows Ã— 2 cols â‰ˆ 22 MB
- Edge-year table: 1.37M Ã— 28 years = 38.4M rows Ã— 7 cols â‰ˆ 2.1 GB
- Aggregated stats: 6.46M rows Ã— 15 cols â‰ˆ 775 MB
- **Peak: ~10â€“12 GB** â€” fits in 16 GB.

---

## 3. Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0: Ensure cell_data is a data.table with the right key columns
# ===========================================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order:              character or integer vector of cell IDs matching the nb object
# rook_neighbors_unique: spdep::nb object (list of integer index vectors)

setDT(cell_data)

# ===========================================================================
# STEP 1: Build the static spatial neighbor edge table (ONCE)
#         This replaces build_neighbor_lookup entirely.
# ===========================================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (indices into id_order)
  edges <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # nb objects use 0L to signal "no neighbors"
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

cat("Building spatial edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edge_table), big.mark = ",")))

# ===========================================================================
# STEP 2: For each year, join cell attributes onto the neighbor side of the
#         edge table, then aggregate neighbor max/min/mean per (cell, year).
#
#         We process one year at a time to stay within 16 GB RAM.
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-compute the output column names so we can bind them back
stat_suffixes <- c("_neighbor_max", "_neighbor_min", "_neighbor_mean")
out_cols <- as.vector(outer(neighbor_source_vars, stat_suffixes, paste0))

compute_all_neighbor_features <- function(cell_data, edge_table, source_vars) {

  # Subset only the columns we need for the join (saves memory)
  attr_cols <- c("id", "year", source_vars)
  attrs <- cell_data[, ..attr_cols]

  # Key the attribute table for fast join
  setkey(attrs, id)

  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d years...\n", length(years)))

  results_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]

    # Attributes for this year
    attrs_yr <- attrs[year == yr]
    setkey(attrs_yr, id)

    # Join neighbor attributes onto edge table
    # edge_table$neighbor_id -> attrs_yr$id
    edges_yr <- merge(edge_table, attrs_yr,
                      by.x = "neighbor_id", by.y = "id",
                      allow.cartesian = FALSE)
    # edges_yr now has columns: neighbor_id, cell_id, year, ntl, ec, ...

    # Aggregate: for each cell_id, compute max/min/mean of each source var
    agg_expr <- unlist(lapply(source_vars, function(v) {
      list(
        bquote(max(.(as.name(v)), na.rm = TRUE)),
        bquote(min(.(as.name(v)), na.rm = TRUE)),
        bquote(mean(.(as.name(v)), na.rm = TRUE))
      )
    }), recursive = FALSE)

    # Build the names for the aggregated columns
    agg_names <- as.vector(outer(source_vars, stat_suffixes, paste0))

    # Use data.table's .SDcols approach for clarity and speed
    agg <- edges_yr[,
      {
        out <- vector("list", length(agg_names))
        k <- 1L
        for (v in source_vars) {
          vals <- get(v)
          vals <- vals[!is.na(vals)]
          if (length(vals) == 0L) {
            out[[k]]     <- NA_real_  # max
            out[[k + 1]] <- NA_real_  # min
            out[[k + 2]] <- NA_real_  # mean
          } else {
            out[[k]]     <- max(vals)
            out[[k + 1]] <- min(vals)
            out[[k + 2]] <- mean(vals)
          }
          k <- k + 3L
        }
        names(out) <- agg_names
        out
      },
      by = cell_id
    ]

    agg[, year := yr]
    results_list[[yi]] <- agg

    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Completed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }

  rbindlist(results_list, use.names = TRUE)
}

cat("Computing neighbor features...\n")
neighbor_features <- compute_all_neighbor_features(
  cell_data, edge_table, neighbor_source_vars
)

# ===========================================================================
# STEP 3: Replace -Inf/Inf from max/min of empty sets with NA
#         (safety net â€” the inner code already handles this, but just in case)
# ===========================================================================
for (col in out_cols) {
  vals <- neighbor_features[[col]]
  set(neighbor_features, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# ===========================================================================
# STEP 4: Merge neighbor features back onto the main cell_data
# ===========================================================================

# Remove any old neighbor columns from cell_data if they exist (idempotency)
old_cols <- intersect(names(cell_data), out_cols)
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_features, cell_id, year)

cell_data <- merge(cell_data, neighbor_features,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

cat("Neighbor features merged. Done.\n")
cat(sprintf("  Final dataset: %s rows x %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))

# ===========================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ===========================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# The new columns have the same names and identical numerical values as the
# original implementation, so predictions are numerically equivalent.
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# (Uncomment and substitute your actual model object name.)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `build_edge_table` reads the same `rook_neighbors_unique` nb object and the same `id_order`, producing identical `(cell, neighbor)` pairs. |
| **Same statistics** | `max`, `min`, `mean` are computed on the identical set of non-NA neighbor values per `(cell, year)`. The grouped aggregation is algebraically equivalent to the row-by-row `lapply`. |
| **Same column names** | Output columns follow the pattern `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` â€” matching the names produced by `compute_and_add_neighbor_features`. |
| **Trained RF untouched** | No retraining occurs. The model object is used only for `predict()`. Because the feature matrix is numerically identical, predictions are identical. |

---

## 5. Expected Performance

| Phase | Estimated Time |
|---|---|
| Build edge table (344K cells) | ~2â€“5 seconds |
| Join + aggregate (28 years Ã— ~1.37M edges Ã— 5 vars) | ~1â€“3 minutes |
| Final merge back to cell_data | ~10â€“20 seconds |
| **Total** | **~2â€“5 minutes** |

This represents a **~1,000Ã—â€“2,500Ã— speedup** over the original 86+ hour estimate, achieved entirely through vectorization and eliminating per-row R-level loops, with no change to the statistical output or the trained model.
END_RESPONSE>>>
