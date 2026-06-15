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
run_id: Anthropic_F04_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clearly in `build_neighbor_lookup`, not `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates over **~6.46 million rows** with an `lapply` call that, for each row:

1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs **string keys** by pasting neighbor IDs with the current year (`paste(..., sep="_")`).
4. Performs **named character vector lookups** (`idx_lookup[neighbor_keys]`) â€” these are hash-table lookups on ~6.46 million-entry named vectors, repeated for every neighbor of every row.

With ~6.46M rows and an average of ~4 rook neighbors each, that's **~25.8 million string constructions and hash lookups**, all inside an interpreted R loop. String allocation, garbage collection, and the overhead of R-level `lapply` make this brutally slow.

**`compute_neighbor_stats`** is a secondary bottleneck: another R-level `lapply` over 6.46M elements, subsetting a numeric vector and computing `max`/`min`/`mean` per row, repeated for 5 variables (i.e., ~32.3M R-level function calls).

**The Random Forest** is already trained and only runs `predict()` once â€” it is not the bottleneck.

---

## Optimization Strategy

### Core Principle: Eliminate all per-row string operations; vectorize everything with `data.table`.

1. **Replace the string-key lookup with an integer join.** Instead of building a 6.46M-entry named character vector and pasting keys, create a `data.table` keyed on `(id, year)` with a pre-assigned integer row index. Then expand the neighbor graph into a two-column edge table `(id, neighbor_id)`, join it with the year dimension, and use `data.table` indexed joins to resolve neighbor row indices in bulk â€” zero per-row string operations.

2. **Replace the per-row `lapply` in `compute_neighbor_stats` with a grouped `data.table` aggregation.** Once we have an edge table `(row_i, neighbor_row_j)`, we can directly index into the variable column, then `group by row_i` and compute `max`, `min`, `mean` in a single vectorized pass per variable.

3. **Memory check:** The edge table will have ~6.46M rows Ã— ~4 neighbors = ~25.8M rows Ã— 2 integer columns â‰ˆ 200 MB. Comfortable within 16 GB.

**Expected speedup:** From 86+ hours to **minutes** (roughly 2â€“10 minutes depending on disk I/O and RAM speed).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table (if not already) and add row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a vectorized edge table from the nb object
#
#   rook_neighbors_unique is a list of length N_cells (344,208).
#   id_order[k] gives the cell id for the k-th element.
#   rook_neighbors_unique[[k]] gives integer indices of neighbors in id_order.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (k in seq_along(neighbors)) {
    nb_k <- neighbors[[k]]
    len_k <- length(nb_k)
    if (len_k == 0L) next
    idx <- pos:(pos + len_k - 1L)
    from_id[idx] <- id_order[k]
    to_id[idx]   <- id_order[nb_k]
    pos <- pos + len_k
  }

  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s rows\n", formatC(nrow(edge_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Expand edges across years and resolve row indices via keyed join
#
#   For every (id, neighbor_id) pair, and for every year in the panel,
#   we need the row index of (id, year) and (neighbor_id, year).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Keyed lookup: cell_data row index by (id, year)
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# Get unique years
years <- sort(unique(cell_data$year))

cat("Expanding edges across years...\n")

# Cross join edges Ã— years
edge_year <- edge_dt[, CJ_idx := TRUE][
  , CJ(edge_row = seq_len(nrow(edge_dt)), year = years)
]
# This CJ approach may be too memory-heavy for 25.8M Ã— 28 = 722M rows.
# Instead, we do it more efficiently: for each row in cell_data, look up its
# neighbors. We join cell_data to edge_dt on 'id', which naturally replicates
# across years.

# More memory-efficient approach:
# Join cell_data (which has id, year, .row_idx) to edge_dt on id.
# This gives us (id, year, .row_idx [of the focal cell], neighbor_id).
# Then join again to row_lookup on (neighbor_id, year) to get neighbor's row idx.

# Clean up the CJ attempt
rm(edge_year)

cat("Joining focal rows to edge table...\n")
focal <- cell_data[, .(id, year, focal_row = .row_idx)]
setkey(edge_dt, id)
setkey(focal, id)

# This join replicates each edge for every year the focal id appears in
# Result columns: id, year, focal_row, neighbor_id
expanded <- edge_dt[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
cat(sprintf("  Expanded edge-year table: %s rows\n",
            formatC(nrow(expanded), big.mark = ",")))

# Now resolve neighbor row indices
setnames(row_lookup, c("id", "year", ".row_idx"),
         c("neighbor_id", "year", "neighbor_row"))
setkey(row_lookup, neighbor_id, year)
setkey(expanded, neighbor_id, year)

cat("Resolving neighbor row indices...\n")
expanded <- row_lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]

# Drop rows where neighbor_row is NA (neighbor cell-year missing from data)
expanded <- expanded[!is.na(neighbor_row)]

cat(sprintf("  Final resolved edge-year table: %s rows\n",
            formatC(nrow(expanded), big.mark = ",")))

# Restore row_lookup names for potential reuse
setnames(row_lookup, c("neighbor_id", "year", "neighbor_row"),
         c("id", "year", ".row_idx"))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Compute neighbor statistics per variable â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  # Pull the variable values into the expanded table via row index
  expanded[, nbr_val := cell_data[[var_name]][neighbor_row]]

  # Aggregate: group by focal_row, compute max/min/mean (excluding NAs)
  agg <- expanded[!is.na(nbr_val),
                  .(nb_max  = max(nbr_val),
                    nb_min  = min(nbr_val),
                    nb_mean = mean(nbr_val)),
                  by = focal_row]

  # Assign back into cell_data by row index
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  # Initialize with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  cell_data[agg$focal_row, (max_col)  := agg$nb_max]
  cell_data[agg$focal_row, (min_col)  := agg$nb_min]
  cell_data[agg$focal_row, (mean_col) := agg$nb_mean]

  # Clean up the temporary column
  expanded[, nbr_val := NULL]

  cat(sprintf("    Done: %s, %s, %s\n", max_col, min_col, mean_col))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cleanup helper column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_idx := NULL]

cat("Neighbor feature construction complete.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (This line is unchanged from the original pipeline.)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code â€” the edge table is derived from the same `rook_neighbors_unique` nb object, and the join logic replicates the original `paste(id, year)`-based matching. No neighbor is added or dropped. |
| **Trained RF model** | The model object is never touched. We only modify the *input feature columns* of `cell_data`, which are the same columns with the same names and identical values. `predict()` is called identically. |
| **NA handling** | Matches the original: neighbors missing from the data (no matching cell-year row) are dropped; variables with all-NA neighbors yield `NA` for all three statistics. |

---

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (6.46M string-key lookups) | ~60â€“80 hours | **Eliminated** |
| `compute_neighbor_stats` (6.46M Ã— 5 R-level lapply iterations) | ~6â€“10 hours | **~1â€“3 min** (vectorized `data.table` groupby) |
| Edge table construction + join | N/A | **~2â€“5 min** |
| Peak RAM for `expanded` (~25.8M Ã— 28 years â‰ˆ 170M rows Ã— 3 int cols) | N/A | **~4â€“6 GB** (fits in 16 GB) |
| **Total** | **86+ hours** | **~5â€“10 minutes** |

The key insight: replacing **millions of interpreted R-level string-paste + hash-lookup operations** with a single **`data.table` keyed equi-join** moves the work from R's interpreter into `data.table`'s optimized C backend, yielding roughly a 500â€“1000Ã— speedup for this stage.
END_RESPONSE>>>
